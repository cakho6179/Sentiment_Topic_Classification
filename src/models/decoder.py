"""Decoder-only classification helpers (prompting, parsing, generation)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from src.preprocessing.text import build_decoder_prompt

SYSTEM_PROMPT = (
    "You are a Vietnamese social-listening classifier. "
    "Return exactly one label from the allowed set and nothing else."
)

SENTIMENT_WORDS = {
    "negative": 0,
    "positive": 1,
    "neutral": 2,
    "tieu cuc": 0,
    "tich cuc": 1,
    "trung lap": 2,
}


def build_chat_prompt(
    tokenizer,
    *,
    task: str,
    target: str,
    context: str = "",
    brand: str | None = None,
) -> str:
    user_prompt = build_decoder_prompt(task, target, context, brand)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        except Exception:
            pass
    return f"System: {SYSTEM_PROMPT}\nUser: {user_prompt}\nAssistant:"


def label_to_answer(task: str, label: int) -> str:
    if task == "sentiment":
        return str(int(label))
    if task == "topic":
        return f"T{int(label) + 1:02d}"
    raise ValueError(task)


def parse_answer(task: str, text: str) -> int | None:
    text = str(text).strip()
    if task == "sentiment":
        match = re.search(r"\b([012])\b", text)
        if match:
            return int(match.group(1))
        lowered = text.lower()
        for word, label in SENTIMENT_WORDS.items():
            if word in lowered:
                return label
        return None
    if task == "topic":
        match = re.search(r"T\s*0?(\d{1,2})\b", text, flags=re.I)
        if match:
            value = int(match.group(1))
            if 1 <= value <= 13:
                return value - 1
        match = re.search(r"\b(1[0-3]|[1-9])\b", text)
        if match:
            value = int(match.group(1))
            if 1 <= value <= 13:
                return value - 1
        return None
    raise ValueError(task)


@dataclass
class DecoderExample:
    task: str
    prompt: str
    answer: str
    meta: dict[str, Any]


class DecoderDataset(torch.utils.data.Dataset):
    def __init__(self, examples: Sequence[DecoderExample], tokenizer, max_length: int = 512) -> None:
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        prompt_ids = self.tokenizer.encode(example.prompt, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(example.answer, add_special_tokens=False)
        eos = self.tokenizer.eos_token_id
        answer_ids = answer_ids + ([eos] if eos is not None else [])
        max_prompt = max(1, self.max_length - len(answer_ids))
        if len(prompt_ids) > max_prompt:
            prompt_ids = prompt_ids[-max_prompt:]
        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }


class DecoderCollator:
    def __init__(self, tokenizer) -> None:
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id

    def __call__(self, features: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_mask = []
        labels = []
        for feature in features:
            pad = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad)
            attention_mask.append(feature["attention_mask"] + [0] * pad)
            labels.append(feature["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


@torch.no_grad()
def generate_predictions(
    model,
    tokenizer,
    examples: Sequence[DecoderExample],
    *,
    batch_size: int = 8,
    max_new_tokens: int = 8,
    max_length: int = 512,
) -> list[dict[str, Any]]:
    model.eval()
    device = next(model.parameters()).device
    outputs: list[dict[str, Any]] = []
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    effective_max_length = max(1, int(max_length) - int(max_new_tokens))
    try:
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            prompts = [example.prompt for example in batch]
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=effective_max_length,
                add_special_tokens=False,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            if model.config.pad_token_id is None:
                model.config.pad_token_id = tokenizer.pad_token_id
            if getattr(model, "generation_config", None) is not None:
                model.generation_config.pad_token_id = tokenizer.pad_token_id
                model.generation_config.eos_token_id = tokenizer.eos_token_id
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            prompt_length = encoded["input_ids"].shape[1]
            new_tokens = generated[:, prompt_length:]
            decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            for example, text in zip(batch, decoded):
                parsed = parse_answer(example.task, text)
                outputs.append(
                    {
                        **example.meta,
                        "task": example.task,
                        "raw_output": text,
                        "parsed_label": parsed,
                        "invalid": parsed is None,
                    }
                )
    finally:
        tokenizer.padding_side = old_padding_side
    return outputs
