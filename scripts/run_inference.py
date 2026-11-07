"""Run the unified inference engine from a config file and CSV input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.inference.engine import InferenceEngine


def load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(text)
    return json.loads(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    engine = InferenceEngine.from_config(config)
    frame = pd.read_csv(args.input)
    records = frame.to_dict(orient="records")
    outputs = engine.predict_batch(records, strict=args.strict)
    output_frame = pd.DataFrame(outputs)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_csv(args.output, index=False, encoding="utf-8")
    print(output_frame.to_string(index=False))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
