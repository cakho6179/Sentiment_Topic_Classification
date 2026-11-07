# Vietnamese Brand-Conditioned Sentiment & Topic Classification

Hệ thống phân loại cảm xúc theo từng thương hiệu và phân loại chủ đề cho comment social listening tiếng Việt.

> **Trạng thái hiện tại:** source code cho toàn bộ pipeline đã có; baseline TF-IDF đã chạy xong 3 seed; encoder/MTL/fusion/decoder đã smoke-test; **chưa chạy full grid trên Kaggle** và **chưa có báo cáo kết quả cuối cùng**. Dataset hiện tại là dữ liệu synthetic 1000 dòng, nên điểm gần 1.0 không đại diện cho production.

## 1. Kết quả đã có

### Baseline TF-IDF + Linear SVM (test set, seed 42)

| Task | Variant | Macro-F1 | Accuracy |
|---|---|---:|---:|
| Sentiment | `full` | 0.9636 | 0.9653 |
| Sentiment | `full_masked_brand` | **0.9709** | **0.9722** |
| Sentiment | `brand_target` | 0.9575 | 0.9583 |
| Sentiment | `target_only` | 0.9512 | 0.9514 |
| Sentiment | `brand_only` | 0.3031 | 0.3125 |
| Sentiment | `context_only` | 0.2929 | 0.3333 |
| Topic | `target_context` | 0.9853 | 0.9854 |
| Topic | `target_only` | 0.9930 | 0.9927 |
| Topic | `target_masked_brand` | **1.0000** | **1.0000** |
| Topic | `context_only` | 0.5643 | 0.5036 |

Nhận xét nhanh:

- Topic gần như deterministic từ `target`; context một mình rất yếu.
- Brand-only gần như đoán đều, nên brand conditioning chưa được kiểm chứng mạnh trên synthetic data.
- `full_masked_brand` nhỉnh hơn `full` ở sentiment, cho thấy model có thể đang dựa vào shortcut thương hiệu.

Các file kết quả:

- `reports/benchmark.md`, `reports/benchmark.csv`
- `reports/error_analysis.md`
- `reports/error_examples_sentiment.csv`, `reports/error_examples_topic.csv`
- `experiments/*_metrics.json`
- `experiments/*_predictions.csv`
- `experiments/*_val_predictions.csv`

## 2. Cấu trúc chính

```text
src/data/                 parse, label mapping, build dataset, leakage-aware split
src/preprocessing/        normalize, build text, Vietnamese word segmentation
src/models/               TF-IDF, encoder, MTL, fusion, decoder helpers
src/training/             train_tfidf, train_encoder, train_mtl, train_fusion, train_decoder
src/evaluation/           metrics, bootstrap, McNemar, error analysis, hybrid
src/inference/            unified inference engine
scripts/                  chạy suite, aggregate, error analysis, hybrid, inference
configs/                  ví dụ config inference
data/processed/           comments.csv, comment_brands.csv
data/splits/folds.csv     fold/group assignment
reports/                  audit, benchmark, error analysis
experiments/              metrics, predictions, model artifacts
```

## 3. Cài đặt

Khuyến nghị Python 3.11+.

```powershell
python -m pip install -r requirements.txt
```

`bitsandbytes` chỉ cần cho QLoRA trên Linux/CUDA. Trên Windows, có thể bỏ qua và chạy decoder bằng LoRA không lượng tử hoá.

## 4. Chạy lại toàn bộ baseline

```powershell
$env:PYTHONIOENCODING='utf-8'

python -m src.data.build_dataset `
  --raw data/raw/Vietnamese_Social_Listening_Synthetic_1000.xlsx `
  --out data/processed `
  --splits data/splits `
  --reports reports

python -m scripts.run_baseline `
  --processed-dir data/processed `
  --output-dir experiments `
  --reports-dir reports `
  --seeds 42 43 44

python -m scripts.aggregate_results `
  --experiments-dir experiments `
  --reports-dir reports

python -m scripts.run_error_analysis `
  --experiments-dir experiments `
  --processed-dir data/processed `
  --reports-dir reports `
  --benchmark reports/benchmark.csv
```

## 5. Chạy encoder

Ví dụ ViSoBERT cho sentiment:

```powershell
python -m src.training.train_encoder `
  --task sentiment `
  --variant full `
  --encoder visobert `
  --epochs 8 `
  --batch-size 16 `
  --eval-batch-size 32 `
  --max-length 256 `
  --save-model
```

Ví dụ PhoBERT cho topic (tự động dùng `pyvi`):

```powershell
python -m src.training.train_encoder `
  --task topic `
  --variant target_context `
  --encoder phobert `
  --epochs 8 `
  --batch-size 16 `
  --max-length 256
```

Các encoder có sẵn trong registry: `visobert`, `phobert`, `xlmr`, `bert_tiny`, `tiny_bert`.

Chạy cả suite và resume:

```powershell
python -m scripts.run_encoder_suite `
  --encoders visobert phobert `
  --tasks sentiment topic `
  --seeds 42 `
  --output-dir experiments
```

Thêm `--force` để chạy lại experiment đã có metrics.

## 6. Chạy multi-task shared encoder

```powershell
python -m src.training.train_mtl `
  --encoder visobert `
  --epochs 8 `
  --batch-size 16 `
  --lambda-topic 1.0 `
  --lambda-sentiment 1.0 `
  --save-model
```

Hoặc chạy suite 3 seed:

```powershell
python -m scripts.run_mtl_suite `
  --encoders visobert phobert `
  --seeds 42 43 44 `
  --output-dir experiments
```

Loss topic chỉ được tính trên `comments.csv` (một dòng / unique comment), không lặp theo từng brand.

## 7. Chạy feature fusion (text encoder + brand embedding)

```powershell
python -m src.training.train_fusion `
  --encoder visobert `
  --text-variant target_context `
  --epochs 8 `
  --save-model
```

Hoặc:

```powershell
python -m scripts.run_fusion_suite `
  --encoders visobert phobert `
  --text-variants target_context target_only `
  --seeds 42 43 44
```

## 8. Chạy decoder QLoRA

Trên Kaggle/Linux:

```bash
pip install -U bitsandbytes
python -m src.training.train_decoder \
  --model Qwen/Qwen3.5-4B \
  --load-in-4bit \
  --epochs 3 \
  --batch-size 2 \
  --grad-accum 8 \
  --eval-batch-size 8 \
  --lr 2e-4 \
  --max-length 512 \
  --save-model
```

Nếu Qwen3.5 gặp lỗi multimodal/text-only, thử:

```bash
python -m src.training.train_decoder --model Qwen/Qwen3-4B --load-in-4bit ...
```

Smoke test decoder bằng model nhỏ:

```powershell
python -m src.training.train_decoder `
  --model hf-internal-testing/tiny-random-gpt2 `
  --epochs 1 --batch-size 1 --grad-accum 1 `
  --max-length 256 --smoke `
  --output-dir experiments_smoke
```

## 9. Inference

### Với TF-IDF đã có

```powershell
python -m scripts.run_inference `
  --config configs/inference_tfidf.json `
  --input experiments/inference_sample.csv `
  --output experiments/inference_sample_predictions.csv
```

Input CSV cần tối thiểu:

```text
comment_id,brand,target,context
C000001,VinFast,Xe chạy rất tốt,...
```

Output mỗi dòng:

```json
{
  "comment_id": "C000001",
  "brand": "VinFast",
  "sentiment": {"label": 1, "name": "Positive", "confidence": 0.98, "low_confidence": false},
  "topic": {"label": 0, "topic_id": "T01", "name": "Pin & Sạc", "confidence": 0.95, "low_confidence": false}
}
```

Topic được dự đoán một lần cho mỗi `comment_id`; sentiment dự đoán theo từng `(comment_id, brand)`.

### Với encoder đã lưu

```json
{
  "sentiment": {
    "type": "encoder",
    "model_dir": "experiments/models/visobert_sentiment_full_seed42"
  },
  "topic": {
    "type": "encoder",
    "model_dir": "experiments/models/visobert_topic_target_context_seed42"
  },
  "sentiment_threshold": 0.6,
  "topic_threshold": 0.6
}
```

### Với MTL đã lưu

```json
{
  "mtl": {
    "model_dir": "experiments/models/mtl_visobert_lt1.0_ls1.0_seed42"
  },
  "sentiment_threshold": 0.6,
  "topic_threshold": 0.6
}
```

Lưu ý: encoder/MTL inference backend đã được code nhưng mới test end-to-end với TF-IDF; cần test lại sau khi có model lưu từ Kaggle.

## 10. Hybrid encoder + decoder

Sau khi có prediction của 2 model:

```powershell
python -m scripts.run_hybrid `
  --task sentiment `
  --primary visobert_sentiment_full_seed42 `
  --verifier decoder_Qwen_Qwen3.5-4B_seed42 `
  --experiments-dir experiments `
  --reports-dir reports
```

Hybrid dùng validation set để chọn `threshold` và `margin`, sau đó mới đánh giá test.

## 11. Những việc còn lại

- [ ] Chạy full encoder/MTL/fusion/decoder trên Kaggle.
- [ ] Test end-to-end inference backend cho encoder/MTL/decoder.
- [ ] Bổ sung `pytest` unit tests.
- [ ] Đóng gói private Kaggle dataset + kernel.
- [ ] Tổng hợp benchmark cuối, calibration, McNemar, hybrid.
- [ ] Viết báo cáo kết quả cuối bằng tiếng Việt.
- [ ] Kiểm tra lại privacy: không publish dữ liệu nội bộ.

## 12. Lưu ý dữ liệu

- Dataset hiện tại là synthetic 1000 dòng, không phải dữ liệu production.
- Split đã chống leakage theo `thread_id`, exact duplicate và near-duplicate.
- Không dùng kết quả gần 1.0 để kết luận model tổng quát tốt.
- Cần chạy lại trên dữ liệu thật trước khi triển khai.
