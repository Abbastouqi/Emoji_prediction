# Large-Scale Multi-Label Emoji Prediction for Urdu

Code and reproduction materials for the paper *"Large-Scale Multi-Label Emoji
Prediction for Urdu Social Media Text Using Transformer Encoders and Asymmetric
Loss."*

The task: given an Urdu tweet, predict the set of emojis it would carry. We treat
this as a **large-vocabulary, severely imbalanced, multi-label** problem over
inventories of the top *K* = 20 / 50 / 100 / 200 most frequent emojis, built from
~1.04M emoji-bearing Urdu tweets.

## Contents

```
src/
  data_prep.py        Emoji normalization, label construction, cleaning, splits
  losses_metrics.py   Asymmetric Loss, multi-label metrics, threshold calibration
  dual_encoder.py     Two-tower cosine-similarity model (comparison baseline)
scripts/
  train_classifier.py     Train XLM-R + ASL classifier for a given K
  train_dual_encoder.py   Train the dual-encoder baseline
  evaluate.py             Threshold calibration + full metric suite; saves preds
  make_figures.py         Generate all paper figures
  predict.py              CLI emoji prediction for a single Urdu sentence
  upload_to_hf.py         Push the trained model to the Hugging Face Hub
notebooks/
  exploration.ipynb       Original end-to-end Kaggle notebook (the scripts above
                          are the cleaned, modular refactoring of this notebook)
requirements.txt
LICENSE
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The model is trained on a corpus of Urdu tweets with emoji-derived labels.
Place the source spreadsheet, then cache it as Parquet (one-time, slow):

```python
from src.data_prep import xlsx_to_parquet
xlsx_to_parquet("cleaned_urdu_dataset_1M_processed.xlsx", "processed.parquet")
```

`processed.parquet` is expected to contain at least the columns `Urdu_Text`
(emoji-free tweet text) and `Unique_Emojis` (stringified list of emoji
characters, e.g. `"['😊', '🦋']"`). Adjust `--text_col` / `--emoji_col` if your
columns differ.

> Dataset note: the corpus is derived from publicly collected Urdu tweets; see
> the paper for collection and licensing details. (TODO: add dataset DOI / link.)

## Reproduce the paper

```bash
# 1. coverage report (Table 2)
python -c "from src.data_prep import coverage_report; coverage_report('processed.parquet')"

# 2. train the classifier for each inventory size (Table 3)
for K in 20 50 100 200; do
  python scripts/train_classifier.py --processed processed.parquet --K $K \
         --out_dir runs/xlmr_k$K
done

# 3. dual-encoder baseline at K=100 (Table 4)
python scripts/train_dual_encoder.py --processed processed.parquet --K 100 \
       --out_dir runs/twotower_k100

# 4. threshold calibration + metrics + saved predictions (Section 6.3)
python scripts/evaluate.py --processed processed.parquet --K 100 \
       --model_dir runs/xlmr_k100/xlmr_k100_best --out_dir runs/eval

# 5. figures
python scripts/make_figures.py dataset --processed processed.parquet --out figures
python scripts/make_figures.py scaling --out figures
python scripts/make_figures.py preds --npz runs/eval/test_preds_k100.npz \
       --vocab runs/eval/vocab_k100.pkl --out figures
```

To reproduce the stronger-encoder variant, add `--model intfloat/multilingual-e5-base --e5_prefix`.

## Inference

```bash
python scripts/predict.py --model_dir runs/xlmr_k100/xlmr_k100_best \
       --text "آج بہت خوشی کا دن ہے"
```

## Hardware / environment notes

- Trains in ~2 epochs; each K run is ~2.5–3 h on 2× NVIDIA T4 (free Kaggle GPU).
- On recent `transformers`, `Trainer` uses `processing_class=` and
  `TrainingArguments(eval_strategy=...)`. On older versions rename to
  `tokenizer=` and `evaluation_strategy=` respectively.
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (the scripts do this) to
  reduce fragmentation OOMs.

## Trained model

A trained checkpoint is available on the Hugging Face Hub: **[TODO: add HF link]**.

## Citation

```bibtex
@article{TODO_emoji_urdu,
  title   = {Large-Scale Multi-Label Emoji Prediction for Urdu Social Media Text
             Using Transformer Encoders and Asymmetric Loss},
  author  = {TODO},
  journal = {ACM Transactions on Asian and Low-Resource Language Information Processing},
  year    = {2026}
}
```

## License

Code released under the MIT License (see `LICENSE`).
