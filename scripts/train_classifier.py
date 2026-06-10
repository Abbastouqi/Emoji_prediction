#!/usr/bin/env python
"""
Train the XLM-RoBERTa + Asymmetric Loss multi-label emoji classifier.

Example:
  python scripts/train_classifier.py \
      --processed processed.parquet --K 100 --model xlm-roberta-base \
      --epochs 2 --out_dir runs/xlmr_k100

Reproduces the per-K rows of Table 3 in the paper.
"""
import os
import sys
import json
import argparse

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    DataCollatorWithPadding,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_prep import build_dataset                       # noqa: E402
from losses_metrics import MultiLabelTrainer, compute_metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True, help="processed.parquet path")
    ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--model", default="xlm-roberta-base")
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--text_col", default="Urdu_Text")
    ap.add_argument("--emoji_col", default="Unique_Emojis")
    ap.add_argument("--out_dir", default="runs/classifier")
    ap.add_argument("--e5_prefix", action="store_true",
                    help="prepend 'query: ' (use with intfloat/multilingual-e5-*)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df, vocab, _ = build_dataset(
        args.processed, args.K, text_col=args.text_col, emoji_col=args.emoji_col,
        out_vocab=os.path.join(args.out_dir, f"vocab_k{args.K}.pkl"), seed=args.seed,
    )

    tok = AutoTokenizer.from_pretrained(args.model)

    def to_ds(split):
        sub = df[df.split == split]
        texts = sub["text"].tolist()
        if args.e5_prefix:
            texts = ["query: " + t for t in texts]
        return Dataset.from_dict(
            {"text": texts, "labels": [np.asarray(x, dtype=np.float32) for x in sub["labels"]]}
        )

    ds = {s: to_ds(s) for s in ["train", "val", "test"]}
    ds = {
        s: d.map(lambda b: tok(b["text"], truncation=True, max_length=args.max_len),
                 batched=True, remove_columns=["text"])
        for s, d in ds.items()
    }

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(vocab), problem_type="multi_label_classification"
    )

    targs = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=True,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs * 2,
        eval_strategy="epoch",          # rename to evaluation_strategy on older transformers
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="acc@3",
        logging_steps=200,
        report_to="none",
        dataloader_num_workers=2,
        seed=args.seed,
    )

    trainer = MultiLabelTrainer(
        model=model, args=targs,
        train_dataset=ds["train"], eval_dataset=ds["val"],
        processing_class=tok,            # rename to tokenizer= on older transformers
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tok),
    )
    trainer.train()

    test_metrics = trainer.evaluate(ds["test"])
    print("\n=== K=%d TEST ===" % args.K)
    print(test_metrics)

    trainer.save_model(os.path.join(args.out_dir, f"xlmr_k{args.K}_best"))
    json.dump(test_metrics, open(os.path.join(args.out_dir, f"test_metrics_k{args.K}.json"), "w"), indent=2)
    json.dump(trainer.state.log_history,
              open(os.path.join(args.out_dir, "log_history.json"), "w"))


if __name__ == "__main__":
    main()
