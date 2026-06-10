#!/usr/bin/env python
"""
Train the dual-encoder (two-tower cosine similarity) baseline.

Example:
  python scripts/train_dual_encoder.py --processed processed.parquet --K 100 \
      --out_dir runs/twotower_k100

Reproduces the dual-encoder row of Table 4 in the paper.
"""
import os
import sys
import json
import argparse

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_prep import build_dataset                # noqa: E402
from losses_metrics import compute_metrics         # noqa: E402
from dual_encoder import TwoTower, build_emoji_init # noqa: E402

try:
    import emoji as emojilib
except Exception:
    os.system("pip -q install emoji")
    import emoji as emojilib


def emoji_name(ch):
    d = emojilib.demojize(ch)
    return d.strip(":").replace("_", " ") if d.startswith(":") else ch


class TwoTowerTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        out = model(**inputs)
        return (out["loss"], out) if return_outputs else out["loss"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True)
    ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--model", default="xlm-roberta-base")
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--out_dir", default="runs/twotower")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df, vocab, _ = build_dataset(args.processed, args.K,
                                 out_vocab=os.path.join(args.out_dir, f"vocab_k{args.K}.pkl"))

    tok = AutoTokenizer.from_pretrained(args.model)

    def to_ds(split):
        sub = df[df.split == split]
        return Dataset.from_dict(
            {"text": sub["text"].tolist(),
             "labels": [np.asarray(x, dtype=np.float32) for x in sub["labels"]]}
        )

    ds = {s: to_ds(s) for s in ["train", "val", "test"]}
    ds = {s: d.map(lambda b: tok(b["text"], truncation=True, max_length=args.max_len),
                   batched=True, remove_columns=["text"]) for s, d in ds.items()}

    descriptions = [emoji_name(e) for e in vocab]
    emoji_init = build_emoji_init(args.model, descriptions, tokenizer=tok)
    model = TwoTower(args.model, emoji_init)

    targs = TrainingArguments(
        output_dir=args.out_dir, num_train_epochs=args.epochs, learning_rate=args.lr,
        fp16=True, per_device_train_batch_size=args.bs, per_device_eval_batch_size=args.bs * 2,
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="acc@3", logging_steps=200, report_to="none", dataloader_num_workers=2,
    )
    trainer = TwoTowerTrainer(
        model=model, args=targs, train_dataset=ds["train"], eval_dataset=ds["val"],
        processing_class=tok, compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tok),
    )
    trainer.train()
    test_metrics = trainer.evaluate(ds["test"])
    print("\n=== TWO-TOWER K=%d TEST ===" % args.K)
    print(test_metrics)
    json.dump(test_metrics, open(os.path.join(args.out_dir, f"test_metrics_k{args.K}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
