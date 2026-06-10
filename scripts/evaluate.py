#!/usr/bin/env python
"""
Evaluate a trained classifier: threshold calibration + full multi-label metrics.

Example:
  python scripts/evaluate.py --processed processed.parquet --K 100 \
      --model_dir runs/xlmr_k100/xlmr_k100_best

Produces the numbers in Section 6.3 (threshold calibration) of the paper:
  - micro-F1 under the global threshold tuned for micro-F1
  - macro-F1 under per-class thresholds
  - threshold-free P@1 and acc@3
Also saves test probabilities (.npz) for figure generation.
"""
import os
import sys
import json
import argparse
import pickle

import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_prep import build_dataset                                  # noqa: E402
from losses_metrics import tune_thresholds, best_global_threshold     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True)
    ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--out_dir", default="runs/eval")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df, vocab, _ = build_dataset(args.processed, args.K)
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)

    def to_ds(split):
        sub = df[df.split == split]
        return Dataset.from_dict(
            {"text": sub["text"].tolist(),
             "labels": [np.asarray(x, dtype=np.float32) for x in sub["labels"]]}
        )

    ds = {s: to_ds(s) for s in ["val", "test"]}
    ds = {s: d.map(lambda b: tok(b["text"], truncation=True, max_length=args.max_len),
                   batched=True, remove_columns=["text"]) for s, d in ds.items()}

    trainer = Trainer(model=model, processing_class=tok,
                      data_collator=DataCollatorWithPadding(tok))
    pv = trainer.predict(ds["val"])
    pt = trainer.predict(ds["test"])
    probs_v = 1 / (1 + np.exp(-pv.predictions)); yv = pv.label_ids.astype(int)
    probs_t = 1 / (1 + np.exp(-pt.predictions)); yt = pt.label_ids.astype(int)

    rows = np.arange(len(yt))[:, None]
    top3 = np.argsort(-probs_t, 1)[:, :3]
    p1 = float(yt[rows, probs_t.argmax(1)[:, None]].mean())
    acc3 = float(yt[rows, top3].any(1).mean())

    gt = best_global_threshold(probs_v, yv)
    micro = f1_score(yt, (probs_t >= gt).astype(int), average="micro", zero_division=0)

    thr = tune_thresholds(probs_v, yv)
    pred_pc = (probs_t >= thr).astype(int)
    macro = f1_score(yt, pred_pc, average="macro", zero_division=0)
    subset = accuracy_score(yt, (probs_t >= gt).astype(int))

    results = {
        "K": args.K, "global_threshold": gt,
        "micro_f1_global": round(float(micro), 4),
        "macro_f1_perclass": round(float(macro), 4),
        "subset_acc": round(float(subset), 4),
        "P@1": round(p1, 4), "acc@3": round(acc3, 4),
    }
    print(json.dumps(results, indent=2))

    np.savez(os.path.join(args.out_dir, f"test_preds_k{args.K}.npz"),
             probs=probs_t, y=yt, thr=thr)
    pickle.dump(vocab, open(os.path.join(args.out_dir, f"vocab_k{args.K}.pkl"), "wb"))
    json.dump(results, open(os.path.join(args.out_dir, f"eval_k{args.K}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
