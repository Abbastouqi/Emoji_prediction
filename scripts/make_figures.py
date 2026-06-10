#!/usr/bin/env python
"""
Generate the figures used in the paper.

Dataset-level figures (need only processed.parquet):
  python scripts/make_figures.py dataset --processed processed.parquet --out figures/

Prediction-level figures (need the .npz saved by evaluate.py):
  python scripts/make_figures.py preds --npz runs/eval/test_preds_k100.npz \
      --vocab runs/eval/vocab_k100.pkl --out figures/
"""
import os
import sys
import argparse
import pickle
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_prep import parse_list, norm_emoji   # noqa: E402

BLUE, RED, GREEN, GRAY = "#1a5fb4", "#c01c28", "#26a269", "#5e5c64"

try:
    import emoji as emojilib
    def ename(ch):
        d = emojilib.demojize(ch)
        return d.strip(":").replace("_", " ") if d.startswith(":") else ch
except Exception:
    def ename(ch):
        return ch


def dataset_figs(processed, out):
    import pandas as pd
    proc = pd.read_parquet(processed)
    uniq = proc["Unique_Emojis"].apply(parse_list).apply(
        lambda l: [norm_emoji(e) for e in l if norm_emoji(e)]
    )
    cnt = Counter(e for r in uniq for e in r)
    freqs = np.array(sorted(cnt.values(), reverse=True))

    # rank-frequency
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.loglog(range(1, len(freqs) + 1), freqs, lw=1.5, color=BLUE)
    for k in [20, 50, 100, 200]:
        ax.axvline(k, ls=":", color=RED, lw=1)
        ax.annotate(f"K={k}", (k, freqs[min(k, len(freqs) - 1)]), fontsize=7,
                    rotation=90, va="bottom", ha="right", color=RED)
    ax.set_xlabel("Emoji rank (log)"); ax.set_ylabel("Frequency (log)")
    ax.grid(alpha=0.3, which="both", lw=0.4); plt.tight_layout()
    plt.savefig(f"{out}/freq_dist.pdf"); plt.savefig(f"{out}/freq_dist.png", dpi=160); plt.close()

    # cardinality
    card = uniq.apply(len).clip(upper=8)
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    vc = card.value_counts().sort_index()
    ax.bar(vc.index, vc.values / len(card) * 100, color=BLUE, width=0.7)
    ax.set_xlabel("Distinct emojis per tweet (8 = 8+)"); ax.set_ylabel("% of tweets")
    ax.grid(alpha=0.3, axis="y", lw=0.4); plt.tight_layout()
    plt.savefig(f"{out}/cardinality.pdf"); plt.savefig(f"{out}/cardinality.png", dpi=160); plt.close()

    # coverage
    cum = np.cumsum(freqs) / freqs.sum()
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot(range(1, len(cum) + 1), cum * 100, lw=1.6, color=BLUE)
    ax.set_xscale("log"); ax.set_xlabel("Inventory size $K$ (log)")
    ax.set_ylabel("% of emoji occurrences covered")
    for k in [20, 50, 100, 200]:
        ax.axvline(k, ls=":", lw=0.8, color=RED)
    ax.grid(alpha=0.3, lw=0.4); plt.tight_layout()
    plt.savefig(f"{out}/coverage.pdf"); plt.savefig(f"{out}/coverage.png", dpi=160); plt.close()
    print("dataset figures ->", out)


def scaling_fig(out):
    """Scaling curve from the paper's four runs (edit values if you re-run)."""
    K = [20, 50, 100, 200]
    p1 = [0.543, 0.435, 0.395, 0.385]
    acc3 = [0.758, 0.616, 0.562, 0.547]
    mif1 = [0.475, 0.380, 0.337, 0.324]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(K, acc3, "o-", color=BLUE, label="accuracy@3", lw=1.8, ms=5)
    ax.plot(K, p1, "s--", color=RED, label="P@1", lw=1.8, ms=5)
    ax.plot(K, mif1, "^:", color=GRAY, label="micro-F1", lw=1.6, ms=5)
    ax.set_xscale("log"); ax.set_xticks(K); ax.set_xticklabels(K)
    ax.set_xlabel("Label inventory size $K$ (log scale)"); ax.set_ylabel("Score")
    ax.set_ylim(0.25, 0.82); ax.grid(alpha=0.3, lw=0.5); ax.legend(frameon=False, fontsize=9)
    plt.tight_layout(); plt.savefig(f"{out}/scaling_curve.pdf"); plt.savefig(f"{out}/scaling_curve.png", dpi=160); plt.close()
    print("scaling figure ->", out)


def pred_figs(npz, vocab_path, out):
    d = np.load(npz)
    probs_t, yt, thr = d["probs"], d["y"], d["thr"]
    vocab = pickle.load(open(vocab_path, "rb"))
    pred_set = (probs_t >= thr)
    freq_order = np.argsort(-yt.sum(0))

    # per-class F1
    f1c = [f1_score(yt[:, c], pred_set[:, c].astype(int), zero_division=0) for c in freq_order[:15]]
    names = [ename(vocab[c])[:18] for c in freq_order[:15]]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.barh(range(len(f1c))[::-1], f1c, color=BLUE)
    ax.set_yticks(range(len(f1c))[::-1]); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Per-class F1 (calibrated)"); ax.grid(alpha=0.3, axis="x", lw=0.4)
    plt.tight_layout(); plt.savefig(f"{out}/perclass_f1.pdf"); plt.savefig(f"{out}/perclass_f1.png", dpi=160); plt.close()

    # precision-recall
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    P, R, _ = precision_recall_curve(yt.ravel(), probs_t.ravel())
    ax.plot(R, P, lw=1.8, color="black",
            label=f'micro (AP={average_precision_score(yt, probs_t, average="micro"):.3f})')
    for c, col in [(freq_order[0], BLUE), (freq_order[len(freq_order) // 2], GREEN), (freq_order[-1], RED)]:
        P, R, _ = precision_recall_curve(yt[:, c], probs_t[:, c])
        ax.plot(R, P, lw=1.2, color=col, label=ename(vocab[c])[:20])
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.3, lw=0.4); plt.tight_layout()
    plt.savefig(f"{out}/pr_curves.pdf"); plt.savefig(f"{out}/pr_curves.png", dpi=160); plt.close()

    # threshold sweep
    ths = np.arange(0.05, 0.91, 0.05)
    mi = [f1_score(yt, (probs_t >= t).astype(int), average="micro", zero_division=0) for t in ths]
    ma = [f1_score(yt, (probs_t >= t).astype(int), average="macro", zero_division=0) for t in ths]
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    ax.plot(ths, mi, "o-", ms=3.5, lw=1.6, color=BLUE, label="micro-F1")
    ax.plot(ths, ma, "s--", ms=3.5, lw=1.6, color=RED, label="macro-F1")
    ax.set_xlabel("Global decision threshold"); ax.set_ylabel("F1")
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.3, lw=0.4)
    plt.tight_layout(); plt.savefig(f"{out}/threshold_sweep.pdf"); plt.savefig(f"{out}/threshold_sweep.png", dpi=160); plt.close()
    print("prediction figures ->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("dataset"); a.add_argument("--processed", required=True); a.add_argument("--out", default="figures")
    b = sub.add_parser("preds"); b.add_argument("--npz", required=True); b.add_argument("--vocab", required=True); b.add_argument("--out", default="figures")
    c = sub.add_parser("scaling"); c.add_argument("--out", default="figures")
    args = ap.parse_args()
    os.makedirs(getattr(args, "out", "figures"), exist_ok=True)
    if args.cmd == "dataset":
        dataset_figs(args.processed, args.out)
    elif args.cmd == "preds":
        pred_figs(args.npz, args.vocab, args.out)
    elif args.cmd == "scaling":
        scaling_fig(args.out)
