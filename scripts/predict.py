#!/usr/bin/env python
"""
Predict emojis for an Urdu sentence with a trained model.

Interactive:   python scripts/predict.py --model_dir runs/xlmr_k100/xlmr_k100_best
One-shot:      python scripts/predict.py --model_dir <dir> --text "آج بہت خوشی کا دن ہے"
"""
import os
import re
import sys
import pickle
import argparse
import unicodedata

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

_URL = re.compile(r"https?://\S+|www\.\S+")


def clean(t):
    t = unicodedata.normalize("NFC", str(t))
    t = _URL.sub(" ", t)
    t = "".join(c for c in t if not (0xE000 <= ord(c) <= 0xF8FF))
    return re.sub(r"\s+", " ", t).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--vocab", default=None, help="vocab pickle (default: <model_dir>/../vocab_k*.pkl)")
    ap.add_argument("--text", default=None)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--e5_prefix", action="store_true")
    args = ap.parse_args()

    # locate vocab
    vocab_path = args.vocab
    if vocab_path is None:
        parent = os.path.dirname(args.model_dir.rstrip("/"))
        cands = [f for f in os.listdir(parent) if f.startswith("vocab_k") and f.endswith(".pkl")]
        if not cands:
            sys.exit("Could not find vocab_k*.pkl; pass --vocab explicitly.")
        vocab_path = os.path.join(parent, cands[0])
    VOCAB = pickle.load(open(vocab_path, "rb"))

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)

    @torch.no_grad()
    def predict(text, k):
        s = clean(text)
        if args.e5_prefix:
            s = "query: " + s
        enc = tok(s, truncation=True, max_length=64, return_tensors="pt").to(dev)
        probs = torch.sigmoid(model(**enc).logits[0]).cpu().numpy()
        idx = np.argsort(-probs)[:k]
        return [(VOCAB[i], round(float(probs[i]), 3)) for i in idx]

    if args.text:
        for e, p in predict(args.text, args.k):
            print(f"  {e}   {p:.3f}")
    else:
        print("Type an Urdu sentence (blank to quit):")
        while True:
            try:
                t = input("\n> ").strip()
            except EOFError:
                break
            if not t:
                break
            for e, p in predict(t, args.k):
                print(f"  {e}   {p:.3f}")


if __name__ == "__main__":
    main()
