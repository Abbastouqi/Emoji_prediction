#!/usr/bin/env python
"""
Upload a trained checkpoint to the Hugging Face Hub, baking the emoji vocab into
the model config as id2label so the Space (and any downstream user) needs no
separate vocab file.

  huggingface-cli login
  python scripts/upload_to_hf.py --model_dir runs/xlmr_k100/xlmr_k100_best \
         --vocab runs/eval/vocab_k100.pkl --repo your-username/urdu-emoji-xlmr-k100
"""
import argparse
import pickle

from transformers import AutoTokenizer, AutoModelForSequenceClassification


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--repo", required=True, help="e.g. your-username/urdu-emoji-xlmr-k100")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    vocab = pickle.load(open(args.vocab, "rb"))
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)

    model.config.id2label = {i: e for i, e in enumerate(vocab)}
    model.config.label2id = {e: i for i, e in enumerate(vocab)}
    model.config.problem_type = "multi_label_classification"

    model.push_to_hub(args.repo, private=args.private)
    tok.push_to_hub(args.repo, private=args.private)
    print(f"Uploaded to https://huggingface.co/{args.repo}")
    print("The Space can now read labels from config.id2label (no vocab.pkl needed).")


if __name__ == "__main__":
    main()
