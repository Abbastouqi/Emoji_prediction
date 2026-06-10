"""
Gradio demo for Hugging Face Spaces: Urdu emoji prediction.

Deploy: create a Space (SDK = Gradio), upload this file as app.py plus a
requirements.txt containing: gradio, torch, transformers, sentencepiece.
Set MODEL_ID below to your uploaded model repo, and make sure vocab.pkl is
either in the model repo or replaced by the model's id2label mapping.
"""
import os
import re
import pickle
import unicodedata

import numpy as np
import torch
import gradio as gr
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = os.environ.get("MODEL_ID", "your-username/urdu-emoji-xlmr-k100")  # TODO

tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()

# Prefer the model's own id2label; fall back to a bundled vocab.pkl.
id2label = model.config.id2label
if id2label and not all(str(v).startswith("LABEL_") for v in id2label.values()):
    VOCAB = [id2label[i] for i in range(len(id2label))]
else:
    path = hf_hub_download(MODEL_ID, "vocab.pkl")
    VOCAB = pickle.load(open(path, "rb"))

_URL = re.compile(r"https?://\S+|www\.\S+")


def clean(t):
    t = unicodedata.normalize("NFC", str(t))
    t = _URL.sub(" ", t)
    t = "".join(c for c in t if not (0xE000 <= ord(c) <= 0xF8FF))
    return re.sub(r"\s+", " ", t).strip()


@torch.no_grad()
def predict(text, k=5):
    if not text or not text.strip():
        return {}
    enc = tok(clean(text), truncation=True, max_length=64, return_tensors="pt")
    probs = torch.sigmoid(model(**enc).logits[0]).numpy()
    idx = np.argsort(-probs)[:k]
    return {VOCAB[i]: float(probs[i]) for i in idx}


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Urdu text", placeholder="آج بہت خوشی کا دن ہے", lines=2),
        gr.Slider(1, 10, value=5, step=1, label="Top-k"),
    ],
    outputs=gr.Label(num_top_classes=10, label="Predicted emojis"),
    title="Urdu Emoji Prediction",
    description="Multi-label emoji prediction for Urdu tweets (XLM-RoBERTa + Asymmetric Loss).",
    examples=[
        ["آج بہت خوشی کا دن ہے", 5],
        ["میں بہت اداس اور پریشان ہوں", 5],
        ["یہ بات بہت مزاحیہ ہے ہا ہا", 5],
        ["میں تم سے بہت محبت کرتا ہوں", 5],
    ],
)

if __name__ == "__main__":
    demo.launch()
