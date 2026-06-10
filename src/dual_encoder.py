"""
Dual-encoder (two-tower) similarity model for emoji prediction.

A text encoder produces a mean-pooled embedding; each emoji has a learnable
embedding initialized from the encoder's representation of its description.
Scores are scaled cosine similarities with a learnable temperature, trained
under the same asymmetric loss as the classifier. This is the learned-similarity
counterpart compared against the discriminative classifier in the paper.
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from losses_metrics import AsymmetricLoss


def mean_pool(model_output, attention_mask):
    h = model_output.last_hidden_state
    m = attention_mask.unsqueeze(-1).float()
    return (h * m).sum(1) / m.sum(1).clamp(min=1e-9)


def build_emoji_init(model_name, descriptions, tokenizer=None, max_len=16, batch=128):
    """Encode emoji descriptions with the base encoder to initialize the emoji tower."""
    tok = tokenizer or AutoTokenizer.from_pretrained(model_name)
    enc = AutoModel.from_pretrained(model_name).eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(descriptions), batch):
            b = tok(
                descriptions[i : i + batch],
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            chunks.append(mean_pool(enc(**b), b["attention_mask"]))
    del enc
    return torch.cat(chunks).float()


class TwoTower(nn.Module):
    def __init__(self, model_name, emoji_init, temp=0.05, loss_fn=None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.emoji = nn.Parameter(emoji_init.clone())
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / temp).log())  # CLIP-style
        self.loss_fn = loss_fn if loss_fn is not None else AsymmetricLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        t = nn.functional.normalize(
            mean_pool(self.encoder(input_ids=input_ids, attention_mask=attention_mask), attention_mask),
            dim=-1,
        )
        e = nn.functional.normalize(self.emoji, dim=-1)
        logits = self.logit_scale.exp().clamp(max=100) * (t @ e.t())
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}
