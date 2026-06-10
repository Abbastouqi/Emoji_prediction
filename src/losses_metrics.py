"""
Losses, metrics, and Trainer subclass for multi-label emoji prediction.
"""
import numpy as np
import torch
from transformers import Trainer
from sklearn.metrics import f1_score, accuracy_score


class AsymmetricLoss(torch.nn.Module):
    """
    Asymmetric Loss for multi-label classification (Ridnik et al., ICCV 2021).
    Down-weights the abundant easy negatives that dominate multi-hot targets.
    """

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        xp = torch.sigmoid(logits)
        xn = 1 - xp
        if self.clip and self.clip > 0:
            xn = (xn + self.clip).clamp(max=1)
        loss = targets * torch.log(xp.clamp(min=self.eps)) + (1 - targets) * torch.log(
            xn.clamp(min=self.eps)
        )
        pt = xp * targets + xn * (1 - targets)
        gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
        loss = loss * (1 - pt).pow(gamma)
        return -loss.sum(1).mean()


class MultiLabelTrainer(Trainer):
    """HF Trainer that uses a custom multi-label loss (default: ASL)."""

    def __init__(self, *args, loss_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn if loss_fn is not None else AsymmetricLoss()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self.loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    """Micro/macro F1, subset accuracy, P@1, accuracy@3 (at threshold 0.5)."""
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    rows = np.arange(len(labels))[:, None]
    top3 = np.argsort(-probs, 1)[:, :3]
    return {
        "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "subset_acc": accuracy_score(labels, preds),
        "P@1": float(labels[rows, probs.argmax(1)[:, None]].mean()),
        "acc@3": float(labels[rows, top3].any(1).mean()),
    }


def tune_thresholds(probs_val, y_val, grid=None):
    """Per-class F1-optimal thresholds (for macro-F1) tuned on validation."""
    if grid is None:
        grid = np.arange(0.05, 0.60, 0.02)
    K = probs_val.shape[1]
    thr = np.array(
        [
            max(grid, key=lambda t: f1_score(y_val[:, c], (probs_val[:, c] >= t).astype(int), zero_division=0))
            for c in range(K)
        ]
    )
    return thr


def best_global_threshold(probs_val, y_val, grid=None):
    """Single global threshold maximizing micro-F1 (for reporting micro-F1)."""
    if grid is None:
        grid = np.arange(0.05, 0.60, 0.02)
    return float(
        max(grid, key=lambda t: f1_score(y_val, (probs_val >= t).astype(int), average="micro", zero_division=0))
    )
