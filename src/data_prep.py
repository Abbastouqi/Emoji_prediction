"""
Data preparation for large-scale multi-label Urdu emoji prediction.

Pipeline:
  raw xlsx  ->  processed.parquet  ->  per-K training parquet + vocab

Emoji normalization merges variation-selector and skin-tone variants so that
e.g. U+2764 and U+2764 U+FE0F collapse to one canonical label. Text cleaning is
deliberately light (NFC, URL/PUA removal) and PRESERVES stopwords and negation,
which carry the affective signal needed for emoji prediction.
"""
import ast
import re
import unicodedata
import pickle
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

# --- emoji normalization ---------------------------------------------------
VARIATION_SELECTORS = {"\uFE0F", "\uFE0E"}
SKIN_TONES = {chr(c) for c in range(0x1F3FB, 0x1F400)}  # U+1F3FB..U+1F3FF


def norm_emoji(e: str) -> str:
    """Strip variation selectors and skin-tone modifiers from an emoji."""
    return "".join(c for c in e if c not in VARIATION_SELECTORS and c not in SKIN_TONES)


def parse_list(x):
    """Safely parse a stringified Python list (e.g. "['😊', '🦋']")."""
    if isinstance(x, list):
        return x
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    try:
        v = ast.literal_eval(x)
        return v if isinstance(v, list) else [v]
    except Exception:
        return []


# --- text cleaning ---------------------------------------------------------
_URL = re.compile(r"https?://\S+|www\.\S+")


def clean_text(t) -> str:
    """Light cleaning only. Keeps stopwords/negation intact."""
    if not isinstance(t, str):
        return ""
    t = unicodedata.normalize("NFC", t)
    t = _URL.sub(" ", t)
    t = "".join(ch for ch in t if not (0xE000 <= ord(ch) <= 0xF8FF))  # drop private-use
    return re.sub(r"\s+", " ", t).strip()


# --- pipeline steps --------------------------------------------------------
def xlsx_to_parquet(xlsx_path: str, out_path: str = "processed.parquet") -> pd.DataFrame:
    """Read the (large) source xlsx once and cache as parquet for fast reuse."""
    proc = pd.read_excel(xlsx_path, engine="openpyxl")
    proc.to_parquet(out_path, index=False)
    return proc


def build_dataset(
    processed_parquet: str,
    K: int,
    text_col: str = "Urdu_Text",
    emoji_col: str = "Unique_Emojis",
    out_parquet: str | None = None,
    out_vocab: str | None = None,
    seed: int = 42,
):
    """
    Build a top-K multi-label dataset.

    Returns (dataframe, vocab, Y) where dataframe has columns
    ['text', 'labels', 'split'] and Y is the multi-hot matrix.
    """
    proc = pd.read_parquet(processed_parquet)

    uniq = proc[emoji_col].apply(parse_list).apply(
        lambda lst: list(dict.fromkeys(ne for e in lst if (ne := norm_emoji(e))))
    )
    counts = Counter(e for row in uniq for e in row)
    vocab = [e for e, _ in counts.most_common(K)]
    vset = set(vocab)

    df = pd.DataFrame(
        {
            "text": proc[text_col].apply(clean_text),
            "emojis": uniq.apply(lambda lst: [e for e in lst if e in vset]),
        }
    )
    df = df[(df["emojis"].apply(len) > 0) & (df["text"].str.len() >= 2)].reset_index(drop=True)

    Y = MultiLabelBinarizer(classes=vocab).fit_transform(df["emojis"]).astype("int8")

    idx = np.arange(len(df))
    tr, tmp = train_test_split(idx, test_size=0.2, random_state=seed)
    va, te = train_test_split(tmp, test_size=0.5, random_state=seed)
    split = np.array(["train"] * len(df), dtype=object)
    split[va] = "val"
    split[te] = "test"

    out = pd.DataFrame({"text": df["text"], "labels": list(Y), "split": split})

    if out_parquet:
        out.to_parquet(out_parquet, index=False)
    if out_vocab:
        pickle.dump(vocab, open(out_vocab, "wb"))

    print(
        f"K={K}: kept {len(df):,} rows | avg labels/row {Y.sum(1).mean():.3f} "
        f"| class freq min {int(Y.sum(0).min())} | "
        f"train/val/test {int((split=='train').sum())}/{int((split=='val').sum())}/{int((split=='test').sum())}"
    )
    return out, vocab, Y


def coverage_report(processed_parquet: str, emoji_col: str = "Unique_Emojis"):
    """Print the coverage curve used to choose K (Table 2 in the paper)."""
    proc = pd.read_parquet(processed_parquet)
    uniq = proc[emoji_col].apply(parse_list).apply(
        lambda lst: list(dict.fromkeys(ne for e in lst if (ne := norm_emoji(e))))
    )
    counts = Counter(e for row in uniq for e in row)
    total = sum(counts.values())
    freqs = np.array(sorted(counts.values(), reverse=True))
    cum = np.cumsum(freqs) / total
    print(f"canonical emojis: {len(counts)} | total occurrences: {total:,}")
    for t in (0.80, 0.90, 0.95, 0.99):
        k = int(np.searchsorted(cum, t) + 1)
        print(f"  top-{k:<4} covers {t*100:.0f}%")
    return counts
