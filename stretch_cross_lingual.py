"""
Stretch 6B-S2: Cross-Lingual Embedding Comparison.

Extracts mean-pooled bert-base-multilingual-cased embeddings for paired
English/Arabic climate texts from data/climate_articles.csv, computes a
20x20 cosine similarity matrix, renders a heatmap, and quantifies how
well the multilingual model captures cross-lingual topic similarity.
"""

from __future__ import annotations

import json
from pathlib import Path

import arabic_reshaper
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from bidi.algorithm import get_display
from matplotlib import font_manager
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "climate_articles.csv"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

MODEL_NAME = "bert-base-multilingual-cased"
MAX_LEN = 256
LABEL_CHARS = 40

# Topic-paired EN/AR ids selected from data/climate_articles.csv.
# Each pair covers the same real-world climate topic in both languages
# so cross-lingual similarity has a meaningful ground truth.
PAIRS = [
    (1, 79, "policy", "IPCC AR6"),
    (2, 81, "policy", "COP28 Dubai"),
    (11, 86, "science", "NASA 2023 warmest year"),
    (12, 87, "science", "Greenland ice loss"),
    (14, 89, "science", "Great Barrier Reef bleaching"),
    (23, 94, "impact", "Dead Sea shrinking"),
    (26, 97, "impact", "Derna Libya floods"),
    (27, 96, "impact", "Jordan water scarcity"),
    (35, 105, "adaptation", "Disi Water Conveyance"),
    (36, 103, "adaptation", "Mafraq solar 200 MW"),
]


def configure_arabic_font() -> None:
    """Make matplotlib pick up Noto Sans Arabic so RTL labels render."""
    candidates = [
        "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
    plt.rcParams["font.family"] = ["DejaVu Sans", "Noto Sans Arabic"]


def shape_label(text: str, lang: str) -> str:
    """Trim to LABEL_CHARS and fix Arabic shaping/bidi for display."""
    snippet = text[:LABEL_CHARS]
    if lang == "ar":
        snippet = get_display(arabic_reshaper.reshape(snippet))
    return snippet


def mean_pool(last_hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token vectors using the attention mask (lab pipeline)."""
    expanded = mask.unsqueeze(-1).float()
    summed = (last_hidden * expanded).sum(dim=1)
    counts = expanded.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def embed(texts: list[str], tokenizer, model, device) -> np.ndarray:
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        out = model(**encoded)
    pooled = mean_pool(out.last_hidden_state, encoded["attention_mask"])
    return pooled.cpu().numpy()


def select_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Build the 20-row table: 10 English first, then 10 Arabic, paired by index."""
    en_ids = [p[0] for p in PAIRS]
    ar_ids = [p[1] for p in PAIRS]
    topics = [p[3] for p in PAIRS]
    cats = [p[2] for p in PAIRS]

    en_df = df.set_index("id").loc[en_ids].reset_index()
    en_df["lang"] = "en"
    en_df["topic"] = topics
    en_df["cat"] = cats

    ar_df = df.set_index("id").loc[ar_ids].reset_index()
    ar_df["lang"] = "ar"
    ar_df["topic"] = topics
    ar_df["cat"] = cats

    return pd.concat([en_df, ar_df], ignore_index=True)


def block_means(sim: np.ndarray, n_per_lang: int = 10) -> dict:
    """Average similarities for the four blocks of the 20x20 matrix."""
    en = sim[:n_per_lang, :n_per_lang]
    ar = sim[n_per_lang:, n_per_lang:]
    cross = sim[:n_per_lang, n_per_lang:]

    # Off-diagonal mean for within-language blocks.
    off = ~np.eye(n_per_lang, dtype=bool)
    return {
        "within_en_mean": float(en[off].mean()),
        "within_ar_mean": float(ar[off].mean()),
        "cross_mean": float(cross.mean()),
        "same_topic_cross_mean": float(np.diag(cross).mean()),
        "diff_topic_cross_mean": float(cross[~np.eye(n_per_lang, dtype=bool)].mean()),
    }


def retrieval_ranks(cross: np.ndarray) -> dict:
    """For each EN text, where does its true AR pair rank among the 10 AR options?"""
    n = cross.shape[0]
    en_ranks = []
    for i in range(n):
        order = np.argsort(-cross[i])  # descending
        en_ranks.append(int(np.where(order == i)[0][0]) + 1)  # 1-indexed
    ar_ranks = []
    for j in range(n):
        order = np.argsort(-cross[:, j])
        ar_ranks.append(int(np.where(order == j)[0][0]) + 1)
    en_top1 = sum(r == 1 for r in en_ranks) / n
    ar_top1 = sum(r == 1 for r in ar_ranks) / n
    return {
        "en_to_ar_ranks": en_ranks,
        "ar_to_en_ranks": ar_ranks,
        "en_to_ar_top1_acc": en_top1,
        "ar_to_en_top1_acc": ar_top1,
        "mean_rank_en_to_ar": float(np.mean(en_ranks)),
        "mean_rank_ar_to_en": float(np.mean(ar_ranks)),
    }


def plot_heatmap(sim: np.ndarray, labels: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(
        sim,
        xticklabels=labels,
        yticklabels=labels,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 6},
        cbar_kws={"label": "Cosine similarity"},
        ax=ax,
    )
    ax.axhline(10, color="white", linewidth=2)
    ax.axvline(10, color="white", linewidth=2)
    ax.set_title(
        "bert-base-multilingual-cased: 10 EN (rows/cols 0-9) vs 10 AR (10-19)\n"
        "Diagonal of the cross block = same-topic EN/AR pair",
        fontsize=11,
    )
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_arabic_font()

    df = pd.read_csv(DATA)
    rows = select_rows(df)
    texts = rows["text"].tolist()
    labels = [shape_label(t, l) for t, l in zip(texts, rows["lang"])]
    # Prefix language tag so the heatmap stays readable even if a glyph fails.
    labels = [f"[{l.upper()}] {lab}" for l, lab in zip(rows["lang"], labels)]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    print(f"Embedding {len(texts)} texts (10 EN + 10 AR)...")
    embs = embed(texts, tokenizer, model, device)
    sim = cosine_similarity(embs)

    cross = sim[:10, 10:]
    stats = block_means(sim)
    ranks = retrieval_ranks(cross)

    sim_df = pd.DataFrame(sim, index=labels, columns=labels)
    sim_df.to_csv(OUT / "cross_lingual_similarity_matrix.csv")

    pair_rows = []
    for i, (en_id, ar_id, cat, topic) in enumerate(PAIRS):
        pair_rows.append(
            {
                "topic": topic,
                "category": cat,
                "en_id": en_id,
                "ar_id": ar_id,
                "same_topic_cross_sim": float(cross[i, i]),
                "best_other_ar_sim": float(np.max(np.delete(cross[i], i))),
                "en_to_ar_rank": ranks["en_to_ar_ranks"][i],
                "ar_to_en_rank": ranks["ar_to_en_ranks"][i],
            }
        )
    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(OUT / "cross_lingual_pair_scores.csv", index=False)

    summary = {
        "model": MODEL_NAME,
        "n_pairs": len(PAIRS),
        **stats,
        "lift_same_vs_diff_topic_cross": stats["same_topic_cross_mean"]
        - stats["diff_topic_cross_mean"],
        "en_to_ar_top1_acc": ranks["en_to_ar_top1_acc"],
        "ar_to_en_top1_acc": ranks["ar_to_en_top1_acc"],
        "mean_rank_en_to_ar": ranks["mean_rank_en_to_ar"],
        "mean_rank_ar_to_en": ranks["mean_rank_ar_to_en"],
    }
    with (OUT / "cross_lingual_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    plot_heatmap(sim, labels, OUT / "cross_lingual_heatmap.png")

    print("\n=== Block similarity (mean) ===")
    for k, v in stats.items():
        print(f"  {k:30s} {v:.4f}")
    print("\n=== Cross-lingual retrieval ===")
    print(f"  EN -> AR top-1 accuracy: {ranks['en_to_ar_top1_acc']:.2f}")
    print(f"  AR -> EN top-1 accuracy: {ranks['ar_to_en_top1_acc']:.2f}")
    print(f"  Mean rank EN->AR: {ranks['mean_rank_en_to_ar']:.2f}")
    print(f"  Mean rank AR->EN: {ranks['mean_rank_ar_to_en']:.2f}")
    print("\n=== Per-pair scores ===")
    print(pair_df.to_string(index=False))
    print(f"\nArtifacts written to {OUT}/")


if __name__ == "__main__":
    main()
