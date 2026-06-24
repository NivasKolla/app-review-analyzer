"""
Phase D: Insights & Visualisation
Generates the full "Management Summary" dashboard:
  1. Sentiment Distribution Pie Chart
  2. Star Rating Distribution Bar Chart
  3. Monthly Negative Sentiment Trend Line
  4. Word Cloud of Pain (1–2 ★ reviews)
  5. Top Bigrams in Negative Reviews (Horizontal Bar)
  6. TF-IDF: Low-star vs High-star terms (Side-by-side)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from wordcloud import WordCloud
from analyzer import get_top_ngrams, monthly_sentiment_trend, tfidf_by_rating

# ── Colour palette ──────────────────────────────────────────
PALETTE = {
    "Positive": "#2ecc71",
    "Neutral":  "#f39c12",
    "Negative": "#e74c3c",
    "bg":       "#0f1117",
    "card":     "#1e2130",
    "text":     "#e8eaf6",
    "accent":   "#7c83fd",
}


def _apply_dark_style():
    plt.rcParams.update({
        "figure.facecolor":  PALETTE["bg"],
        "axes.facecolor":    PALETTE["card"],
        "axes.edgecolor":    "#3a3f5c",
        "axes.labelcolor":   PALETTE["text"],
        "xtick.color":       PALETTE["text"],
        "ytick.color":       PALETTE["text"],
        "text.color":        PALETTE["text"],
        "grid.color":        "#2a2f4c",
        "grid.linestyle":    "--",
        "grid.alpha":        0.5,
        "font.family":       "DejaVu Sans",
        "font.size":         10,
    })


# ────────────────────────────────────────────────────────────
# Individual chart functions
# ────────────────────────────────────────────────────────────

def plot_sentiment_distribution(ax: plt.Axes, df: pd.DataFrame):
    counts = df["sentiment_label"].value_counts()
    labels = counts.index.tolist()
    colors = [PALETTE.get(l, "#888") for l in labels]
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        wedgeprops={"edgecolor": PALETTE["bg"], "linewidth": 2},
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")
    ax.set_title("Sentiment Distribution", color=PALETTE["text"], fontsize=13, pad=12)


def plot_star_distribution(ax: plt.Axes, df: pd.DataFrame):
    counts = df["star_rating"].value_counts().sort_index()
    bar_colors = [
        PALETTE["Negative"] if i in [1, 2]
        else PALETTE["Neutral"] if i == 3
        else PALETTE["Positive"]
        for i in counts.index
    ]
    bars = ax.bar(counts.index.astype(str), counts.values, color=bar_colors, edgecolor=PALETTE["bg"], linewidth=1.5)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts.values) * 0.01,
            str(int(bar.get_height())),
            ha="center", va="bottom", color=PALETTE["text"], fontsize=9,
        )
    ax.set_xlabel("Star Rating")
    ax.set_ylabel("Number of Reviews")
    ax.set_title("Star Rating Distribution", color=PALETTE["text"], fontsize=13)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)


def plot_trend_line(ax: plt.Axes, df: pd.DataFrame):
    trend = monthly_sentiment_trend(df)
    if trend.empty:
        ax.text(0.5, 0.5, "Not enough date data", ha="center", va="center", transform=ax.transAxes)
        return

    x = range(len(trend))
    ax.plot(
        list(x), trend["negative_pct"].tolist(),
        color=PALETTE["Negative"], linewidth=2.5, marker="o", markersize=5, label="Negative %",
    )
    ax.fill_between(
        list(x), trend["negative_pct"].tolist(),
        alpha=0.15, color=PALETTE["Negative"],
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(trend["month"].tolist(), rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Negative Reviews (%)")
    ax.set_title("Monthly Negative Sentiment Trend", color=PALETTE["text"], fontsize=13)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)


def plot_word_cloud(ax: plt.Axes, df: pd.DataFrame):
    """Word cloud of pain: words from 1–2 star reviews."""
    low_star_text = " ".join(
        df[df["star_rating"].isin([1, 2])]["cleaned_text"].dropna().tolist()
    )
    if not low_star_text.strip():
        ax.text(0.5, 0.5, "No low-star reviews found", ha="center", va="center", transform=ax.transAxes)
        return

    wc = WordCloud(
        width=800, height=400,
        background_color="#1e2130",
        colormap="RdYlGn_r",
        max_words=80,
        collocations=False,
    ).generate(low_star_text)

    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Word Cloud of Pain  (1–2 ★ Reviews)", color=PALETTE["text"], fontsize=13)


def plot_top_bigrams(ax: plt.Axes, df: pd.DataFrame):
    bigrams = get_top_ngrams(df, n=2, top_k=15, rating_filter=[1, 2])
    if bigrams.empty:
        ax.text(0.5, 0.5, "No bigrams found", ha="center", va="center", transform=ax.transAxes)
        return

    bigrams = bigrams.sort_values("count")
    bars = ax.barh(bigrams["ngram"], bigrams["count"], color=PALETTE["accent"], edgecolor=PALETTE["bg"])
    for bar in bars:
        ax.text(
            bar.get_width() + max(bigrams["count"]) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            str(int(bar.get_width())),
            va="center", fontsize=8, color=PALETTE["text"],
        )
    ax.set_xlabel("Frequency")
    ax.set_title("Top Bigrams in 1–2 ★ Reviews", color=PALETTE["text"], fontsize=13)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)


def plot_tfidf_comparison(ax_neg: plt.Axes, ax_pos: plt.Axes, df: pd.DataFrame):
    tfidf = tfidf_by_rating(df)

    for ax, key, color, label in [
        (ax_neg, "negative", PALETTE["Negative"], "Complaint Terms  (1–2 ★)"),
        (ax_pos, "positive", PALETTE["Positive"], "Praise Terms  (4–5 ★)"),
    ]:
        data = tfidf[key].sort_values("tfidf_score")
        ax.barh(data["term"], data["tfidf_score"], color=color, edgecolor=PALETTE["bg"])
        ax.set_xlabel("Avg TF-IDF Score")
        ax.set_title(label, color=PALETTE["text"], fontsize=11)
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)


# ────────────────────────────────────────────────────────────
# Main Dashboard
# ────────────────────────────────────────────────────────────

def generate_dashboard(df: pd.DataFrame, app_name: str = "App", output_dir: str = "outputs"):
    """Render the full management-summary dashboard and save as PNG."""
    _apply_dark_style()
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(20, 24), facecolor=PALETTE["bg"])
    fig.suptitle(
        f"📊  {app_name} — Review Intelligence Dashboard",
        fontsize=18, fontweight="bold", color=PALETTE["text"], y=0.98,
    )

    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])   # Sentiment pie
    ax2 = fig.add_subplot(gs[0, 1])   # Star distribution
    ax3 = fig.add_subplot(gs[1, :])   # Trend line (full width)
    ax4 = fig.add_subplot(gs[2, :])   # Word cloud (full width)
    ax5 = fig.add_subplot(gs[3, 0])   # Top bigrams
    ax6 = fig.add_subplot(gs[3, 1])   # TF-IDF negative terms

    plot_sentiment_distribution(ax1, df)
    plot_star_distribution(ax2, df)
    plot_trend_line(ax3, df)
    plot_word_cloud(ax4, df)
    plot_top_bigrams(ax5, df)

    # TF-IDF — show only the negative panel in slot 6 (positive in a separate figure)
    tfidf = tfidf_by_rating(df)
    neg_data = tfidf["negative"].sort_values("tfidf_score")
    ax6.barh(neg_data["term"], neg_data["tfidf_score"], color=PALETTE["Negative"], edgecolor=PALETTE["bg"])
    ax6.set_xlabel("Avg TF-IDF Score")
    ax6.set_title("Key Complaint Terms — TF-IDF (1–2 ★)", color=PALETTE["text"], fontsize=11)
    ax6.xaxis.grid(True)
    ax6.set_axisbelow(True)

    out_path = os.path.join(output_dir, f"{app_name.lower().replace(' ', '_')}_dashboard.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"\n[Visualizer] Dashboard saved → {out_path}")
    return out_path


def generate_tfidf_comparison_chart(df: pd.DataFrame, app_name: str = "App", output_dir: str = "outputs"):
    """Separate side-by-side TF-IDF chart for complaints vs praise."""
    _apply_dark_style()
    os.makedirs(output_dir, exist_ok=True)

    fig, (ax_neg, ax_pos) = plt.subplots(1, 2, figsize=(16, 7), facecolor=PALETTE["bg"])
    fig.suptitle("TF-IDF: Complaint Terms vs Praise Terms", fontsize=15, color=PALETTE["text"])
    plot_tfidf_comparison(ax_neg, ax_pos, df)

    out_path = os.path.join(output_dir, f"{app_name.lower().replace(' ', '_')}_tfidf.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"[Visualizer] TF-IDF chart saved → {out_path}")
    return out_path
