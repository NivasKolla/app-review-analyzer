"""
Phase C: Sentiment & Frequency Analysis
  - Sentiment scoring (TextBlob)
  - N-Gram (Bigram) analysis
  - Rating correlation (TF-IDF)
  - Actionable complaint extraction
"""

from __future__ import annotations

import pandas as pd
from collections import Counter
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.util import ngrams


# ──────────────────────────────────────────────
# 1. Sentiment Scoring
# ──────────────────────────────────────────────

def get_sentiment(text: str) -> tuple[float, str]:
    """Return (polarity_score, label) using TextBlob."""
    if not isinstance(text, str) or not text.strip():
        return 0.0, "Neutral"
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        label = "Positive"
    elif polarity < -0.1:
        label = "Negative"
    else:
        label = "Neutral"
    return round(polarity, 4), label


def add_sentiment(df: pd.DataFrame, text_col: str = "cleaned_text") -> pd.DataFrame:
    """Add 'sentiment_score' and 'sentiment_label' columns."""
    print("\n[Analyzer] Scoring sentiment …")
    df = df.copy()
    results = df[text_col].apply(get_sentiment)
    df["sentiment_score"] = results.apply(lambda x: x[0])
    df["sentiment_label"] = results.apply(lambda x: x[1])
    print(df["sentiment_label"].value_counts().to_string())
    return df


# ──────────────────────────────────────────────
# 2. N-Gram (Bigram) Analysis
# ──────────────────────────────────────────────

def get_top_ngrams(
    df: pd.DataFrame,
    n: int = 2,
    top_k: int = 20,
    rating_filter: list[int] | None = None,
    token_col: str = "tokens",
) -> pd.DataFrame:
    """
    Extract the top-k most frequent n-grams.

    Args:
        df           : DataFrame with a 'tokens' column (list of words)
        n            : n-gram size (2 = bigrams, 3 = trigrams)
        top_k        : number of results to return
        rating_filter: optional list of star ratings to filter by (e.g. [1, 2])
    """
    if rating_filter:
        subset = df[df["star_rating"].isin(rating_filter)]
    else:
        subset = df

    all_tokens: list[str] = []
    for token_list in subset[token_col]:
        if isinstance(token_list, list):
            all_tokens.extend(token_list)

    gram_counter: Counter = Counter()
    # Rebuild per-review n-grams to avoid cross-review boundaries
    for token_list in subset[token_col]:
        if isinstance(token_list, list) and len(token_list) >= n:
            gram_counter.update(ngrams(token_list, n))

    top = gram_counter.most_common(top_k)
    result = pd.DataFrame(top, columns=["ngram", "count"])
    result["ngram"] = result["ngram"].apply(lambda g: " ".join(g))
    return result


# ──────────────────────────────────────────────
# 3. Rating Correlation — TF-IDF per star bucket
# ──────────────────────────────────────────────

def tfidf_by_rating(
    df: pd.DataFrame,
    low_stars: list[int] = [1, 2],
    high_stars: list[int] = [4, 5],
    top_k: int = 15,
    text_col: str = "cleaned_text",
) -> dict[str, pd.DataFrame]:
    """
    Compute TF-IDF scores for low-star vs high-star reviews.

    Returns a dict with keys 'negative' and 'positive', each a DataFrame
    of (term, tfidf_score) sorted descending.
    """
    print("\n[Analyzer] Running TF-IDF correlation analysis …")
    low_df  = df[df["star_rating"].isin(low_stars)][text_col].dropna()
    high_df = df[df["star_rating"].isin(high_stars)][text_col].dropna()

    def _top_terms(corpus: pd.Series, label: str) -> pd.DataFrame:
        if corpus.empty:
            return pd.DataFrame(columns=["term", "tfidf_score"])
        vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(corpus)
        scores = matrix.mean(axis=0).A1
        terms  = vectorizer.get_feature_names_out()
        top_idx = scores.argsort()[::-1][:top_k]
        return pd.DataFrame({
            "term": terms[top_idx],
            "tfidf_score": scores[top_idx].round(4),
            "group": label,
        })

    return {
        "negative": _top_terms(low_df,  "Negative (1–2 ★)"),
        "positive": _top_terms(high_df, "Positive (4–5 ★)"),
    }


# ──────────────────────────────────────────────
# 4. Actionable Complaint Extraction
# ──────────────────────────────────────────────

COMPLAINT_KEYWORDS = [
    "crash", "bug", "fix", "broken", "error", "fail", "freeze", "slow",
    "lag", "issue", "problem", "login", "load", "payment", "subscription",
    "expensive", "charge", "battery", "drain", "uninstall", "refund",
    "glitch", "ads", "ad", "annoying", "worst", "terrible", "awful",
]


def extract_actionable_complaints(
    df: pd.DataFrame,
    text_col: str = "cleaned_text",
    rating_threshold: int = 3,
) -> pd.DataFrame:
    """
    Filter reviews that are:
      - Low star (≤ rating_threshold)
      - Contain at least one complaint keyword
    Returns ranked by thumbsUpCount (most upvoted first).
    """
    print("\n[Analyzer] Extracting actionable complaints …")
    mask = df["star_rating"] <= rating_threshold

    def _has_keyword(text: str) -> bool:
        if not isinstance(text, str):
            return False
        return any(kw in text for kw in COMPLAINT_KEYWORDS)

    mask &= df[text_col].apply(_has_keyword)
    complaints = df[mask].copy()

    if "thumbsUpCount" in complaints.columns:
        complaints.sort_values("thumbsUpCount", ascending=False, inplace=True)

    print(f"[Analyzer] Found {len(complaints)} actionable complaints.")
    return complaints.reset_index(drop=True)


# ──────────────────────────────────────────────
# 5. Trend: Monthly Negative Sentiment
# ──────────────────────────────────────────────

def monthly_sentiment_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates average sentiment score and negative % by month.
    Requires 'date' column parseable as datetime.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = df["date"].dt.to_period("M")

    monthly = (
        df.groupby("month")
        .agg(
            avg_sentiment=("sentiment_score", "mean"),
            total=("sentiment_score", "count"),
            negative=("sentiment_label", lambda s: (s == "Negative").sum()),
        )
        .reset_index()
    )
    monthly["negative_pct"] = (monthly["negative"] / monthly["total"] * 100).round(2)
    monthly["month"] = monthly["month"].astype(str)
    return monthly


if __name__ == "__main__":
    # Smoke test
    sample = pd.DataFrame({
        "cleaned_text": [
            "app crash login fail terrible",
            "love music great sound quality",
            "subscription expensive cancel refund",
            "amazing playlist discover song",
        ],
        "star_rating": [1, 5, 2, 4],
        "tokens": [
            ["app", "crash", "login", "fail"],
            ["love", "music", "great", "sound"],
            ["subscription", "expensive", "cancel", "refund"],
            ["amazing", "playlist", "discover", "song"],
        ],
        "thumbsUpCount": [120, 5, 88, 3],
        "date": pd.to_datetime(["2024-01-10", "2024-01-15", "2024-02-05", "2024-02-20"]),
    })
    df_with_sentiment = add_sentiment(sample)
    print(df_with_sentiment[["cleaned_text", "sentiment_score", "sentiment_label"]])

    bigrams = get_top_ngrams(df_with_sentiment, n=2, top_k=5)
    print("\nTop Bigrams:\n", bigrams)

    tfidf = tfidf_by_rating(df_with_sentiment)
    print("\nNegative TF-IDF:\n", tfidf["negative"])

    complaints = extract_actionable_complaints(df_with_sentiment)
    print("\nActionable Complaints:\n", complaints[["cleaned_text", "star_rating", "thumbsUpCount"]])
