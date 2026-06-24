"""
App Review Analyzer — Main Orchestrator
=========================================
Runs the complete pipeline:
  Phase A → Scrape reviews
  Phase B → Preprocess text
  Phase C → Sentiment + frequency analysis
  Phase D → Generate dashboard

Usage:
  python main.py --app com.spotify.music --count 2000 --name Spotify
  python main.py --app com.whatsapp --count 1500 --name WhatsApp
  python main.py --csv data/my_reviews.csv --name MyApp   # skip scraping, use local CSV
"""

import argparse
import os
import sys
import pandas as pd

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from scraper import scrape_reviews
from preprocessor import preprocess_dataframe
from analyzer import add_sentiment, get_top_ngrams, extract_actionable_complaints
from visualizer import generate_dashboard, generate_tfidf_comparison_chart


def run_pipeline(
    app_id: str | None,
    app_name: str,
    review_count: int = 2000,
    csv_path: str | None = None,
):
    # ── Phase A: Data Collection ─────────────────────────
    if csv_path and os.path.exists(csv_path):
        print(f"\n[Main] Loading reviews from CSV: {csv_path}")
        df = pd.read_csv(csv_path)
    elif app_id:
        df = scrape_reviews(app_id, count=review_count)
    else:
        raise ValueError("Provide either --app <package_id> or --csv <path>")

    print(f"\n[Main] Total reviews loaded: {len(df)}")
    print(df[["review_text", "star_rating"]].head(3))

    # ── Phase B: Pre-processing ──────────────────────────
    df = preprocess_dataframe(df)

    # ── Phase C: Analysis ────────────────────────────────
    df = add_sentiment(df)

    print("\n── Top 10 Bigrams (All Reviews) ──")
    bigrams_all = get_top_ngrams(df, n=2, top_k=10)
    print(bigrams_all.to_string(index=False))

    print("\n── Top 10 Bigrams (1–2 ★ Only) ──")
    bigrams_neg = get_top_ngrams(df, n=2, top_k=10, rating_filter=[1, 2])
    print(bigrams_neg.to_string(index=False))

    complaints = extract_actionable_complaints(df)
    print(f"\n── Top 5 Actionable Complaints ──")
    cols = ["review_text", "star_rating", "sentiment_label"]
    if "thumbsUpCount" in complaints.columns:
        cols.append("thumbsUpCount")
    print(complaints[cols].head(5).to_string(index=False))

    # ── Phase D: Visualisation ───────────────────────────
    generate_dashboard(df, app_name=app_name)
    generate_tfidf_comparison_chart(df, app_name=app_name)

    # ── Save enriched CSV ────────────────────────────────
    os.makedirs("outputs", exist_ok=True)
    out_csv = f"outputs/{app_name.lower().replace(' ', '_')}_enriched.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[Main] Enriched dataset saved → {out_csv}")
    print("\n✅  Pipeline complete!")


def parse_args():
    parser = argparse.ArgumentParser(description="App Review Analyzer")
    parser.add_argument("--app",   type=str, help="Google Play package ID, e.g. com.spotify.music")
    parser.add_argument("--name",  type=str, default="App", help="Human-readable app name")
    parser.add_argument("--count", type=int, default=2000, help="Number of reviews to scrape")
    parser.add_argument("--csv",   type=str, default=None, help="Path to existing CSV (skips scraping)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        app_id=args.app,
        app_name=args.name,
        review_count=args.count,
        csv_path=args.csv,
    )
