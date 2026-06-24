"""
Demo script — runs the full pipeline on synthetic reviews.
No internet / API key required. Great for testing and submission demos.

Usage:
  python demo_with_mock_data.py
"""

import os
import sys
import random
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocessor import preprocess_dataframe
from analyzer import add_sentiment, get_top_ngrams, extract_actionable_complaints
from visualizer import generate_dashboard, generate_tfidf_comparison_chart

# ── Synthetic review corpus ──────────────────────────────────
NEGATIVE_REVIEWS = [
    "The app crashes every single time I try to log in. Completely broken after the update.",
    "Login fails constantly. I've tried reinstalling three times and nothing works.",
    "My subscription was charged but I can't access premium features. Want a refund now.",
    "App freezes on the home screen. Waste of time and money. Uninstalling.",
    "Terrible performance after latest update. Battery drain is insane.",
    "Ads are so annoying and intrusive. The free version is basically unusable.",
    "Keeps logging me out. Password reset loop is a nightmare. Fix this bug!",
    "Slow loading, constant buffering, and now it won't even open. 1 star.",
    "Payment went through but no premium access. Customer support never replied.",
    "Interface is confusing and buggy. Buttons don't respond half the time.",
    "Downloaded the update and now it crashes immediately. Broken and awful.",
    "Why did you remove the dark mode? The bright screen hurts my eyes at night.",
    "Subscription is way too expensive for what you actually get. Cancel!",
    "Error 503 every morning. How is this still not fixed?",
    "Glitch makes the app restart mid-session. So frustrating!",
    "Can't download songs for offline play despite paying for premium.",
    "The login page is broken on Android 14. Please fix it urgently.",
    "Worst update ever. Lost all my playlists. The app is a disaster.",
    "App drains 30% battery in one hour. Totally unacceptable.",
    "Search is completely broken. Can't find anything anymore.",
]

POSITIVE_REVIEWS = [
    "Absolutely love this app! Best music experience I've ever had.",
    "The playlist recommendations are spot-on. Discover Weekly is amazing.",
    "Seamless offline playback. Works perfectly on my commute.",
    "Clean interface, great sound quality, and the library is massive.",
    "Worth every penny of the subscription. Can't imagine life without it.",
    "Customer support resolved my issue in minutes. Impressed!",
    "The new update is smooth and fast. Love the redesigned player.",
    "Cross-device sync works flawlessly. Started on phone, finished on laptop.",
    "Podcasts + music in one app is genius. 5 stars no question.",
    "Sound quality is exceptional especially with headphones.",
    "Family plan is a great deal. All my kids love it too.",
    "The collaborative playlist feature is so fun to use with friends.",
    "Lyrics are now synced in real time. Such a nice touch!",
    "Downloaded 500 songs with no issues. Rock solid offline mode.",
    "The equalizer settings are fantastic for audiophiles.",
]

NEUTRAL_REVIEWS = [
    "It's okay. Does what it's supposed to do, nothing more.",
    "Average app. Some features work, others feel half-baked.",
    "Decent for the price but not the best in the market.",
    "Works fine on my device. Haven't had any major issues so far.",
    "Middling experience. Some days great, other days buggy.",
]


def generate_mock_data(n: int = 300) -> pd.DataFrame:
    records = []
    base_date = datetime(2024, 1, 1)

    for i in range(n):
        # Weight distribution towards negative for realism
        roll = random.random()
        if roll < 0.40:
            text   = random.choice(NEGATIVE_REVIEWS)
            rating = random.choice([1, 2])
        elif roll < 0.65:
            text   = random.choice(NEUTRAL_REVIEWS)
            rating = 3
        else:
            text   = random.choice(POSITIVE_REVIEWS)
            rating = random.choice([4, 5])

        date = base_date + timedelta(days=random.randint(0, 180))

        records.append({
            "reviewId":      f"mock_{i:04d}",
            "userName":      f"User_{i}",
            "review_text":   text,
            "star_rating":   rating,
            "thumbsUpCount": random.randint(0, 200) if rating in [1, 2] else random.randint(0, 20),
            "date":          date.strftime("%Y-%m-%d"),
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    print("=" * 60)
    print("  App Review Analyzer — Demo (Synthetic Data)")
    print("=" * 60)

    # Phase A (mocked)
    df = generate_mock_data(n=300)
    print(f"\n[Demo] Generated {len(df)} synthetic reviews.")

    # Phase B
    df = preprocess_dataframe(df)

    # Phase C
    df = add_sentiment(df)

    print("\n── Top Bigrams (1–2 ★ Reviews) ──")
    neg_bigrams = get_top_ngrams(df, n=2, top_k=10, rating_filter=[1, 2])
    print(neg_bigrams.to_string(index=False))

    complaints = extract_actionable_complaints(df)
    print(f"\n── Actionable Complaints (top 5) ──")
    print(complaints[["review_text", "star_rating", "thumbsUpCount"]].head(5).to_string(index=False))

    # Phase D
    generate_dashboard(df, app_name="DemoApp")
    generate_tfidf_comparison_chart(df, app_name="DemoApp")

    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/demo_enriched.csv", index=False)
    print("\n✅  Demo complete! Check the 'outputs/' folder for charts.")
