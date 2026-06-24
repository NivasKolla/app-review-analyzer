"""
Phase A: Data Collection
Scrapes Google Play Store reviews for a given app.
"""

import pandas as pd
from google_play_scraper import reviews, Sort
from tqdm import tqdm
import os


def scrape_reviews(app_id: str, count: int = 2000, lang: str = "en", country: str = "us") -> pd.DataFrame:
    """
    Scrapes reviews from Google Play Store.

    Args:
        app_id  : Package name, e.g. 'com.spotify.music'
        count   : Total number of reviews to fetch
        lang    : Language filter
        country : Country filter

    Returns:
        DataFrame with columns: reviewId, userName, content, score, thumbsUpCount, at
    """
    print(f"\n[Scraper] Fetching up to {count} reviews for '{app_id}' ...")

    all_reviews = []
    batch_size = 200
    continuation_token = None

    with tqdm(total=count, desc="Scraping", unit="reviews") as pbar:
        while len(all_reviews) < count:
            fetch_count = min(batch_size, count - len(all_reviews))
            result, continuation_token = reviews(
                app_id,
                lang=lang,
                country=country,
                sort=Sort.NEWEST,
                count=fetch_count,
                continuation_token=continuation_token,
            )

            if not result:
                break

            all_reviews.extend(result)
            pbar.update(len(result))

            if continuation_token is None:
                break

    df = pd.DataFrame(all_reviews)

    # Keep only the columns we need
    columns_needed = ["reviewId", "userName", "content", "score", "thumbsUpCount", "at"]
    df = df[[c for c in columns_needed if c in df.columns]]
    df.rename(columns={"content": "review_text", "score": "star_rating", "at": "date"}, inplace=True)
    df.dropna(subset=["review_text"], inplace=True)
    df.drop_duplicates(subset=["reviewId"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    os.makedirs("data", exist_ok=True)
    out_path = f"data/{app_id.replace('.', '_')}_raw.csv"
    df.to_csv(out_path, index=False)
    print(f"[Scraper] Saved {len(df)} reviews → {out_path}")
    return df


if __name__ == "__main__":
    # Quick test with Spotify
    df = scrape_reviews("com.spotify.music", count=500)
    print(df.head())
    print(df["star_rating"].value_counts())
