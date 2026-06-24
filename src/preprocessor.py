"""
Phase B: Text Pre-processing
Cleans and normalises raw review text for NLP analysis.
"""

import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK assets (only on first run)
def _download_nltk():
    resources = ["punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"]
    for r in resources:
        try:
            nltk.download(r, quiet=True)
        except Exception:
            pass

_download_nltk()

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# Extra domain-specific noise words to ignore
CUSTOM_NOISE = {"app", "apps", "update", "version", "please", "really", "would", "could"}


def remove_noise(text: str) -> str:
    """Strip emojis, URLs, HTML tags, punctuation, and special characters."""
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    # Remove emojis (broad Unicode range)
    text = re.sub(
        r"[\U00010000-\U0010ffff"
        r"\U0001F600-\U0001F64F"
        r"\U0001F300-\U0001F5FF"
        r"\U0001F680-\U0001F6FF"
        r"\U0001F1E0-\U0001F1FF]+",
        "",
        text,
        flags=re.UNICODE,
    )
    # Remove special characters and digits
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_text(text: str) -> str:
    """Full pipeline: lowercase → denoise → tokenize → remove stops → lemmatize."""
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = remove_noise(text)

    tokens = word_tokenize(text)

    cleaned = [
        LEMMATIZER.lemmatize(tok)
        for tok in tokens
        if tok not in STOP_WORDS
        and tok not in CUSTOM_NOISE
        and tok not in string.punctuation
        and len(tok) > 2
    ]

    return " ".join(cleaned)


def preprocess_dataframe(df: pd.DataFrame, text_col: str = "review_text") -> pd.DataFrame:
    """
    Apply full preprocessing pipeline to every review in the DataFrame.

    Adds two new columns:
      - cleaned_text : lemmatised, stop-word-free text
      - tokens       : list of individual tokens
    """
    print("\n[Preprocessor] Cleaning review text …")
    df = df.copy()
    df["cleaned_text"] = df[text_col].apply(preprocess_text)
    df["tokens"] = df["cleaned_text"].apply(lambda t: t.split() if t else [])

    # Drop rows where cleaning produced nothing useful
    df = df[df["cleaned_text"].str.strip().astype(bool)].reset_index(drop=True)
    print(f"[Preprocessor] {len(df)} reviews remain after cleaning.")
    return df


if __name__ == "__main__":
    sample = pd.DataFrame({
        "review_text": [
            "This app 😡 CRASHES every time I open it!!! #Broken @devs fix it NOW",
            "Absolutely love it! Best music app ever 🎵",
            "Login keeps failing after the latest update. Very frustrating.",
        ],
        "star_rating": [1, 5, 2],
    })
    result = preprocess_dataframe(sample)
    print(result[["review_text", "cleaned_text", "tokens"]])
