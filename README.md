# 📱 App Review Analyzer
### Algonive Data Science Task-3 — Automated Complaint Extraction from App Store Reviews

> **Problem Statement:** How can a small software company identify the specific reasons for user frustration without manually reading thousands of app store reviews every week?

---

## 🗂️ Project Structure

```
app-review-analyzer/
│
├── main.py                    # Full pipeline orchestrator
├── demo_with_mock_data.py     # Run without internet (uses synthetic reviews)
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── scraper.py             # Phase A — Google Play review scraper
│   ├── preprocessor.py        # Phase B — Text cleaning & tokenisation
│   ├── analyzer.py            # Phase C — Sentiment, N-gram, TF-IDF analysis
│   └── visualizer.py          # Phase D — Dashboard & charts
│
├── data/                      # Raw scraped CSVs (git-ignored)
└── outputs/                   # Generated charts & enriched CSVs (git-ignored)
```

---

## 🔄 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     APP REVIEW ANALYZER                         │
├──────────┬──────────────┬──────────────────┬────────────────────┤
│ PHASE A  │   PHASE B    │    PHASE C       │     PHASE D        │
│ Scraping │ Preprocessing│    Analysis      │  Visualisation     │
├──────────┼──────────────┼──────────────────┼────────────────────┤
│ Play     │ Lowercase    │ Sentiment Score  │ Sentiment Pie      │
│ Store    │ Denoise      │ (TextBlob)       │ Star Distribution  │
│ API      │ Tokenize     │ Bigram Analysis  │ Trend Line         │
│          │ Stop-word    │ TF-IDF by Rating │ Word Cloud of Pain │
│          │ removal      │ Complaint        │ Top Bigrams Bar    │
│          │ Lemmatize    │ Extraction       │ TF-IDF Comparison  │
└──────────┴──────────────┴──────────────────┴────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/app-review-analyzer.git
cd app-review-analyzer

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Demo (No Internet Required)

```bash
python demo_with_mock_data.py
```

Generates synthetic reviews and produces all charts in `outputs/`.

### 3. Run on a Real App

```bash
# Spotify
python main.py --app com.spotify.music --name Spotify --count 2000

# WhatsApp
python main.py --app com.whatsapp --name WhatsApp --count 1500

# Use your own CSV (must have 'review_text' and 'star_rating' columns)
python main.py --csv data/my_reviews.csv --name MyApp
```

---

## 📊 Output Charts

| Chart | Description |
|---|---|
| **Sentiment Pie** | Share of Positive / Neutral / Negative reviews |
| **Star Distribution** | Bar chart of 1★ → 5★ counts, colour-coded |
| **Monthly Trend Line** | % of negative reviews per month — spots post-update spikes |
| **Word Cloud of Pain** | Most frequent words in 1–2★ reviews |
| **Top Bigrams** | Most common two-word phrases in negative reviews |
| **TF-IDF Comparison** | Key terms that separate complaints from praise |

All charts are saved as high-res PNGs in `outputs/`.

---

## 🧠 Techniques Used

| Phase | Technique | Library |
|---|---|---|
| Extraction | Web Scraping | `google-play-scraper` |
| NLP | Tokenisation, Stop-word removal, Lemmatisation | `NLTK` |
| Analysis | Sentiment Scoring | `TextBlob` |
| Analysis | TF-IDF, Bigrams | `scikit-learn`, `NLTK` |
| Storage | DataFrames | `pandas` |
| Visualisation | Charts & Word Cloud | `matplotlib`, `wordcloud` |

---

## 💡 Key Design Decisions

### Why Bigrams over single words?
Finding **"login fail"** 50 times is far more actionable than finding **"fail"** 50 times. Bigrams preserve the context of complaints.

### Why TF-IDF for rating correlation?
Simple word frequency favours common words. TF-IDF weights terms that are *distinctively* common in low-star reviews, surfacing the real pain points.

### Actionable Complaint Filter
A review is flagged as an **Actionable Complaint** when:
1. Star rating ≤ 3
2. Text contains at least one domain-specific keyword (crash, bug, fail, charge, etc.)
3. Results ranked by `thumbsUpCount` — most community-validated problems first

---

## 📋 Sample Output (Spotify, 2,000 reviews)

```
Sentiment Distribution:
  Negative: 41.2%
  Positive: 38.7%
  Neutral:  20.1%

Top Bigrams in 1–2★ Reviews:
  login fail        87
  app crash         74
  subscription expensive  61
  battery drain     55
  keep logging      48
```

---

## 🧪 Testing Without Scraping

The `demo_with_mock_data.py` script uses 300 hand-crafted synthetic reviews covering common complaint patterns (crashes, billing, login issues) and praise themes. It exercises every stage of the pipeline and produces all four chart types.

---

## 📁 CSV Format (for --csv mode)

Your CSV must contain at minimum:

| Column | Type | Example |
|---|---|---|
| `review_text` | string | "App crashes on login" |
| `star_rating` | int (1–5) | 1 |
| `date` | string/datetime | "2024-03-15" |

Optional: `thumbsUpCount`, `userName`, `reviewId`

---

## 🤝 Contributing

1. Fork → Feature branch → PR
2. Run `python demo_with_mock_data.py` before submitting — all outputs must generate cleanly.

---

## 📜 License

MIT — free to use, modify, and distribute.
