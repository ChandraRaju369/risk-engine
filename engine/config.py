"""Central configuration: universe, aliases, feeds, paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "signals.db"

# 15-stock mock index (S&P 100 members, spread across sectors)
UNIVERSE = {
    "AAPL": ["apple", "iphone", "tim cook"],
    "MSFT": ["microsoft", "azure", "satya nadella"],
    "GOOGL": ["google", "alphabet", "youtube"],
    "AMZN": ["amazon", "aws", "andy jassy"],
    "NVDA": ["nvidia", "jensen huang"],
    "JPM": ["jpmorgan", "jp morgan", "jamie dimon"],
    "BAC": ["bank of america"],
    "XOM": ["exxon", "exxonmobil"],
    "CVX": ["chevron"],
    "JNJ": ["johnson & johnson", "johnson and johnson"],
    "PFE": ["pfizer"],
    "WMT": ["walmart"],
    "KO": ["coca-cola", "coca cola"],
    "DIS": ["disney"],
    "BA": ["boeing"],
}

# Public RSS feeds used as the "news" source (no API key needed)
NEWS_FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
]

# Source credibility used later by the impact scorer and aggregator
SOURCE_CREDIBILITY = {"news": 1.0, "social": 0.5}

# Zero-shot event labels (used by events.py in the next step)
EVENT_LABELS = [
    "Geopolitical", "Macroeconomic", "Credit Event", "Merger/Acquisition",
    "Earnings", "Product Launch", "Regulatory", "Leadership Change",
]
OTHER_LABEL = "Other"  # no clear event (general chatter)

# Zero-shot fallback model. Set env var RISK_ZERO_SHOT=0 to disable it
# (then anything the keyword rules miss becomes "Other").
ZERO_SHOT_MODEL = "facebook/bart-large-mnli"
ZERO_SHOT_MIN_SCORE = 0.35

# How market-moving each event type tends to be (0..1). Tunable.
BASE_SEVERITY = {
    "Credit Event": 0.95,
    "Geopolitical": 0.90,
    "Macroeconomic": 0.80,
    "Regulatory": 0.70,
    "Merger/Acquisition": 0.70,
    "Earnings": 0.55,
    "Leadership Change": 0.50,
    "Product Launch": 0.40,
    "Other": 0.25,
}
