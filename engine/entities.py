"""Entity linking: map free text to tickers in our universe via aliases."""
import re
from .config import UNIVERSE

_PATTERNS = {}
for _ticker, _aliases in UNIVERSE.items():
    _terms = [re.escape(a) for a in _aliases]
    # $AAPL / AAPL cashtag-or-symbol, or any alias as a whole word
    _terms.append(r"\$?" + re.escape(_ticker))
    _PATTERNS[_ticker] = re.compile(r"\b(?:" + "|".join(_terms) + r")\b", re.IGNORECASE)


def link_tickers(text: str) -> list[str]:
    """Return every universe ticker mentioned in `text` (may be empty)."""
    return [t for t, pat in _PATTERNS.items() if pat.search(text)]
