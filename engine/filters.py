"""Guardrails: drop obvious spam / pump-and-dump posts before they move the index."""
import re

_SPAM_PATTERNS = [re.compile(p, re.IGNORECASE) for p in (
    r"\bguaranteed\b", r"\btelegram\b", r"\bwhatsapp\b", r"\bdm me\b", r"\bjoin my\b",
    r"\bfree signals?\b", r"\b\d{3,}%\s*(gains?|returns?|profit)", r"\b\d{2,}x\b.*\bguaranteed\b",
    r"\blink in bio\b", r"\bpromo code\b", r"\bpump\b", r"\bsign up now\b",
)]
_CASHTAG = re.compile(r"\$[A-Za-z]{1,5}\b")


def is_spam(text: str) -> bool:
    if any(p.search(text) for p in _SPAM_PATTERNS):
        return True
    # a post tagging 4+ different tickers is almost always promotion
    return len({t.upper() for t in _CASHTAG.findall(text)}) >= 4
