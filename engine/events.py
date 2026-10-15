"""Event classification.

Two stages:
  1. Keyword rules  - fast and precise for obvious headlines.
  2. Zero-shot NLI  - fallback for text the rules do not recognise.

classify(text) -> (label, confidence, method)
"""
import os
import re
from functools import lru_cache

from .config import EVENT_LABELS, OTHER_LABEL, ZERO_SHOT_MIN_SCORE, ZERO_SHOT_MODEL

_RULES = {
    "Earnings": [
        r"\bearnings\b", r"\bEPS\b", r"\brevenue\b", r"\bquarterly (results|profit|loss)\b",
        r"\bguidance\b", r"\bbeats?\b.*\b(estimates|expectations|forecasts?)\b",
        r"\bmisses?\b.*\b(estimates|expectations|forecasts?)\b", r"\bprofit warning\b",
    ],
    "Merger/Acquisition": [
        r"\bacqui(re|res|red|sition)\b", r"\bmerger\b", r"\bmerges?\b", r"\btakeover\b",
        r"\bbuyout\b", r"\bdeal to buy\b", r"\bbid for\b", r"\bto buy\b.*\bfor \$",
    ],
    "Credit Event": [
        r"\bdowngrad(e|es|ed)\b", r"\bdefault(s|ed)?\b", r"\bbankruptcy\b", r"\bchapter 11\b",
        r"\bcredit rating\b", r"\bjunk\b", r"\b(loan|credit) losses\b", r"\bwrite-?downs?\b",
        r"\bdebt (crisis|load|restructuring)\b", r"\bbond (selloff|yields? spike)\b",
    ],
    "Geopolitical": [
        r"\bwar\b", r"\bsanctions?\b", r"\bceasefire\b", r"\bmiddle east\b", r"\bukraine\b",
        r"\brussia\b", r"\btariffs?\b", r"\bembargo\b", r"\bmilitary\b", r"\bgeopolitic\w*",
        r"\btrade war\b", r"\bconflict\b", r"\bmissile\b",
    ],
    "Macroeconomic": [
        r"\binflation\b", r"\binterest rates?\b", r"\bfederal reserve\b", r"\bthe fed\b",
        r"\brate (cut|hike)s?\b", r"\bGDP\b", r"\bunemployment\b", r"\brecession\b",
        r"\bjobs report\b", r"\boil prices?\b", r"\bconsumer spending\b", r"\bCPI\b",
    ],
    "Regulatory": [
        r"\bFAA\b", r"\bSEC\b", r"\bFDA\b", r"\bDOJ\b", r"\bantitrust\b", r"\bfined?\b",
        r"\binvestigation\b", r"\blawsuits?\b", r"\bsues?\b", r"\brecalls?\b", r"\bprobe\b",
        r"\bregulators?\b", r"\bsettlement\b", r"\bapproval\b",
    ],
    "Product Launch": [
        r"\blaunch(es|ed)?\b", r"\bunveil(s|ed)?\b", r"\bnew (iphone|product|model|chip|service)\b",
        r"\breleases?\b", r"\bannounces? (a )?new\b", r"\brolls? out\b",
    ],
    "Leadership Change": [
        r"\bsteps? down\b", r"\bresign(s|ed)?\b", r"\bappoint(s|ed)?\b", r"\bnew CEO\b",
        r"\bCEO (exit|departure|succession)\b", r"\bfired\b", r"\bnamed (chief|CEO|CFO)\b",
    ],
}

_COMPILED = {
    label: [re.compile(p, re.IGNORECASE) for p in patterns]
    for label, patterns in _RULES.items()
}

# Ties between event types are broken in this order (most severe first).
_PRIORITY = ["Credit Event", "Geopolitical", "Macroeconomic", "Regulatory",
             "Merger/Acquisition", "Earnings", "Leadership Change", "Product Launch"]


def _rule_match(text: str):
    """Return (label, confidence) from keyword rules, or None."""
    hits = {label: sum(1 for p in pats if p.search(text))
            for label, pats in _COMPILED.items()}
    best = max(hits.values())
    if best == 0:
        return None
    label = next(l for l in _PRIORITY if hits[l] == best)
    return label, round(min(0.95, 0.6 + 0.15 * (best - 1)), 2)


@lru_cache(maxsize=1)
def _zero_shot():
    from transformers import pipeline
    return pipeline("zero-shot-classification", model=ZERO_SHOT_MODEL)


def _model_match(text: str):
    result = _zero_shot()(text, EVENT_LABELS, hypothesis_template="This text is about {}.")
    label, score = result["labels"][0], float(result["scores"][0])
    if score < ZERO_SHOT_MIN_SCORE:
        return OTHER_LABEL, round(score, 2)
    return label, round(score, 2)


def classify(text: str, use_model: bool | None = None):
    """Return (event_label, confidence, method) where method is rules|zero-shot|none."""
    if use_model is None:
        use_model = os.environ.get("RISK_ZERO_SHOT", "1") != "0"
    matched = _rule_match(text)
    if matched:
        return matched[0], matched[1], "rules"
    if use_model:
        label, conf = _model_match(text)
        return label, conf, "zero-shot"
    return OTHER_LABEL, 0.0, "none"
