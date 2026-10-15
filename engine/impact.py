"""Impact score (1-10): how market-moving a signal is likely to be.

impact = 1 + 9 * base_severity[event]
             * (0.35 + 0.65 * |sentiment|)      # stronger tone -> bigger impact
             * (0.70 + 0.30 * source_credibility)
             * (0.60 + 0.40 * model_confidence)

Deliberately a transparent formula (not a black box) so every score can be
explained on the dashboard and tuned in config.py.
"""
from .config import BASE_SEVERITY, OTHER_LABEL, SOURCE_CREDIBILITY


def impact_score(event_type: str, sentiment: float, confidence: float, source: str) -> float:
    base = BASE_SEVERITY.get(event_type, BASE_SEVERITY[OTHER_LABEL])
    cred = SOURCE_CREDIBILITY.get(source, 0.5)
    raw = (base
           * (0.35 + 0.65 * abs(sentiment))
           * (0.70 + 0.30 * cred)
           * (0.60 + 0.40 * confidence))
    return round(max(1.0, min(10.0, 1 + 9 * raw)), 1)
