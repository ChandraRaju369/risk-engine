"""Step 2 of the accuracy check: compare your labels with the engine's answers.

  python -m validation.score
  python -m validation.score --labels output/label_me.xlsx --threshold 0.25

Writes output/validation.md (paste-ready for the slides) and output/validation.json.
"""
import argparse
import json
import math
from collections import Counter

import pandas as pd

from engine.config import DATA_DIR, EVENT_LABELS, OTHER_LABEL, ROOT

OUTPUT_DIR = ROOT / "output"
CLASSES = ["positive", "neutral", "negative"]


def to_class(score: float, threshold: float) -> str:
    return "positive" if score > threshold else "negative" if score < -threshold else "neutral"


def wilson(correct: int, n: int, z: float = 1.96):
    """95% confidence interval for an accuracy measured on n items."""
    if n == 0:
        return 0.0, 0.0
    p = correct / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def per_class(truth, pred):
    rows = {}
    for c in CLASSES:
        tp = sum(1 for t, p in zip(truth, pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(truth, pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(truth, pred) if t == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows[c] = {"precision": prec, "recall": rec, "f1": f1, "support": sum(1 for t in truth if t == c)}
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default=str(OUTPUT_DIR / "label_me.xlsx"))
    ap.add_argument("--engine", default=str(DATA_DIR / "validation_engine.csv"))
    ap.add_argument("--threshold", type=float, default=0.25,
                    help="|score| below this counts as neutral when turning the -1..1 score into a class")
    args = ap.parse_args()

    human = pd.read_excel(args.labels, sheet_name="Label these")
    engine = pd.read_csv(args.engine)
    df = engine.merge(human[["id", "your_sentiment", "your_event"]], on="id", how="left")
    df["your_sentiment"] = df["your_sentiment"].astype(str).str.strip().str.lower()
    df["your_event"] = df["your_event"].astype(str).str.strip()

    # ---- sentiment ----
    s = df[df["your_sentiment"].isin(CLASSES)].copy()
    if len(s) < 10:
        raise SystemExit(f"Only {len(s)} rows have a sentiment label. Fill column D in the sheet and save it first.")
    s["engine_class"] = s["engine_sentiment"].apply(lambda x: to_class(x, args.threshold))
    truth, pred = s["your_sentiment"].tolist(), s["engine_class"].tolist()
    correct = sum(t == p for t, p in zip(truth, pred))
    lo, hi = wilson(correct, len(s))
    pc = per_class(truth, pred)
    macro_f1 = sum(v["f1"] for v in pc.values() if v["support"]) / max(1, sum(1 for v in pc.values() if v["support"]))
    majority = Counter(truth).most_common(1)[0]
    polar = s[s["your_sentiment"].isin(["positive", "negative"])]
    polar_ok = int(((polar["engine_sentiment"] > 0) == (polar["your_sentiment"] == "positive")).sum())
    plo, phi = wilson(polar_ok, len(polar))
    confusion = {t: {p: sum(1 for a, b in zip(truth, pred) if a == t and b == p) for p in CLASSES} for t in CLASSES}
    by_source = {src: {"n": len(g), "accuracy": float((g["your_sentiment"] == g["engine_class"]).mean())}
                 for src, g in s.groupby("source")}

    wrong = s[s["your_sentiment"] != s["engine_class"]].copy()
    wrong["abs"] = wrong["engine_sentiment"].abs()
    sent_failures = wrong.sort_values("abs", ascending=False).head(8)

    # ---- events ----
    valid_events = set(EVENT_LABELS + [OTHER_LABEL])
    e = df[df["your_event"].isin(valid_events)].copy()
    ev = None
    if len(e) >= 10:
        e["ok"] = e["your_event"] == e["engine_event"]
        elo, ehi = wilson(int(e["ok"].sum()), len(e))
        by_method = {m: {"n": len(g), "accuracy": float(g["ok"].mean())} for m, g in e.groupby("engine_event_method")}
        confusions = Counter((r.your_event, r.engine_event) for r in e[~e["ok"]].itertuples()).most_common(5)
        ev = {"n": len(e), "accuracy": float(e["ok"].mean()), "ci": [elo, ehi], "by_method": by_method,
              "top_confusions": [{"human": h, "engine": g, "count": c} for (h, g), c in confusions],
              "failures": e[~e["ok"]].head(8)[["text", "your_event", "engine_event"]].to_dict("records")}

    result = {
        "threshold": args.threshold,
        "sentiment": {"n": len(s), "accuracy": correct / len(s), "ci": [lo, hi], "macro_f1": macro_f1,
                      "majority_class": majority[0], "majority_baseline": majority[1] / len(s),
                      "polarity_accuracy": polar_ok / max(1, len(polar)), "polarity_n": len(polar),
                      "polarity_ci": [plo, phi], "per_class": pc, "confusion": confusion, "by_source": by_source,
                      "failures": sent_failures[["text", "your_sentiment", "engine_sentiment"]].to_dict("records")},
        "event": ev,
    }
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "validation.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")

    sa = result["sentiment"]
    md = ["# Engine accuracy against human labels", "",
          f"Labelled by hand: {sa['n']} items for sentiment" + (f", {ev['n']} for event type." if ev else "."), "",
          "## Sentiment", "",
          f"- Agreement with the human label: **{sa['accuracy']:.0%}** (95% interval {lo:.0%}–{hi:.0%}), "
          f"macro-F1 {macro_f1:.2f}",
          f"- Guessing the most common label ('{majority[0]}') would score {sa['majority_baseline']:.0%}",
          f"- When the human saw clear good or bad news ({sa['polarity_n']} items), the engine got the direction right "
          f"**{sa['polarity_accuracy']:.0%}** of the time ({plo:.0%}–{phi:.0%})",
          f"- Scores within ±{args.threshold} are counted as neutral", "",
          "| Human label | engine positive | engine neutral | engine negative |", "|---|---|---|---|"]
    for t in CLASSES:
        md.append(f"| {t} | " + " | ".join(str(confusion[t][p]) for p in CLASSES) + " |")
    md += ["", "| Class | Precision | Recall | F1 | Items |", "|---|---|---|---|---|"]
    for c in CLASSES:
        v = pc[c]
        md.append(f"| {c} | {v['precision']:.2f} | {v['recall']:.2f} | {v['f1']:.2f} | {v['support']} |")
    md += ["", "Worst misses (most confident wrong answers):", ""]
    for r in sa["failures"]:
        md.append(f"- \"{r['text'][:110]}\" — human: {r['your_sentiment']}, engine score: {r['engine_sentiment']:+.2f}")
    if ev:
        md += ["", "## Event type", "",
               f"- Agreement with the human label: **{ev['accuracy']:.0%}** (95% interval {ev['ci'][0]:.0%}–{ev['ci'][1]:.0%}) "
               f"on {ev['n']} items"]
        for m, v in ev["by_method"].items():
            md.append(f"- Answered by {m}: {v['n']} items, {v['accuracy']:.0%} correct")
        md += ["", "Most common mix-ups (human → engine):", ""]
        for c in ev["top_confusions"]:
            md.append(f"- {c['human']} → {c['engine']}: {c['count']}")
    md += ["", "_With about 100 items the interval is roughly ±9 percentage points: treat the numbers as an estimate, "
               "not a precise score._"]
    (OUTPUT_DIR / "validation.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nSaved: {OUTPUT_DIR / 'validation.md'}")


if __name__ == "__main__":
    main()
