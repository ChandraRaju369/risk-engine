# Engine accuracy against human labels

Labelled by hand: 100 items for sentiment, 100 for event type.

## Sentiment

- Agreement with the human label: **63%** (95% interval 53%–72%), macro-F1 0.61
- Guessing the most common label ('neutral') would score 57%
- When the human saw clear good or bad news (43 items), the engine got the direction right **86%** of the time (73%–93%)
- Scores within ±0.25 are counted as neutral

| Human label | engine positive | engine neutral | engine negative |
|---|---|---|---|
| positive | 15 | 10 | 2 |
| neutral | 9 | 37 | 11 |
| negative | 0 | 5 | 11 |

| Class | Precision | Recall | F1 | Items |
|---|---|---|---|---|
| positive | 0.62 | 0.56 | 0.59 | 27 |
| neutral | 0.71 | 0.65 | 0.68 | 57 |
| negative | 0.46 | 0.69 | 0.55 | 16 |

Worst misses (most confident wrong answers):

- "Despite Tariff Fears, Market Shows Signs Of Resilience As Retail Earnings Continue" — human: neutral, engine score: +0.93
- "Mid-Afternoon Market Update: Dow Rises Over 200 Points; Ocugen Shares Spike Higher" — human: neutral, engine score: +0.90
- "Pete Najarian Sees Unusual Activity In Bank Of America And Macy's" — human: neutral, engine score: -0.90
- "Study: ViiV Healthcare's 2-Drug HIV Regimen As Effective As 3" — human: neutral, engine score: +0.88
- "Semiconductor Equipment Billings Stay above $2B in June" — human: neutral, engine score: +0.88
- "How 7 New Laws Affected Stocks In The Last Week" — human: neutral, engine score: -0.85
- "Mid-Morning Market Update: Markets Open Higher; Deere Earnings Miss Views" — human: neutral, engine score: +0.85
- "Buy Square (SQ) Stock Still on the Dip Ahead of Q1 2019 Earnings?" — human: neutral, engine score: -0.84

## Event type

- Agreement with the human label: **67%** (95% interval 57%–75%) on 100 items
- Answered by none: 69 items, 74% correct
- Answered by rules: 31 items, 52% correct

Most common mix-ups (human → engine):

- Regulatory → Other: 6
- Other → Credit Event: 5
- Product Launch → Other: 3
- Macroeconomic → Other: 3
- Geopolitical → Other: 2

_With about 100 items the interval is roughly ±9 percentage points: treat the numbers as an estimate, not a precise score._