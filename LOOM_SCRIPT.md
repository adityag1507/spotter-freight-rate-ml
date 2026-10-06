# 2–3 Minute Loom Walkthrough Script

## 0:00–0:25 — Problem and data

“I treated this as a future freight-rate forecasting problem rather than a random regression split. The labeled development data contains 48,000 loads from January through October 2025, while the 12,000 unlabeled validation loads are from November and December.”

## 0:25–0:55 — Data quality

“I found three main data-quality issues. There are 292 negative weights, 300 missing weights, and 374 missing market-index values in development. I converted negative weights to absolute values, added missingness indicators, and imputed numeric values using development-set statistics. I also found a small heavy tail in rate-per-mile, so I used a log target and an L1 objective instead of deleting those observations.”

## 0:55–1:25 — Feature engineering

“The target was normalized to rate per mile, which separates the very strong distance effect from route and market effects. Features include pickup and delivery, route and route-equipment keys, weight-per-mile, calendar seasonality, geographic relationships such as haversine distance and coordinate deltas, and market and quote signals.”

## 1:25–1:55 — Validation and model

“I avoided random cross-validation because it would mix future observations into training. My primary backtest trains on January through August and validates on September and October. I also ran a more recent January through September to October stress test. The selected LightGBM approach achieved about 104 dollars MAE on the primary holdout and 97 dollars MAE on the October stress test.”

## 1:55–2:25 — Final prediction design

“The final validation prediction is a 70/30 blend of a market-aware LightGBM model and a market-agnostic LightGBM model. Both predict log rate-per-mile using a robust L1 objective. For the December chart, the input schema does not contain market or quote signals, so I use a separate chart-safe model that only consumes information actually available in that scenario rather than inventing hidden inputs.”

## 2:25–2:45 — Output and reproducibility

“The repository contains the complete training pipeline, dependencies, the 12,000 validation predictions, the 31 December predictions, and the chart produced by the supplied scorer. Running the documented commands reproduces the output contract.”

## 2:45–3:00 — Close

“The key design principles were leakage-safe chronological validation, robust handling of noisy freight-rate observations, distance-normalized modeling, native categorical route structure, and strict adherence to the submission schema.”
