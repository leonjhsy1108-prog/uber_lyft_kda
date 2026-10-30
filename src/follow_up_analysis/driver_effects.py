import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (FAST_TRAIN_ROWS, FAST_TREES, FEATURES, OUT_ROOT, REFERENCE_ROWS, TARGET,
                    chronological_split, load_data, make_forest, sample_rows)

OUT = OUT_ROOT / "effects"
OUT.mkdir(parents=True, exist_ok=True)
data = load_data()
train, test, _ = chronological_split(data)

def fit_forest(frame):
    sample = sample_rows(frame, FAST_TRAIN_ROWS)
    model = make_forest(FEATURES, n_estimators=FAST_TREES)
    model.fit(sample[FEATURES], sample[TARGET])
    return model

def adjusted_curve(model, reference, column, values):
    rows, modified = [], reference[FEATURES].copy()
    for value in values:
        modified[column] = value
        pred = model.predict(modified)
        rows.append({column: value, "adjusted_predicted_price": float(pred.mean()),
                     "adjusted_predicted_price_sd": float(pred.std())})
    return pd.DataFrame(rows)

pooled = fit_forest(train)
reference = sample_rows(test, REFERENCE_ROWS)
grid = np.linspace(train["distance"].quantile(0.05), train["distance"].quantile(0.95), 16)
distance_effect = adjusted_curve(pooled, reference, "distance", grid)
distance_effect.to_csv(OUT / "distance_adjusted_curve.csv", index=False)
print("Adjusted distance curve:\n", distance_effect.round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(distance_effect["distance"], distance_effect["adjusted_predicted_price"], marker="o", markersize=3)
ax.set(xlabel="Distance (miles)", ylabel="Mean model-predicted fare ($)",
       title="Distance: adjusted model predictions")
fig.tight_layout()
fig.savefig(OUT / "distance_effect.png", dpi=160)

lyft_train, lyft_test = train.loc[train["cab_type"] == "Lyft"], test.loc[test["cab_type"] == "Lyft"]
lyft_forest = fit_forest(lyft_train)
lyft_reference = sample_rows(lyft_test, REFERENCE_ROWS)
levels = lyft_train["surge_multiplier"].value_counts().sort_index()
surge_values = sorted(levels[levels >= 30].index.tolist())
surge_effect = adjusted_curve(lyft_forest, lyft_reference, "surge_multiplier", surge_values)
surge_effect["train_count_at_level"] = surge_effect["surge_multiplier"].map(levels).astype(int)
surge_effect.to_csv(OUT / "lyft_surge_adjusted_curve.csv", index=False)
print("\nLyft adjusted surge curve:\n", surge_effect.round(3).to_string(index=False))
if surge_effect["adjusted_predicted_price"].round(6).duplicated().any():
    print("WARNING: two surge levels give identical predictions - the forest is not "
          "splitting between them (too few training rows at the top levels).")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(surge_effect["surge_multiplier"], surge_effect["adjusted_predicted_price"], marker="o")
for _, r in surge_effect.iterrows():
    ax.annotate(f"n={int(r['train_count_at_level']):,}", (r["surge_multiplier"], r["adjusted_predicted_price"]),
                textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
ax.set(xlabel="Recorded Lyft surge multiplier", ylabel="Mean model-predicted fare ($)",
       title="Lyft surge: adjusted model predictions (n = training rows at level)")
fig.tight_layout()
fig.savefig(OUT / "lyft_surge_effect.png", dpi=160)

product_prices = (data.groupby(["cab_type", "name"], observed=True)["price"]
                  .agg(count="size", average_price="mean", median_price="median")
                  .sort_values("average_price"))
product_prices.to_csv(OUT / "product_descriptive_prices.csv")
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh([f"{c}: {n}" for c, n in product_prices.index], product_prices["average_price"])
ax.set(xlabel="Average observed fare ($)", title="Ride-product prices (unadjusted descriptive comparison)")
fig.tight_layout()
fig.savefig(OUT / "product_average_price.png", dpi=160)

print(f"\nSaved effect curves to: {OUT}")
print("Caution: adjusted curves change one feature inside a model; they are not causal estimates.")
