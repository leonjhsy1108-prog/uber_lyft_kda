import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import OUT_ROOT, PRICE_BANDS, SEED, regression_metrics

PRED = OUT_ROOT / "initial_analysis" / "test_predictions.csv"
OUT = OUT_ROOT / "diagnostics"
OUT.mkdir(parents=True, exist_ok=True)
if not PRED.exists():
    raise SystemExit(f"{PRED} not found - run analysis.py first.")

test = pd.read_csv(PRED, parse_dates=["local_date"])
MODELS = {"pred_rf": "Random forest",
          "pred_linear_interacted": "Linear + product x distance",
          "pred_linear": "Linear regression"}
for col in MODELS:
    test[f"residual_{col}"] = test["price"] - test[col]        # positive = underprediction
    test[f"abs_error_{col}"] = test[f"residual_{col}"].abs()
test["residual"], test["absolute_error"] = test["residual_pred_rf"], test["abs_error_pred_rf"]
test["price_band"] = pd.cut(test["price"], **PRICE_BANDS)

overall = pd.DataFrame([{"model": label, "test_rows": len(test),
                         **regression_metrics(test["price"], test[col]),
                         "mean_residual": test[f"residual_{col}"].mean()}
                        for col, label in MODELS.items()])
print("Overall held-out diagnostics:\n", overall.round(4).to_string(index=False))
overall.to_csv(OUT / "overall_metrics.csv", index=False)

by_product = (test.groupby("name", observed=True)
              .agg(count=("price", "size"), MAE=("absolute_error", "mean"),
                   mean_residual=("residual", "mean"), mean_actual=("price", "mean"))
              .sort_values("MAE", ascending=False))
print("\nForest error by ride product:\n", by_product.round(3).to_string())
by_product.to_csv(OUT / "error_by_product.csv")

by_band = (test.groupby("price_band", observed=True)
           .agg(count=("price", "size"), MAE=("absolute_error", "mean"),
                mean_residual=("residual", "mean"), mean_actual=("price", "mean")))
print("\nForest error by actual price range:\n", by_band.round(3).to_string())
by_band.to_csv(OUT / "error_by_price_band.csv")

band_compare = test.groupby("price_band", observed=True).agg(
    count=("price", "size"),
    **{f"MAE_{label}": (f"abs_error_{col}", "mean") for col, label in MODELS.items()},
    **{f"mean_residual_{label}": (f"residual_{col}", "mean") for col, label in MODELS.items()},
)
print("\nAll models by price band:\n", band_compare.round(3).to_string())
band_compare.to_csv(OUT / "error_by_price_band_all_models.csv")

by_company = (test.groupby("cab_type", observed=True)
              .agg(count=("price", "size"), MAE=("absolute_error", "mean"),
                   mean_residual=("residual", "mean")))
print("\nForest error by company:\n", by_company.round(3).to_string())
by_company.to_csv(OUT / "error_by_company.csv")

lyft = test.loc[test["cab_type"] == "Lyft"]
by_surge = (lyft.groupby("surge_multiplier", observed=True)
            .agg(count=("price", "size"), MAE=("absolute_error", "mean"),
                 mean_residual=("residual", "mean"), mean_actual=("price", "mean")))
print("\nForest error for Lyft by recorded surge:\n", by_surge.round(3).to_string())
by_surge.to_csv(OUT / "lyft_error_by_surge.csv")

plot_sample = test.sample(n=min(5_000, len(test)), random_state=SEED)
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(plot_sample["price"], plot_sample["pred_rf"], s=8, alpha=0.25)
lims = [0, max(plot_sample["price"].max(), plot_sample["pred_rf"].max()) * 1.02]
ax.plot(lims, lims, linestyle="--", color="black", linewidth=1)
ax.set(xlabel="Actual fare ($)", ylabel="Predicted fare ($)",
       title="Actual vs predicted fares (held-out dates)")
fig.tight_layout()
fig.savefig(OUT / "actual_vs_predicted.png", dpi=160)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(plot_sample["pred_rf"], plot_sample["residual"], s=8, alpha=0.25)
ax.axhline(0, color="black", linestyle="--", linewidth=1)
ax.set(xlabel="Predicted fare ($)", ylabel="Actual minus predicted ($)",
       title="Residuals vs predicted fares")
fig.tight_layout()
fig.savefig(OUT / "residuals.png", dpi=160)

fig, ax = plt.subplots(figsize=(9, 5))
by_product.sort_values("MAE")["MAE"].plot(kind="barh", ax=ax)
ax.set(xlabel="Mean absolute error ($)", ylabel="Ride product",
       title="Forest prediction error by ride product")
fig.tight_layout()
fig.savefig(OUT / "mae_by_product.png", dpi=160)

fig, ax = plt.subplots(figsize=(9, 5))
pos, width = np.arange(len(band_compare)), 0.27
for i, (col, label) in enumerate(MODELS.items()):
    ax.bar(pos + (i - 1) * width, band_compare[f"MAE_{label}"], width, label=label)
ax.set_xticks(pos, band_compare.index.astype(str))
ax.set(ylabel="Held-out MAE ($)", title="Prediction error by fare band, all models")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "mae_by_price_band_all_models.png", dpi=160)

print(f"\nSaved diagnostics to: {OUT}")
