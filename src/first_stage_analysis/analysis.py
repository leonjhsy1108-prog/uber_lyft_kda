import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from common import (CATEGORICAL, FEATURES, GROUPS, MAIN_TRAIN_ROWS, NUMERICAL, OUT_ROOT,
                    PERM_REPEATS, PERM_ROWS, SEED, SHAP_ROWS, TARGET, barh_with_error,
                    chronological_split, driver_of_output_columns,
                    grouped_permutation_importance, load_data, make_forest,
                    regression_metrics, sample_rows)

OUT = OUT_ROOT / "initial_analysis"
OUT.mkdir(parents=True, exist_ok=True)

data = load_data()
train, test, split_day = chronological_split(data)
train_sample = sample_rows(train, MAIN_TRAIN_ROWS)
print(f"All models below are fitted on the same {len(train_sample):,}-row training sample.")

PRODUCTS = sorted(train_sample["name"].unique())
INTERACTION_COLS = [f"dist_x_{p}" for p in PRODUCTS]

def add_interactions(frame):
    out = frame.copy()
    for product, col in zip(PRODUCTS, INTERACTION_COLS):
        out[col] = frame["distance"] * (frame["name"] == product)
    return out

def linear_pipeline(numerical, categorical, fit_intercept=True):
    return Pipeline([
        ("preprocessor", ColumnTransformer([
            ("num", "passthrough", numerical),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ])),
        ("model", LinearRegression(fit_intercept=fit_intercept)),
    ])

train_int, test_int = add_interactions(train_sample), add_interactions(test)
other_numerical = [c for c in NUMERICAL if c != "distance"]

specs = {
    "Linear regression": (
        linear_pipeline(NUMERICAL, CATEGORICAL), FEATURES, train_sample, test),
    "Linear + product x distance": (
        linear_pipeline(other_numerical + INTERACTION_COLS, CATEGORICAL),
        other_numerical + INTERACTION_COLS + CATEGORICAL, train_int, test_int),
    "Random forest": (
        make_forest(FEATURES), FEATURES, train_sample, test),
}

rows, predictions = [], test[["id", "local_date", "cab_type", "name", "price",
                              "distance", "surge_multiplier", "local_hour"]].copy()
for label, (model, cols, tr, te) in specs.items():
    start = time.time()
    model.fit(tr[cols], tr[TARGET])
    pred = model.predict(te[cols])
    key = {"Linear regression": "pred_linear",
           "Linear + product x distance": "pred_linear_interacted",
           "Random forest": "pred_rf"}[label]
    predictions[key] = pred
    rows.append({"model": label, "train_rows": len(tr), "test_rows": len(te),
                 **regression_metrics(te[TARGET], pred), "fit_seconds": time.time() - start})
    print(f"  {label}: fitted in {time.time() - start:,.0f}s")

metrics = pd.DataFrame(rows).sort_values("RMSE")
print("\nHeld-out performance (chronological test period):")
print(metrics[["model", "train_rows", "MAE", "RMSE", "R2"]].round(4).to_string(index=False))
metrics.to_csv(OUT / "model_performance.csv", index=False)
predictions.to_csv(OUT / "test_predictions.csv", index=False)
forest = specs["Random forest"][0]

for frame in (train_int, test_int):
    frame["surge_excess"] = frame["surge_multiplier"] - 1
card_cols = ["surge_excess"] + INTERACTION_COLS + ["name"]
rate_card = linear_pipeline(["surge_excess"] + INTERACTION_COLS, ["name"], fit_intercept=False)
rate_card.fit(train_int[card_cols], train_int[TARGET])
coef = dict(zip(rate_card.named_steps["preprocessor"].get_feature_names_out(),
                rate_card.named_steps["model"].coef_))
card = pd.DataFrame({
    "product": PRODUCTS,
    "base_fare_usd": [coef[f"cat__name_{p}"] for p in PRODUCTS],
    "per_mile_usd": [coef[f"num__dist_x_{p}"] for p in PRODUCTS],
})
card["usd_per_unit_surge_above_1"] = coef["num__surge_excess"]
card = card.sort_values("base_fare_usd")
card_pred = rate_card.predict(test_int[card_cols])
card_metrics = regression_metrics(test[TARGET], card_pred)
print(f"\nRate-card linear model (product + product x distance + surge only): "
      f"test R² = {card_metrics['R2']:.4f}, MAE = ${card_metrics['MAE']:.3f}")
print(card.round(2).to_string(index=False))
card.to_csv(OUT / "linear_rate_card.csv", index=False)

perm_rows = sample_rows(test, PERM_ROWS)
perm = permutation_importance(forest, perm_rows[FEATURES], perm_rows[TARGET],
                              scoring="r2", n_repeats=PERM_REPEATS, random_state=SEED, n_jobs=1)
ranking = pd.DataFrame({"driver": FEATURES,
                        "importance_mean_r2_drop": perm.importances_mean,
                        "importance_std": perm.importances_std}
                       ).sort_values("importance_mean_r2_drop", ascending=False)
print(f"\nPermutation importance ({len(perm_rows):,} held-out rows x {PERM_REPEATS} repeats):")
print(ranking.round(4).to_string(index=False))
ranking.to_csv(OUT / "driver_importance.csv", index=False)

group_rank = grouped_permutation_importance(forest, perm_rows[FEATURES], perm_rows[TARGET], GROUPS)
print("\nGrouped permutation importance (columns of a group shuffled together):")
print(group_rank.round(4).to_string(index=False))
group_rank.to_csv(OUT / "group_importance.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
d = ranking.sort_values("importance_mean_r2_drop")
barh_with_error(axes[0], d["driver"], d["importance_mean_r2_drop"], d["importance_std"],
                "Mean drop in held-out R² (± SD)", "Individual drivers")
g = group_rank.sort_values("importance_mean_r2_drop")
barh_with_error(axes[1], g["group"], g["importance_mean_r2_drop"], g["importance_std"],
                "Mean drop in held-out R² (± SD)", "Driver groups")
fig.suptitle("Random forest permutation importance")
fig.tight_layout()
fig.savefig(OUT / "driver_importance.png", dpi=160)

try:
    import shap
except ImportError:
    shap = None
    print("\n`shap` is not installed - skipping additive importance shares. "
          "Install with:  python -m pip install shap")

if shap is not None:
    start = time.time()
    pre, rf = forest.named_steps["preprocessor"], forest.named_steps["model"]
    shap_rows = sample_rows(test, SHAP_ROWS, seed=SEED + 1)
    X_t = pre.transform(shap_rows[FEATURES])
    X_t = X_t.toarray() if hasattr(X_t, "toarray") else np.asarray(X_t)
    values = shap.TreeExplainer(rf).shap_values(X_t, check_additivity=False)
    drivers = driver_of_output_columns(pre)
    per_driver = pd.DataFrame(values, columns=drivers).T.groupby(level=0).sum().T
    mean_abs = per_driver.abs().mean()
    shap_table = pd.DataFrame({"driver": mean_abs.index, "mean_abs_shap_usd": mean_abs.values})
    shap_table["share_pct"] = 100 * shap_table["mean_abs_shap_usd"] / shap_table["mean_abs_shap_usd"].sum()
    shap_table = shap_table.sort_values("share_pct", ascending=False)
    group_of = {c: g for g, cols in GROUPS.items() for c in cols}
    shap_groups = (shap_table.assign(group=shap_table["driver"].map(group_of))
                   .groupby("group")[["mean_abs_shap_usd", "share_pct"]].sum()
                   .sort_values("share_pct", ascending=False).reset_index())
    print(f"\nSHAP mean |contribution| on {len(shap_rows):,} held-out rows "
          f"({time.time() - start:,.0f}s). Shares sum to 100%:")
    print(shap_table.round(3).to_string(index=False))
    print(shap_groups.round(3).to_string(index=False))
    shap_table.to_csv(OUT / "shap_importance.csv", index=False)
    shap_groups.to_csv(OUT / "shap_group_importance.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    s = shap_table.sort_values("share_pct")
    axes[0].barh(s["driver"], s["share_pct"])
    axes[0].set(xlabel="Share of mean |SHAP| (%)", title="Individual drivers")
    s = shap_groups.sort_values("share_pct")
    axes[1].barh(s["group"], s["share_pct"])
    axes[1].set(xlabel="Share of mean |SHAP| (%)", title="Driver groups")
    fig.suptitle("Additive driver importance (SHAP shares sum to 100%)")
    fig.tight_layout()
    fig.savefig(OUT / "shap_importance.png", dpi=160)

print("\nInterpretation notes:")
print("- Permutation drops are predictive, not causal, and not additive; SHAP shares are additive.")
print("- Correlated predictors (route and distance) can share importance under either method.")
print("- Recorded surge > 1 appears only for Lyft, so pooled surge importance is Lyft-driven.")
print(f"\nSaved results to: {OUT}")
