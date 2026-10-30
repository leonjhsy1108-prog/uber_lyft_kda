import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from common import (FAST_TRAIN_ROWS, FAST_TREES, FEATURES, OUT_ROOT, PERM_REPEATS, PERM_ROWS,
                    SEED, TARGET, chronological_split, load_data, make_forest,
                    regression_metrics, sample_rows)

OUT = OUT_ROOT / "companies"
OUT.mkdir(parents=True, exist_ok=True)
data = load_data()
train_all, test_all, split_day = chronological_split(data)

results, tables = [], []
for company in ["Uber", "Lyft"]:
    train = train_all.loc[train_all["cab_type"] == company]
    test = test_all.loc[test_all["cab_type"] == company]
    features = [c for c in FEATURES if train[c].nunique() > 1]
    dropped = sorted(set(FEATURES) - set(features))
    train_sample = sample_rows(train, FAST_TRAIN_ROWS)
    model = make_forest(features, n_estimators=FAST_TREES)
    model.fit(train_sample[features], train_sample[TARGET])
    m = regression_metrics(test[TARGET], model.predict(test[features]))
    results.append({"company": company, "train_rows_available": len(train),
                    "train_rows_used": len(train_sample), "test_rows": len(test),
                    "MAE": m["MAE"], "RMSE": m["RMSE"], "R2": m["R2"],
                    "test_start": str(split_day), "dropped_constant_features": ",".join(dropped)})

    perm_rows = sample_rows(test, PERM_ROWS)
    perm = permutation_importance(model, perm_rows[features], perm_rows[TARGET], scoring="r2",
                                  n_repeats=PERM_REPEATS, random_state=SEED, n_jobs=1)
    scores = pd.DataFrame({"company": company, "driver": features,
                           "importance_mean_r2_drop": perm.importances_mean,
                           "importance_std": perm.importances_std})
    tables.append(scores)
    print(f"\n{company}: test MAE=${m['MAE']:.3f}, R²={m['R2']:.4f}"
          + (f" (dropped constant: {dropped})" if dropped else ""))
    print(scores.sort_values("importance_mean_r2_drop", ascending=False).round(4).to_string(index=False))

performance, importance = pd.DataFrame(results), pd.concat(tables, ignore_index=True)
performance.to_csv(OUT / "company_model_performance.csv", index=False)
importance.to_csv(OUT / "company_driver_importance.csv", index=False)

pivot = importance.pivot(index="driver", columns="company", values="importance_mean_r2_drop")
pivot = pivot.loc[pivot.max(axis=1).sort_values().index]
err = importance.pivot(index="driver", columns="company", values="importance_std").loc[pivot.index]
fig, ax = plt.subplots(figsize=(10, 6))
pos, width = np.arange(len(pivot)), 0.38
for offset, company in [(-width / 2, "Uber"), (width / 2, "Lyft")]:
    vals = pivot[company].to_numpy()
    ax.barh(pos + offset, np.nan_to_num(vals), height=width, label=company,
            xerr=np.nan_to_num(err[company].to_numpy()), capsize=2)
    for i, v in enumerate(vals):
        if np.isnan(v):
            ax.text(0.008, pos[i] + offset, "N/A (constant)", va="center", fontsize=8)
ax.set_yticks(pos, pivot.index)
ax.set(xlabel="Decrease in held-out R² after permutation (± SD)",
       title="Uber vs Lyft: within-company driver importance")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "company_importance_comparison.png", dpi=160)

print(f"\nSaved company results to: {OUT}")
print("Note: compare rankings within a company; magnitudes are not directly comparable across platforms.")
