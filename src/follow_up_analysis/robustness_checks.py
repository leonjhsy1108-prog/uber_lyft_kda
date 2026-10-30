import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score

from common import (FAST_TRAIN_ROWS, FAST_TREES, FEATURES, GROUPS, OUT_ROOT, PERM_REPEATS,
                    PERM_ROWS, SEED, TARGET, barh_with_error, chronological_split,
                    load_data, make_forest, regression_metrics, sample_rows)

OUT = OUT_ROOT / "robustness"
OUT.mkdir(parents=True, exist_ok=True)
data = load_data()

temporal, latest = [], None
for fraction in [0.65, 0.75, 0.80]:
    train, test, split_day = chronological_split(data, fraction)
    train_sample = sample_rows(train, FAST_TRAIN_ROWS)
    model = make_forest(FEATURES, n_estimators=FAST_TREES)
    model.fit(train_sample[FEATURES], train_sample[TARGET])
    m = regression_metrics(test[TARGET], model.predict(test[FEATURES]))
    temporal.append({"train_day_fraction": fraction, "test_start": str(split_day),
                     "train_rows_available": len(train), "train_rows_used": len(train_sample),
                     "test_rows": len(test), "MAE": m["MAE"], "R2": m["R2"]})
    print(f"Temporal split {fraction:.0%}: test from {split_day}, R²={m['R2']:.4f}, MAE=${m['MAE']:.3f}")
    latest = (model, train_sample, test)
temporal = pd.DataFrame(temporal)
temporal.to_csv(OUT / "temporal_stability.csv", index=False)

model, train_sample, test = latest

perm_rows = sample_rows(test, PERM_ROWS)
perm = permutation_importance(model, perm_rows[FEATURES], perm_rows[TARGET], scoring="r2",
                              n_repeats=PERM_REPEATS, random_state=SEED, n_jobs=1)
importance = pd.DataFrame({"driver": FEATURES, "importance_mean_r2_drop": perm.importances_mean,
                           "importance_std": perm.importances_std}
                          ).sort_values("importance_mean_r2_drop", ascending=False)
importance.to_csv(OUT / "repeated_permutation_importance.csv", index=False)
print(f"\nPermutation importance ({len(perm_rows):,} rows x {PERM_REPEATS} repeats):\n",
      importance.round(4).to_string(index=False))

full_r2 = r2_score(test[TARGET], model.predict(test[FEATURES]))
ablations = []
for group, removed in GROUPS.items():
    kept = [c for c in FEATURES if c not in removed]
    reduced = make_forest(kept, n_estimators=FAST_TREES)
    reduced.fit(train_sample[kept], train_sample[TARGET])
    reduced_r2 = r2_score(test[TARGET], reduced.predict(test[kept]))
    ablations.append({"removed_group": group, "full_model_R2": full_r2,
                      "without_group_R2": reduced_r2, "R2_loss": full_r2 - reduced_r2})
    print(f"Removing {group}: R² drops by {full_r2 - reduced_r2:.4f}")
ablation = pd.DataFrame(ablations).sort_values("R2_loss", ascending=False)
ablation.to_csv(OUT / "feature_ablation.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(temporal["test_start"], temporal["R2"], marker="o")
ax.set(xlabel="Held-out period begins", ylabel="Test R²",
       title="Performance across chronological cutoffs")
fig.tight_layout()
fig.savefig(OUT / "temporal_stability.png", dpi=160)

fig, ax = plt.subplots(figsize=(9, 5))
d = importance.sort_values("importance_mean_r2_drop")
barh_with_error(ax, d["driver"], d["importance_mean_r2_drop"], d["importance_std"],
                "Mean drop in held-out R² (± SD)", "Repeated permutation importance")
fig.tight_layout()
fig.savefig(OUT / "repeated_importance.png", dpi=160)

fig, ax = plt.subplots(figsize=(8, 5))
d = ablation.sort_values("R2_loss")
barh_with_error(ax, d["removed_group"], d["R2_loss"], None,
                "R² of full model minus R² after removal",
                "Group ablation: retrain without each driver group")
fig.tight_layout()
fig.savefig(OUT / "ablation.png", dpi=160)

print(f"\nSaved robustness results to: {OUT}")
print("Note: ablation and permutation scores are model-based, not causal effects.")
