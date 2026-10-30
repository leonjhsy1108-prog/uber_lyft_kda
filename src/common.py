from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "rideshare_model.csv"
OUT_ROOT = ROOT / "kda_outputs"

SEED = 42
TEST_FRACTION_OF_DAYS = 0.8

MAIN_TRAIN_ROWS = 200_000
FAST_TRAIN_ROWS = 50_000
MAIN_TREES = 100
FAST_TREES = 40

PERM_ROWS = 5_000
PERM_REPEATS = 10
SHAP_ROWS = 1_000
REFERENCE_ROWS = 2_000

TARGET = "price"
NUMERICAL = ["distance", "surge_multiplier", "temperature", "precipIntensity", "windSpeed"]
CATEGORICAL = ["name", "source", "destination", "local_hour", "weekday"]
FEATURES = NUMERICAL + CATEGORICAL

GROUPS = {
    "ride_product": ["name"],
    "distance": ["distance"],
    "surge": ["surge_multiplier"],
    "route": ["source", "destination"],
    "time": ["local_hour", "weekday"],
    "weather": ["temperature", "precipIntensity", "windSpeed"],
}

PRICE_BANDS = dict(bins=[-np.inf, 10, 20, 35, np.inf],
                   labels=["Under $10", "$10–20", "$20–35", "Over $35"], right=False)

def load_data():
    data = pd.read_csv(DATA_PATH, parse_dates=["local_date"])
    required = FEATURES + [TARGET, "local_date", "cab_type", "id"]
    missing = data[required].isna().any()
    if missing.any():
        raise ValueError(f"SQL output has missing values in: {missing[missing].index.tolist()}")
    first, last = data["local_date"].min().date(), data["local_date"].max().date()
    print(f"Clean rows: {len(data):,}; Boston-local dates {first} to {last}")
    return data

def chronological_split(data, fraction=TEST_FRACTION_OF_DAYS):
    days = np.array(sorted(data["local_date"].dt.date.unique()))
    if len(days) < 5:
        raise ValueError("Need at least 5 distinct dates for a chronological holdout.")
    split_day = days[int(len(days) * fraction)]
    mask = data["local_date"].dt.date < split_day
    train, test = data.loc[mask].copy(), data.loc[~mask].copy()
    print(f"Train rows: {len(train):,}; test rows: {len(test):,}; test starts {split_day}")
    return train, test, split_day

def sample_rows(frame, n, seed=SEED):
    return frame.sample(n=min(n, len(frame)), random_state=seed)

def make_preprocessor(columns):
    nums = [c for c in NUMERICAL if c in columns]
    cats = [c for c in CATEGORICAL if c in columns]
    steps = []
    if nums:
        steps.append(("num", "passthrough", nums))
    if cats:
        steps.append(("cat", OneHotEncoder(handle_unknown="ignore"), cats))
    return ColumnTransformer(steps)

def make_forest(columns, n_estimators=MAIN_TREES, seed=SEED):
    return Pipeline([
        ("preprocessor", make_preprocessor(columns)),
        ("model", RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=20,
            min_samples_leaf=5,
            max_features=0.5,
            random_state=seed,
            n_jobs=-1,
        )),
    ])

def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {"MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": float(np.sqrt(mse)),
            "R2": r2_score(y_true, y_pred),
            "MSE": mse}

def driver_of_output_columns(fitted_preprocessor):
    cats = []
    for name, _, cols in fitted_preprocessor.transformers_:
        if name == "cat":
            cats = list(cols)
    drivers = []
    for out_name in fitted_preprocessor.get_feature_names_out():
        prefix, rest = out_name.split("__", 1)
        if prefix == "cat":
            match = [c for c in cats if rest.startswith(c + "_")]
            if len(match) != 1:
                raise ValueError(f"Could not map output column {out_name!r} to a driver.")
            drivers.append(match[0])
        else:
            drivers.append(rest)
    return drivers

def grouped_permutation_importance(model, X, y, groups, n_repeats=PERM_REPEATS, seed=SEED):
    rng = np.random.default_rng(seed)
    base = r2_score(y, model.predict(X))
    rows = []
    for group, cols in groups.items():
        drops = []
        for _ in range(n_repeats):
            order = rng.permutation(len(X))
            permuted = X.copy()
            permuted.loc[:, cols] = X.iloc[order][cols].to_numpy()
            drops.append(base - r2_score(y, model.predict(permuted)))
        rows.append({"group": group,
                     "importance_mean_r2_drop": float(np.mean(drops)),
                     "importance_std": float(np.std(drops, ddof=1))})
    return pd.DataFrame(rows).sort_values("importance_mean_r2_drop", ascending=False)

def barh_with_error(ax, labels, values, errors=None, xlabel="", title=""):
    ax.barh(labels, values, xerr=errors, capsize=3 if errors is not None else 0)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(xlabel=xlabel, title=title)
