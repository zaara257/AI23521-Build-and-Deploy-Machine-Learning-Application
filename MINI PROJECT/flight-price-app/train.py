"""Train, evaluate, tune and save the flight-price model.

    python train.py

Steps: load -> clean -> feature engineering -> baseline models -> hyper-parameter tuning
-> final test-set evaluation -> ablation (do the peak/festival features help?) -> save.
"""
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from features import CATEGORICAL, MAIN_EXTRAS, NUMERIC, RAW_TIME_FEATURES, clean_raw, engineer_features
from festivals import FESTIVALS

ROOT = Path(__file__).parent
SEED = 42
CV = 3


def make_pipeline(model, numeric=NUMERIC, categorical=CATEGORICAL):
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
         ("num", "passthrough", numeric)]
    )
    return Pipeline([("pre", pre), ("model", model)])


def score(pipe, X, y):
    pred = pipe.predict(X)
    return {"mae": round(float(mean_absolute_error(y, pred)), 1),
            "rmse": round(float(root_mean_squared_error(y, pred)), 1),
            "r2": round(float(r2_score(y, pred)), 4)}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- 1. data
raw = pd.read_excel(ROOT / "data" / "Data_Train.xlsx")
tidy, cleaning_log = clean_raw(raw)
X = engineer_features(tidy)
y = tidy["Price"]
log(f"cleaned: {cleaning_log['raw_rows']} -> {cleaning_log['final_rows']} rows, {X.shape[1]} features")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
log(f"train={len(X_train)} test={len(X_test)}")

# ---------------------------------------------------------------- 2. baselines
baseline_models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=2),
    "Gradient Boosting": GradientBoostingRegressor(random_state=SEED),
}
results = {}
for name, model in baseline_models.items():
    t = time.time()
    pipe = make_pipeline(model).fit(X_train, y_train)
    results[name] = {"baseline": score(pipe, X_test, y_test)}
    log(f"baseline {name}: {results[name]['baseline']} ({time.time() - t:.0f}s)")

# ---------------------------------------------------------------- 3. tuning
# Grid search for Gradient Boosting (small grid), random search for Random Forest (bigger space).
log("tuning Gradient Boosting with GridSearchCV ...")
gb_search = GridSearchCV(
    make_pipeline(GradientBoostingRegressor(random_state=SEED)),
    {"model__n_estimators": [200, 400],
     "model__learning_rate": [0.05, 0.1],
     "model__max_depth": [3, 5, 7]},
    cv=CV, scoring="neg_root_mean_squared_error", n_jobs=2,
).fit(X_train, y_train)
log(f"GB best: {gb_search.best_params_} cv_rmse={-gb_search.best_score_:.0f}")

log("tuning Random Forest with RandomizedSearchCV ...")
rf_search = RandomizedSearchCV(
    make_pipeline(RandomForestRegressor(random_state=SEED, n_jobs=2)),
    {"model__n_estimators": [150, 250, 400],
     "model__max_depth": [15, 25, None],
     "model__min_samples_leaf": [1, 2, 4],
     "model__max_features": [0.4, 0.6, 1.0]},
    n_iter=10, cv=CV, scoring="neg_root_mean_squared_error", random_state=SEED,
).fit(X_train, y_train)
log(f"RF best: {rf_search.best_params_} cv_rmse={-rf_search.best_score_:.0f}")

for name, search in [("Gradient Boosting", gb_search), ("Random Forest", rf_search)]:
    results[name]["tuned"] = score(search.best_estimator_, X_test, y_test)
    results[name]["best_params"] = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    results[name]["cv_rmse"] = round(float(-search.best_score_), 1)
    log(f"tuned {name} on test: {results[name]['tuned']}")

# Pick the winner by cross-validated RMSE on the training data (not by the test set).
best_name = min(["Gradient Boosting", "Random Forest"], key=lambda n: results[n]["cv_rmse"])
best_search = gb_search if best_name == "Gradient Boosting" else rf_search
final = best_search.best_estimator_
final_scores = results[best_name]["tuned"]
log(f"final model: {best_name} -> {final_scores}")

# ---------------------------------------------------------------- 4. ablation
# Do the peak-time / festival / season features actually help? Same model, same split,
# with and without them (5-fold CV on the training set + the held-out test set).
log("ablation: what do the date/time and peak/festival features add? ...")
best_params = {k.replace("model__", ""): v for k, v in best_search.best_params_.items()}
base_model = (GradientBoostingRegressor if best_name == "Gradient Boosting" else RandomForestRegressor)
extra = {"n_jobs": 2} if best_name == "Random Forest" else {}
core = ["Total_Stops", "Duration_Mins"]
ablation = {}
for label, numeric in [
    ("1_route_airline_stops_duration_only", core),
    ("2_plus_raw_date_and_departure_time", core + RAW_TIME_FEATURES),
    ("3_plus_peak_and_festival_flags", NUMERIC),
]:
    cols = CATEGORICAL + numeric
    pipe = make_pipeline(base_model(random_state=SEED, **best_params, **extra), numeric)
    cv_rmse = -cross_val_score(pipe, X_train[cols], y_train, cv=5, scoring="neg_root_mean_squared_error").mean()
    pipe.fit(X_train[cols], y_train)
    ablation[label] = {"cv_rmse": round(float(cv_rmse), 1), **score(pipe, X_test[cols], y_test)}
    log(f"  {label}: {ablation[label]}")

# ---------------------------------------------------------------- 5. importances + plots
names = final.named_steps["pre"].get_feature_names_out()
imp = pd.Series(final.named_steps["model"].feature_importances_, index=names).sort_values(ascending=False)
imp.index = [n.replace("cat__", "").replace("num__", "") for n in imp.index]
top_features = {k: round(float(v), 4) for k, v in imp.head(10).items()}

BLUE, INK, MUTED, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e3e2de"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True})
(ROOT / "reports").mkdir(exist_ok=True)

pred = final.predict(X_test)
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(y_test, pred, s=10, alpha=0.35, color=BLUE, linewidths=0)
lim = [0, max(y_test.max(), pred.max()) * 1.02]
ax.plot(lim, lim, color=MUTED, lw=1, ls="--")
ax.set(xlim=lim, ylim=lim, xlabel="Actual price (₹)", ylabel="Predicted price (₹)")
ax.set_title(f"{best_name}: predicted vs actual (test set)", color=INK, loc="left", fontsize=11)
ax.text(0.98, 0.04, f"MAE ₹{final_scores['mae']:,.0f}  ·  RMSE ₹{final_scores['rmse']:,.0f}",
        transform=ax.transAxes, ha="right", color=MUTED)
fig.tight_layout()
fig.savefig(ROOT / "reports" / "predicted_vs_actual.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4))
top = imp.head(10)[::-1]
ax.barh(top.index, top.values, color=BLUE, height=0.6)
ax.set_xlabel("Importance")
ax.grid(axis="y", visible=False)
ax.set_title("Top 10 features", color=INK, loc="left", fontsize=11)
fig.tight_layout()
fig.savefig(ROOT / "reports" / "feature_importance.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- 6. save
(ROOT / "models").mkdir(exist_ok=True)
joblib.dump(final, ROOT / "models" / "flight_model.joblib", compress=3)

routes = tidy.groupby("Source")["Destination"].unique().map(sorted).to_dict()
metadata = {
    "best_model": best_name,
    "metrics": final_scores,
    "all_results": results,
    "ablation": ablation,
    "top_features": top_features,
    "cleaning_log": cleaning_log,
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "date_range": [str(tidy["Journey_Date"].min().date()), str(tidy["Journey_Date"].max().date())],
    "duration_range_mins": [int(tidy["Duration_Mins"].min()), int(tidy["Duration_Mins"].max())],
    "airlines": sorted(tidy["Airline"].unique()),
    "extras": MAIN_EXTRAS,
    "routes": routes,
    "festivals": [{"date": str(d), "name": n} for d, n in sorted(FESTIVALS.items())],
}
(ROOT / "models" / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
size_mb = (ROOT / "models" / "flight_model.joblib").stat().st_size / 1e6
log(f"saved model ({size_mb:.1f} MB) + metadata + 2 plots")
