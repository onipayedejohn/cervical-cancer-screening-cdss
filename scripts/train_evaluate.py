"""Train, compare and evaluate the risk models.

Order of operations (this order is the whole point):
  1. Clean and de-duplicate, then split once. The test set is locked away.
  2. Compare five candidates on the development set with nested cross-validation.
  3. Measure the energy and carbon cost of each candidate.
  4. Refit the chosen model on the development set, calibrate it, and pick the
     decision threshold from out-of-fold predictions (still no test data).
  5. Score the locked test set exactly once.

Run from the project root:  python scripts/train_evaluate.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
import warnings
from datetime import date
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from codecarbon import OfflineEmissionsTracker
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cervical_cdss.features import (  # noqa: E402
    FEATURES,
    LABELS,
    LEAKY_COLUMNS,
    MODEL_DIR,
    PROCESSED_DIR,
    REPORTS_DIR,
    TARGET,
    assert_no_leakage,
    clean,
    load_raw,
    xy,
)

warnings.filterwarnings("ignore")
logging.getLogger("codecarbon").setLevel(logging.ERROR)

SEED = 42
TARGET_SENSITIVITY = 0.80
FIG_DIR = REPORTS_DIR / "figures"
MET_DIR = REPORTS_DIR / "metrics"
for d in (FIG_DIR, MET_DIR, MODEL_DIR, PROCESSED_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Chart styling: validated categorical palette, fixed order per model
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
COLORS = {
    "Logistic regression": "#2a78d6",
    "Random forest": "#eb6834",
    "Gradient boosting": "#1baf7a",
    "SVM (RBF)": "#4a3aa7",
    "Baseline (no skill)": "#9a9892",
}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def imputer():
    # add_indicator keeps "she chose not to answer" as information
    return SimpleImputer(strategy="median", add_indicator=True)


def candidates():
    return {
        "Baseline (no skill)": (
            Pipeline([("impute", imputer()), ("model", DummyClassifier(strategy="prior"))]),
            {},
        ),
        "Logistic regression": (
            Pipeline([("impute", imputer()), ("scale", StandardScaler()),
                      ("model", LogisticRegression(class_weight="balanced", max_iter=5000))]),
            {"model__C": [0.01, 0.1, 1.0]},
        ),
        "Random forest": (
            Pipeline([("impute", imputer()),
                      ("model", RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                                       random_state=SEED, n_jobs=1))]),
            {"model__min_samples_leaf": [5, 10, 20], "model__max_features": ["sqrt", 0.5]},
        ),
        "Gradient boosting": (
            Pipeline([("impute", imputer()),
                      ("model", HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED))]),
            {"model__learning_rate": [0.03, 0.1], "model__max_depth": [2, 3], "model__max_iter": [100]},
        ),
        "SVM (RBF)": (
            Pipeline([("impute", imputer()), ("scale", StandardScaler()),
                      ("model", SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=SEED))]),
            {"model__C": [0.1, 1.0], "model__gamma": ["scale", 0.01]},
        ),
    }


def tracker():
    return OfflineEmissionsTracker(country_iso_code="GHA", log_level="error", save_to_file=False,
                                   measure_power_secs=1, tracking_mode="process")


def metrics_at(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    f2 = (5 * ppv * sens / (4 * ppv + sens)) if (ppv and sens and not np.isnan(ppv)) else 0.0
    return {"sensitivity": sens, "specificity": spec, "ppv": ppv, "npv": npv, "f2": f2,
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}


def bootstrap_ci(y, p, thr, n=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    keys = ["roc_auc", "pr_auc", "brier", "sensitivity", "specificity", "ppv", "npv"]
    rows = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        yb, pb = y[idx], p[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        m = metrics_at(yb, pb, thr)
        rows.append([roc_auc_score(yb, pb), average_precision_score(yb, pb), brier_score_loss(yb, pb),
                     m["sensitivity"], m["specificity"], m["ppv"], m["npv"]])
    arr = np.array(rows, dtype=float)
    return {k: (float(np.nanpercentile(arr[:, i], 2.5)), float(np.nanpercentile(arr[:, i], 97.5)))
            for i, k in enumerate(keys)}


def net_benefit(y, p, thresholds):
    y = np.asarray(y); n = len(y)
    out = []
    for t in thresholds:
        pred = p >= t
        tp = np.sum(pred & (y == 1)); fp = np.sum(pred & (y == 0))
        out.append(tp / n - fp / n * t / (1 - t))
    return np.array(out)


def main():
    t0 = time.time()

    # 1. Clean, split once, lock the test set --------------------------------
    raw = load_raw()
    df = clean(raw)
    n_dupes = len(raw) - len(df)
    dev, test = train_test_split(df, test_size=0.25, stratify=df[TARGET], random_state=SEED)
    dev[FEATURES + [TARGET]].to_csv(PROCESSED_DIR / "dev.csv", index=False)
    test[FEATURES + [TARGET]].to_csv(PROCESSED_DIR / "test.csv", index=False)
    X_dev, y_dev = xy(dev)
    X_test, y_test = xy(test)
    assert_no_leakage(X_dev.columns)
    prevalence = float(y_dev.mean())
    print(f"rows: raw {len(raw)}, after de-duplication {len(df)} ({n_dupes} duplicates removed)")
    print(f"dev {len(dev)} ({y_dev.sum()} positive), test {len(test)} ({y_test.sum()} positive)")

    # 2. Nested CV comparison on the development set -------------------------
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    fold_rows, green_rows = [], []
    for name, (pipe, grid) in candidates().items():
        tr = tracker(); tr.start(); start = time.perf_counter()
        for k, (i_tr, i_va) in enumerate(outer.split(X_dev, y_dev)):
            est = GridSearchCV(pipe, grid, scoring="average_precision", cv=inner, n_jobs=1) if grid else pipe
            est.fit(X_dev.iloc[i_tr], y_dev.iloc[i_tr])
            p = est.predict_proba(X_dev.iloc[i_va])[:, 1]
            yv = y_dev.iloc[i_va]
            fold_rows.append({"model": name, "fold": k, "roc_auc": roc_auc_score(yv, p),
                              "pr_auc": average_precision_score(yv, p), "brier": brier_score_loss(yv, p)})
        tr.stop(); cv_seconds = time.perf_counter() - start
        cv_energy = tr.final_emissions_data.energy_consumed
        cv_co2 = tr.final_emissions_data.emissions

        # Energy to fit once and to score 10,000 women
        tr = tracker(); tr.start(); s = time.perf_counter()
        final = GridSearchCV(pipe, grid, scoring="average_precision", cv=inner, n_jobs=1) if grid else pipe
        final.fit(X_dev, y_dev)
        fit_seconds = time.perf_counter() - s
        tr.stop(); fit_energy = tr.final_emissions_data.energy_consumed; fit_co2 = tr.final_emissions_data.emissions
        # Scoring is fast, so repeat it for ~5 s to give the energy meter something to measure
        big = X_dev.sample(10_000, replace=True, random_state=SEED)
        tr = tracker(); tr.start(); s = time.perf_counter(); n_scored = 0
        while time.perf_counter() - s < 5 or n_scored == 0:
            final.predict_proba(big); n_scored += len(big)
        inf_seconds = time.perf_counter() - s
        tr.stop(); inf_energy = tr.final_emissions_data.energy_consumed
        tmp = MODEL_DIR / "_size_check.joblib"
        joblib.dump(final.best_estimator_ if grid else final, tmp)
        size_kb = tmp.stat().st_size / 1024
        tmp.unlink()
        green_rows.append({
            "model": name,
            "cv_seconds": cv_seconds, "cv_energy_wh": cv_energy * 1000, "cv_co2_g": cv_co2 * 1000,
            "fit_seconds": fit_seconds, "fit_energy_wh": fit_energy * 1000, "fit_co2_g": fit_co2 * 1000,
            "inference_ms_per_1000": inf_seconds / n_scored * 1000 * 1000,
            "inference_energy_mwh_per_1000": inf_energy * 1e6 / n_scored * 1000,
            "model_size_kb": size_kb,
            "best_params": json.dumps(final.best_params_) if grid else "{}",
        })
        print(f"  {name:22s} done in {cv_seconds:5.1f}s")

    folds = pd.DataFrame(fold_rows)
    folds.to_csv(MET_DIR / "nested_cv_folds.csv", index=False)
    summary = folds.groupby("model")[["roc_auc", "pr_auc", "brier"]].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    green = pd.DataFrame(green_rows).set_index("model")
    comparison = summary.join(green).sort_values("pr_auc_mean", ascending=False)
    comparison.round(5).to_csv(MET_DIR / "model_comparison.csv")
    print(comparison[["roc_auc_mean", "pr_auc_mean", "brier_mean", "cv_energy_wh"]].round(4))

    # Selection rule, fixed before looking at the results: highest mean PR-AUC.
    # If another model is within 0.01, take the one that used less energy.
    ranked = comparison.drop(index="Baseline (no skill)")
    best_pr = ranked["pr_auc_mean"].max()
    close = ranked[ranked["pr_auc_mean"] >= best_pr - 0.01]
    chosen = close["cv_energy_wh"].idxmin()
    print(f"chosen model: {chosen}")

    # 3. Refit chosen model on dev, calibrate, choose threshold -------------------
    pipe, grid = candidates()[chosen]
    tuned = GridSearchCV(pipe, grid, scoring="average_precision",
                         cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=1).fit(X_dev, y_dev)
    base = tuned.best_estimator_
    model = CalibratedClassifierCV(base, method="sigmoid", cv=StratifiedKFold(5, shuffle=True, random_state=SEED))

    # Out-of-fold probabilities on dev, averaged over 5 different shuffles
    oof = np.zeros(len(y_dev))
    for r in range(5):
        oof += cross_val_predict(model, X_dev, y_dev, method="predict_proba",
                                 cv=StratifiedKFold(5, shuffle=True, random_state=100 + r))[:, 1]
    oof /= 5
    # Highest threshold that still reaches the target sensitivity on dev
    cand = np.unique(np.round(oof, 4))[::-1]
    threshold = float(next(t for t in cand if metrics_at(y_dev, oof, t)["sensitivity"] >= TARGET_SENSITIVITY))
    dev_at_thr = metrics_at(y_dev, oof, threshold)
    print(f"threshold {threshold:.4f} -> dev OOF sens {dev_at_thr['sensitivity']:.2f}, spec {dev_at_thr['specificity']:.2f}")

    model.fit(X_dev, y_dev)

    # Risk bands for the app: cut points from dev out-of-fold predictions
    band_cuts = [float(np.quantile(oof, 0.40)), float(np.quantile(oof, 0.80))]

    # Subgroup check by age band on dev OOF predictions (bias check)
    bands = pd.cut(X_dev["Age"], [0, 29, 49, 120], labels=["under 30", "30 to 49", "50 and over"])
    sub_rows = []
    for band in bands.cat.categories:
        m = bands == band
        yb, pb = y_dev[m].values, oof[m.values]
        row = {"age_band": band, "n": int(m.sum()), "positives": int(yb.sum())}
        if 0 < yb.sum() < len(yb):
            row["roc_auc"] = roc_auc_score(yb, pb)
        row.update({k: v for k, v in metrics_at(yb, pb, threshold).items() if k in ("sensitivity", "specificity")})
        sub_rows.append(row)
    pd.DataFrame(sub_rows).round(3).to_csv(MET_DIR / "subgroup_age_dev_oof.csv", index=False)

    # 4. One look at the test set -------------------------------------------------
    p_test = model.predict_proba(X_test)[:, 1]
    test_metrics = {"roc_auc": roc_auc_score(y_test, p_test),
                    "pr_auc": average_precision_score(y_test, p_test),
                    "brier": brier_score_loss(y_test, p_test),
                    "prevalence": float(y_test.mean())}
    test_metrics.update(metrics_at(y_test, p_test, threshold))
    ci = bootstrap_ci(y_test, p_test, threshold)
    band_names = ["Lower", "Average", "Higher"]
    test_band = np.digitize(p_test, band_cuts)
    bands_test = [{"band": band_names[b], "n": int((test_band == b).sum()),
                   "positives": int(y_test.values[test_band == b].sum()),
                   "observed_rate": float(y_test.values[test_band == b].mean()) if (test_band == b).any() else None}
                  for b in range(3)]
    pd.DataFrame(bands_test).to_csv(MET_DIR / "risk_bands_test.csv", index=False)
    print("test:", {k: round(v, 3) if isinstance(v, float) else v for k, v in test_metrics.items()})

    # What published "high accuracy" usually hides: same model with the leaky columns
    leak_cols = FEATURES + [c for c in LEAKY_COLUMNS if c in dev.columns]
    leak_pipe = Pipeline([("impute", imputer()),
                          ("model", RandomForestClassifier(300, min_samples_leaf=5, class_weight="balanced_subsample",
                                                           random_state=SEED, n_jobs=1))]).fit(dev[leak_cols], y_dev)
    p_leak = leak_pipe.predict_proba(test[leak_cols])[:, 1]
    leakage_demo = {"with_leaky_columns_roc_auc": roc_auc_score(y_test, p_leak),
                    "with_leaky_columns_pr_auc": average_precision_score(y_test, p_leak),
                    "leaky_columns": list(LEAKY_COLUMNS)}

    # Permutation importance on dev (so the test set stays a clean one-shot score)
    pi = permutation_importance(model, X_dev, y_dev, scoring="average_precision", n_repeats=20,
                                random_state=SEED, n_jobs=1)
    importance = (pd.DataFrame({"feature": FEATURES, "mean": pi.importances_mean, "std": pi.importances_std})
                  .sort_values("mean", ascending=False))
    importance.round(4).to_csv(MET_DIR / "permutation_importance.csv", index=False)

    # 5. Save model and metadata --------------------------------------------------
    joblib.dump(model, MODEL_DIR / "cervical_risk_model.joblib", compress=3)
    meta = {
        "model_name": chosen,
        "best_params": tuned.best_params_,
        "features": FEATURES,
        "labels": LABELS,
        "threshold": threshold,
        "target_sensitivity": TARGET_SENSITIVITY,
        "training_prevalence": prevalence,
        "dev_medians": X_dev.median(numeric_only=True).to_dict(),
        "dev_oof_at_threshold": dev_at_thr,
        "risk_band_cuts": band_cuts,
        "risk_bands_test": bands_test,
        "test_metrics": test_metrics,
        "test_ci_95": ci,
        "leakage_demo": leakage_demo,
        "n_raw": len(raw), "n_after_dedup": len(df), "n_dev": len(dev), "n_test": len(test),
        "positives_dev": int(y_dev.sum()), "positives_test": int(y_test.sum()),
        "sklearn_version": sklearn.__version__,
        "trained_on": date.today().isoformat(),
        "seed": SEED,
    }
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(meta, indent=2, default=float))
    (MET_DIR / "test_metrics.json").write_text(json.dumps({"metrics": test_metrics, "ci_95": ci,
                                                           "leakage_demo": leakage_demo}, indent=2, default=float))

    # 6. Figures ------------------------------------------------------------------
    order = list(comparison.index)

    # Model comparison: PR-AUC across outer folds
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for i, name in enumerate(order[::-1]):
        vals = folds.loc[folds.model == name, "pr_auc"].values
        ax.scatter(vals, np.full(len(vals), i) + np.random.default_rng(i).uniform(-0.12, 0.12, len(vals)),
                   s=14, color=COLORS[name], alpha=0.55, linewidths=0)
        ax.plot([vals.mean()] * 2, [i - 0.28, i + 0.28], color=COLORS[name], lw=2.5, solid_capstyle="round")
    ax.set_yticks(range(len(order)), order[::-1])
    ax.axvline(prevalence, color=INK2, lw=1, ls=(0, (3, 3)))
    ax.text(prevalence, -0.75, "  no-skill line", color=INK2, fontsize=8, va="center")
    ax.set_ylim(-1.0, len(order) - 0.5)
    ax.set_xlabel("PR-AUC per outer fold (bar = mean)")
    ax.set_title("Nested cross-validation on the development set")
    fig.tight_layout(); fig.savefig(FIG_DIR / "model_comparison.png", dpi=200); plt.close(fig)

    # Greenness vs performance
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for name in order:
        r = comparison.loc[name]
        ax.scatter(r["cv_energy_wh"], r["pr_auc_mean"], s=70, color=COLORS[name],
                   edgecolors=SURFACE, linewidths=2, zorder=3)
        below = name == "SVM (RBF)"
        ax.annotate(name, (r["cv_energy_wh"], r["pr_auc_mean"]), xytext=(7, -12 if below else 4),
                    textcoords="offset points", fontsize=8.5, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("Energy for full nested CV (Wh, log scale)")
    ax.set_ylabel("Mean PR-AUC")
    ax.set_title("Performance against energy use")
    fig.tight_layout(); fig.savefig(FIG_DIR / "green_vs_performance.png", dpi=200); plt.close(fig)

    # ROC and PR on the test set
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9))
    fpr, tpr, _ = roc_curve(y_test, p_test)
    axes[0].plot(fpr, tpr, color=COLORS[chosen], lw=2)
    axes[0].plot([0, 1], [0, 1], color=INK2, lw=1, ls=(0, (3, 3)))
    axes[0].set(xlabel="1 - specificity", ylabel="Sensitivity", title=f"ROC, test set (AUC {test_metrics['roc_auc']:.2f})")
    prec, rec, _ = precision_recall_curve(y_test, p_test)
    axes[1].plot(rec, prec, color=COLORS[chosen], lw=2)
    axes[1].axhline(test_metrics["prevalence"], color=INK2, lw=1, ls=(0, (3, 3)))
    axes[1].set(xlabel="Sensitivity (recall)", ylabel="PPV (precision)", title=f"Precision-recall (AP {test_metrics['pr_auc']:.2f})")
    fig.tight_layout(); fig.savefig(FIG_DIR / "test_roc_pr.png", dpi=200); plt.close(fig)

    # Calibration on dev OOF (more positives than the test set)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    frac, mean_pred = calibration_curve(y_dev, oof, n_bins=4, strategy="quantile")
    lim = max(mean_pred.max(), frac.max()) * 1.15
    ax.plot([0, lim], [0, lim], color=INK2, lw=1, ls=(0, (3, 3)))
    ax.plot(mean_pred, frac, color=COLORS[chosen], lw=2, marker="o", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=2)
    ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="Predicted risk", ylabel="Observed rate",
           title="Calibration (dev, out-of-fold)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "calibration.png", dpi=200); plt.close(fig)

    # Decision curve on test
    ts = np.linspace(0.01, 0.30, 60)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(ts, net_benefit(y_test, p_test, ts), color=COLORS[chosen], lw=2, label="Model")
    prev_t = test_metrics["prevalence"]
    ax.plot(ts, prev_t - (1 - prev_t) * ts / (1 - ts), color=INK2, lw=1.2, label="Treat all as high risk")
    ax.axhline(0, color=INK, lw=1, label="Treat none")
    ax.set_ylim(-0.02, max(0.08, prev_t * 1.2))
    ax.set(xlabel="Threshold probability", ylabel="Net benefit", title="Decision curve, test set")
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout(); fig.savefig(FIG_DIR / "decision_curve.png", dpi=200); plt.close(fig)

    # Permutation importance
    top = importance.head(10)[::-1]
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    ax.barh([LABELS[f] for f in top.feature], top["mean"], xerr=top["std"], color=COLORS[chosen],
            height=0.6, error_kw={"ecolor": INK2, "lw": 1})
    ax.set_xlabel("Drop in PR-AUC when shuffled")
    ax.set_title("What the model relies on (dev set)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "feature_importance.png", dpi=200); plt.close(fig)

    print(f"finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
