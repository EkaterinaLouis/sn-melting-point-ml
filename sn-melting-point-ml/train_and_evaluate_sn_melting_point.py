#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train a melting-point regressor from a phase map (Sn-centered multicomponent alloys).
- Target: For each unique composition (12 fractions summing to 1), the minimum T with any LIQUID phase.
- Features: 12 fractions + physics-inspired descriptors (entropy-like, diversity, Sn emphasis).
- Split: Grouped by rounded composition (2 decimal places) to avoid near-duplicate leakage.
- Models: HistGradientBoosting (strong baseline), RandomForest, ElasticNet(+PolynomialFeatures).
- Metrics: MAE, RMSE, R^2 on a held-out test set (20% groups).
- Outputs: best model .joblib, prediction scatter plot, top features by permutation importance.
"""

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold, GridSearchCV, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
import joblib

ELEMENTS = ['Ag','Al','Au','Bi','Cu','Ga','In','Ni','Pb','Sb','Sn','Zn']

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="SnMeltingPoint.xlsx",
                   help="Path to SnMeltingPoint.xlsx")
    p.add_argument("--outdir", type=str, default="outputs",
                   help="Directory to save artifacts")
    p.add_argument("--compare_models", action="store_true",
                   help="If set, compare HGBR/RF/ElasticNet via small grid-search. Otherwise use HGBR only.")
    p.add_argument("--random_state", type=int, default=42)
    return p.parse_args()

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def read_raw(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    # Basic sanity checks
    assert set(ELEMENTS).issubset(df.columns), f"Missing element columns in {path}"
    assert "Phase" in df.columns and "T" in df.columns, "Expect columns 'Phase' and 'T'"
    return df

def build_regression_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Group rows by unique composition and take the minimal T where Phase contains LIQUID."""
    mask_liquid = df["Phase"].astype(str).str.contains("LIQUID", case=False, na=False)
    liq = (df.loc[mask_liquid]
             .groupby(ELEMENTS, as_index=False)["T"]
             .min()
             .rename(columns={"T": "T_melt"}))
    return liq

def add_domain_features(df_reg: pd.DataFrame) -> pd.DataFrame:
    """Add physics-inspired descriptors that do not require external tables."""
    X = df_reg.copy()
    x = X[ELEMENTS].values
    x_safe = np.where(x > 0, x, 1.0)   # for log only, to avoid -inf
    X["mixing_entropy"] = -np.sum(x * np.log(x_safe), axis=1)
    X["num_components"] = (x > 0).sum(axis=1)
    X["max_frac"] = x.max(axis=1)
    X["min_frac"] = x.min(axis=1)
    X["var_frac"] = x.var(axis=1)
    X["gini_diversity"] = 1.0 - (x**2).sum(axis=1)
    X["sn_frac"] = X["Sn"]
    X["sn_major"] = (X["Sn"] >= 0.5).astype(int)
    return X

def make_group_key(df_reg: pd.DataFrame, decimals: int = 2) -> pd.Series:
    """Round composition to a coarse grid to define groups; prevents near-duplicate leakage."""
    rounded = df_reg[ELEMENTS].round(decimals)
    return rounded.apply(lambda row: "|".join(f"{v:.{decimals}f}" for v in row.values), axis=1)

def summarize_regression_table(df_reg: pd.DataFrame):
    n_unique = df_reg[ELEMENTS].drop_duplicates().shape[0]
    print(f"[INFO] Unique compositions with T_melt labels: {len(df_reg):,} (from {n_unique:,} total unique compositions with LIQUID available)")

def get_pipelines(random_state: int, feature_kind: str, domain_cols):
    """
    Return dict of name->(pipeline, param_grid).
    feature_kind controls whether linear model uses polynomial expansion.
    """
    # Transformers
    ct_raw = ColumnTransformer([
        ("frac", "passthrough", ELEMENTS),
        ("domain", "passthrough", domain_cols)
    ])

    ct_poly = ColumnTransformer([
        ("poly", PolynomialFeatures(degree=2, include_bias=False), ELEMENTS),
        ("domain", "passthrough", domain_cols)
    ])

    # Models
    pipe_hgb = Pipeline([
        ("feat", ct_raw),
        ("model", HistGradientBoostingRegressor(random_state=random_state))
    ])
    grid_hgb = {
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [None, 8],
        "model__max_iter": [500]
    }

    pipe_rf = Pipeline([
        ("feat", ct_raw),
        ("model", RandomForestRegressor(random_state=random_state, n_jobs=-1))
    ])
    grid_rf = {
        "model__n_estimators": [300, 600],
        "model__max_depth": [None, 16],
        "model__min_samples_leaf": [1, 5]
    }

    pipe_enet = Pipeline([
        ("feat", ct_poly if feature_kind == "poly" else ct_raw),
        ("scale", StandardScaler(with_mean=True)),
        ("model", ElasticNet(max_iter=20000, random_state=random_state))
    ])
    grid_enet = {
        "model__alpha": [0.1, 1.0, 10.0],
        "model__l1_ratio": [0.2, 0.5, 0.8]
    }

    return {
        "HistGradientBoosting": (pipe_hgb, grid_hgb),
        "RandomForest": (pipe_rf, grid_rf),
        "ElasticNet+Poly": (pipe_enet, grid_enet)
    }

def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Read & build regression target
    raw = read_raw(Path(args.data))
    reg = build_regression_dataset(raw)        # columns: ELEMENTS + T_melt
    reg = add_domain_features(reg)
    reg["group_key"] = make_group_key(reg, decimals=2)
    summarize_regression_table(reg)

    # 2) Train/test split by groups (20% groups -> test)
    rng = np.random.default_rng(args.random_state)
    unique_groups = reg["group_key"].unique()
    n_test_groups = max(1, int(0.2 * len(unique_groups)))
    test_groups = set(rng.choice(unique_groups, size=n_test_groups, replace=False))

    train_df = reg.loc[~reg["group_key"].isin(test_groups)].reset_index(drop=True)
    test_df  = reg.loc[ reg["group_key"].isin(test_groups)].reset_index(drop=True)

    domain_cols = [c for c in reg.columns if c not in ELEMENTS + ["T_melt", "group_key"]]
    X_train = train_df[ELEMENTS + domain_cols]
    y_train = train_df["T_melt"].values
    groups  = train_df["group_key"].values

    X_test  = test_df[ELEMENTS + domain_cols]
    y_test  = test_df["T_melt"].values

    print(f"[INFO] Train size: {X_train.shape}, Test size: {X_test.shape}, Features: {X_train.shape[1]}")

    # 3) Model selection
    if args.compare_models:
        models = get_pipelines(args.random_state, feature_kind="poly", domain_cols=domain_cols)
        cv = GroupKFold(n_splits=3)  # grouped CV
        results = {}
        for name, (pipe, grid) in models.items():
            gs = GridSearchCV(pipe, grid, scoring="neg_mean_absolute_error", cv=cv, n_jobs=-1, verbose=0)
            gs.fit(X_train, y_train, groups=groups)
            results[name] = gs
            print(f"[CV] {name}: best MAE = {-gs.best_score_:.3f}, params = {gs.best_params_}")

        # pick the best by CV
        best_name, best_gs = min(results.items(), key=lambda kv: -kv[1].best_score_)
        best_model = best_gs.best_estimator_
        print(f"[SELECT] Best by CV: {best_name}")
    else:
        # Strong baseline: HGBR with sensible defaults
        models = get_pipelines(args.random_state, feature_kind="raw", domain_cols=domain_cols)
        pipe_hgb, _ = models["HistGradientBoosting"]
        best_model = pipe_hgb.set_params(model__learning_rate=0.1,
                                         model__max_iter=600,
                                         model__max_depth=None)
        # Grouped CV quick check
        cv = GroupKFold(n_splits=3)
        from sklearn.model_selection import cross_val_score
        cv_mae = -cross_val_score(best_model, X_train, y_train,
                                  scoring="neg_mean_absolute_error",
                                  cv=cv.split(X_train, y_train, groups),
                                  n_jobs=1)
        print(f"[CV] HGBR 3-fold MAE: {cv_mae.mean():.3f} ± {cv_mae.std():.3f}")
        best_model.fit(X_train, y_train)

    # 4) Evaluate on held-out test set
    y_pred = best_model.predict(X_test)
    metrics = {
        "test_MAE": float(mean_absolute_error(y_test, y_pred)),
        "test_RMSE": rmse(y_test, y_pred),
        "test_R2": float(r2_score(y_test, y_pred)),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1])
    }
    print("[TEST]", json.dumps(metrics, indent=2))
    # 新增保存文件（放在 outputs/metrics.json）
    metrics_path = outdir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[SAVE] Metrics -> {metrics_path}")

    # 5) Plot predictions vs ground truth
    fig_path = outdir / "pred_vs_actual.png"
    plt.figure(figsize=(5,5))
    plt.scatter(y_test, y_pred, s=10, alpha=0.7)
    mn, mx = float(np.min([y_test.min(), y_pred.min()])), float(np.max([y_test.max(), y_pred.max()]))
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel("Actual T_melt")
    plt.ylabel("Predicted T_melt")
    plt.title("Predicted vs Actual (Test)")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    print(f"[SAVE] Scatter plot -> {fig_path}")

    # 6) Permutation importance on test set (model-agnostic)
    #    Get feature names back from ColumnTransformer
    feat_step = best_model.named_steps["feat"]
    names = []
    for name, trans, cols in feat_step.transformers_:
        if name == "poly" and hasattr(trans, "get_feature_names_out"):
            # Polynomial on element fractions
            poly_names = trans.get_feature_names_out(ELEMENTS)
            names.extend(poly_names.tolist())
        elif name in ("frac", "domain", "poly"):
            # Passthrough
            if isinstance(cols, list):
                names.extend(cols)
            else:
                names.extend(list(cols))
    # Compute permutation importance (can take ~seconds)
    from sklearn.utils import Bunch
    imp = permutation_importance(best_model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=1)
    importances = pd.DataFrame({
        "feature": names,
        "importance_mean": imp.importances_mean,
        "importance_std": imp.importances_std
    }).sort_values("importance_mean", ascending=False)
    top_path = outdir / "permutation_importance_top20.csv"
    importances.head(20).to_csv(top_path, index=False)
    print(f"[SAVE] Top-20 permutation importance -> {top_path}")

    # 7) Save model
    model_path = outdir / "best_model.joblib"
    joblib.dump(best_model, model_path)
    print(f"[SAVE] Model pipeline -> {model_path}")

if __name__ == "__main__":
    main()
