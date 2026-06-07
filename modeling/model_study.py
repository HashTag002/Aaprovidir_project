#!/usr/bin/env python3
"""
Windowed model study for AaPROVIDIR price forecasting.

Each supervised example uses the previous N months (12 by default) to predict
the next month. This is the same inference contract used by the Dash app.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Dataset.csv"
MODELS_DIR = BASE_DIR / "models"
TARGET_COL = "Prix_Vente_FCFA_kg"
HELPER_COLUMNS = {"Score_Choc"}
RANDOM_STATE = 42


@dataclass
class StudyResult:
    name: str
    estimator_type: str
    status: str
    path: str | None
    cv_mae: float | None
    holdout_mae: float | None
    holdout_rmse: float | None
    holdout_mape: float | None
    holdout_r2: float | None
    beats_baseline: bool
    window_size: int
    selected_features: str
    details: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train windowed AaPROVIDIR forecasting models.")
    parser.add_argument("--data", default=str(DATA_PATH), help="Path to the semicolon-separated dataset.")
    parser.add_argument("--models-dir", default=str(MODELS_DIR), help="Directory where trained models are saved.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Temporal holdout ratio.")
    parser.add_argument("--cv-splits", type=int, default=4, help="Number of TimeSeriesSplit folds.")
    parser.add_argument("--window-size", type=int, default=12, help="Historical months used to predict next month.")
    parser.add_argument("--top-features", type=int, default=10, help="Number of key base features kept before windowing.")
    parser.add_argument("--pca-components", type=int, default=10, help="PCA components for linear/MLP models.")
    parser.add_argument("--quick", action="store_true", help="Use a reduced grid for faster validation.")
    parser.add_argument("--include-lstm", action="store_true", help="Train the optional TensorFlow LSTM model.")
    parser.add_argument("--allow-no-improvement", action="store_true", help="Exit 0 even if no model beats baseline.")
    return parser.parse_args()


def load_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    df["Date"] = pd.to_datetime(df["Date"])
    choc_cols = [col for col in df.columns if col.startswith("Choc_")]
    if choc_cols and "Score_Choc" not in df.columns:
        df["Score_Choc"] = df[choc_cols].sum(axis=1)
    return df.sort_values(["Produit_ID", "Region_Vente", "Date"]).reset_index(drop=True)


def base_numeric_features(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in df.select_dtypes(include=[np.number]).columns
        if col != TARGET_COL and col not in HELPER_COLUMNS
    ]


def rank_features(df: pd.DataFrame, features: list[str], top_k: int) -> pd.DataFrame:
    x = df[features].replace([np.inf, -np.inf], np.nan).fillna(df[features].median())
    y = df[TARGET_COL]
    scores = mutual_info_regression(x, y, random_state=RANDOM_STATE)
    ranking = pd.DataFrame({"feature": features, "importance": scores})
    ranking = ranking.sort_values("importance", ascending=False).reset_index(drop=True)
    return ranking.head(min(top_k, len(ranking)))


def make_windowed_dataset(
    df: pd.DataFrame,
    features: list[str],
    window_size: int,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    targets: list[float] = []
    meta_rows: list[dict[str, Any]] = []
    window_features = features + [TARGET_COL]

    for (produit, region), group in df.groupby(["Produit_ID", "Region_Vente"], sort=False):
        group = group.sort_values("Date").reset_index(drop=True)
        if len(group) <= window_size:
            continue
        for idx in range(window_size, len(group)):
            history = group.iloc[idx - window_size : idx]
            row: dict[str, Any] = {}
            for lag in range(1, window_size + 1):
                source = history.iloc[-lag]
                for feature in window_features:
                    row[f"lag_{lag:02d}__{feature}"] = source[feature]
            rows.append(row)
            targets.append(float(group.iloc[idx][TARGET_COL]))
            meta_rows.append(
                {
                    "Date": group.iloc[idx]["Date"],
                    "Produit_ID": produit,
                    "Region_Vente": region,
                    "Prix_T-1": group.iloc[idx].get("Prix_T-1", history.iloc[-1][TARGET_COL]),
                }
            )

    return pd.DataFrame(rows), pd.Series(targets, name=TARGET_COL), pd.DataFrame(meta_rows)


def temporal_split(x: pd.DataFrame, y: pd.Series, meta: pd.DataFrame, test_size: float):
    order = meta["Date"].sort_values().index
    x, y, meta = x.loc[order].reset_index(drop=True), y.loc[order].reset_index(drop=True), meta.loc[order].reset_index(drop=True)
    n_test = max(1, int(len(x) * test_size))
    n_test = min(n_test, len(x) - 1)
    return x.iloc[:-n_test], x.iloc[-n_test:], y.iloc[:-n_test], y.iloc[-n_test:], meta.iloc[:-n_test], meta.iloc[-n_test:]


def regression_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    non_zero = y_true_arr != 0
    mape = (
        float(np.mean(np.abs((y_true_arr[non_zero] - y_pred_arr[non_zero]) / y_true_arr[non_zero])) * 100)
        if non_zero.any()
        else math.nan
    )
    return {
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))),
        "mape": mape,
        "r2": float(r2_score(y_true_arr, y_pred_arr)) if len(y_true_arr) > 1 else math.nan,
    }


def build_candidate_models(quick: bool, pca_components: int, n_features: int):
    pca_n = max(1, min(pca_components, n_features))
    rf_grid = (
        {"model__n_estimators": [120], "model__max_depth": [8, None], "model__min_samples_leaf": [1, 3]}
        if quick
        else {"model__n_estimators": [180, 280], "model__max_depth": [8, 14, None], "model__min_samples_leaf": [1, 2, 4]}
    )
    mlp_grid = (
        {"model__hidden_layer_sizes": [(48,), (64, 24)], "model__alpha": [0.0005]}
        if quick
        else {"model__hidden_layer_sizes": [(64,), (96, 48)], "model__alpha": [0.0001, 0.0005]}
    )

    candidates = {
        "mlp_model": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=pca_n)),
                    (
                        "model",
                        MLPRegressor(
                            random_state=RANDOM_STATE,
                            max_iter=600 if quick else 1200,
                            early_stopping=True,
                        ),
                    ),
                ]
            ),
            mlp_grid,
            f"MLP fenetre avec PCA({pca_n})",
        ),
        "random_forest_model": (
            Pipeline([("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))]),
            rf_grid,
            "Random Forest fenetre",
        ),
    }

    try:
        from xgboost import XGBRegressor

        xgb_grid = (
            {"model__n_estimators": [160], "model__max_depth": [3, 5], "model__learning_rate": [0.05, 0.1]}
            if quick
            else {
                "model__n_estimators": [220, 320],
                "model__max_depth": [3, 5],
                "model__learning_rate": [0.03, 0.08, 0.12],
                "model__subsample": [0.85, 1.0],
            }
        )
        candidates["xgboost_model"] = (
            Pipeline(
                [
                    (
                        "model",
                        XGBRegressor(
                            objective="reg:squarederror",
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                            verbosity=0,
                        ),
                    )
                ]
            ),
            xgb_grid,
            "XGBoost fenetre",
        )
    except Exception as exc:
        print(f"[skip] XGBoost indisponible: {exc}")

    return candidates


def train_row_regression_model(
    df: pd.DataFrame,
    features: list[str],
    train_cutoff: pd.Timestamp,
    baseline_mae: float,
    models_dir: Path,
    window_size: int,
) -> StudyResult:
    train_df = df[df["Date"] <= train_cutoff].copy()
    test_df = df[df["Date"] > train_cutoff].copy()
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )
    model.fit(train_df[features], train_df[TARGET_COL])
    predictions = model.predict(test_df[features])
    metrics = regression_metrics(test_df[TARGET_COL], predictions)
    model_path = models_dir / "regression_model.joblib"
    joblib.dump(model, model_path)
    return StudyResult(
        "regression_model",
        "Regression lineaire multiple explicable",
        "trained",
        str(model_path.relative_to(BASE_DIR)),
        None,
        metrics["mae"],
        metrics["rmse"],
        metrics["mape"],
        metrics["r2"],
        metrics["mae"] < baseline_mae,
        0,
        ", ".join(features),
        "Modele tabulaire explicable conserve comme reference de production, sans PCA pour garder les coefficients lisibles.",
    )


def cross_validate_model(
    name: str,
    pipeline: Pipeline,
    grid: dict[str, list[Any]],
    description: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    baseline_mae: float,
    models_dir: Path,
    cv_splits: int,
    window_size: int,
    selected_features: list[str],
) -> StudyResult:
    cv = TimeSeriesSplit(n_splits=max(2, cv_splits))
    search = GridSearchCV(
        pipeline,
        grid,
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )
    search.fit(x_train, y_train)
    best_model = search.best_estimator_
    predictions = best_model.predict(x_test)
    metrics = regression_metrics(y_test, predictions)
    model_path = models_dir / f"{name}.joblib"
    joblib.dump(best_model, model_path)
    return StudyResult(
        name=name,
        estimator_type=description,
        status="trained",
        path=str(model_path.relative_to(BASE_DIR)),
        cv_mae=float(-search.best_score_),
        holdout_mae=metrics["mae"],
        holdout_rmse=metrics["rmse"],
        holdout_mape=metrics["mape"],
        holdout_r2=metrics["r2"],
        beats_baseline=metrics["mae"] < baseline_mae,
        window_size=window_size,
        selected_features=", ".join(selected_features),
        details=json.dumps(search.best_params_, ensure_ascii=False),
    )


def make_lstm_sequences(x: np.ndarray, y: np.ndarray, sequence_length: int):
    xs, ys = [], []
    for idx in range(sequence_length, len(x)):
        xs.append(x[idx - sequence_length : idx])
        ys.append(y[idx])
    return np.asarray(xs), np.asarray(ys)


def train_lstm_model(df: pd.DataFrame, train_cutoff: pd.Timestamp, features: list[str], baseline_mae: float, models_dir: Path, window_size: int, quick: bool):
    try:
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    except Exception as exc:
        return StudyResult("lstm_model", "LSTM TensorFlow", "skipped", None, None, None, None, None, None, False, window_size, ", ".join(features), f"TensorFlow indisponible: {exc}")

    tf.random.set_seed(RANDOM_STATE)
    lstm_features = features + [TARGET_COL]
    train_df = df[df["Date"] <= train_cutoff].copy()
    test_df = df[df["Date"] > train_cutoff].copy()
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_scaled = x_scaler.fit_transform(train_df[lstm_features])
    y_scaled = y_scaler.fit_transform(train_df[[TARGET_COL]]).ravel()
    x_seq, y_seq = make_lstm_sequences(x_scaled, y_scaled, window_size)
    if len(x_seq) < 20 or test_df.empty:
        return StudyResult("lstm_model", "LSTM TensorFlow", "skipped", None, None, None, None, None, None, False, window_size, ", ".join(features), "Historique insuffisant.")

    model = Sequential(
        [
            Input(shape=(window_size, len(lstm_features))),
            LSTM(32 if quick else 64),
            Dropout(0.15),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mae")
    model.fit(
        x_seq,
        y_seq,
        epochs=8 if quick else 30,
        batch_size=32,
        validation_split=0.2,
        verbose=0,
        callbacks=[EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
    )

    combined = pd.concat([train_df, test_df], ignore_index=True)
    all_x_scaled = x_scaler.transform(combined[lstm_features])
    start_idx = len(train_df)
    predictions_scaled = []
    for idx in range(start_idx, len(combined)):
        seq = all_x_scaled[idx - window_size : idx]
        predictions_scaled.append(float(model.predict(seq[np.newaxis, ...], verbose=0)[0][0]))
    predictions = y_scaler.inverse_transform(np.asarray(predictions_scaled).reshape(-1, 1)).ravel()
    metrics = regression_metrics(test_df[TARGET_COL], predictions)
    model_path = models_dir / "lstm_model.keras"
    preprocessor_path = models_dir / "lstm_preprocessor.joblib"
    model.save(model_path)
    joblib.dump({"features": lstm_features, "window_size": window_size, "x_scaler": x_scaler, "y_scaler": y_scaler}, preprocessor_path)
    return StudyResult(
        "lstm_model",
        "LSTM TensorFlow fenetre",
        "trained",
        str(model_path.relative_to(BASE_DIR)),
        None,
        metrics["mae"],
        metrics["rmse"],
        metrics["mape"],
        metrics["r2"],
        metrics["mae"] < baseline_mae,
        window_size,
        ", ".join(features),
        f"preprocessor={preprocessor_path.relative_to(BASE_DIR)}",
    )


def save_results(results: list[StudyResult], models_dir: Path, feature_importance: pd.DataFrame) -> None:
    records = [asdict(result) for result in results]
    metrics_df = pd.DataFrame(records)
    metrics_df.to_csv(models_dir / "model_study_metrics.csv", index=False)
    (models_dir / "model_study_results.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    feature_importance.to_csv(models_dir / "feature_importance.csv", index=False)
    trained_joblib = [
        result
        for result in results
        if result.status == "trained" and result.path and result.path.endswith(".joblib") and result.holdout_mae is not None
    ]
    if trained_joblib:
        best = min(trained_joblib, key=lambda result: result.holdout_mae or float("inf"))
        shutil.copyfile(BASE_DIR / best.path, models_dir / "best_model.joblib")
        (models_dir / "best_model_metadata.json").write_text(json.dumps(asdict(best), ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    df = load_dataset(args.data)
    all_features = base_numeric_features(df)
    feature_importance = rank_features(df, all_features, args.top_features)
    selected_features = feature_importance["feature"].tolist()
    x, y, meta = make_windowed_dataset(df, selected_features, args.window_size)
    x_train, x_test, y_train, y_test, meta_train, meta_test = temporal_split(x, y, meta, args.test_size)

    baseline_pred = meta_test["Prix_T-1"].astype(float).to_numpy()
    baseline_metrics = regression_metrics(y_test, baseline_pred)
    baseline_mae = baseline_metrics["mae"]
    results = [
        StudyResult(
            "baseline_prix_t_1",
            "Baseline naive Prix_T-1",
            "reference",
            None,
            None,
            baseline_metrics["mae"],
            baseline_metrics["rmse"],
            baseline_metrics["mape"],
            baseline_metrics["r2"],
            False,
            args.window_size,
            ", ".join(selected_features),
            "Reference: prediction = Prix_T-1 sur le holdout temporel.",
        )
    ]
    train_cutoff = meta_train["Date"].max()
    results.append(
        train_row_regression_model(
            df,
            all_features,
            train_cutoff,
            baseline_mae,
            models_dir,
            args.window_size,
        )
    )

    for name, (pipeline, grid, description) in build_candidate_models(args.quick, args.pca_components, x_train.shape[1]).items():
        try:
            results.append(
                cross_validate_model(
                    name,
                    pipeline,
                    grid,
                    description,
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    baseline_mae,
                    models_dir,
                    args.cv_splits,
                    args.window_size,
                    selected_features,
                )
            )
        except Exception as exc:
            results.append(StudyResult(name, description, "failed", None, None, None, None, None, None, False, args.window_size, ", ".join(selected_features), str(exc)))

    if args.include_lstm:
        results.append(train_lstm_model(df, train_cutoff, selected_features, baseline_mae, models_dir, args.window_size, args.quick))

    save_results(results, models_dir, feature_importance)
    print(pd.DataFrame([asdict(result) for result in results]).to_string(index=False))
    improved = [result for result in results if result.beats_baseline]
    if not improved and not args.allow_no_improvement:
        print("Aucun modele n'a battu la baseline Prix_T-1.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
