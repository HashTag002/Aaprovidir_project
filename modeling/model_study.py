#!/usr/bin/env python3
"""
Model study and training pipeline for AaPROVIDIR price forecasting.

The script compares:
- multiple linear regression;
- random forest;
- XGBoost when installed;
- LSTM when TensorFlow is installed and explicitly requested.

It uses temporal validation, cross-validation on the train split, and saves the
best usable models into ../models for the Dash forecasting dashboard.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
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
    details: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and compare AaPROVIDIR forecasting models.")
    parser.add_argument("--data", default=str(DATA_PATH), help="Path to the semicolon-separated dataset.")
    parser.add_argument("--models-dir", default=str(MODELS_DIR), help="Directory where trained models are saved.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Temporal holdout ratio.")
    parser.add_argument("--cv-splits", type=int, default=4, help="Number of TimeSeriesSplit folds.")
    parser.add_argument("--quick", action="store_true", help="Use a reduced search grid for faster validation.")
    parser.add_argument("--include-lstm", action="store_true", help="Train the optional TensorFlow LSTM model.")
    parser.add_argument("--allow-no-improvement", action="store_true", help="Exit 0 even if no model beats baseline.")
    return parser.parse_args()


def load_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    df["Date"] = pd.to_datetime(df["Date"])
    choc_cols = [col for col in df.columns if col.startswith("Choc_")]
    if choc_cols and "Score_Choc" not in df.columns:
        df["Score_Choc"] = df[choc_cols].sum(axis=1)
    return df.sort_values(["Date", "Produit_ID", "Region_Vente"]).reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in df.select_dtypes(include=[np.number]).columns
        if col != TARGET_COL and col not in HELPER_COLUMNS
    ]


def temporal_split(df: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_rows = max(1, int(len(df) * test_size))
    test_rows = min(test_rows, len(df) - 1)
    return df.iloc[:-test_rows].copy(), df.iloc[-test_rows:].copy()


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


def baseline_predictions(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    if "Prix_T-1" in test_df.columns:
        return test_df["Prix_T-1"].astype(float).to_numpy()
    return np.repeat(float(train_df[TARGET_COL].iloc[-1]), len(test_df))


def build_candidate_models(quick: bool) -> dict[str, tuple[Pipeline, dict[str, list[Any]], str]]:
    rf_grid = (
        {"model__n_estimators": [120], "model__max_depth": [8, None], "model__min_samples_leaf": [1, 3]}
        if quick
        else {
            "model__n_estimators": [160, 260],
            "model__max_depth": [8, 14, None],
            "model__min_samples_leaf": [1, 2, 4],
        }
    )

    candidates: dict[str, tuple[Pipeline, dict[str, list[Any]], str]] = {
        "regression_model": (
            Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
            {},
            "Regression lineaire multiple",
        ),
        "random_forest_model": (
            Pipeline(
                [
                    (
                        "model",
                        RandomForestRegressor(
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                        ),
                    )
                ]
            ),
            rf_grid,
            "Random Forest avec optimisation grille",
        ),
    }

    try:
        from xgboost import XGBRegressor

        xgb_grid = (
            {
                "model__n_estimators": [160],
                "model__max_depth": [3, 5],
                "model__learning_rate": [0.05, 0.1],
            }
            if quick
            else {
                "model__n_estimators": [180, 280],
                "model__max_depth": [3, 5],
                "model__learning_rate": [0.03, 0.08, 0.12],
                "model__subsample": [0.85, 1.0],
                "model__colsample_bytree": [0.85, 1.0],
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
            "XGBoost avec optimisation grille",
        )
    except Exception as exc:
        print(f"[skip] XGBoost indisponible: {exc}")

    return candidates


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
        details=json.dumps(search.best_params_, ensure_ascii=False),
    )


def make_lstm_sequences(x: np.ndarray, y: np.ndarray, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for idx in range(sequence_length, len(x)):
        xs.append(x[idx - sequence_length : idx])
        ys.append(y[idx])
    return np.asarray(xs), np.asarray(ys)


def train_lstm_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    baseline_mae: float,
    models_dir: Path,
    quick: bool,
) -> StudyResult:
    try:
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.layers import LSTM, Dense, Dropout
    except Exception as exc:
        return StudyResult(
            name="lstm_model",
            estimator_type="LSTM TensorFlow",
            status="skipped",
            path=None,
            cv_mae=None,
            holdout_mae=None,
            holdout_rmse=None,
            holdout_mape=None,
            holdout_r2=None,
            beats_baseline=False,
            details=f"TensorFlow indisponible: {exc}",
        )

    tf.random.set_seed(RANDOM_STATE)
    sequence_length = 6
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train_scaled = x_scaler.fit_transform(train_df[features])
    y_train_scaled = y_scaler.fit_transform(train_df[[TARGET_COL]]).ravel()
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    x_all_scaled = x_scaler.transform(combined_df[features])
    y_all_scaled = y_scaler.transform(combined_df[[TARGET_COL]]).ravel()

    x_seq, y_seq = make_lstm_sequences(x_train_scaled, y_train_scaled, sequence_length)
    if len(x_seq) < 20:
        return StudyResult(
            name="lstm_model",
            estimator_type="LSTM TensorFlow",
            status="skipped",
            path=None,
            cv_mae=None,
            holdout_mae=None,
            holdout_rmse=None,
            holdout_mape=None,
            holdout_r2=None,
            beats_baseline=False,
            details="Historique insuffisant pour sequence LSTM.",
        )

    model = Sequential(
        [
            LSTM(32 if quick else 64, input_shape=(sequence_length, len(features))),
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

    test_start = len(train_df)
    predictions_scaled = []
    for idx in range(test_start, len(combined_df)):
        seq = x_all_scaled[idx - sequence_length : idx]
        predictions_scaled.append(float(model.predict(seq[np.newaxis, ...], verbose=0)[0][0]))

    predictions = y_scaler.inverse_transform(np.asarray(predictions_scaled).reshape(-1, 1)).ravel()
    metrics = regression_metrics(test_df[TARGET_COL], predictions)
    model_path = models_dir / "lstm_model.keras"
    preprocessor_path = models_dir / "lstm_preprocessor.joblib"
    model.save(model_path)
    joblib.dump(
        {
            "features": features,
            "sequence_length": sequence_length,
            "x_scaler": x_scaler,
            "y_scaler": y_scaler,
        },
        preprocessor_path,
    )
    return StudyResult(
        name="lstm_model",
        estimator_type="LSTM TensorFlow",
        status="trained",
        path=str(model_path.relative_to(BASE_DIR)),
        cv_mae=None,
        holdout_mae=metrics["mae"],
        holdout_rmse=metrics["rmse"],
        holdout_mape=metrics["mape"],
        holdout_r2=metrics["r2"],
        beats_baseline=metrics["mae"] < baseline_mae,
        details=f"preprocessor={preprocessor_path.relative_to(BASE_DIR)}",
    )


def save_results(results: list[StudyResult], models_dir: Path) -> None:
    records = [result.__dict__ for result in results]
    metrics_df = pd.DataFrame(records)
    metrics_df.to_csv(models_dir / "model_study_metrics.csv", index=False)
    (models_dir / "model_study_results.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    trained_joblib = [
        result
        for result in results
        if result.status == "trained" and result.path and result.path.endswith(".joblib") and result.holdout_mae is not None
    ]
    if trained_joblib:
        best = min(trained_joblib, key=lambda result: result.holdout_mae or float("inf"))
        shutil.copyfile(BASE_DIR / best.path, models_dir / "best_model.joblib")
        (models_dir / "best_model_metadata.json").write_text(
            json.dumps(best.__dict__, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def main() -> int:
    args = parse_args()
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.data)
    features = feature_columns(df)
    train_df, test_df = temporal_split(df, args.test_size)
    x_train, y_train = train_df[features], train_df[TARGET_COL]
    x_test, y_test = test_df[features], test_df[TARGET_COL]

    baseline_pred = baseline_predictions(train_df, test_df)
    baseline_metrics = regression_metrics(y_test, baseline_pred)
    results = [
        StudyResult(
            name="baseline_prix_t_1",
            estimator_type="Baseline naive Prix_T-1",
            status="reference",
            path=None,
            cv_mae=None,
            holdout_mae=baseline_metrics["mae"],
            holdout_rmse=baseline_metrics["rmse"],
            holdout_mape=baseline_metrics["mape"],
            holdout_r2=baseline_metrics["r2"],
            beats_baseline=False,
            details="Reference: prediction = Prix_T-1 sur le holdout temporel.",
        )
    ]
    baseline_mae = baseline_metrics["mae"]

    for name, (pipeline, grid, description) in build_candidate_models(args.quick).items():
        try:
            result = cross_validate_model(
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
            )
        except Exception as exc:
            result = StudyResult(
                name=name,
                estimator_type=description,
                status="failed",
                path=None,
                cv_mae=None,
                holdout_mae=None,
                holdout_rmse=None,
                holdout_mape=None,
                holdout_r2=None,
                beats_baseline=False,
                details=str(exc),
            )
        results.append(result)

    if args.include_lstm:
        results.append(train_lstm_model(train_df, test_df, features, baseline_mae, models_dir, args.quick))

    save_results(results, models_dir)
    print(pd.DataFrame([result.__dict__ for result in results]).to_string(index=False))

    improved = [result for result in results if result.beats_baseline]
    if not improved and not args.allow_no_improvement:
        print("Aucun modele n'a battu la baseline Prix_T-1.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
