import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_FEATURE_COLUMNS = [
    "net_sentiment",
    "bullish_share",
    "sentiment_change",
    "message_count_5m",
    "abnormal_density",
    "one_min_return",
    "trailing_volatility",
    "volume_zscore_30m",
]


def load_model_data(input_path: str) -> pd.DataFrame:
    """
    Load merged sentiment + market data from CSV or parquet.
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Model data file not found: {input_path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError("Input file must be .csv or .parquet")

    if "minute" not in df.columns:
        raise ValueError("Expected 'minute' column in model dataset.")

    df["minute"] = pd.to_datetime(df["minute"], errors="coerce")
    df = df.dropna(subset=["minute"]).sort_values("minute").reset_index(drop=True)

    return df


def validate_columns(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> None:
    """
    Ensure required columns exist.
    """
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")

    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")


def prepare_model_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "direction_5m",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Keep only rows with valid feature/target information.
    Returns:
        X, y, aligned_dataframe
    """
    validate_columns(df, feature_cols, target_col)

    keep_cols = ["minute"] + feature_cols + [target_col]
    model_df = df[keep_cols].copy()

    model_df[target_col] = pd.to_numeric(model_df[target_col], errors="coerce")
    model_df = model_df.dropna(subset=[target_col]).reset_index(drop=True)

    X = model_df[feature_cols].copy()
    y = model_df[target_col].astype(int).copy()

    return X, y, model_df


def chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
    meta_df: pd.DataFrame,
    train_fraction: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Chronological train/test split to avoid leakage.
    """
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")

    split_idx = int(len(X) * train_fraction)
    if split_idx <= 0 or split_idx >= len(X):
        raise ValueError("Not enough rows for train/test split.")

    X_train = X.iloc[:split_idx].reset_index(drop=True)
    X_test = X.iloc[split_idx:].reset_index(drop=True)
    y_train = y.iloc[:split_idx].reset_index(drop=True)
    y_test = y.iloc[split_idx:].reset_index(drop=True)
    meta_train = meta_df.iloc[:split_idx].reset_index(drop=True)
    meta_test = meta_df.iloc[split_idx:].reset_index(drop=True)

    return X_train, X_test, y_train, y_test, meta_train, meta_test


def build_logistic_pipeline() -> Pipeline:
    """
    Logistic regression with median imputation + scaling.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def build_random_forest() -> Pipeline:
    """
    Random forest with median imputation.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=6,
                    min_samples_leaf=10,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def evaluate_binary_classifier(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    """
    Compute binary classification metrics.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["auc"] = np.nan

    return metrics


def fit_and_evaluate_model(
    model_name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, Pipeline]:
    """
    Train a model and return metrics, predictions, probabilities, and fitted pipeline.
    """
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
    else:
        # fallback if needed
        y_prob = y_pred.astype(float)

    metrics = evaluate_binary_classifier(y_test, y_pred, y_prob)
    metrics["model"] = model_name

    return metrics, y_pred, y_prob, pipeline


def extract_logistic_coefficients(
    pipeline: Pipeline,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Extract logistic regression coefficients after fitting.
    """
    model = pipeline.named_steps["model"]
    coef = model.coef_[0]

    coef_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "coefficient": coef,
            "abs_coefficient": np.abs(coef),
        }
    ).sort_values("abs_coefficient", ascending=False)

    return coef_df.reset_index(drop=True)


def extract_random_forest_importance(
    pipeline: Pipeline,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Extract feature importances from a fitted random forest.
    """
    model = pipeline.named_steps["model"]
    imp = model.feature_importances_

    imp_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": imp,
        }
    ).sort_values("importance", ascending=False)

    return imp_df.reset_index(drop=True)


def build_predictions_output(
    meta_test: pd.DataFrame,
    y_test: pd.Series,
    logistic_pred: np.ndarray,
    logistic_prob: np.ndarray,
    rf_pred: np.ndarray,
    rf_prob: np.ndarray,
) -> pd.DataFrame:
    """
    Build a dataframe of out-of-sample predictions.
    """
    out = pd.DataFrame(
        {
            "minute": meta_test["minute"],
            "actual": y_test,
            "logistic_pred": logistic_pred,
            "logistic_prob_up": logistic_prob,
            "rf_pred": rf_pred,
            "rf_prob_up": rf_prob,
        }
    )
    return out.reset_index(drop=True)


def save_json(obj: dict, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)


def save_dataframe(df: pd.DataFrame, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError("Output file must end in .csv or .parquet")


def run_modeling_pipeline(
    input_path: str,
    metrics_output: str,
    predictions_output: str,
    logistic_coef_output: str,
    rf_importance_output: str,
    target_col: str = "direction_5m",
    feature_cols: Optional[list[str]] = None,
    train_fraction: float = 0.8,
) -> dict[str, dict[str, float]]:
    """
    End-to-end modeling pipeline.
    """
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLUMNS.copy()

    df = load_model_data(input_path)
    X, y, model_df = prepare_model_data(df, feature_cols=feature_cols, target_col=target_col)

    X_train, X_test, y_train, y_test, meta_train, meta_test = chronological_split(
        X=X,
        y=y,
        meta_df=model_df[["minute"]],
        train_fraction=train_fraction,
    )

    logistic_pipeline = build_logistic_pipeline()
    rf_pipeline = build_random_forest()

    logistic_metrics, logistic_pred, logistic_prob, fitted_logistic = fit_and_evaluate_model(
        model_name="logistic_regression",
        pipeline=logistic_pipeline,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    rf_metrics, rf_pred, rf_prob, fitted_rf = fit_and_evaluate_model(
        model_name="random_forest",
        pipeline=rf_pipeline,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    metrics_summary = {
        "logistic_regression": logistic_metrics,
        "random_forest": rf_metrics,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_columns": feature_cols,
        "target_column": target_col,
    }

    predictions_df = build_predictions_output(
        meta_test=meta_test,
        y_test=y_test,
        logistic_pred=logistic_pred,
        logistic_prob=logistic_prob,
        rf_pred=rf_pred,
        rf_prob=rf_prob,
    )

    logistic_coef_df = extract_logistic_coefficients(
        pipeline=fitted_logistic,
        feature_cols=feature_cols,
    )

    rf_importance_df = extract_random_forest_importance(
        pipeline=fitted_rf,
        feature_cols=feature_cols,
    )

    save_json(metrics_summary, metrics_output)
    save_dataframe(predictions_df, predictions_output)
    save_dataframe(logistic_coef_df, logistic_coef_output)
    save_dataframe(rf_importance_df, rf_importance_output)

    return metrics_summary


def parse_feature_columns(raw: Optional[str]) -> Optional[list[str]]:
    """
    Parse comma-separated feature names from CLI.
    """
    if raw is None:
        return None

    cols = [x.strip() for x in raw.split(",") if x.strip()]
    return cols if cols else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train prediction models on merged StockTwits sentiment + market data."
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to merged model dataset (.csv or .parquet)",
    )
    parser.add_argument(
        "--metrics-output",
        type=str,
        required=True,
        help="Path to JSON file for metrics output",
    )
    parser.add_argument(
        "--predictions-output",
        type=str,
        required=True,
        help="Path to CSV/parquet file for test predictions",
    )
    parser.add_argument(
        "--logistic-coef-output",
        type=str,
        required=True,
        help="Path to CSV/parquet file for logistic coefficients",
    )
    parser.add_argument(
        "--rf-importance-output",
        type=str,
        required=True,
        help="Path to CSV/parquet file for random forest importances",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="direction_5m",
        help="Binary target column",
    )
    parser.add_argument(
        "--feature-cols",
        type=str,
        default=None,
        help="Optional comma-separated feature list",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Fraction of rows used for chronological training split",
    )

    args = parser.parse_args()

    feature_cols = parse_feature_columns(args.feature_cols)

    metrics = run_modeling_pipeline(
        input_path=args.input_path,
        metrics_output=args.metrics_output,
        predictions_output=args.predictions_output,
        logistic_coef_output=args.logistic_coef_output,
        rf_importance_output=args.rf_importance_output,
        target_col=args.target_col,
        feature_cols=feature_cols,
        train_fraction=args.train_fraction,
    )

    print("Modeling complete.")
    print(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    main()