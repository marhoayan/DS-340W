import json
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def load_messages(json_path: str) -> pd.DataFrame:
    """
    Load StockTwits messages from the scraper JSON output.

    Expected JSON structure: a list of dicts with keys like
    id, author, time, post, sentiment
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {json_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected JSON file to contain a list of messages.")

    df = pd.DataFrame(data)

    required_cols = {"id", "time", "post", "sentiment"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in input JSON: {missing}")

    return df


def clean_messages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize the raw message dataframe.
    """
    df = df.copy()

    # Parse timestamps
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    df = df.dropna(subset=["time"])

    # Normalize sentiment labels
    df["sentiment"] = (
        df["sentiment"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"null": "unlabeled", "none": "unlabeled", "nan": "unlabeled"})
    )

    allowed = {"bullish", "bearish", "unlabeled"}
    df.loc[~df["sentiment"].isin(allowed), "sentiment"] = "unlabeled"

    # Drop duplicate message ids if present
    df = df.drop_duplicates(subset=["id"]).sort_values("time").reset_index(drop=True)

    return df


def build_minute_message_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert message-level data into minute-level counts.

    Output columns:
    - bullish_count
    - bearish_count
    - unlabeled_count
    - total_count
    """
    df = df.copy()
    df["minute"] = df["time"].dt.floor("min")

    minute_counts = (
        df.groupby(["minute", "sentiment"])
        .size()
        .unstack(fill_value=0)
        .rename_axis(None, axis=1)
    )

    for col in ["bullish", "bearish", "unlabeled"]:
        if col not in minute_counts.columns:
            minute_counts[col] = 0

    minute_counts = minute_counts[["bullish", "bearish", "unlabeled"]].rename(
        columns={
            "bullish": "bullish_count",
            "bearish": "bearish_count",
            "unlabeled": "unlabeled_count",
        }
    )

    minute_counts["total_count"] = (
        minute_counts["bullish_count"]
        + minute_counts["bearish_count"]
        + minute_counts["unlabeled_count"]
    )

    minute_counts = minute_counts.sort_index()

    # Fill missing minutes so rolling windows behave correctly
    full_index = pd.date_range(
        start=minute_counts.index.min(),
        end=minute_counts.index.max(),
        freq="min",
        tz=minute_counts.index.tz,
    )
    minute_counts = minute_counts.reindex(full_index, fill_value=0)
    minute_counts.index.name = "minute"

    return minute_counts.reset_index()


def build_sentiment_features(
    minute_df: pd.DataFrame,
    rolling_window_minutes: int = 5,
    abnormal_density_lookback_minutes: int = 60,
    eps: float = 1e-6,
) -> pd.DataFrame:
    """
    Build rolling sentiment and density features from minute-level counts.

    Features created:
    - bullish_count_{window}m
    - bearish_count_{window}m
    - unlabeled_count_{window}m
    - message_count_{window}m
    - net_sentiment
    - bullish_share
    - sentiment_change
    - abnormal_density

    Definitions:
    - net_sentiment = (bullish - bearish) / (bullish + bearish + 1)
    - bullish_share = bullish / (bullish + bearish + 1)
    - sentiment_change = net_sentiment_t - net_sentiment_{t-1}
    - abnormal_density = message_count_{window}m / avg(total_count over prior lookback)
    """
    df = minute_df.copy()

    required_cols = {
        "minute",
        "bullish_count",
        "bearish_count",
        "unlabeled_count",
        "total_count",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Minute dataframe is missing required columns: {missing}")

    window = rolling_window_minutes

    df[f"bullish_count_{window}m"] = (
        df["bullish_count"].rolling(window=window, min_periods=1).sum()
    )
    df[f"bearish_count_{window}m"] = (
        df["bearish_count"].rolling(window=window, min_periods=1).sum()
    )
    df[f"unlabeled_count_{window}m"] = (
        df["unlabeled_count"].rolling(window=window, min_periods=1).sum()
    )
    df[f"message_count_{window}m"] = (
        df["total_count"].rolling(window=window, min_periods=1).sum()
    )

    bull = df[f"bullish_count_{window}m"]
    bear = df[f"bearish_count_{window}m"]
    total_labeled = bull + bear

    df["net_sentiment"] = (bull - bear) / (total_labeled + 1.0)
    df["bullish_share"] = bull / (total_labeled + 1.0)
    df["sentiment_change"] = df["net_sentiment"].diff().fillna(0.0)

    prior_avg_density = (
        df["total_count"]
        .shift(1)
        .rolling(window=abnormal_density_lookback_minutes, min_periods=5)
        .mean()
    )

    df["abnormal_density"] = df[f"message_count_{window}m"] / (prior_avg_density + eps)

    # Optional helpful extras
    df["labeled_message_count"] = total_labeled
    df["labeled_fraction"] = total_labeled / (df[f"message_count_{window}m"] + eps)

    # Replace infinite values from division edge cases
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def filter_market_hours(
    df: pd.DataFrame,
    market_open: str = "09:30",
    market_close: str = "16:00",
    timezone: Optional[str] = "America/New_York",
) -> pd.DataFrame:
    """
    Optional filter to keep only regular U.S. market hours.

    Assumes 'minute' is timezone-aware UTC or already tz-aware.
    """
    out = df.copy()

    if "minute" not in out.columns:
        raise ValueError("Expected column 'minute' in dataframe.")

    minute_series = pd.to_datetime(out["minute"], utc=True, errors="coerce")
    if timezone:
        minute_series = minute_series.dt.tz_convert(timezone)

    out["minute"] = minute_series
    out = out.set_index("minute")
    out = out.between_time(market_open, market_close, inclusive="left").reset_index()

    return out


def save_features(df: pd.DataFrame, output_path: str) -> None:
    """
    Save features to CSV or parquet based on extension.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError("Output file must end in .csv or .parquet")


def run_feature_pipeline(
    input_json: str,
    output_path: str,
    rolling_window_minutes: int = 5,
    abnormal_density_lookback_minutes: int = 60,
    market_hours_only: bool = False,
    market_timezone: str = "America/New_York",
) -> pd.DataFrame:
    """
    End-to-end feature pipeline:
    1. Load raw messages JSON
    2. Clean messages
    3. Build minute-level counts
    4. Build rolling sentiment features
    5. Optionally filter to market hours
    6. Save output
    """
    raw_df = load_messages(input_json)
    clean_df = clean_messages(raw_df)
    minute_df = build_minute_message_table(clean_df)
    features_df = build_sentiment_features(
        minute_df=minute_df,
        rolling_window_minutes=rolling_window_minutes,
        abnormal_density_lookback_minutes=abnormal_density_lookback_minutes,
    )

    if market_hours_only:
        features_df = filter_market_hours(
            features_df,
            timezone=market_timezone,
        )

    save_features(features_df, output_path)
    return features_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build minute-level rolling sentiment and density features from StockTwits JSON."
    )
    parser.add_argument("input_json", type=str, help="Path to input StockTwits JSON file")
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path (.csv or .parquet)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Rolling window size in minutes for sentiment features",
    )
    parser.add_argument(
        "--density-lookback",
        type=int,
        default=60,
        help="Lookback window in minutes for abnormal density baseline",
    )
    parser.add_argument(
        "--market-hours-only",
        action="store_true",
        help="Keep only regular U.S. market hours",
    )
    parser.add_argument(
        "--market-timezone",
        type=str,
        default="America/New_York",
        help="Timezone used for market-hours filtering",
    )

    args = parser.parse_args()

    features_df = run_feature_pipeline(
        input_json=args.input_json,
        output_path=args.output,
        rolling_window_minutes=args.window,
        abnormal_density_lookback_minutes=args.density_lookback,
        market_hours_only=args.market_hours_only,
        market_timezone=args.market_timezone,
    )

    print(f"Built features with {len(features_df)} rows.")
    print("Columns:")
    print(list(features_df.columns))


if __name__ == "__main__":
    main()