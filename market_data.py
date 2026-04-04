import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf


VALID_INTERVALS = {
    "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h",
    "1d", "5d", "1wk", "1mo", "3mo"
}


def download_price_data(
    ticker: str,
    interval: str = "1m",
    period: str = "30d",
    auto_adjust: bool = False,
    prepost: bool = False,
) -> pd.DataFrame:
    """
    Download OHLCV price data for a ticker using yfinance.

    Notes:
    - Intraday data supports intervals like 1m, 2m, 5m, etc.
    - Intraday history cannot extend beyond the last 60 days.
    """
    if interval not in VALID_INTERVALS:
        raise ValueError(f"Unsupported interval: {interval}")

    df = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        auto_adjust=auto_adjust,
        prepost=prepost,
        progress=False,
        group_by="column",
        threads=True,
    )

    if df.empty:
        raise ValueError(
            f"No market data returned for ticker={ticker}, interval={interval}, period={period}"
        )

    return df


def clean_price_data(
    df: pd.DataFrame,
    timezone: str = "America/New_York",
) -> pd.DataFrame:
    """
    Clean yfinance output into a flat minute-level dataframe.

    Output columns:
    - minute
    - open
    - high
    - low
    - close
    - volume
    """
    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)

    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    out = out.rename(columns=rename_map)

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Missing required price columns: {missing}")

    out = out.reset_index()

    time_col = None
    for candidate in ["Datetime", "Date"]:
        if candidate in out.columns:
            time_col = candidate
            break

    if time_col is None:
        raise ValueError("Could not find Datetime/Date column in price data.")

    out = out.rename(columns={time_col: "minute"})
    out["minute"] = pd.to_datetime(out["minute"], utc=True, errors="coerce")
    out = out.dropna(subset=["minute"])

    if timezone:
        out["minute"] = out["minute"].dt.tz_convert(timezone)

    out = out.sort_values("minute").drop_duplicates(subset=["minute"]).reset_index(drop=True)

    keep_cols = ["minute", "open", "high", "low", "close", "volume"]
    return out[keep_cols]


def filter_market_hours(
    df: pd.DataFrame,
    market_open: str = "09:30",
    market_close: str = "16:00",
) -> pd.DataFrame:
    """
    Keep only regular U.S. market hours.
    Assumes 'minute' is timezone-aware and already in local market time.
    """
    out = df.copy()
    if "minute" not in out.columns:
        raise ValueError("Expected column 'minute' in dataframe.")

    out = out.set_index("minute")
    out = out.between_time(market_open, market_close, inclusive="left")
    return out.reset_index()


def add_return_features(
    df: pd.DataFrame,
    horizon_minutes: int = 5,
    neutral_band: float = 0.0,
) -> pd.DataFrame:
    """
    Add forward returns and direction labels.

    Creates:
    - future_close
    - forward_return
    - log_forward_return
    - direction_5m (or corresponding horizon)
    - direction_3class_5m:
        1  = up
        0  = neutral
       -1  = down

    neutral_band:
        If > 0, returns inside [-neutral_band, neutral_band] are labeled neutral.
        Example: neutral_band=0.0005 means +/- 5 bps.
    """
    out = df.copy().sort_values("minute").reset_index(drop=True)

    out["future_close"] = out["close"].shift(-horizon_minutes)
    out["forward_return"] = (out["future_close"] / out["close"]) - 1.0
    out["log_forward_return"] = np.log(out["future_close"] / out["close"])

    binary_col = f"direction_{horizon_minutes}m"
    out[binary_col] = (out["forward_return"] > 0).astype("Int64")

    multiclass_col = f"direction_3class_{horizon_minutes}m"
    out[multiclass_col] = 0
    out.loc[out["forward_return"] > neutral_band, multiclass_col] = 1
    out.loc[out["forward_return"] < -neutral_band, multiclass_col] = -1
    out[multiclass_col] = out[multiclass_col].astype("Int64")

    return out


def add_market_microstructure_features(
    df: pd.DataFrame,
    trailing_vol_window: int = 30,
) -> pd.DataFrame:
    """
    Add simple trailing market features for baseline comparisons.

    Creates:
    - one_min_return
    - log_return_1m
    - trailing_volatility
    - dollar_volume
    - volume_zscore_30m
    """
    out = df.copy().sort_values("minute").reset_index(drop=True)

    out["one_min_return"] = out["close"].pct_change()
    out["log_return_1m"] = np.log(out["close"] / out["close"].shift(1))
    out["trailing_volatility"] = (
        out["log_return_1m"].rolling(window=trailing_vol_window, min_periods=5).std()
    )
    out["dollar_volume"] = out["close"] * out["volume"]

    rolling_vol_mean = out["volume"].rolling(window=trailing_vol_window, min_periods=5).mean()
    rolling_vol_std = out["volume"].rolling(window=trailing_vol_window, min_periods=5).std()

    out["volume_zscore_30m"] = (out["volume"] - rolling_vol_mean) / (rolling_vol_std + 1e-6)

    return out


def load_sentiment_features(features_path: str) -> pd.DataFrame:
    """
    Load sentiment features generated by features.py from CSV or parquet.
    """
    path = Path(features_path)
    if not path.exists():
        raise FileNotFoundError(f"Sentiment features file not found: {features_path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError("Sentiment features file must be .csv or .parquet")

    if "minute" not in df.columns:
        raise ValueError("Sentiment features file must contain a 'minute' column.")

    df["minute"] = pd.to_datetime(df["minute"], utc=False, errors="coerce")
    df = df.dropna(subset=["minute"]).sort_values("minute").reset_index(drop=True)
    return df


def merge_sentiment_and_market(
    sentiment_df: pd.DataFrame,
    market_df: pd.DataFrame,
    how: str = "inner",
) -> pd.DataFrame:
    """
    Merge sentiment features with market data on exact minute timestamp.
    """
    left = sentiment_df.copy()
    right = market_df.copy()

    left["minute"] = pd.to_datetime(left["minute"], errors="coerce")
    right["minute"] = pd.to_datetime(right["minute"], errors="coerce")

    merged = pd.merge(left, right, on="minute", how=how)
    merged = merged.sort_values("minute").reset_index(drop=True)
    return merged


def save_dataframe(df: pd.DataFrame, output_path: str) -> None:
    """
    Save dataframe to CSV or parquet based on extension.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError("Output file must end in .csv or .parquet")


def run_market_pipeline(
    ticker: str,
    output_price_path: str,
    output_merged_path: Optional[str] = None,
    sentiment_features_path: Optional[str] = None,
    interval: str = "1m",
    period: str = "30d",
    horizon_minutes: int = 5,
    market_hours_only: bool = True,
    timezone: str = "America/New_York",
    auto_adjust: bool = False,
    prepost: bool = False,
    neutral_band: float = 0.0,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    End-to-end market data pipeline:
    1. Download prices
    2. Clean prices
    3. Optionally filter market hours
    4. Add market features and labels
    5. Save price dataset
    6. Optionally merge with sentiment features and save merged dataset
    """
    raw_prices = download_price_data(
        ticker=ticker,
        interval=interval,
        period=period,
        auto_adjust=auto_adjust,
        prepost=prepost,
    )

    price_df = clean_price_data(raw_prices, timezone=timezone)

    if market_hours_only:
        price_df = filter_market_hours(price_df)

    price_df = add_market_microstructure_features(price_df)
    price_df = add_return_features(
        price_df,
        horizon_minutes=horizon_minutes,
        neutral_band=neutral_band,
    )

    save_dataframe(price_df, output_price_path)

    merged_df = None
    if sentiment_features_path is not None and output_merged_path is not None:
        sentiment_df = load_sentiment_features(sentiment_features_path)
        merged_df = merge_sentiment_and_market(sentiment_df, price_df, how="inner")
        save_dataframe(merged_df, output_merged_path)

    return price_df, merged_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download intraday market data, build labels, and optionally merge with sentiment features."
    )
    parser.add_argument("ticker", type=str, help="Ticker symbol, e.g. AAPL")
    parser.add_argument(
        "--output-price",
        type=str,
        required=True,
        help="Output file for cleaned market data (.csv or .parquet)",
    )
    parser.add_argument(
        "--sentiment-features",
        type=str,
        help="Optional path to sentiment features file (.csv or .parquet)",
    )
    parser.add_argument(
        "--output-merged",
        type=str,
        help="Optional output path for merged sentiment + market dataset",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1m",
        help="Price interval, e.g. 1m, 5m, 15m, 1d",
    )
    parser.add_argument(
        "--period",
        type=str,
        default="30d",
        help="History period, e.g. 5d, 30d, 60d, 3mo",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="Forward return horizon in minutes",
    )
    parser.add_argument(
        "--market-hours-only",
        action="store_true",
        help="Keep only regular U.S. market hours",
    )
    parser.add_argument(
        "--timezone",
        type=str,
        default="America/New_York",
        help="Timezone for timestamps",
    )
    parser.add_argument(
        "--auto-adjust",
        action="store_true",
        help="Use auto-adjusted prices from yfinance",
    )
    parser.add_argument(
        "--prepost",
        action="store_true",
        help="Include premarket and postmarket data",
    )
    parser.add_argument(
        "--neutral-band",
        type=float,
        default=0.0,
        help="Neutral return band for 3-class labeling, e.g. 0.0005 = 5 bps",
    )

    args = parser.parse_args()

    if (args.sentiment_features is None) != (args.output_merged is None):
        raise ValueError(
            "Provide both --sentiment-features and --output-merged together, or neither."
        )

    price_df, merged_df = run_market_pipeline(
        ticker=args.ticker,
        output_price_path=args.output_price,
        output_merged_path=args.output_merged,
        sentiment_features_path=args.sentiment_features,
        interval=args.interval,
        period=args.period,
        horizon_minutes=args.horizon,
        market_hours_only=args.market_hours_only,
        timezone=args.timezone,
        auto_adjust=args.auto_adjust,
        prepost=args.prepost,
        neutral_band=args.neutral_band,
    )

    print(f"Saved price dataset with {len(price_df)} rows.")
    print("Price columns:")
    print(list(price_df.columns))

    if merged_df is not None:
        print(f"Saved merged dataset with {len(merged_df)} rows.")
        print("Merged columns:")
        print(list(merged_df.columns))


if __name__ == "__main__":
    main()