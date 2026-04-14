import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load_dataframe(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    elif p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")

    return df


def ensure_dir(path: str) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_figure(fig: plt.Figure, output_dir: Path, filename_stem: str) -> None:
    png_path = output_dir / f"{filename_stem}.png"
    pdf_path = output_dir / f"{filename_stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def standardize_minute_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "minute" in out.columns:
        out["minute"] = pd.to_datetime(out["minute"], errors="coerce")
        out = out.dropna(subset=["minute"]).sort_values("minute").reset_index(drop=True)
    return out


def filter_market_hours_for_plot(
    df: pd.DataFrame,
    time_col: str = "minute",
    market_open: str = "09:30",
    market_close: str = "16:00",
) -> pd.DataFrame:
    out = standardize_minute_column(df)

    if time_col not in out.columns:
        raise ValueError(f"Expected '{time_col}' column in dataframe.")

    out = out.set_index(time_col)
    out = out.between_time(market_open, market_close, inclusive="left")
    return out.reset_index()


def plot_rolling_window_demo(
    features_df: pd.DataFrame,
    output_dir: Path,
    ticker: str,
    rolling_window_minutes: int = 5,
    max_points: int = 300,
) -> None:
    df = filter_market_hours_for_plot(features_df)

    raw_col = "total_count"
    rolling_col = f"message_count_{rolling_window_minutes}m"

    if raw_col not in df.columns or rolling_col not in df.columns:
        raise ValueError(
            f"Expected columns '{raw_col}' and '{rolling_col}' in features file."
        )

    if len(df) > max_points:
        df = df.tail(max_points).copy()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        df["minute"],
        df[raw_col],
        label="Raw messages per minute",
        color="tab:blue",
        linestyle="-",
        linewidth=1.8,
    )
    ax.plot(
        df["minute"],
        df[rolling_col],
        label=f"Rolling {rolling_window_minutes}-minute message count",
        color="tab:orange",
        linestyle="--",
        linewidth=2.2,
    )
    ax.set_title(f"{ticker}: Rolling Window Message Density")
    ax.set_xlabel("Time")
    ax.set_ylabel("Message Count")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_figure(fig, output_dir, f"{ticker}_rolling_window_demo")


def plot_net_sentiment(
    features_df: pd.DataFrame,
    output_dir: Path,
    ticker: str,
    max_points: int = 500,
) -> None:
    df = filter_market_hours_for_plot(features_df)

    if "net_sentiment" not in df.columns:
        raise ValueError("Expected 'net_sentiment' column in features file.")

    if len(df) > max_points:
        df = df.tail(max_points).copy()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        df["minute"],
        df["net_sentiment"],
        color="tab:green",
        linestyle="-",
        linewidth=2.0,
    )
    ax.set_title(f"{ticker}: Net Sentiment Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Net Sentiment")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_figure(fig, output_dir, f"{ticker}_net_sentiment")


def plot_message_density(
    features_df: pd.DataFrame,
    output_dir: Path,
    ticker: str,
    rolling_window_minutes: int = 5,
    max_points: int = 500,
) -> None:
    df = filter_market_hours_for_plot(features_df)

    density_col = f"message_count_{rolling_window_minutes}m"
    abnormal_col = "abnormal_density"

    if density_col not in df.columns or abnormal_col not in df.columns:
        raise ValueError(
            f"Expected columns '{density_col}' and '{abnormal_col}' in features file."
        )

    if len(df) > max_points:
        df = df.tail(max_points).copy()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        df["minute"],
        df[density_col],
        label=density_col,
        color="tab:purple",
        linestyle="-",
        linewidth=2.0,
    )
    ax.plot(
        df["minute"],
        df[abnormal_col],
        label="abnormal_density",
        color="tab:red",
        linestyle="--",
        linewidth=2.0,
    )
    ax.set_title(f"{ticker}: Message Density Features")
    ax.set_xlabel("Time")
    ax.set_ylabel("Feature Value")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_figure(fig, output_dir, f"{ticker}_message_density")


def plot_rolling_sentiment_vs_price(
    features_df: pd.DataFrame,
    model_df: pd.DataFrame,
    output_dir: Path,
    ticker: str,
    sentiment_col: str = "net_sentiment",
    price_col: str = "close",
    max_points: int = 300,
) -> None:
    fdf = filter_market_hours_for_plot(features_df)
    mdf = filter_market_hours_for_plot(model_df)

    if "minute" not in fdf.columns or sentiment_col not in fdf.columns:
        raise ValueError(
            f"Features file must contain 'minute' and '{sentiment_col}'."
        )

    if "minute" not in mdf.columns or price_col not in mdf.columns:
        raise ValueError(
            f"Model data file must contain 'minute' and '{price_col}'."
        )

    merged = pd.merge(
        fdf[["minute", sentiment_col]],
        mdf[["minute", price_col]],
        on="minute",
        how="inner",
    ).sort_values("minute")

    if merged.empty:
        raise ValueError("No overlapping timestamps found between sentiment and price data.")

    if len(merged) > max_points:
        merged = merged.tail(max_points).copy()

    fig, ax1 = plt.subplots(figsize=(10, 5))
    line1 = ax1.plot(
        merged["minute"],
        merged[price_col],
        label="Stock Price",
        color="tab:blue",
        linestyle="-",
        linewidth=2.2,
    )
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Stock Price", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    line2 = ax2.plot(
        merged["minute"],
        merged[sentiment_col],
        label="Rolling Sentiment",
        color="tab:orange",
        linestyle="--",
        linewidth=2.2,
    )
    ax2.set_ylabel("Rolling Sentiment", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", frameon=True)

    ax1.set_title(f"{ticker}: Rolling Sentiment vs Stock Price")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()

    save_figure(fig, output_dir, f"{ticker}_rolling_sentiment_vs_price")


def plot_price_with_signal(
    model_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    output_dir: Path,
    ticker: str,
    model_prob_col: str = "logistic_prob_up",
    max_points: int = 300,
) -> None:
    mdf = filter_market_hours_for_plot(model_df)
    pdf = filter_market_hours_for_plot(predictions_df)

    if "minute" not in mdf.columns or "close" not in mdf.columns:
        raise ValueError("Model data must contain 'minute' and 'close' columns.")

    if "minute" not in pdf.columns or model_prob_col not in pdf.columns:
        raise ValueError(f"Predictions file must contain 'minute' and '{model_prob_col}'.")

    merged = pd.merge(
        mdf[["minute", "close"]],
        pdf[["minute", model_prob_col]],
        on="minute",
        how="inner",
    ).sort_values("minute")

    if len(merged) > max_points:
        merged = merged.tail(max_points).copy()

    fig, ax1 = plt.subplots(figsize=(10, 5))
    line1 = ax1.plot(
        merged["minute"],
        merged["close"],
        label="Close Price",
        color="tab:blue",
        linestyle="-",
        linewidth=2.2,
    )
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Close Price", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    line2 = ax2.plot(
        merged["minute"],
        merged[model_prob_col],
        label="Predicted Probability Up",
        color="tab:red",
        linestyle="--",
        linewidth=2.2,
    )
    ax2.set_ylabel("Predicted Probability Up", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", frameon=True)

    ax1.set_title(f"{ticker}: Price and Predicted Up Probability")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()

    save_figure(fig, output_dir, f"{ticker}_price_with_signal")


def plot_actual_vs_predicted(
    predictions_df: pd.DataFrame,
    output_dir: Path,
    ticker: str,
    model_prob_col: str = "logistic_prob_up",
    max_points: int = 300,
) -> None:
    df = filter_market_hours_for_plot(predictions_df)

    required = {"minute", "actual", model_prob_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Predictions file missing columns: {missing}")

    if len(df) > max_points:
        df = df.tail(max_points).copy()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        df["minute"],
        df[model_prob_col],
        label="Predicted probability up",
        color="tab:green",
        linestyle="-",
        linewidth=2.0,
    )
    ax.plot(
        df["minute"],
        df["actual"],
        label="Actual direction",
        color="tab:red",
        linestyle="--",
        linewidth=2.0,
    )
    ax.set_title(f"{ticker}: Actual Direction vs Predicted Probability")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_figure(fig, output_dir, f"{ticker}_actual_vs_predicted")


def plot_logistic_coefficients(
    coef_df: pd.DataFrame,
    output_dir: Path,
    ticker: str,
    top_n: int = 15,
) -> None:
    if "feature" not in coef_df.columns or "coefficient" not in coef_df.columns:
        raise ValueError("Logistic coefficient file must contain 'feature' and 'coefficient'.")

    sort_col = "abs_coefficient" if "abs_coefficient" in coef_df.columns else "coefficient"
    df = coef_df.copy().sort_values(sort_col, ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df["feature"], df["coefficient"])
    ax.set_title(f"{ticker}: Logistic Regression Coefficients")
    ax.set_xlabel("Coefficient")
    ax.set_ylabel("Feature")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_figure(fig, output_dir, f"{ticker}_logistic_coefficients")


def plot_rf_importance(
    rf_df: pd.DataFrame,
    output_dir: Path,
    ticker: str,
    top_n: int = 15,
) -> None:
    if "feature" not in rf_df.columns or "importance" not in rf_df.columns:
        raise ValueError("Random forest importance file must contain 'feature' and 'importance'.")

    df = rf_df.copy().sort_values("importance", ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df["feature"], df["importance"])
    ax.set_title(f"{ticker}: Random Forest Feature Importances")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_figure(fig, output_dir, f"{ticker}_rf_importance")


def run_plotting_pipeline(
    ticker: str,
    features_path: str,
    model_data_path: str,
    predictions_path: str,
    logistic_coef_path: str,
    rf_importance_path: str,
    output_dir: str,
    rolling_window_minutes: int = 5,
) -> None:
    outdir = ensure_dir(output_dir)

    features_df = load_dataframe(features_path)
    model_df = load_dataframe(model_data_path)
    predictions_df = load_dataframe(predictions_path)
    logistic_coef_df = load_dataframe(logistic_coef_path)
    rf_importance_df = load_dataframe(rf_importance_path)

    plot_rolling_window_demo(
        features_df=features_df,
        output_dir=outdir,
        ticker=ticker,
        rolling_window_minutes=rolling_window_minutes,
    )

    plot_net_sentiment(
        features_df=features_df,
        output_dir=outdir,
        ticker=ticker,
    )

    plot_message_density(
        features_df=features_df,
        output_dir=outdir,
        ticker=ticker,
        rolling_window_minutes=rolling_window_minutes,
    )

    plot_rolling_sentiment_vs_price(
        features_df=features_df,
        model_df=model_df,
        output_dir=outdir,
        ticker=ticker,
    )

    plot_actual_vs_predicted(
        predictions_df=predictions_df,
        output_dir=outdir,
        ticker=ticker,
    )

    plot_price_with_signal(
        model_df=model_df,
        predictions_df=predictions_df,
        output_dir=outdir,
        ticker=ticker,
    )

    plot_logistic_coefficients(
        coef_df=logistic_coef_df,
        output_dir=outdir,
        ticker=ticker,
    )

    plot_rf_importance(
        rf_df=rf_importance_df,
        output_dir=outdir,
        ticker=ticker,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create paper-ready figures for StockTwits sentiment project."
    )
    parser.add_argument("ticker", type=str, help="Ticker symbol, e.g. AAPL")
    parser.add_argument("--features-path", type=str, required=True, help="Path to features file")
    parser.add_argument("--model-data-path", type=str, required=True, help="Path to merged model data file")
    parser.add_argument("--predictions-path", type=str, required=True, help="Path to predictions file")
    parser.add_argument("--logistic-coef-path", type=str, required=True, help="Path to logistic coefficients file")
    parser.add_argument("--rf-importance-path", type=str, required=True, help="Path to random forest importance file")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save figures")
    parser.add_argument("--window", type=int, default=5, help="Rolling window size in minutes")

    args = parser.parse_args()

    run_plotting_pipeline(
        ticker=args.ticker,
        features_path=args.features_path,
        model_data_path=args.model_data_path,
        predictions_path=args.predictions_path,
        logistic_coef_path=args.logistic_coef_path,
        rf_importance_path=args.rf_importance_path,
        output_dir=args.output_dir,
        rolling_window_minutes=args.window,
    )

    print(f"Saved figures to: {args.output_dir}")


if __name__ == "__main__":
    main()