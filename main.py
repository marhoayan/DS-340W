import argparse
from pathlib import Path

from curl_scraper_2 import run_scraper
from features import run_feature_pipeline
from market_data import run_market_pipeline
from modeling import run_modeling_pipeline
from plotting import run_plotting_pipeline


def ensure_parent_dir(path_str: str) -> None:
    """
    Ensure the parent directory for an output path exists.
    """
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Main pipeline runner for StockTwits sentiment + market data project."
    )

    parser.add_argument(
        "mode",
        choices=["scrape", "features", "market", "model", "plot", "all"],
        help="Which stage of the pipeline to run",
    )

    parser.add_argument(
        "ticker",
        type=str,
        help="Ticker symbol, e.g. AAPL",
    )

    # -----------------------------
    # File paths
    # -----------------------------
    parser.add_argument(
        "--raw-output",
        type=str,
        default=None,
        help="Path to raw StockTwits JSON output",
    )

    parser.add_argument(
        "--features-output",
        type=str,
        default=None,
        help="Path to processed sentiment features file (.csv or .parquet)",
    )

    parser.add_argument(
        "--price-output",
        type=str,
        default=None,
        help="Path to cleaned market data output (.csv or .parquet)",
    )

    parser.add_argument(
        "--merged-output",
        type=str,
        default=None,
        help="Path to merged modeling dataset (.csv or .parquet)",
    )

    parser.add_argument(
        "--metrics-output",
        type=str,
        default=None,
        help="Path to JSON file for model metrics output",
    )

    parser.add_argument(
        "--predictions-output",
        type=str,
        default=None,
        help="Path to CSV/parquet file for test predictions",
    )

    parser.add_argument(
        "--logistic-coef-output",
        type=str,
        default=None,
        help="Path to CSV/parquet file for logistic coefficients",
    )

    parser.add_argument(
        "--rf-importance-output",
        type=str,
        default=None,
        help="Path to CSV/parquet file for random forest importances",
    )

    parser.add_argument(
        "--figures-output-dir",
        type=str,
        default=None,
        help="Directory where generated figures should be saved",
    )

    # -----------------------------
    # Scraper options
    # -----------------------------
    parser.add_argument(
        "--impersonate",
        type=str,
        default="chrome",
        help='Browser fingerprint target for scraper, e.g. "chrome"',
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Request timeout in seconds for scraper",
    )

    parser.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Optional number of scraper cycles. In scrape mode, default is infinite.",
    )

    # -----------------------------
    # Feature engineering options
    # -----------------------------
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Rolling sentiment window in minutes",
    )

    parser.add_argument(
        "--density-lookback",
        type=int,
        default=60,
        help="Lookback window in minutes for abnormal density",
    )

    # -----------------------------
    # Market data options
    # -----------------------------
    parser.add_argument(
        "--interval",
        type=str,
        default="1m",
        help="Market data interval, e.g. 1m, 5m, 15m",
    )

    parser.add_argument(
        "--period",
        type=str,
        default="5d",
        help="Market data download period, e.g. 5d, 8d, 30d",
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="Forward return horizon in minutes",
    )

    parser.add_argument(
        "--neutral-band",
        type=float,
        default=0.0,
        help="Neutral return band for 3-class direction label",
    )

    parser.add_argument(
        "--market-hours-only",
        action="store_true",
        help="Restrict outputs to regular U.S. market hours",
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
        help="Use auto-adjusted prices in yfinance",
    )

    parser.add_argument(
        "--prepost",
        action="store_true",
        help="Include premarket and postmarket bars",
    )

    parser.add_argument(
        "--merge-sentiment",
        action="store_true",
        help="Merge market data with sentiment features during market stage",
    )

    # -----------------------------
    # Modeling options
    # -----------------------------
    parser.add_argument(
        "--target-col",
        type=str,
        default=None,
        help="Target column for modeling. Defaults to direction_{horizon}m",
    )

    parser.add_argument(
        "--feature-cols",
        type=str,
        default=None,
        help="Optional comma-separated feature list for modeling",
    )

    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Fraction of rows used for chronological training split",
    )

    args = parser.parse_args()

    ticker = args.ticker.upper()
    args.ticker = ticker

    # -----------------------------
    # Default paths
    # -----------------------------
    if args.raw_output is None:
        args.raw_output = f"data/raw/{ticker}_tweets.json"

    if args.features_output is None:
        args.features_output = f"data/processed/{ticker}_features.csv"

    if args.price_output is None:
        args.price_output = f"data/processed/{ticker}_prices.csv"

    if args.merged_output is None:
        args.merged_output = f"data/processed/{ticker}_model_data.csv"

    if args.metrics_output is None:
        args.metrics_output = f"data/models/{ticker}_metrics.json"

    if args.predictions_output is None:
        args.predictions_output = f"data/models/{ticker}_predictions.csv"

    if args.logistic_coef_output is None:
        args.logistic_coef_output = f"data/models/{ticker}_logistic_coef.csv"

    if args.rf_importance_output is None:
        args.rf_importance_output = f"data/models/{ticker}_rf_importance.csv"

    if args.figures_output_dir is None:
        args.figures_output_dir = f"data/figures/{ticker}"

    if args.target_col is None:
        args.target_col = f"direction_{args.horizon}m"

    return args


def parse_feature_columns(raw: str | None) -> list[str] | None:
    """
    Parse optional comma-separated feature columns.
    """
    if raw is None:
        return None

    cols = [c.strip() for c in raw.split(",") if c.strip()]
    return cols if cols else None


def validate_args(args: argparse.Namespace) -> None:
    """
    Validate required files for each pipeline stage.
    """
    if args.mode == "features":
        if not Path(args.raw_output).exists():
            raise FileNotFoundError(
                f"Raw StockTwits JSON not found: {args.raw_output}\n"
                "Run scrape mode first."
            )

    if args.mode == "market" and args.merge_sentiment:
        if not Path(args.features_output).exists():
            raise FileNotFoundError(
                f"Sentiment features file not found: {args.features_output}\n"
                "Run features mode first."
            )

    if args.mode == "model":
        if not Path(args.merged_output).exists():
            raise FileNotFoundError(
                f"Merged modeling dataset not found: {args.merged_output}\n"
                "Run market mode with --merge-sentiment first."
            )

    if args.mode == "plot":
        required_files = [
            args.features_output,
            args.merged_output,
            args.predictions_output,
            args.logistic_coef_output,
            args.rf_importance_output,
        ]

        missing = [path for path in required_files if not Path(path).exists()]

        if missing:
            raise FileNotFoundError(
                "Cannot run plot mode because these files are missing:\n"
                + "\n".join(missing)
                + "\nRun all mode or run features, market, and model first."
            )


def print_pipeline_header(args: argparse.Namespace) -> None:
    print("==========================================")
    print("StockTwits Sentiment Project Pipeline")
    print("==========================================")
    print(f"Mode: {args.mode}")
    print(f"Ticker: {args.ticker}")
    print(f"Raw sentiment file: {args.raw_output}")
    print(f"Features file: {args.features_output}")
    print(f"Price file: {args.price_output}")
    print(f"Merged file: {args.merged_output}")
    print(f"Metrics file: {args.metrics_output}")
    print(f"Predictions file: {args.predictions_output}")
    print(f"Figures directory: {args.figures_output_dir}")
    print()


def run_scrape_stage(args: argparse.Namespace) -> None:
    print("SCRAPE STAGE")
    print("------------")

    run_scraper(
        symbol=args.ticker,
        output_filename=args.raw_output,
        impersonate=args.impersonate,
        timeout=args.timeout,
        max_cycles=args.max_cycles,
    )

    print()


def run_features_stage(args: argparse.Namespace) -> None:
    print("FEATURES STAGE")
    print("--------------")

    if not Path(args.raw_output).exists():
        raise FileNotFoundError(
            f"Raw StockTwits JSON not found: {args.raw_output}\n"
            "Run scrape mode first."
        )

    ensure_parent_dir(args.features_output)

    features_df = run_feature_pipeline(
        input_json=args.raw_output,
        output_path=args.features_output,
        rolling_window_minutes=args.window,
        abnormal_density_lookback_minutes=args.density_lookback,
        market_hours_only=args.market_hours_only,
        market_timezone=args.timezone,
    )

    print(f"Saved sentiment features to: {args.features_output}")
    print(f"Feature rows: {len(features_df)}")
    print()


def run_market_stage(args: argparse.Namespace) -> None:
    print("MARKET DATA STAGE")
    print("-----------------")

    ensure_parent_dir(args.price_output)

    if args.merge_sentiment:
        if not Path(args.features_output).exists():
            raise FileNotFoundError(
                f"Sentiment features file not found: {args.features_output}\n"
                "Run features mode first."
            )

        ensure_parent_dir(args.merged_output)

    price_df, merged_df = run_market_pipeline(
        ticker=args.ticker,
        output_price_path=args.price_output,
        output_merged_path=args.merged_output if args.merge_sentiment else None,
        sentiment_features_path=args.features_output if args.merge_sentiment else None,
        interval=args.interval,
        period=args.period,
        horizon_minutes=args.horizon,
        market_hours_only=args.market_hours_only,
        timezone=args.timezone,
        auto_adjust=args.auto_adjust,
        prepost=args.prepost,
        neutral_band=args.neutral_band,
    )

    print(f"Saved market data to: {args.price_output}")
    print(f"Price rows: {len(price_df)}")

    if merged_df is not None:
        print(f"Saved merged modeling dataset to: {args.merged_output}")
        print(f"Merged rows: {len(merged_df)}")

    print()


def run_model_stage(args: argparse.Namespace) -> None:
    print("MODELING STAGE")
    print("--------------")

    if not Path(args.merged_output).exists():
        raise FileNotFoundError(
            f"Merged modeling dataset not found: {args.merged_output}\n"
            "Run market mode with --merge-sentiment first."
        )

    ensure_parent_dir(args.metrics_output)
    ensure_parent_dir(args.predictions_output)
    ensure_parent_dir(args.logistic_coef_output)
    ensure_parent_dir(args.rf_importance_output)

    metrics = run_modeling_pipeline(
        input_path=args.merged_output,
        metrics_output=args.metrics_output,
        predictions_output=args.predictions_output,
        logistic_coef_output=args.logistic_coef_output,
        rf_importance_output=args.rf_importance_output,
        target_col=args.target_col,
        feature_cols=parse_feature_columns(args.feature_cols),
        train_fraction=args.train_fraction,
    )

    print(f"Saved metrics to: {args.metrics_output}")
    print(f"Saved predictions to: {args.predictions_output}")
    print(f"Saved logistic coefficients to: {args.logistic_coef_output}")
    print(f"Saved random forest importances to: {args.rf_importance_output}")
    print("Metrics summary:")
    print(metrics)
    print()


def run_plot_stage(args: argparse.Namespace) -> None:
    print("PLOTTING STAGE")
    print("--------------")

    required_files = [
        args.features_output,
        args.merged_output,
        args.predictions_output,
        args.logistic_coef_output,
        args.rf_importance_output,
    ]

    missing = [path for path in required_files if not Path(path).exists()]

    if missing:
        raise FileNotFoundError(
            "Cannot create plots because these files are missing:\n"
            + "\n".join(missing)
        )

    Path(args.figures_output_dir).mkdir(parents=True, exist_ok=True)

    run_plotting_pipeline(
        ticker=args.ticker,
        features_path=args.features_output,
        model_data_path=args.merged_output,
        predictions_path=args.predictions_output,
        logistic_coef_path=args.logistic_coef_output,
        rf_importance_path=args.rf_importance_output,
        output_dir=args.figures_output_dir,
        rolling_window_minutes=args.window,
    )

    print(f"Saved figures to: {args.figures_output_dir}")
    print()


def main() -> None:
    args = parse_args()

    # In all mode, prevent the common mistake where scraping runs forever.
    if args.mode == "all" and args.max_cycles is None:
        raise ValueError(
            "In all mode, you must provide --max-cycles so the scraper eventually stops.\n"
            "Example: python main.py all NVDA --max-cycles 10 --market-hours-only --period 5d"
        )

    validate_args(args)
    print_pipeline_header(args)

    if args.mode == "scrape":
        run_scrape_stage(args)

    elif args.mode == "features":
        run_features_stage(args)

    elif args.mode == "market":
        run_market_stage(args)

    elif args.mode == "model":
        run_model_stage(args)

    elif args.mode == "plot":
        run_plot_stage(args)

    elif args.mode == "all":
        # In all mode, force merge sentiment so model and plots can run.
        args.merge_sentiment = True

        run_scrape_stage(args)
        run_features_stage(args)
        run_market_stage(args)
        run_model_stage(args)
        run_plot_stage(args)

    print("Pipeline step(s) completed.")


if __name__ == "__main__":
    main()