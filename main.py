import argparse
from pathlib import Path

from curl_scraper_2 import run_scraper
from features import run_feature_pipeline
from market_data import run_market_pipeline
from modeling import run_modeling_pipeline


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
        choices=["scrape", "features", "market", "model", "all"],
        help="Which stage of the pipeline to run",
    )
    parser.add_argument(
        "ticker",
        type=str,
        help="Ticker symbol, e.g. AAPL",
    )

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

    args = parser.parse_args()

    if args.raw_output is None:
        args.raw_output = f"data/raw/{args.ticker}_tweets.json"

    if args.features_output is None:
        args.features_output = f"data/processed/{args.ticker}_features.csv"

    if args.price_output is None:
        args.price_output = f"data/processed/{args.ticker}_prices.csv"

    if args.merged_output is None:
        args.merged_output = f"data/processed/{args.ticker}_model_data.csv"

    if args.metrics_output is None:
        args.metrics_output = f"data/models/{args.ticker}_metrics.json"

    if args.predictions_output is None:
        args.predictions_output = f"data/models/{args.ticker}_predictions.csv"

    if args.logistic_coef_output is None:
        args.logistic_coef_output = f"data/models/{args.ticker}_logistic_coef.csv"

    if args.rf_importance_output is None:
        args.rf_importance_output = f"data/models/{args.ticker}_rf_importance.csv"

    if args.target_col is None:
        args.target_col = f"direction_{args.horizon}m"

    return args


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
        feature_cols=None if args.feature_cols is None else [c.strip() for c in args.feature_cols.split(",") if c.strip()],
        train_fraction=args.train_fraction,
    )

    print(f"Saved metrics to: {args.metrics_output}")
    print(f"Saved predictions to: {args.predictions_output}")
    print(f"Saved logistic coefficients to: {args.logistic_coef_output}")
    print(f"Saved random forest importances to: {args.rf_importance_output}")
    print("Metrics summary:")
    print(metrics)
    print()


def main() -> None:
    args = parse_args()
    validate_args(args)

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
    print()

    if args.mode == "scrape":
        run_scrape_stage(args)

    elif args.mode == "features":
        run_features_stage(args)

    elif args.mode == "market":
        run_market_stage(args)

    elif args.mode == "model":
        run_model_stage(args)

    elif args.mode == "all":
        run_scrape_stage(args)
        run_features_stage(args)

        # In all mode, automatically merge sentiment and market data
        if not args.merge_sentiment:
            args.merge_sentiment = True

        run_market_stage(args)
        run_model_stage(args)

    print("Pipeline step(s) completed.")


if __name__ == "__main__":
    main()