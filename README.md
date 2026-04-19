# DS-340W

# StockTwits Sentiment and Short-Term Stock Price Prediction

This project investigates whether unstructured financial sentiment from StockTwits can help predict short-term stock price direction. The core idea is to scrape StockTwits posts for a given ticker, transform the posts into rolling sentiment and message-density features, merge those features with intraday market data, and train machine learning models to predict whether the stock price will move upward over the next five minutes.

This project was built for DS 340W at Penn State.

---

## Project Overview

The project tests the following research question:

> Can rolling StockTwits sentiment and message-density features improve short-horizon prediction of stock price direction?

The pipeline follows a full data science workflow:

1. Scrape StockTwits messages for a ticker.
2. Store the raw messages as JSON.
3. Convert raw messages into rolling five-minute sentiment features.
4. Download intraday stock price data.
5. Create future return labels.
6. Merge sentiment features with market data.
7. Train predictive models.
8. Evaluate performance.
9. Generate paper-ready visualizations.

---

## Project Structure

```text
.
├── main.py
├── curl_scraper_2.py
├── features.py
├── market_data.py
├── modeling.py
├── plotting.py
├── data/
│   ├── raw/
│   ├── processed/
│   ├── models/
│   └── figures/
└── README.md
````

---

## File Descriptions

### `main.py`

Main controller for the full pipeline.

It supports the following modes:

```text
scrape
features
market
model
plot
all
```

The `all` mode runs the full project from scraping through plotting.

---

### `curl_scraper_2.py`

Scrapes StockTwits messages using `curl_cffi` with browser TLS impersonation.

The scraper collects:

* message ID
* author
* timestamp
* post text
* bullish/bearish sentiment label, if available

The scraper performs both:

* live updates using `since_id`
* historical backfill using `max_id`

Raw output is saved as:

```text
data/raw/{TICKER}_tweets.json
```

Timestamps are saved in `America/New_York` time.

---

### `features.py`

Transforms raw StockTwits JSON into minute-level rolling sentiment features.

Key generated features include:

```text
bullish_count_5m
bearish_count_5m
unlabeled_count_5m
message_count_5m
net_sentiment
bullish_share
sentiment_change
abnormal_density
labeled_message_count
labeled_fraction
```

The main sentiment formula is:

```text
net_sentiment = (bullish_count_5m - bearish_count_5m)
                / (bullish_count_5m + bearish_count_5m + 1)
```

Processed output is saved as:

```text
data/processed/{TICKER}_features.csv
```

---

### `market_data.py`

Downloads intraday stock price data using `yfinance`.

It creates market features such as:

```text
one_min_return
log_return_1m
trailing_volatility
dollar_volume
volume_zscore_30m
```

It also creates the prediction target:

```text
direction_5m
```

where:

```text
direction_5m = 1 if the stock price increases over the next 5 minutes
direction_5m = 0 otherwise
```

Outputs:

```text
data/processed/{TICKER}_prices.csv
data/processed/{TICKER}_model_data.csv
```

---

### `modeling.py`

Trains and evaluates machine learning models.

Current models:

* logistic regression
* random forest

Default features:

```text
net_sentiment
bullish_share
sentiment_change
message_count_5m
abnormal_density
one_min_return
trailing_volatility
volume_zscore_30m
```

Evaluation metrics:

```text
accuracy
precision
recall
f1
auc
```

Outputs:

```text
data/models/{TICKER}_metrics.json
data/models/{TICKER}_predictions.csv
data/models/{TICKER}_logistic_coef.csv
data/models/{TICKER}_rf_importance.csv
```

---

### `plotting.py`

Creates paper-ready figures from the model outputs.

Generated plots include:

* rolling window message density
* net sentiment over time
* message density and abnormal density
* rolling sentiment vs stock price
* actual direction vs predicted probability
* stock price vs predicted probability
* logistic regression coefficients
* random forest feature importances

Figures are saved as PNG and PDF in:

```text
data/figures/{TICKER}/
```

---

## Installation

Install the required Python packages:

```bash
pip install pandas numpy matplotlib scikit-learn yfinance curl_cffi
```

This project uses Python 3.10 or later.

---

## Running the Full Pipeline

To run the entire pipeline for one ticker:

```bash
python main.py all NVDA --max-cycles 10 --market-hours-only --period 5d
```

This will run:

1. scraping
2. feature engineering
3. market-data processing
4. modeling
5. plotting

The `--max-cycles` argument is required in `all` mode so the scraper eventually stops.

---

## Running Each Stage Separately

### 1. Scrape StockTwits Data

```bash
python main.py scrape NVDA --max-cycles 10
```

This creates:

```text
data/raw/NVDA_tweets.json
```

Without `--max-cycles`, the scraper will run indefinitely until stopped manually.

---

### 2. Build Sentiment Features

```bash
python main.py features NVDA --market-hours-only
```

This creates:

```text
data/processed/NVDA_features.csv
```

---

### 3. Download Market Data and Merge

```bash
python main.py market NVDA --merge-sentiment --market-hours-only --period 5d
```

This creates:

```text
data/processed/NVDA_prices.csv
data/processed/NVDA_model_data.csv
```

---

### 4. Train Models

```bash
python main.py model NVDA
```

This creates:

```text
data/models/NVDA_metrics.json
data/models/NVDA_predictions.csv
data/models/NVDA_logistic_coef.csv
data/models/NVDA_rf_importance.csv
```

---

### 5. Generate Figures

```bash
python main.py plot NVDA
```

This creates figures in:

```text
data/figures/NVDA/
```

---

## Important Command-Line Arguments

### `--max-cycles`

Controls how many scraper loops run.

Example:

```bash
--max-cycles 10
```

A larger number collects more StockTwits messages.

---

### `--period`

Controls how much historical stock price data is downloaded from Yahoo Finance.

Example:

```bash
--period 5d
```

This means download five days of price data.

For one-minute price data, use shorter periods such as:

```text
1d
5d
8d
```

Yahoo Finance usually does not allow long lookback periods for one-minute data.

---

### `--market-hours-only`

Restricts features and market data to regular U.S. market hours.

Regular market hours are:

```text
9:30 AM to 4:00 PM Eastern Time
```

---

### `--window`

Controls the rolling sentiment window size in minutes.

Default:

```bash
--window 5
```

---

### `--horizon`

Controls the prediction horizon in minutes.

Default:

```bash
--horizon 5
```

This creates the target:

```text
direction_5m
```

---

## Example Workflow

A typical workflow for a ticker is:

```bash
python main.py all TSLA --max-cycles 10 --market-hours-only --period 5d
```

Then inspect:

```text
data/models/TSLA_metrics.json
data/figures/TSLA/
```

---

## Output Interpretation

The most important file is:

```text
data/models/{TICKER}_metrics.json
```

Example metrics:

```json
{
    "logistic_regression": {
        "accuracy": 0.51,
        "precision": 0.52,
        "recall": 0.50,
        "f1": 0.51,
        "auc": 0.54
    },
    "random_forest": {
        "accuracy": 0.50,
        "precision": 0.51,
        "recall": 0.48,
        "f1": 0.49,
        "auc": 0.52
    }
}
```

AUC interpretation:

```text
AUC = 0.50     random guessing
AUC > 0.50     some positive ranking ability
AUC < 0.50     worse than random ranking
```

In this project, many tickers produced results close to random chance, suggesting that StockTwits sentiment is not consistently predictive across all stocks.

---

## Current Findings

Preliminary results suggest:

* StockTwits sentiment is not uniformly predictive across tickers.
* Many stocks have sparse or intermittent StockTwits activity.
* Message density is often as important as sentiment direction.
* Market-based variables such as one-minute return, volatility, and volume z-score often dominate model importance.
* Some tickers show modest predictive structure, but the signal is weak.
* The usefulness of sentiment depends heavily on ticker-level message volume and attention.

---

## Known Limitations

### Sparse Sentiment Data

Many tickers have few StockTwits messages during market hours. This can cause sentiment features to be flat or noisy.

---

### Abnormal Density Spikes

The abnormal-density feature can become extremely large when prior message volume is close to zero.

Current formula:

```text
abnormal_density = message_count_5m / prior_average_message_count
```

Future versions should consider:

```text
log_abnormal_density = log(1 + abnormal_density)
```

or capping extreme values.

---

### Short Prediction Horizon

Predicting five-minute stock direction is very noisy. Even if sentiment contains useful information, the signal may be small relative to market noise.

---

### Exact Timestamp Merge

Sentiment and market data are currently merged on exact minute timestamps. This can reduce the number of usable observations if timestamps do not align well.

---

### No Market-Only Baseline Yet

The current models use both sentiment and market features. A future improvement should compare against a market-only model to measure whether sentiment adds incremental predictive value.

---

## Future Work

Possible extensions include:

1. Run the pipeline across more high-attention tickers.
2. Add a batch runner for multiple tickers.
3. Create a cross-stock results summary table.
4. Add market-only baseline models.
5. Tune probability thresholds instead of always using 0.50.
6. Improve abnormal-density construction.
7. Add event-window analysis around earnings announcements.
8. Use more advanced NLP methods for sentiment classification.
9. Add calibration plots and ROC curves.
10. Compare high-attention stocks against low-attention stocks.

---

## Research Summary

This project demonstrates a complete end-to-end data science pipeline for transforming unstructured financial text into predictive features. The results suggest that StockTwits sentiment may contain weak short-horizon information for some stocks, but the relationship is inconsistent and highly dependent on message volume, ticker attention, and feature construction.

The strongest conclusion is that social-media sentiment should be treated as a supplementary signal rather than a standalone predictor of short-term stock price direction.

```
```

