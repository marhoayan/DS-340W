import json
import argparse
import os
import time
import random
from pathlib import Path
from typing import Optional, Any
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from curl_cffi import requests


DEFAULT_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/136.0.0.0 Safari/537.36"
    ),
}


EASTERN_TZ = ZoneInfo("America/New_York")


def convert_to_eastern_time(timestamp: str) -> str:
    """
    Convert a StockTwits timestamp to America/New_York time.

    StockTwits usually returns timestamps in UTC.
    This function stores them in Eastern Time with timezone offset.

    Example:
        Input:  2026-04-19T14:35:00Z
        Output: 2026-04-19T10:35:00-04:00
    """
    if not timestamp:
        return ""

    try:
        cleaned = timestamp.strip()

        # Handle common UTC format ending in Z
        if cleaned.endswith("Z"):
            cleaned = cleaned.replace("Z", "+00:00")

        dt = datetime.fromisoformat(cleaned)

        # If timestamp has no timezone, assume UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        eastern_dt = dt.astimezone(EASTERN_TZ)
        return eastern_dt.isoformat()

    except Exception:
        # If conversion fails, return the original timestamp
        return timestamp


def get_symbol_stream(
    symbol: str,
    max_id: Optional[int] = None,
    since_id: Optional[int] = None,
    impersonate: str = "chrome",
    timeout: int = 30,
) -> Optional[dict[str, Any]]:
    """
    Fetch messages for a StockTwits symbol stream.

    max_id: fetch messages older than this ID (backfill)
    since_id: fetch messages newer than this ID (live update)
    """
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
    params = {"limit": 50}

    if max_id is not None:
        params["max"] = max_id
    if since_id is not None:
        params["since"] = since_id

    headers = DEFAULT_HEADERS.copy()
    headers["Referer"] = f"https://stocktwits.com/symbol/{symbol}"

    print(f"Fetching {symbol} (max={max_id}, since={since_id})...")

    try:
        response = requests.get(
            url,
            params=params,
            impersonate=impersonate,
            timeout=timeout,
            headers=headers,
        )

        if response.status_code == 200:
            return response.json()

        if response.status_code == 429:
            print("Rate limited. Waiting 60s...")
            time.sleep(60)
            return None

        print(f"Error {response.status_code}: {response.text[:300]}")
        return None

    except requests.RequestsError as e:
        print(f"Request error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


def normalize_message(msg: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a raw StockTwits API message into the stored schema.

    The timestamp is converted to America/New_York time before saving.
    """
    user = msg.get("user", {}).get("username", "Unknown")
    body = msg.get("body", "")
    created_at = msg.get("created_at", "")
    eastern_time = convert_to_eastern_time(created_at)

    sentiment = msg.get("entities", {}).get("sentiment", {})
    sentiment_val = sentiment.get("basic", "null") if sentiment else "null"

    return {
        "id": msg["id"],
        "author": user,
        "time": eastern_time,
        "post": body,
        "sentiment": sentiment_val,
    }


def load_existing_data(
    output_filename: str,
) -> tuple[list[dict[str, Any]], set[int], Optional[int], Optional[int]]:
    """
    Load existing saved messages and compute ID bounds.

    Returns:
        existing_data, existing_ids, min_id, max_id
    """
    existing_data: list[dict[str, Any]] = []
    existing_ids: set[int] = set()
    min_id: Optional[int] = None
    max_id: Optional[int] = None

    if not os.path.exists(output_filename):
        return existing_data, existing_ids, min_id, max_id

    try:
        with open(output_filename, "r", encoding="utf-8") as f:
            existing_data = json.load(f)

        ids = [item.get("id") for item in existing_data if item.get("id") is not None]
        existing_ids = set(ids)

        if ids:
            min_id = min(ids)
            max_id = max(ids)

        print(f"Loaded {len(existing_data)} msgs. Range: {min_id} (old) <-> {max_id} (new)")

    except Exception as e:
        print(f"Error loading file: {e}")

    return existing_data, existing_ids, min_id, max_id


def save_data(
    output_filename: str,
    existing_data: list[dict[str, Any]],
    min_id: Optional[int],
    max_id: Optional[int],
) -> None:
    """
    Save messages sorted newest-first by ID.
    """
    Path(output_filename).parent.mkdir(parents=True, exist_ok=True)
    existing_data.sort(key=lambda x: x["id"], reverse=True)

    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=4, ensure_ascii=False)

        print(f"Saved total {len(existing_data)} messages. (Newest: {max_id}, Oldest: {min_id})")

    except Exception as e:
        print(f"Save error: {e}")


def run_scraper(
    symbol: str,
    output_filename: Optional[str] = None,
    impersonate: str = "chrome",
    timeout: int = 30,
    max_cycles: Optional[int] = None,
) -> list[dict[str, Any]]:
    """
    Run the StockTwits scraper.

    Sleep behavior intentionally matches the original scraper:
    - 1.0 to 2.0 seconds between live and backfill
    - 2.0 to 4.0 seconds at the end of each loop
    - 60 seconds on HTTP 429
    """
    if output_filename is None:
        output_filename = f"data/raw/{symbol}_tweets.json"

    existing_data, existing_ids, min_id, max_id = load_existing_data(output_filename)

    print(f"Starting Hybrid Scraper for {symbol} (Live + Backfill).")
    print(f"Using impersonation profile: {impersonate}")
    print(f"Output file: {output_filename}")
    print("Saving message timestamps in America/New_York time.")

    backfill_active = True
    cycle = 0

    try:
        while True:
            cycle += 1
            print(f"\n--- Cycle {cycle} ---")

            new_msgs_count = 0
            hist_msgs_count = 0
            changed = False

            # --- Phase 1: Live Update (Newest) ---
            data_live = get_symbol_stream(
                symbol=symbol,
                since_id=max_id,
                impersonate=impersonate,
                timeout=timeout,
            )

            if data_live:
                msgs = data_live.get("messages", [])

                for msg in msgs:
                    msg_id = msg.get("id")

                    if msg_id is None or msg_id in existing_ids:
                        continue

                    existing_data.append(normalize_message(msg))
                    existing_ids.add(msg_id)
                    new_msgs_count += 1
                    changed = True

                if existing_ids:
                    max_id = max(existing_ids)

            if new_msgs_count > 0:
                print(f"Live: Added {new_msgs_count} new messages.")

            # Small sleep between phases
            time.sleep(random.uniform(1.0, 2.0))

            # --- Phase 2: Backfill (History) ---
            if backfill_active:
                if min_id is None and existing_ids:
                    min_id = min(existing_ids)

                if min_id is not None:
                    data_hist = get_symbol_stream(
                        symbol=symbol,
                        max_id=min_id,
                        impersonate=impersonate,
                        timeout=timeout,
                    )

                    if data_hist:
                        msgs = data_hist.get("messages", [])

                        if not msgs:
                            print("Backfill: End of history reached.")
                            backfill_active = False
                        else:
                            for msg in msgs:
                                msg_id = msg.get("id")

                                if msg_id is None or msg_id in existing_ids:
                                    continue

                                existing_data.append(normalize_message(msg))
                                existing_ids.add(msg_id)
                                hist_msgs_count += 1
                                changed = True

                            if existing_ids:
                                min_id = min(existing_ids)

                            if hist_msgs_count > 0:
                                print(f"Backfill: Added {hist_msgs_count} older messages.")

            # --- Save & Sort ---
            if changed:
                save_data(output_filename, existing_data, min_id, max_id)

            if max_cycles is not None and cycle >= max_cycles:
                print(f"Reached max_cycles={max_cycles}. Stopping scraper.")
                break

            # Global Sleep
            wait = random.uniform(2.0, 4.0)
            print(f"Sleeping {wait:.2f}s...")
            time.sleep(wait)

    except KeyboardInterrupt:
        print("\nStopped by user.")

    except Exception as e:
        print(f"Critical error: {e}")

    return existing_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape StockTwits history + live updates using curl_cffi impersonation."
    )

    parser.add_argument(
        "symbol",
        type=str,
        help="Ticker symbol, e.g. AAPL",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON filename",
    )

    parser.add_argument(
        "--impersonate",
        type=str,
        default="chrome",
        help='Browser fingerprint target, e.g. "chrome", "safari", "edge"',
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Request timeout in seconds",
    )

    parser.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Optional number of loop cycles before exiting",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_scraper(
        symbol=args.symbol,
        output_filename=args.output,
        impersonate=args.impersonate,
        timeout=args.timeout,
        max_cycles=args.max_cycles,
    )


if __name__ == "__main__":
    main()