import json
import argparse
import os
import time
import random
from typing import Optional, Any

from curl_cffi import requests


def get_symbol_stream(
    symbol: str,
    max_id: Optional[int] = None,
    since_id: Optional[int] = None,
    impersonate: str = "chrome",
    timeout: int = 30,
) -> Optional[dict[str, Any]]:
    """
    Fetch messages for a symbol using curl-impersonate via curl_cffi.

    max_id: fetch messages older than this ID (backfill)
    since_id: fetch messages newer than this ID (live update)
    impersonate: browser TLS fingerprint profile, e.g. "chrome", "safari"
    """
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
    params = {"limit": 50}

    if max_id is not None:
        params["max"] = max_id
    if since_id is not None:
        params["since"] = since_id

    print(f"Fetching {symbol} (max={max_id}, since={since_id})...")

    try:
        response = requests.get(
            url,
            params=params,
            impersonate=impersonate,
            timeout=timeout,
            headers={
                "Accept": "application/json, text/plain, */*",
                "Referer": f"https://stocktwits.com/symbol/{symbol}",
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/136.0.0.0 Safari/537.36"
                ),
            },
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
    Convert Stocktwits API message into your saved JSON structure.
    """
    user = msg.get("user", {}).get("username", "Unknown")
    body = msg.get("body", "")
    created_at = msg.get("created_at", "")
    sentiment = msg.get("entities", {}).get("sentiment", {})
    sentiment_val = sentiment.get("basic", "null") if sentiment else "null"

    return {
        "id": msg["id"],
        "author": user,
        "time": created_at,
        "post": body,
        "sentiment": sentiment_val,
    }


def load_existing_data(output_filename: str) -> tuple[list[dict[str, Any]], set[int], Optional[int], Optional[int]]:
    """
    Load existing saved messages and compute ID bounds.
    Returns: (existing_data, existing_ids, min_id, max_id)
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


def save_data(output_filename: str, existing_data: list[dict[str, Any]], min_id: Optional[int], max_id: Optional[int]) -> None:
    """
    Save messages sorted newest-first by ID.
    """
    existing_data.sort(key=lambda x: x["id"], reverse=True)

    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=4, ensure_ascii=False)
        print(f"Saved total {len(existing_data)} messages. (Newest: {max_id}, Oldest: {min_id})")
    except Exception as e:
        print(f"Save error: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape Stocktwits history + live updates using curl-impersonate TLS fingerprinting."
    )
    parser.add_argument("symbol", type=str, help="The stock symbol (e.g., AAPL)")
    parser.add_argument("--output", type=str, help="Output JSON filename (optional)")
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

    args = parser.parse_args()
    output_filename = args.output if args.output else f"{args.symbol}_tweets.json"

    existing_data, existing_ids, min_id, max_id = load_existing_data(output_filename)

    print(f"Starting Hybrid Scraper for {args.symbol} (Live + Backfill).")
    print(f"Using impersonation profile: {args.impersonate}")

    backfill_active = True

    try:
        while True:
            changed = False
            hist_msgs_count = 0
            new_msgs_count = 0

            # Phase 1: Live update
            data_live = get_symbol_stream(
                args.symbol,
                since_id=max_id,
                impersonate=args.impersonate,
                timeout=args.timeout,
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

            time.sleep(random.uniform(1.0, 2.0))

            # Phase 2: Backfill
            if backfill_active:
                if min_id is None and existing_ids:
                    min_id = min(existing_ids)

                if min_id is not None:
                    data_hist = get_symbol_stream(
                        args.symbol,
                        max_id=min_id,
                        impersonate=args.impersonate,
                        timeout=args.timeout,
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

            if changed:
                save_data(output_filename, existing_data, min_id, max_id)

            wait = random.uniform(2.0, 4.0)
            print(f"Sleeping {wait:.2f}s...")
            time.sleep(wait)

    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        print(f"Critical error: {e}")


if __name__ == "__main__":
    main()