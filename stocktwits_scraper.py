import cloudscraper
import json
import argparse
import sys
import os
import time
import random

def get_symbol_stream(symbol, max_id=None, since_id=None):
    """
    Fetches messages for a symbol.
    max_id: fetches messages older than this ID (backfill).
    since_id: fetches messages newer than this ID (live update).
    """
    scraper = cloudscraper.create_scraper()
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
    params = {'limit': 50}
    if max_id:
        params['max'] = max_id
    if since_id:
        params['since'] = since_id
    
    print(f"Fetching {symbol} (max={max_id}, since={since_id})...")
    
    try:
        response = scraper.get(url, params=params)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
             print("Rate limited. Waiting 60s...")
             time.sleep(60)
             return None
        else:
            print(f"Error {response.status_code}")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Scrape Stocktwits history + live updates.")
    parser.add_argument("symbol", type=str, help="The stock symbol (e.g., AAPL)")
    parser.add_argument("--output", type=str, help="Output JSON filename (optional)")
    
    args = parser.parse_args()
    output_filename = args.output if args.output else f"{args.symbol}_tweets.json"
    
    existing_ids = set()
    existing_data = []
    
    # Cursors
    min_id = None # For backfill (older)
    max_id = None # For live (newer)
    
    # Load state
    if os.path.exists(output_filename):
        try:
            with open(output_filename, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                if existing_data:
                    ids = [item.get('id') for item in existing_data if item.get('id')]
                    existing_ids = set(ids)
                    if ids:
                        min_id = min(ids)
                        max_id = max(ids)
                    print(f"Loaded {len(existing_data)} msgs. Range: {min_id} (old) <-> {max_id} (new)")
        except Exception as e:
            print(f"Error loading file: {e}")

    print(f"Starting Hybrid Scraper for {args.symbol} (Live + Backfill).")
    
    backfill_active = True
    
    try:
        while True:
            # --- Phase 1: Live Update (Newest) ---
            # If we have no max_id, this just fetches the latest 30 default
            data_live = get_symbol_stream(args.symbol, since_id=max_id)
            new_msgs_count = 0
            
            if data_live:
                msgs = data_live.get('messages', [])
                for msg in msgs:
                    if msg['id'] not in existing_ids:
                        # Extract and Add
                        user = msg.get('user', {}).get('username', 'Unknown')
                        body = msg.get('body', '')
                        created_at = msg.get('created_at', '')
                        sentiment = msg.get('entities', {}).get('sentiment', {})
                        sentiment_val = sentiment.get('basic', 'null') if sentiment else 'null'
                        
                        existing_data.append({
                            "id": msg['id'],
                            "author": user,
                            "time": created_at,
                            "post": body,
                            "sentiment": sentiment_val
                        })
                        existing_ids.add(msg['id'])
                        new_msgs_count += 1
                
                # Update max_id if we have new stuff
                if existing_ids:
                    max_id = max(existing_ids)
            
            if new_msgs_count > 0:
                print(f"Live: Added {new_msgs_count} new messages.")

            # Small sleep between phases
            time.sleep(random.uniform(1.0, 2.0))
            
            # --- Phase 2: Backfill (History) ---
            if backfill_active:
                # If min_id is None, it means we have NO data yet, so the Live phase 
                # just fetched the latest. We set min_id from that batch.
                if min_id is None and existing_ids:
                    min_id = min(existing_ids)
                
                if min_id:
                    data_hist = get_symbol_stream(args.symbol, max_id=min_id)
                    hist_msgs_count = 0
                    if data_hist:
                        msgs = data_hist.get('messages', [])
                        if not msgs:
                            print("Backfill: End of history reached.")
                            backfill_active = False # Stop trying to backfill
                        else:
                            for msg in msgs:
                                if msg['id'] not in existing_ids:
                                    user = msg.get('user', {}).get('username', 'Unknown')
                                    body = msg.get('body', '')
                                    created_at = msg.get('created_at', '')
                                    sentiment = msg.get('entities', {}).get('sentiment', {})
                                    sentiment_val = sentiment.get('basic', 'null') if sentiment else 'null'
                                    
                                    existing_data.append({
                                        "id": msg['id'],
                                        "author": user,
                                        "time": created_at,
                                        "post": body,
                                        "sentiment": sentiment_val
                                    })
                                    existing_ids.add(msg['id'])
                                    hist_msgs_count += 1
                            
                            # Update min_id
                            if existing_ids:
                                min_id = min(existing_ids)
                            
                            if hist_msgs_count > 0:
                                print(f"Backfill: Added {hist_msgs_count} older messages.")
            
            # --- Save & Sort ---
            if new_msgs_count > 0 or (backfill_active and 'hist_msgs_count' in locals() and hist_msgs_count > 0):
                # Sort: Newest at top (Reverse ID sort is a good proxy for time and faster)
                existing_data.sort(key=lambda x: x['id'], reverse=True)
                
                try:
                    with open(output_filename, 'w', encoding='utf-8') as f:
                        json.dump(existing_data, f, indent=4)
                    print(f"Saved total {len(existing_data)} messages. (Newest: {max_id}, Oldest: {min_id})")
                except Exception as e:
                    print(f"Save error: {e}")
            
            # Global Sleep
            wait = random.uniform(2.0, 4.0)
            print(f"Sleeping {wait:.2f}s...")
            time.sleep(wait)

    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        print(f"Critical error: {e}")

if __name__ == "__main__":
    main()
