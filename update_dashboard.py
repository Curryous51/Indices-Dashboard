"""
Daily updater for the TJI Indices Dashboard.

What this does, every time it runs (via GitHub Actions, once a day):
  1. Loads data/companies.json  -> the fixed list (name, sector, link). Never changes.
  2. Resolves each company to an NSE ticker (yfinance format, e.g. "TCS.NS"):
       - checks overrides.csv first (manual fixes always win)
       - checks the cache in data/ticker_cache.json (so we don't re-guess every day)
       - otherwise tries a few common symbol guesses and validates them live
       - anything that still fails gets written to needs_review.csv for you to fix by hand
  3. Downloads ~5 years of daily closing prices for every resolved ticker in one batch call.
  4. From that single price history, computes every return column the dashboard needs:
       1D, 1W, 1M, 3M, 6M, 1YR, 2YR, 3YR, 5YR, and LTP vs 52-week high.
     WEIGHT is untouched -- it comes straight from companies.json since it's static.
  5. Writes data/dashboard_data.json with the fresh numbers.
  6. Injects that JSON into index.html in place of the old data block.

Run manually the first time with:  python update_dashboard.py
"""

import json
import csv
import os
import re
import io
import difflib
import pandas as pd
import requests
import yfinance as yf

BASE = os.path.dirname(os.path.abspath(__file__))
COMPANIES_PATH = os.path.join(BASE, "data", "companies.json")
CACHE_PATH = os.path.join(BASE, "data", "ticker_cache.json")
OVERRIDES_PATH = os.path.join(BASE, "overrides.csv")
NEEDS_REVIEW_PATH = os.path.join(BASE, "needs_review.csv")
NEW_LISTINGS_PATH = os.path.join(BASE, "new_listings_detected.csv")
OUTPUT_JSON_PATH = os.path.join(BASE, "data", "dashboard_data.json")
INDEX_HTML_PATH = os.path.join(BASE, "index.html")

NSE_EQUITY_LIST_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
FUZZY_MATCH_THRESHOLD = 0.72  # below this, we don't trust the match -> needs_review

CALENDAR_OFFSETS = {
    "1D": pd.DateOffset(days=1),
    "1W": pd.DateOffset(weeks=1),
    "1M": pd.DateOffset(months=1),
    "3M": pd.DateOffset(months=3),
    "6M": pd.DateOffset(months=6),
    "1YR": pd.DateOffset(years=1),
    "2YR": pd.DateOffset(years=2),
    "3YR": pd.DateOffset(years=3),
    "5YR": pd.DateOffset(years=5),
}


def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def load_overrides():
    overrides = {}
    if os.path.exists(OVERRIDES_PATH):
        with open(OVERRIDES_PATH, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("Company") and row.get("NSE_Symbol"):
                    overrides[row["Company"].strip()] = row["NSE_Symbol"].strip()
    return overrides


def clean_name(name):
    """Strip legal-entity suffixes/punctuation so names compare cleanly."""
    n = re.sub(r"[^A-Za-z0-9& ]", " ", name)
    n = re.sub(
        r"\b(Ltd|Limited|Co|Company|India|Industries|Inds|Corp|Corporation|The|Amalgamated|Amalgamat)\b",
        "", n, flags=re.I,
    )
    return re.sub(r"\s+", " ", n).strip().upper()


def fetch_nse_master_list():
    """Download NSE's official symbol <-> company-name list once per run."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
        "Accept": "text/csv,*/*",
    }
    session = requests.Session()
    session.get("https://www.nseindia.com", headers=headers, timeout=15)  # sets cookies
    resp = session.get(NSE_EQUITY_LIST_URL, headers=headers, timeout=15)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = [c.strip() for c in df.columns]
    df["CLEAN_NAME"] = df["NAME OF COMPANY"].apply(clean_name)
    return df


def best_fuzzy_match(target_clean, nse_df):
    """Return (symbol, score) for the closest company-name match, or (None, 0)."""
    choices = nse_df["CLEAN_NAME"].tolist()
    matches = difflib.get_close_matches(target_clean, choices, n=1, cutoff=FUZZY_MATCH_THRESHOLD)
    if not matches:
        return None, 0
    row = nse_df[nse_df["CLEAN_NAME"] == matches[0]].iloc[0]
    score = difflib.SequenceMatcher(None, target_clean, matches[0]).ratio()
    return row["SYMBOL"], score


def resolve_ticker(name, cache, overrides, nse_df):
    if name in overrides:
        val = overrides[name].strip()
        # user can specify an exchange suffix explicitly (e.g. "500166.BO" for
        # a BSE-only listing); otherwise assume NSE and add ".NS" for them
        return val if "." in val else f"{val}.NS"
    if name in cache and cache[name] != "UNRESOLVED":
        return cache[name]

    symbol, score = best_fuzzy_match(clean_name(name), nse_df)
    if symbol:
        full = f"{symbol}.NS"
        cache[name] = full
        return full

    cache[name] = "UNRESOLVED"
    return None


def compute_touch_signals(closes, lows):
    """
    A different signal from a crossover: today's LOW dipped below the
    average, but the CLOSE still finished above it -- a same-day
    dip-and-recover / support test, independent of yesterday's position.
    Returns Touch50/Touch200 (EMA) and TouchSMA50/TouchSMA200 (SMA), each
    True/False.
    """
    out = {"Touch50": False, "Touch200": False, "TouchSMA50": False, "TouchSMA200": False}
    if len(closes) < 3:
        return out
    closes = closes.sort_index()
    lows = lows.sort_index()
    today_close = closes.iloc[-1]
    today_low = lows.iloc[-1]

    configs = [
        ("Touch50", 50, "ema"), ("Touch200", 200, "ema"),
        ("TouchSMA50", 50, "sma"), ("TouchSMA200", 200, "sma"),
    ]
    for label, span, kind in configs:
        if len(closes) < span * 3:
            continue
        ma = closes.ewm(span=span, adjust=False).mean() if kind == "ema" else closes.rolling(span).mean()
        today_ma = ma.iloc[-1]
        if today_low < today_ma < today_close:
            out[label] = True
    return out


def compute_ma_crossovers(closes):
    """
    Returns Crossed50/Crossed200 (EMA-based) and CrossedSMA50/CrossedSMA200
    (SMA-based), each 'up'|'down'|None for today's session.
    'up'   = price closed above the average today, having been at/below it yesterday
    'down' = price closed below the average today, having been at/above it yesterday
    None   = no crossover today (or not enough history to trust the average yet)
    """
    out = {"Crossed50": None, "Crossed200": None, "CrossedSMA50": None, "CrossedSMA200": None}
    if len(closes) < 3:
        return out
    closes = closes.sort_index()
    today_px, yday_px = closes.iloc[-1], closes.iloc[-2]

    configs = [
        ("Crossed50", 50, "ema"), ("Crossed200", 200, "ema"),
        ("CrossedSMA50", 50, "sma"), ("CrossedSMA200", 200, "sma"),
    ]
    for label, span, kind in configs:
        # A moving average needs enough history before it's trustworthy.
        # EMA in particular is still heavily anchored to the first price in
        # the window until several multiples of its span have passed. SMA
        # is exact from `span` rows onward, but we hold it to the same bar
        # for consistency and to avoid noisy day-one-eligible signals.
        if len(closes) < span * 3:
            continue
        ma = closes.ewm(span=span, adjust=False).mean() if kind == "ema" else closes.rolling(span).mean()
        today_ma, yday_ma = ma.iloc[-1], ma.iloc[-2]
        if yday_px <= yday_ma and today_px > today_ma:
            out[label] = "up"
        elif yday_px >= yday_ma and today_px < today_ma:
            out[label] = "down"
    return out


STALENESS_TOLERANCE_DAYS = {
    "1D": 4, "1W": 10, "1M": 45, "3M": 100,
    "6M": 200, "1YR": 380, "2YR": 750, "3YR": 1120, "5YR": 1850,
}


def compute_returns(closes):
    """closes: pandas Series of daily close prices, indexed by date, most recent last."""
    if closes.empty:
        return {}
    closes = closes.sort_index()
    latest_date = closes.index[-1]
    latest = closes.iloc[-1]
    out = {}
    for label, offset in CALENDAR_OFFSETS.items():
        target_date = latest_date - offset
        # only look back as far as we actually have history for
        if target_date < closes.index[0]:
            out[label] = None
            continue
        eligible = closes.index[closes.index <= target_date]
        if len(eligible) == 0:
            out[label] = None
            continue
        past_idx = eligible[-1]
        # Guard against illiquid stocks: if the nearest available price is
        # much older than the target date (stock didn't trade for a while),
        # that gap isn't really "1D" (or whatever label) anymore -- treat
        # it as unavailable rather than silently mislabeling a multi-day
        # gap as a same-day move.
        staleness = (target_date - past_idx).days
        if staleness > STALENESS_TOLERANCE_DAYS[label]:
            out[label] = None
            continue
        past = closes.loc[past_idx]
        out[label] = round((latest / past) - 1, 4) if pd.notna(past) and past else None
    one_year_ago = latest_date - pd.DateOffset(years=1)
    window = closes[closes.index >= one_year_ago]
    high_52w = window.max() if not window.empty else closes.max()
    out["LTP VS 52W HIGH"] = round((latest / high_52w) - 1, 4) if high_52w else None
    return out


def detect_new_listings(nse_df, companies, days_back=270):
    """
    Find NSE-listed companies that aren't in companies.json yet, restricted
    to ones listed recently -- this deliberately does NOT try to pull in
    every stock NSE has ever listed (that would balloon the dataset with
    names outside your curated universe). It's aimed specifically at
    catching new IPOs you haven't added yet.
    """
    if "DATE OF LISTING" not in nse_df.columns:
        print("  [new-listing check] NSE's file didn't include a 'DATE OF LISTING' "
              f"column this run (columns seen: {list(nse_df.columns)}). "
              "Can't determine what's recently listed -- skipping this check.")
        return []

    existing_clean = {clean_name(c["name"]) for c in companies}
    cutoff = pd.Timestamp.today() - pd.Timedelta(days=days_back)

    df = nse_df.copy()
    df["LISTING_DATE"] = pd.to_datetime(df["DATE OF LISTING"], errors="coerce")
    recent = df[df["LISTING_DATE"] >= cutoff]
    print(f"  [new-listing check] {len(recent)} NSE companies listed in the last "
          f"{days_back} days (cutoff: {cutoff.date()})")

    new_ones = []
    for _, row in recent.iterrows():
        if row["CLEAN_NAME"] not in existing_clean:
            new_ones.append({
                "name": str(row["NAME OF COMPANY"]).strip(),
                "sector": "New Listings - Uncategorized",
                "link": None,
                "weight": 0.0001,
            })
    print(f"  [new-listing check] {len(new_ones)} of those aren't in companies.json yet")
    return new_ones


def main():
    companies = load_json(COMPANIES_PATH, [])
    cache = load_json(CACHE_PATH, {})
    overrides = load_overrides()

    print("Downloading NSE official symbol list...")
    nse_df = fetch_nse_master_list()
    print(f"  loaded {len(nse_df)} listed companies from NSE")

    new_listings = detect_new_listings(nse_df, companies)
    if new_listings:
        print(f"  found {len(new_listings)} new/recently-listed companies not yet in companies.json")
        companies.extend(new_listings)
        with open(COMPANIES_PATH, "w", encoding="utf-8") as f:
            json.dump(companies, f, indent=1)
        with open(NEW_LISTINGS_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Company", "Added under sector"])
            for c in new_listings:
                w.writerow([c["name"], c["sector"]])
        print("  added to companies.json under 'New Listings - Uncategorized'")
        print("  see new_listings_detected.csv -- move these into a proper sector when you get a chance")

    # Step 1: resolve tickers
    resolved = {}
    unresolved = []
    for c in companies:
        symbol = resolve_ticker(c["name"], cache, overrides, nse_df)
        if symbol:
            resolved[c["name"]] = symbol
        else:
            unresolved.append(c["name"])

    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=1)

    if unresolved:
        with open(NEEDS_REVIEW_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Company", "NSE_Symbol"])
            for name in unresolved:
                w.writerow([name, ""])
        print(f"{len(unresolved)} companies need manual review -> needs_review.csv")
        print("Fill in the NSE_Symbol column there, then move those rows into overrides.csv.")

    # Step 2: batch-download price history for everything resolved.
    tickers = sorted(set(resolved.values()))
    print(f"Downloading price history for {len(tickers)} tickers...")
    raw = yf.download(tickers, period="6y", group_by="ticker", auto_adjust=True, threads=True)

    # Step 3: compute returns per company, grouped by sector so we can also
    # build a weighted sector-summary row (IsIndex: true) ahead of each group,
    # matching the schema index.html expects.
    from collections import OrderedDict
    by_sector = OrderedDict()
    for c in companies:
        by_sector.setdefault(c["sector"], []).append(c)

    output_rows = []
    for sector, members in by_sector.items():
        sector_rows = []
        for c in members:
            symbol = resolved.get(c["name"])
            row = {
                "Category": c["sector"],
                "Company": c["name"],
                "Link": c["link"],
                "IsIndex": False,
                "Weight": c.get("weight"),
            }
            returns = {}
            crossovers = {"Crossed50": None, "Crossed200": None, "CrossedSMA50": None, "CrossedSMA200": None}
            touches = {"Touch50": False, "Touch200": False, "TouchSMA50": False, "TouchSMA200": False}
            last_close = None
            if symbol:
                try:
                    sub = raw[symbol][["Close", "Low"]].dropna()
                    closes = sub["Close"]
                    returns = compute_returns(closes)
                    crossovers = compute_ma_crossovers(closes)
                    touches = compute_touch_signals(closes, sub["Low"])
                    if not closes.empty:
                        last_close = round(float(closes.iloc[-1]), 2)
                except Exception as e:
                    print(f"  skipping {c['name']} ({symbol}): {e}")
            row["Close"] = last_close
            for label in list(CALENDAR_OFFSETS.keys()) + ["LTP VS 52W HIGH"]:
                row[label] = returns.get(label)
            row["Crossed50"] = crossovers["Crossed50"]
            row["Crossed200"] = crossovers["Crossed200"]
            row["CrossedSMA50"] = crossovers["CrossedSMA50"]
            row["CrossedSMA200"] = crossovers["CrossedSMA200"]
            row["Touch50"] = touches["Touch50"]
            row["Touch200"] = touches["Touch200"]
            row["TouchSMA50"] = touches["TouchSMA50"]
            row["TouchSMA200"] = touches["TouchSMA200"]
            sector_rows.append(row)

        # weighted sector summary row (weight-normalized average of each
        # column across the companies that had a value for it)
        summary = {
            "Category": sector, "Company": sector, "Link": None,
            "IsIndex": True, "Weight": None, "Close": None,
            "Crossed50": None, "Crossed200": None,
            "CrossedSMA50": None, "CrossedSMA200": None,
            "Touch50": False, "Touch200": False,
            "TouchSMA50": False, "TouchSMA200": False,
        }
        for label in list(CALENDAR_OFFSETS.keys()) + ["LTP VS 52W HIGH"]:
            num, den = 0.0, 0.0
            for c, r in zip(members, sector_rows):
                v, w = r.get(label), c.get("weight") or 0
                if v is not None and w:
                    num += v * w
                    den += w
            summary[label] = round(num / den, 4) if den else None

        output_rows.append(summary)
        output_rows.extend(sector_rows)

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(output_rows, f)

    # Step 4: inject into index.html
    with open(INDEX_HTML_PATH, "r", encoding="utf-8") as f:
        html = f.read()

    new_block = "const rawData = " + json.dumps(output_rows) + ";"
    html = re.sub(r"const rawData\s*=\s*\[.*?\];", new_block, html, flags=re.S)

    with open(INDEX_HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    print("Done. index.html updated with today's numbers.")


if __name__ == "__main__":
    main()
