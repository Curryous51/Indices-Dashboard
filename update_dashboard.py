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
import time
import pandas as pd
import yfinance as yf

BASE = os.path.dirname(os.path.abspath(__file__))
COMPANIES_PATH = os.path.join(BASE, "data", "companies.json")
CACHE_PATH = os.path.join(BASE, "data", "ticker_cache.json")
OVERRIDES_PATH = os.path.join(BASE, "overrides.csv")
NEEDS_REVIEW_PATH = os.path.join(BASE, "needs_review.csv")
OUTPUT_JSON_PATH = os.path.join(BASE, "data", "dashboard_data.json")
INDEX_HTML_PATH = os.path.join(BASE, "index.html")

TRADING_DAYS = {
    "1D": 1, "1W": 5, "1M": 21, "3M": 63,
    "6M": 126, "1YR": 252, "2YR": 504, "3YR": 756, "5YR": 1260,
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


def guess_candidates(name):
    """Generate a few plausible NSE ticker guesses for a company name."""
    clean = re.sub(r"[^A-Za-z0-9& ]", "", name)
    clean = re.sub(
        r"\b(Ltd|Limited|Co|Company|India|Industries|Inds|Corp|Corporation|The)\b",
        "", clean, flags=re.I,
    ).strip()
    words = clean.split()
    candidates = []
    if words:
        candidates.append("".join(words).upper())          # e.g. HCLTECHNOLOGIES
        candidates.append("".join(w[0] for w in words).upper())  # initials, e.g. HCL
        if len(words) > 1:
            candidates.append(words[0].upper())             # first word only
    # de-dupe, keep order
    seen = set()
    out = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def resolve_ticker(name, cache, overrides):
    if name in overrides:
        return overrides[name]
    if name in cache and cache[name] != "UNRESOLVED":
        return cache[name]

    for guess in guess_candidates(name):
        symbol = f"{guess}.NS"
        try:
            hist = yf.Ticker(symbol).history(period="5d")
            if not hist.empty:
                cache[name] = symbol
                return symbol
        except Exception:
            pass
        time.sleep(0.1)  # be polite to Yahoo's endpoint

    cache[name] = "UNRESOLVED"
    return None


def compute_returns(closes):
    """closes: pandas Series of daily close prices, most recent last."""
    if closes.empty:
        return {}
    latest = closes.iloc[-1]
    out = {}
    for label, days in TRADING_DAYS.items():
        if len(closes) > days:
            past = closes.iloc[-1 - days]
            out[label] = round((latest / past) - 1, 4) if past else None
        else:
            out[label] = None
    window = closes.iloc[-252:] if len(closes) >= 252 else closes
    high_52w = window.max()
    out["LTP VS 52W HIGH"] = round((latest / high_52w) - 1, 4) if high_52w else None
    return out


def main():
    companies = load_json(COMPANIES_PATH, [])
    cache = load_json(CACHE_PATH, {})
    overrides = load_overrides()

    # Step 1: resolve tickers
    resolved = {}
    unresolved = []
    for c in companies:
        symbol = resolve_ticker(c["name"], cache, overrides)
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

    # Step 2: batch-download 5 years of price history for everything resolved
    tickers = sorted(set(resolved.values()))
    print(f"Downloading price history for {len(tickers)} tickers...")
    raw = yf.download(tickers, period="5y", group_by="ticker", auto_adjust=True, threads=True)

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
            if symbol:
                try:
                    closes = raw[symbol]["Close"].dropna()
                    returns = compute_returns(closes)
                except Exception as e:
                    print(f"  skipping {c['name']} ({symbol}): {e}")
            for label in list(TRADING_DAYS.keys()) + ["LTP VS 52W HIGH"]:
                row[label] = returns.get(label)
            sector_rows.append(row)

        # weighted sector summary row (weight-normalized average of each
        # column across the companies that had a value for it)
        summary = {
            "Category": sector, "Company": sector, "Link": None,
            "IsIndex": True, "Weight": None,
        }
        for label in list(TRADING_DAYS.keys()) + ["LTP VS 52W HIGH"]:
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
