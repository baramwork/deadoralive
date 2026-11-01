"""
fetch_prices.py — raw daily OHLCV (+ split dates) for the panic_dip study (SPEC.md §2).

For every distinct ticker in constituents.csv (plus SPY, the buy&hold benchmark — fetched
explicitly, never part of the universe) this GETs Yahoo's chart API and writes

    data/panic_dip/<TICKER>.csv          date,open,high,low,close,volume   (raw OHLC, NOT adjclose)
    data/panic_dip/<TICKER>.splits.json  ["YYYY-MM-DD", ...]               (only if the response
                                                                            carried an `events` key)

`<TICKER>` is the ORIGINAL constituents.csv spelling (BRK.B); Yahoo is asked for BRK-B. When the API
returns no `events` key at all, no .splits.json is written and the study falls back to SPEC §2's
volume/drop rule for that ticker. NOTE (documented, not "fixed"): Yahoo may omit `events` for a
ticker that simply has no splits in range — the fallback rule is then applied to it, which is the
conservative direction (it can only exclude more signals, never fewer).

Also writes research/panic_dip/results/data_coverage.json (requested / received / missing /
per-ticker first+last date + row count / how many tickers had split events).

The OHLC parser is fetch_data._parse_yahoo_chart_json (SPEC §2: "파서 재사용"); this file adds only
the split-event extraction that parser drops. Runs on GitHub Actions — this dev sandbox cannot
reach Yahoo, so every network-free piece (symbol mapping, split parsing, backoff) is unit-tested in
tests_panic_dip.py against fake responses.

Usage
    python research/panic_dip/fetch_prices.py                # all tickers + SPY
    python research/panic_dip/fetch_prices.py --limit 20     # smoke run: first 20 tickers + SPY
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

CONSTITUENTS_PATH = os.path.join(HERE, "constituents.csv")
DEFAULT_OUT_DIR = os.path.join(ROOT, "data", "panic_dip")
DEFAULT_RESULTS_DIR = os.path.join(HERE, "results")

PERIOD1_ISO = "2015-06-01"   # earlier than the study start (2015-11-01) so the 20-day liquidity warm-up exists
BENCHMARK = "SPY"
CHART_URL = ("https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
             "?period1={period1}&period2={period2}&interval=1d&events=splits")
HEADERS = {"User-Agent": "Mozilla/5.0 (deadoralive/0.1; +github.com/getjogak-sketch/deadoralive)"}
TIMEOUT_S = 30
BACKOFF_S = (2, 4, 8, 16)    # exponential backoff on HTTP 429 / 5xx / network errors; <= 4 retries
SLEEP_BETWEEN_TICKERS_S = 0.25


def _parse_ohlc(payload: dict) -> pd.DataFrame:
    """SPEC §2: reuse fetch_data's Yahoo chart parser (lazy import — keeps `--help` and the pure
    helpers below importable without the repo root's other modules)."""
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    import fetch_data  # noqa: E402
    return fetch_data._parse_yahoo_chart_json(payload)


def period1_epoch() -> int:
    return int(datetime.strptime(PERIOD1_ISO, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def yahoo_symbol(ticker: str) -> str:
    """BRK.B -> BRK-B, BF.B -> BF-B (SPEC §2: one `.`->`-` conversion, nothing else is tried)."""
    return ticker.replace(".", "-")


def read_universe(path: str = CONSTITUENTS_PATH) -> list:
    """Distinct tickers of constituents.csv in first-appearance order."""
    seen, out = set(), []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            t = (row.get("ticker") or "").strip()
            if t and t not in seen:
                seen.add(t)
                out.append(t)
    return out


def parse_split_dates(payload: dict):
    """Split dates (sorted ISO strings) from a chart-API payload's `events.splits`.

    Returns None when the response has no usable `events.splits` dict (=> "no split information":
    nothing is written and the study's fallback rule applies). Returns [] when the dict exists but
    is empty. Never raises."""
    try:
        result = (payload or {}).get("chart", {}).get("result")
        if not result:
            return None
        events = (result[0] or {}).get("events")
        if not isinstance(events, dict):
            return None
        splits = events.get("splits")
        if not isinstance(splits, dict):
            return None
    except AttributeError:
        return None
    dates = set()
    for key, ev in splits.items():
        ts = None
        if isinstance(ev, dict) and ev.get("date") is not None:
            ts = ev.get("date")
        else:
            ts = key
        try:
            dates.add(pd.Timestamp(int(ts), unit="s", tz="UTC").strftime("%Y-%m-%d"))
        except (TypeError, ValueError):
            continue
    return sorted(dates)


def request_with_backoff(url: str, get=requests.get, sleep=time.sleep):
    """GET with SPEC/task retry policy: HTTP 429, 5xx and network errors are retried after
    2,4,8,16 s (at most 4 retries); 404 and every other status return immediately.
    Returns (response_or_None, last_error_string_or_None)."""
    attempt = 0
    while True:
        resp, err = None, None
        try:
            resp = get(url, timeout=TIMEOUT_S, headers=HEADERS)
            status = resp.status_code
        except requests.exceptions.RequestException as e:
            status, err = None, f"{type(e).__name__}: {e}"
        retryable = status is None or status == 429 or 500 <= status < 600
        if retryable and attempt < len(BACKOFF_S):
            sleep(BACKOFF_S[attempt])
            attempt += 1
            continue
        return resp, err


def fetch_symbol(ticker: str, period2: int, get=requests.get, sleep=time.sleep) -> dict:
    """One ticker -> {"status": "ok"|"missing", "reason": str, "df": DataFrame|None,
    "splits": list|None}. Never raises on a network/HTTP/JSON problem."""
    url = CHART_URL.format(symbol=yahoo_symbol(ticker), period1=period1_epoch(), period2=period2)
    resp, err = request_with_backoff(url, get=get, sleep=sleep)
    if resp is None:
        return {"status": "missing", "reason": f"network error after retries: {err}", "df": None, "splits": None}
    if resp.status_code == 404:
        return {"status": "missing", "reason": "HTTP 404", "df": None, "splits": None}
    if resp.status_code != 200:
        return {"status": "missing", "reason": f"HTTP {resp.status_code}", "df": None, "splits": None}
    try:
        payload = resp.json()
    except ValueError as e:
        return {"status": "missing", "reason": f"bad JSON: {e}", "df": None, "splits": None}
    df = _parse_ohlc(payload)
    if df is None or df.empty:
        return {"status": "missing", "reason": "empty data", "df": None, "splits": None}
    return {"status": "ok", "reason": "", "df": df, "splits": parse_split_dates(payload)}


def write_ticker(out_dir: str, ticker: str, res: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    res["df"].to_csv(os.path.join(out_dir, f"{ticker}.csv"), index=False, date_format="%Y-%m-%d")
    spl_path = os.path.join(out_dir, f"{ticker}.splits.json")
    if res["splits"] is not None:
        with open(spl_path, "w") as f:
            json.dump(res["splits"], f)
    elif os.path.exists(spl_path):
        os.remove(spl_path)  # stale file from an earlier run: "no split info" must mean no file


def _coverage_entry(df: pd.DataFrame) -> dict:
    return {"first_date": str(pd.Timestamp(df["date"].iloc[0]).date()),
            "last_date": str(pd.Timestamp(df["date"].iloc[-1]).date()), "rows": int(len(df))}


def run(limit, out_dir: str, results_dir: str, get=requests.get, sleep=time.sleep) -> dict:
    universe = read_universe()
    if limit is not None:
        universe = universe[:limit]
    period2 = int(time.time())
    per_ticker, missing, missing_reasons, n_splits_files, n_with_split_events = {}, [], {}, 0, 0
    print(f"[fetch_prices] requesting {len(universe)} tickers + {BENCHMARK} (period1={PERIOD1_ISO}, "
          f"out={out_dir})")
    t0 = time.time()
    for n, ticker in enumerate(universe, 1):
        res = fetch_symbol(ticker, period2, get=get, sleep=sleep)
        if res["status"] == "ok":
            write_ticker(out_dir, ticker, res)
            per_ticker[ticker] = _coverage_entry(res["df"])
            if res["splits"] is not None:
                n_splits_files += 1
                if res["splits"]:
                    n_with_split_events += 1
        else:
            missing.append(ticker)
            missing_reasons[ticker] = res["reason"]
            print(f"[fetch_prices] MISSING {ticker}: {res['reason']}")
        if n % 25 == 0 or n == len(universe):
            print(f"[fetch_prices] {n}/{len(universe)} done ({len(per_ticker)} ok, {len(missing)} missing, "
                  f"{time.time() - t0:.0f}s)")
        sleep(SLEEP_BETWEEN_TICKERS_S)

    bench = {}
    res = fetch_symbol(BENCHMARK, period2, get=get, sleep=sleep)
    if res["status"] == "ok":
        write_ticker(out_dir, BENCHMARK, res)
        bench[BENCHMARK] = _coverage_entry(res["df"])
    else:
        bench[BENCHMARK] = {"missing": True, "reason": res["reason"]}
        print(f"[fetch_prices] WARNING: benchmark {BENCHMARK} missing: {res['reason']}")

    coverage = {
        "run_date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "period1": PERIOD1_ISO, "requested": len(universe), "received": len(per_ticker),
        "missing": missing, "missing_reasons": missing_reasons,
        "tickers_with_splits_file": n_splits_files,
        "tickers_with_split_events": n_with_split_events,
        "tickers_without_splits_file": len(per_ticker) - n_splits_files,
        "benchmark": bench, "per_ticker": per_ticker,
    }
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "data_coverage.json"), "w") as f:
        json.dump(coverage, f, indent=2)
    print(f"[fetch_prices] done: requested={coverage['requested']} received={coverage['received']} "
          f"missing={len(missing)} split-events={n_with_split_events} ({time.time() - t0:.0f}s)")
    return coverage


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="fetch only the first N tickers (smoke run)")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="default: data/panic_dip (gitignored)")
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR,
                    help="where data_coverage.json goes (default research/panic_dip/results)")
    args = ap.parse_args()
    run(args.limit, args.out_dir, args.results_dir)
    return 0   # missing tickers are data (reported in data_coverage.json), never a failure


if __name__ == "__main__":
    sys.exit(main())
