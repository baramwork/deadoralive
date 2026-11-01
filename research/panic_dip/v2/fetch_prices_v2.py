"""
fetch_prices_v2.py — raw daily OHLCV (+ split dates) for the panic_dip v2 study (SPEC_v2.md §2).

Differences to v1's fetch_prices.py (which is imported, never modified):
  * separate data dir  data/panic_dip_v2/<TICKER>.csv (+ .splits.json)  — v1's data/panic_dip stays as is
  * retries:  a ticker whose request ended "missing" (404 / empty / network) is retried ONCE after 20 s;
              a ticker that came back SHORT (rows < 50 while its point-in-time membership span inside the
              study window is > 60 days — v1's one-row stubs) is retried after 10 s and, if still short,
              again after 30 s.  Waits are per ROUND, not per ticker (all 404s of the first pass wait once),
              which is the same "at least N seconds after its failure" guarantee at ~1/100 of the sleep time.
  * alias recovery (aliases.csv, written before any result): a ticker that is STILL missing/short after the
              retries and has kind=rename gets its data from new_ticker (fetched too if it was not part of the
              universe).  The mapping {old: new} is written to  <out-dir>/_aliases_resolved.json  for the study.
              Direct data always wins: an old ticker that returns >= 50 rows on a (re)try is never aliased.
  * coverage written to  research/panic_dip/v2/results/data_coverage_v2.json

Runs on GitHub Actions — the dev sandbox cannot reach Yahoo, so every piece is tested with fake HTTP in
tests_panic_dip_v2.py.

Usage
    python research/panic_dip/v2/fetch_prices_v2.py                 # everything
    python research/panic_dip/v2/fetch_prices_v2.py --limit 20      # smoke run: first 20 tickers + SPY
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
V1_DIR = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(V1_DIR))
if V1_DIR not in sys.path:
    sys.path.insert(0, V1_DIR)

import fetch_prices as F1  # noqa: E402  (v1, frozen — reused for request/parse/split/write helpers)

ALIASES_PATH = os.path.join(HERE, "aliases.csv")
CONSTITUENTS_PATH = F1.CONSTITUENTS_PATH
DEFAULT_OUT_DIR = os.path.join(ROOT, "data", "panic_dip_v2")
DEFAULT_RESULTS_DIR = os.path.join(HERE, "results")
RESOLVED_FILE = "_aliases_resolved.json"
COVERAGE_FILE = "data_coverage_v2.json"

BENCHMARK = F1.BENCHMARK
KINDS = ("rename", "acquired", "unknown")

# ---- SPEC_v2 §2 constants -------------------------------------------------------------------------
MISSING_RETRY_WAIT_S = 20          # 404 / missing: wait 20 s, retry once
SHORT_RETRY_WAITS_S = (10, 30)     # short: wait 10 s retry, still short: wait 30 s retry
SHORT_ROWS = 50                    # rows < 50 ...
SHORT_MIN_SPAN_DAYS = 60           # ... while the membership span (inside the study window) > 60 days
STUDY_START = "2015-11-01"
STUDY_END = "2026-10-31"


def log(msg: str) -> None:
    print(f"[fetch_prices_v2] {msg}", flush=True)


# ---------------------------------------------------------------------------
# aliases / membership
# ---------------------------------------------------------------------------

def read_aliases(path: str = ALIASES_PATH) -> dict:
    """old_ticker -> {"new": str, "kind": str, "note": str}. Unknown kinds are rejected loudly."""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            old = (row.get("old_ticker") or "").strip()
            if not old:
                continue
            kind = (row.get("kind") or "").strip()
            if kind not in KINDS:
                raise ValueError(f"aliases.csv: {old} has invalid kind {kind!r} (allowed {KINDS})")
            new = (row.get("new_ticker") or "").strip()
            if kind == "rename" and not new:
                raise ValueError(f"aliases.csv: {old} is kind=rename but has no new_ticker")
            if old in out:
                raise ValueError(f"aliases.csv: duplicate old_ticker {old}")
            out[old] = {"new": new, "kind": kind, "note": (row.get("note") or "").strip()}
    return out


def read_membership(path: str = CONSTITUENTS_PATH) -> dict:
    """ticker -> [(start np.datetime64[D], end np.datetime64[D] | None)]  (end exclusive; blank = still member)."""
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            t, s, e = (row.get("ticker") or "").strip(), (row.get("start_date") or "").strip(), (row.get("end_date") or "").strip()
            if not t or not s:
                continue
            out.setdefault(t, []).append((np.datetime64(s, "D"), np.datetime64(e, "D") if e else None))
    return out


def span_days(intervals, lo: str = STUDY_START, hi: str = STUDY_END) -> int:
    """Calendar days covered by the UNION of [start, end) intervals, clipped to [lo, hi] (inclusive)."""
    lo64, hi64 = np.datetime64(lo, "D"), np.datetime64(hi, "D") + 1
    segs = []
    for s, e in intervals or []:
        a = max(s, lo64)
        b = hi64 if e is None else min(e, hi64)
        if b > a:
            segs.append((a, b))
    segs.sort()
    total, cur_a, cur_b = 0, None, None
    for a, b in segs:
        if cur_b is None or a > cur_b:
            if cur_b is not None:
                total += int((cur_b - cur_a).astype(int))
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    if cur_b is not None:
        total += int((cur_b - cur_a).astype(int))
    return total


# ---------------------------------------------------------------------------
# Fetcher: state per ticker + retry rounds
# ---------------------------------------------------------------------------

class Fetcher:
    """Holds one run's per-ticker state. `state[t]` = {"status": "ok"|"missing", "reason": str, "rows": int,
    "first": iso, "last": iso, "splits": list|None, "attempts": int}."""

    def __init__(self, out_dir: str, period2: int, get=requests.get, sleep=time.sleep):
        self.out_dir, self.period2, self.get, self.sleep = out_dir, period2, get, sleep
        self.state = {}
        self.span = {}              # ticker -> membership span days that make a short response matter
        self.log = {"missing_retried": [], "missing_recovered_on_retry": [], "short_retried": [],
                    "short_recovered_on_retry": []}

    # -- one request ------------------------------------------------------------------------------
    def fetch_one(self, ticker: str) -> None:
        res = F1.fetch_symbol(ticker, self.period2, get=self.get, sleep=self.sleep)
        prev = self.state.get(ticker)
        attempts = (prev["attempts"] if prev else 0) + 1
        if res["status"] == "ok":
            rows = int(len(res["df"]))
            if prev and prev["status"] == "ok" and prev["rows"] > rows:
                prev["attempts"] = attempts          # keep the better (longer) earlier response on disk
                return
            F1.write_ticker(self.out_dir, ticker, res)
            ent = F1._coverage_entry(res["df"])
            self.state[ticker] = {"status": "ok", "reason": "", "rows": ent["rows"], "first": ent["first_date"],
                                  "last": ent["last_date"], "splits": res["splits"], "attempts": attempts}
        elif prev and prev["status"] == "ok":
            prev["attempts"] = attempts              # an earlier OK response stays
        else:
            self.state[ticker] = {"status": "missing", "reason": res["reason"], "rows": 0, "first": None,
                                  "last": None, "splits": None, "attempts": attempts}

    def fetch_many(self, tickers, label: str = "") -> None:
        t0, n = time.time(), len(tickers)
        for k, t in enumerate(tickers, 1):
            self.fetch_one(t)
            if k % 25 == 0 or k == n:
                ok = sum(1 for x in tickers[:k] if self.state[x]["status"] == "ok")
                log(f"{label}{k}/{n} done ({ok} ok, {k - ok} missing, {time.time() - t0:.0f}s)")
            self.sleep(F1.SLEEP_BETWEEN_TICKERS_S)

    # -- classification -----------------------------------------------------------------------------
    def is_missing(self, t: str) -> bool:
        return self.state.get(t, {}).get("status") != "ok"

    def is_short(self, t: str) -> bool:
        s = self.state.get(t)
        return bool(s and s["status"] == "ok" and s["rows"] < SHORT_ROWS and self.span.get(t, 0) > SHORT_MIN_SPAN_DAYS)

    def usable(self, t: str) -> bool:
        s = self.state.get(t)
        return bool(s and s["status"] == "ok" and s["rows"] >= SHORT_ROWS)

    # -- SPEC_v2 §2 retry rounds --------------------------------------------------------------------
    def settle(self, tickers, label: str = "") -> None:
        """404/missing -> wait 20 s, retry once.  short -> wait 10 s retry; still short -> wait 30 s retry."""
        miss = [t for t in tickers if self.is_missing(t)]
        if miss:
            log(f"{label}{len(miss)} missing ticker(s): waiting {MISSING_RETRY_WAIT_S}s, then one retry each")
            self.log["missing_retried"] += miss
            self.sleep(MISSING_RETRY_WAIT_S)
            self.fetch_many(miss, label=f"{label}retry-missing ")
            self.log["missing_recovered_on_retry"] += [t for t in miss if not self.is_missing(t)]
        for wait in SHORT_RETRY_WAITS_S:
            short = [t for t in tickers if self.is_short(t)]
            if not short:
                break
            log(f"{label}{len(short)} short ticker(s): waiting {wait}s, then retry")
            self.log["short_retried"] += [t for t in short if t not in self.log["short_retried"]]
            self.sleep(wait)
            self.fetch_many(short, label=f"{label}retry-short({wait}s) ")
        self.log["short_recovered_on_retry"] = sorted(
            t for t in self.log["short_retried"] if self.state[t]["status"] == "ok" and self.state[t]["rows"] >= SHORT_ROWS)


def resolve_aliases(fx: Fetcher, universe: list, aliases: dict, membership: dict) -> tuple:
    """Alias step. Returns (resolved {old: new}, failed [(old, new, why)], alias_targets_fetched [new...]).
    Only kind=rename rows of tickers that are STILL unusable (missing or short) are considered."""
    candidates = [t for t in universe if (fx.is_missing(t) or fx.is_short(t))
                  and aliases.get(t, {}).get("kind") == "rename"]
    targets = []
    for old in candidates:
        new = aliases[old]["new"]
        if new not in targets:
            targets.append(new)
        # a short response of the NEW ticker matters as much as for any other: use the old ticker's span
        fx.span[new] = max(fx.span.get(new, 0), fx.span.get(old, 0))
    to_fetch = [n for n in targets if n not in fx.state]
    if to_fetch:
        log(f"alias step: {len(candidates)} rename candidate(s), fetching {len(to_fetch)} new ticker(s) not yet fetched")
        fx.fetch_many(to_fetch, label="alias-fetch ")
    fx.settle(to_fetch, label="alias ")      # universe targets were already settled in the main pass
    resolved, failed = {}, []
    for old in candidates:
        new = aliases[old]["new"]
        if fx.usable(new):
            resolved[old] = new
        else:
            st = fx.state.get(new, {})
            failed.append((old, new, "target missing: " + st.get("reason", "not fetched") if st.get("status") != "ok"
                           else f"target short: {st.get('rows')} rows"))
    return resolved, failed, to_fetch


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------

def late_start_list(fx: Fetcher, universe: list, membership: dict, slack_days: int = 60) -> list:
    """Informational: tickers whose first data date is > 60 days after the start of their first membership
    interval inside the study window (data exists but does not cover the membership — e.g. a reused ticker such as
    FB). NOT used for recovery."""
    out = []
    lo = np.datetime64(STUDY_START, "D")
    hi = np.datetime64(STUDY_END, "D") + 1
    for t in universe:
        s = fx.state.get(t)
        if not s or s["status"] != "ok":
            continue
        starts = []
        for a, b in membership.get(t, []):
            end = hi if b is None else min(b, hi)
            if end > max(a, lo):
                starts.append(max(a, lo))
        if not starts:
            continue
        first = np.datetime64(s["first"], "D")
        gap = int((first - min(starts)).astype(int))
        if gap > slack_days:
            out.append({"ticker": t, "first_data": s["first"], "membership_start_in_window": str(min(starts)),
                        "gap_days": gap})
    return out


def build_coverage(fx: Fetcher, universe: list, aliases: dict, membership: dict, resolved: dict, failed: list,
                   alias_targets: list, bench: dict) -> dict:
    final_missing, final_short = [], []
    for t in universe:
        if fx.is_missing(t):
            final_missing.append(t)
        elif fx.is_short(t):
            final_short.append(t)
    unusable = [t for t in universe if t in final_missing or t in final_short]
    not_recovered = [t for t in unusable if t not in resolved]
    failed_old = {o for o, _, _ in failed}
    by_kind = {"rename-failed": [], "acquired": [], "unknown": []}
    for t in not_recovered:
        a = aliases.get(t)
        if a is None:
            by_kind["unknown"].append(t)
        elif a["kind"] == "rename":
            by_kind["rename-failed"].append(t)
        else:
            by_kind[a["kind"]].append(t)
    received_direct = [t for t in universe if fx.state.get(t, {}).get("status") == "ok" and not fx.is_short(t)]
    per_ticker = {t: {"rows": fx.state[t]["rows"], "first_date": fx.state[t]["first"], "last_date": fx.state[t]["last"]}
                  for t in universe if fx.state.get(t, {}).get("status") == "ok"}
    alias_target_rows = {n: {"rows": fx.state[n]["rows"], "first_date": fx.state[n]["first"], "last_date": fx.state[n]["last"]}
                         for n in alias_targets if fx.state.get(n, {}).get("status") == "ok"}
    n_with_events = sum(1 for t in per_ticker if fx.state[t]["splits"])
    n_files = sum(1 for t in per_ticker if fx.state[t]["splits"] is not None)
    kinds_in_table = {k: sum(1 for t in universe if aliases.get(t, {}).get("kind") == k) for k in KINDS}
    return {
        "run_date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "period1": F1.PERIOD1_ISO,
        "requested": len(universe),
        "received_direct": len(received_direct),
        "recovered_via_alias": [{"old": o, "new": n, "new_rows": fx.state[n]["rows"],
                                 "old_was": "missing" if fx.is_missing(o) else "short"} for o, n in resolved.items()],
        "still_missing": by_kind,
        "still_missing_counts": {k: len(v) for k, v in by_kind.items()},
        "short": [{"ticker": t, "rows": fx.state[t]["rows"], "kind": aliases.get(t, {}).get("kind", "unknown")}
                  for t in final_short],
        "missing_after_retry": final_missing,
        "missing_reasons": {t: fx.state[t]["reason"] for t in final_missing},
        "rename_failed_detail": [{"old": o, "new": n, "why": why} for o, n, why in failed],
        "retries": {"missing_retried": fx.log["missing_retried"],
                    "missing_recovered_on_retry": fx.log["missing_recovered_on_retry"],
                    "short_retried": fx.log["short_retried"],
                    "short_recovered_on_retry": fx.log["short_recovered_on_retry"]},
        "aliases_table_counts_for_requested": kinds_in_table,
        "tickers_with_splits_file": n_files,
        "tickers_with_split_events": n_with_events,
        "tickers_without_splits_file": len(per_ticker) - n_files,
        "benchmark": bench,
        "per_ticker": per_ticker,
        "alias_targets_fetched": alias_target_rows,
        "late_start_informational": late_start_list(fx, universe, membership),
    }


def run(limit, out_dir: str, results_dir: str, get=requests.get, sleep=time.sleep, universe=None,
        aliases=None, membership=None) -> dict:
    t0 = time.time()
    universe = list(universe) if universe is not None else F1.read_universe()
    if limit is not None:
        universe = universe[:limit]
    aliases = aliases if aliases is not None else read_aliases()
    membership = membership if membership is not None else read_membership()
    fx = Fetcher(out_dir, int(time.time()), get=get, sleep=sleep)
    for t in universe:
        fx.span[t] = span_days(membership.get(t, []))
    log(f"requesting {len(universe)} tickers + {BENCHMARK} (period1={F1.PERIOD1_ISO}, out={out_dir})")
    fx.fetch_many(universe, label="")
    fx.span[BENCHMARK] = 10_000          # SPY must never be short
    fx.fetch_many([BENCHMARK], label="benchmark ")
    # SPEC_v2 §2 retries for the whole first pass (+ SPY)
    fx.settle(universe + [BENCHMARK], label="")
    resolved, failed, alias_targets = resolve_aliases(fx, universe, aliases, membership)
    bench = ({BENCHMARK: {"rows": fx.state[BENCHMARK]["rows"], "first_date": fx.state[BENCHMARK]["first"],
                          "last_date": fx.state[BENCHMARK]["last"]}}
             if fx.state[BENCHMARK]["status"] == "ok" else
             {BENCHMARK: {"missing": True, "reason": fx.state[BENCHMARK]["reason"]}})
    if "missing" in bench[BENCHMARK]:
        log(f"WARNING: benchmark {BENCHMARK} missing: {bench[BENCHMARK]['reason']}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, RESOLVED_FILE), "w") as f:
        json.dump(resolved, f, indent=2, sort_keys=True)
    coverage = build_coverage(fx, universe, aliases, membership, resolved, failed, alias_targets, bench)
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, COVERAGE_FILE), "w") as f:
        json.dump(coverage, f, indent=2)
    sm = coverage["still_missing_counts"]
    log(f"done: requested={coverage['requested']} received_direct={coverage['received_direct']} "
        f"recovered_via_alias={len(resolved)} still_missing(rename-failed/acquired/unknown)="
        f"{sm['rename-failed']}/{sm['acquired']}/{sm['unknown']} short={len(coverage['short'])} "
        f"recovered_on_retry={len(fx.log['missing_recovered_on_retry'])} ({time.time() - t0:.0f}s)")
    return coverage


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="fetch only the first N tickers (smoke run)")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="default: data/panic_dip_v2 (gitignored)")
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR,
                    help="where data_coverage_v2.json goes (default research/panic_dip/v2/results)")
    args = ap.parse_args()
    run(args.limit, args.out_dir, args.results_dir)
    return 0   # missing tickers are data (reported in data_coverage_v2.json), never a failure


if __name__ == "__main__":
    sys.exit(main())
