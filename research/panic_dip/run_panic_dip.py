"""
run_panic_dip.py — panic_dip study (SPEC.md, pre-registered 2026-11-01). Implements the spec
mechanically; no threshold in this file is a free parameter (every one is a SPEC constant below).

Question: after an S&P 500 constituent closes down >= DROP on the day, does buying that close and
exiting by a fixed STOP / TARGET / HOLD rule beat both the 9월 pass criteria and a placebo that
enters random (day, ticker) cells with the SAME exit rule and the SAME number of trades?

This is a stand-alone event-driven backtest: it does not touch registry.py / engine.py / the
walk-forward machinery of research/edge. Everything is vectorised over one flat (ticker-major) panel:

  1. For each of the 24 (STOP, TARGET, HOLD) exit rules, precompute for EVERY cell (ticker, day) the
     exit return as if one entered at that day's close, plus an exit-reason code and the holding
     days (entry-independent; uses days t+1..t+HOLD only, HOLD <= 5, so at most 5 shifted views).
  2. Real signals and placebo samples are then plain lookups into those arrays.
  3. A day-by-day portfolio simulation (SPEC §4) turns the real trades into an equity curve for
     MDD / Sharpe / return.

Runs in GitHub Actions (`.github/workflows/research-panic-dip.yml`) on data written by
fetch_prices.py. `--synthetic` is the self-test and needs no files at all.

Usage
    python research/panic_dip/run_panic_dip.py                    # full run (needs data/panic_dip/*.csv)
    python research/panic_dip/run_panic_dip.py --synthetic        # fast end-to-end self-test
    python research/panic_dip/run_panic_dip.py --placebo-n 20     # fewer placebo runs (smoke)
"""
from __future__ import annotations
import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import time
from datetime import date

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CONSTITUENTS_PATH = os.path.join(HERE, "constituents.csv")
DEFAULT_DATA_DIR = os.path.join(ROOT, "data", "panic_dip")
DEFAULT_OUT_DIR = os.path.join(HERE, "results")

# ---------------------------------------------------------------------------
# Fixed parameters — SPEC.md. Nowhere else in this program may redefine one of these.
# ---------------------------------------------------------------------------
DATA_START = "2015-06-01"          # SPEC §2 + fetch_prices (liquidity warm-up before the study start)
DATA_END = "2026-10-31"
PERIODS = {                        # SPEC §5
    "IS": ("2015-11-01", "2021-10-31"),
    "OOS": ("2021-11-01", "2024-10-31"),
    "HOLDOUT": ("2024-11-01", "2026-10-31"),
}
DROPS = (0.08, 0.10, 0.15)         # SPEC §3
STOPS = (-0.01, -0.03, -0.05, None)
TARGETS = (0.03, 0.05)
HOLDS = (1, 3, 5)
PROPOSAL = (0.10, -0.01, 0.03, 1)  # the original (Gemini) proposal cell

LIQ_WINDOW = 20                    # SPEC §3: trailing 20 bars strictly BEFORE t
LIQ_MIN_DOLLAR_VOL = 50e6
SPLIT_FALLBACK_DROP = 0.40         # SPEC §2 fallback rule (no splits file for the ticker)
SPLIT_FALLBACK_VOL_RATIO = 0.5

COST_PRIMARY = 0.0010              # one-way, SPEC §4 (the only cost the verdict uses)
COST_SENS = (0.0005, 0.0020)
POS_FRACTION = 0.10                # SPEC §4: each position = 10% of current equity
MAX_POSITIONS = 10

MIN_TRADES_IS = 100                # SPEC §6 (i)-(vi)
MIN_PF_IS = 1.3
MAX_MDD_IS = 0.30
MIN_SHARPE_IS = 0.8
MIN_PF_OOS = 1.1
PLACEBO_PERCENTILE = 95            # SPEC §6 (vii)/(viii)
PLACEBO_N = 100
SEED = 20261101

# ---- implementation constants (not SPEC thresholds; see "Implementation notes" in the report) ----
DROP_EPS = 1e-9                    # float tolerance so a true -10.000% close counts as a 10% drop
MAX_CAL_GAP_DAYS = 5               # "전일 종가" guard: previous bar must be <= 5 calendar days earlier
CAL_MIN_FRACTION = 0.2             # a calendar day needs bars from >= 20% of the busiest day's tickers
MIN_BARS = LIQ_WINDOW + 2
TRADING_DAYS = 252
REASONS = {0: "target", 1: "stop", 2: "gap", 3: "time"}
PF_CAP = 1e9                       # only for percentile arithmetic on a loss-free (inf) placebo PF
PROPOSAL_SLUG = "drop10_stop1_target3_hold1"


def log(msg: str) -> None:
    print(f"[panic_dip] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Variant grid
# ---------------------------------------------------------------------------

def build_variants() -> list:
    """72 variants in the order drop, stop, target, hold (stop None last within a drop)."""
    return [(d, s, t, h) for d in DROPS for s in STOPS for t in TARGETS for h in HOLDS]


def combos() -> list:
    return [(s, t, h) for s in STOPS for t in TARGETS for h in HOLDS]


def stop_label(stop) -> str:
    return "none" if stop is None else f"{stop:.2f}"


def variant_slug(drop, stop, target, hold) -> str:
    st = "none" if stop is None else f"{abs(stop) * 100:g}"
    return f"drop{drop * 100:g}_stop{st}_target{target * 100:g}_hold{hold}"


# ---------------------------------------------------------------------------
# Loading: constituents (point-in-time), per-ticker CSVs, split files, SPY
# ---------------------------------------------------------------------------

def load_constituents(path: str = CONSTITUENTS_PATH) -> dict:
    """ticker -> [(start np.datetime64[D], end np.datetime64[D] | None), ...]. A ticker may have
    several intervals; a blank end_date means "still a member"."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    out = {}
    for t, s, e in zip(df["ticker"], df["start_date"], df["end_date"]):
        t, s, e = t.strip(), s.strip(), e.strip()
        if not t or not s:
            continue
        out.setdefault(t, []).append((np.datetime64(s, "D"), np.datetime64(e, "D") if e else None))
    return out


def _read_ohlcv_csv(path: str):
    df = pd.read_csv(path, parse_dates=["date"])
    need = ["date", "open", "high", "low", "close", "volume"]
    if any(c not in df.columns for c in need):
        return None
    return df[need]


def load_series(data_dir: str, tickers: list):
    """-> (series {ticker: DataFrame}, splits {ticker: [iso dates] | None}, missing [tickers])."""
    series, splits, missing = {}, {}, []
    for t in tickers:
        p = os.path.join(data_dir, f"{t}.csv")
        if not os.path.exists(p):
            missing.append(t)
            continue
        try:
            df = _read_ohlcv_csv(p)
        except Exception as e:  # noqa: BLE001 — one bad file must not kill the run
            log(f"WARNING: failed to read {p}: {e}")
            df = None
        if df is None or df.empty:
            missing.append(t)
            continue
        series[t] = df
        sp = os.path.join(data_dir, f"{t}.splits.json")
        splits[t] = None
        if os.path.exists(sp):
            try:
                with open(sp) as f:
                    splits[t] = [str(x) for x in json.load(f)]
            except Exception as e:  # noqa: BLE001
                log(f"WARNING: failed to read {sp}: {e} (treated as 'no splits file')")
    return series, splits, missing


def load_spy(data_dir: str):
    p = os.path.join(data_dir, "SPY.csv")
    if not os.path.exists(p):
        return None
    try:
        df = _read_ohlcv_csv(p)
    except Exception:  # noqa: BLE001
        return None
    return None if df is None or df.empty else df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Panel: one flat ticker-major array set + calendar
# ---------------------------------------------------------------------------

class Panel:
    """Flat panel. Cells of one ticker are contiguous and date-sorted; `off[j]:off[j+1]` is ticker j.
    gidx = calendar index of the cell's date; fut_ok[k][i] = bars i+1..i+k exist in the same ticker
    (each at most MAX_CAL_GAP_DAYS after its predecessor)."""

    def cell(self, ticker: str, iso_date: str) -> int:
        j = self.tickers.index(ticker)
        lo, hi = int(self.off[j]), int(self.off[j + 1])
        d = int(np.datetime64(iso_date, "D").astype("int64"))
        k = int(np.searchsorted(self.dnum[lo:hi], d))
        if k >= hi - lo or self.dnum[lo + k] != d:
            raise KeyError(f"{ticker} has no bar on {iso_date}")
        return lo + k

    def iso(self, gidx: int) -> str:
        return str(self.cal[int(gidx)])


def build_panel(series: dict, membership: dict, splits: dict, lo: str = DATA_START,
                hi: str = DATA_END) -> Panel:
    lo64, hi64 = np.datetime64(lo, "D"), np.datetime64(hi, "D")
    stage = {}
    for t in sorted(series):
        df = series[t]
        d = df["date"].to_numpy().astype("datetime64[D]")
        o, h, l, c = (df[x].to_numpy(dtype=np.float64) for x in ("open", "high", "low", "close"))
        v = np.nan_to_num(df["volume"].to_numpy(dtype=np.float64), nan=0.0)
        ok = (np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c)
              & (o > 0) & (h > 0) & (l > 0) & (c > 0) & (d >= lo64) & (d <= hi64))
        d, o, h, l, c, v = d[ok], o[ok], h[ok], l[ok], c[ok], v[ok]
        d_u, first = np.unique(d, return_index=True)        # sorted, first occurrence of duplicates
        stage[t] = (d_u, o[first], h[first], l[first], c[first], v[first])

    # Calendar: days on which a meaningful share of the busiest day's tickers have a bar. Drops a
    # stray bar some vendor emits on a non-trading day for one ticker.
    all_d = np.concatenate([s[0] for s in stage.values()]) if stage else np.array([], "datetime64[D]")
    cal_days, counts = np.unique(all_d, return_counts=True)
    if len(cal_days) == 0:
        raise ValueError("no usable bars in any series")
    cal = cal_days[counts >= max(1, math.ceil(CAL_MIN_FRACTION * counts.max()))]

    P = Panel()
    P.cal = cal
    tickers, parts, skipped = [], [], []
    for t, (d, o, h, l, c, v) in stage.items():
        keep = np.isin(d, cal)
        d, o, h, l, c, v = d[keep], o[keep], h[keep], l[keep], c[keep], v[keep]
        if len(d) < MIN_BARS:
            skipped.append(t)
            continue
        n = len(d)
        gap_ok = np.zeros(n, bool)
        gap_ok[1:] = (d[1:] - d[:-1]).astype("int64") <= MAX_CAL_GAP_DAYS
        ret1 = np.full(n, np.nan)
        ret1[1:] = c[1:] / c[:-1] - 1.0
        # liquidity guard: mean(close*volume) over bars t-20..t-1 (strictly before t) >= $50M
        liq_ok = np.zeros(n, bool)
        dv = c * v
        if n > LIQ_WINDOW:
            wm = sliding_window_view(dv, LIQ_WINDOW).mean(axis=1)      # wm[i] = mean(dv[i:i+20])
            liq_ok[LIQ_WINDOW:] = wm[: n - LIQ_WINDOW] >= LIQ_MIN_DOLLAR_VOL
        member = np.zeros(n, bool)
        for s, e in membership.get(t, []):
            member |= (d >= s) if e is None else ((d >= s) & (d < e))   # start inclusive, end exclusive
        # split-day exclusion (SPEC §2)
        excl_a = np.zeros(n, bool)
        excl_b = np.zeros(n, bool)
        sp = splits.get(t)
        if sp is not None:
            excl_a = np.isin(d, np.array(sp, dtype="datetime64[D]")) if len(sp) else excl_a
        else:
            for i in np.flatnonzero(ret1 <= -SPLIT_FALLBACK_DROP):
                if i >= LIQ_WINDOW and v[i] < SPLIT_FALLBACK_VOL_RATIO * np.median(v[i - LIQ_WINDOW:i]):
                    excl_b[i] = True
        tickers.append(t)
        parts.append(dict(d=d, o=o, h=h, l=l, c=c, v=v, gap_ok=gap_ok, ret1=ret1, liq_ok=liq_ok,
                          member=member, excl_a=excl_a, excl_b=excl_b,
                          has_splits=sp is not None))
    P.tickers = tickers
    P.skipped = skipped
    lens = np.array([len(p["d"]) for p in parts], dtype=np.int64)
    P.off = np.concatenate([[0], np.cumsum(lens)])
    P.N = int(P.off[-1])
    cat = lambda key, dt=None: (np.concatenate([p[key] for p in parts]) if parts else np.array([], dt))  # noqa: E731
    P.open, P.high, P.low, P.close, P.volume = (cat(k, np.float64) for k in ("o", "h", "l", "c", "v"))
    P.ret1 = cat("ret1", np.float64)
    P.gap_ok, P.liq_ok, P.member = (cat(k, bool) for k in ("gap_ok", "liq_ok", "member"))
    P.excl_a, P.excl_b = cat("excl_a", bool), cat("excl_b", bool)
    P.split_excl = P.excl_a | P.excl_b
    P.has_splits = {t: p["has_splits"] for t, p in zip(tickers, parts)}
    dd = cat("d", "datetime64[D]")
    P.dnum = dd.astype("int64").astype(np.int32)
    P.gidx = np.searchsorted(cal, dd).astype(np.int32)
    P.tick = np.repeat(np.arange(len(parts), dtype=np.int32), lens)
    # future-bar availability, progressive in k (all of t+1..t+k must exist, same ticker, no gap hole)
    P.fut_ok = {}
    prev = np.ones(P.N, bool)
    for k in range(1, max(HOLDS) + 1):
        ok = np.zeros(P.N, bool)
        m = P.N - k
        if m > 0:
            ok[:m] = prev[:m] & (P.tick[k:] == P.tick[:m]) & P.gap_ok[k:]
        P.fut_ok[k] = ok
        prev = ok
    return P


# ---------------------------------------------------------------------------
# Signals and exits
# ---------------------------------------------------------------------------

def candidate_cells(P: Panel, drop: float) -> np.ndarray:
    """SPEC §3 entry: close[t]/close[t-1]-1 <= -DROP, liquidity guard, point-in-time membership,
    split-day exclusion. Flat cell indices (ascending)."""
    m = P.gap_ok & (P.ret1 <= -drop + DROP_EPS) & P.member & P.liq_ok & ~P.split_excl
    return np.flatnonzero(m)


class Exits:
    """ret: float32 gross return of entering at close[t] under one (STOP, TARGET, HOLD) rule;
    code: int8 (0 target, 1 stop, 2 gap, 3 time, -1 insufficient/invalid future bars);
    k: int8 holding days used (exit bar = cell + k)."""
    __slots__ = ("ret", "code", "k")

    def __init__(self, ret, code, k):
        self.ret, self.code, self.k = ret, code, k


def compute_exit_arrays(P: Panel, stop, target: float, hold: int) -> Exits:
    """SPEC §3 exits for every cell at once. Per holding day k=1..HOLD, in this order:
    gap (open <= stop -> exit at the OPEN), stop (low <= stop -> exit at the stop price),
    target (high >= target -> exit at the target price; stop-first on a same-day tie), and at
    k == HOLD the close. Day t itself is never tested."""
    N = P.N
    ret = np.full(N, np.nan)
    code = np.full(N, -1, np.int8)
    kk = np.zeros(N, np.int8)
    undone = P.fut_ok[hold].copy()
    c = P.close
    tgt_px = c * (1.0 + target)
    stop_px = None if stop is None else c * (1.0 + stop)
    for k in range(1, hold + 1):
        m = N - k
        if m <= 0:
            break
        o_k, h_k, l_k, c_k = P.open[k:], P.high[k:], P.low[k:], P.close[k:]
        act = undone[:m]
        if stop_px is not None:
            sp = stop_px[:m]
            gap = act & (o_k <= sp)
            stp = act & ~gap & (l_k <= sp)
            tg = act & ~gap & ~stp & (h_k >= tgt_px[:m])
        else:
            gap = stp = np.zeros(m, bool)
            tg = act & (h_k >= tgt_px[:m])
        c0 = c[:m]
        for mask, cd in ((gap, 2), (stp, 1), (tg, 0)):
            if not mask.any():
                continue
            if cd == 2:
                ret[:m][mask] = o_k[mask] / c0[mask] - 1.0
            elif cd == 1:
                ret[:m][mask] = stop
            else:
                ret[:m][mask] = target
            code[:m][mask] = cd
            kk[:m][mask] = k
        undone[:m] = act & ~(gap | stp | tg)
        if k == hold:
            tm = undone[:m]
            if tm.any():
                ret[:m][tm] = c_k[tm] / c0[tm] - 1.0
                code[:m][tm] = 3
                kk[:m][tm] = hold
    return Exits(ret.astype(np.float32), code, kk)


def compute_all_exits(P: Panel) -> dict:
    return {(s, t, h): compute_exit_arrays(P, s, t, h) for (s, t, h) in combos()}


def exit_price(P: Panel, i: int, code: int, k: int, stop, target: float) -> float:
    """Float64 exit price of one trade (for the trades file; mirrors compute_exit_arrays)."""
    c0 = float(P.close[i])
    if code == 2:
        return float(P.open[i + k])
    if code == 1:
        return c0 * (1.0 + stop)
    if code == 0:
        return c0 * (1.0 + target)
    return float(P.close[i + k])


# ---------------------------------------------------------------------------
# Periods
# ---------------------------------------------------------------------------

def period_bounds(P: Panel, periods: dict) -> dict:
    """name -> (lo calendar idx, hi calendar idx) inclusive; lo > hi means an empty period."""
    out = {}
    for name, (a, b) in periods.items():
        lo = int(np.searchsorted(P.cal, np.datetime64(a, "D"), side="left"))
        hi = int(np.searchsorted(P.cal, np.datetime64(b, "D"), side="right")) - 1
        out[name] = (lo, hi)
    return out


def select_trades(P: Panel, X: Exits, cand: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Real trades of one variant in one period: signal cells whose entry day is in [lo, hi] and
    whose exit is determinable. Ordered by entry day, then largest drop first (SPEC §4 fill order;
    ties broken by flat index, i.e. ticker name — deterministic, no look-ahead)."""
    g = P.gidx[cand]
    sel = cand[(g >= lo) & (g <= hi) & (X.code[cand] >= 0)]
    order = np.lexsort((P.ret1[sel], P.gidx[sel]))
    return sel[order]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def profit_factor(net: np.ndarray) -> float:
    """SPEC §6: sum of positive net returns / |sum of negative net returns|. inf if loss-free with a
    gain, nan if there are no trades or nothing but zeros."""
    if len(net) == 0:
        return float("nan")
    pos = float(net[net > 0].sum())
    neg = float(-net[net < 0].sum())
    if neg == 0.0:
        return float("inf") if pos > 0 else float("nan")
    return pos / neg


def max_drawdown(equity) -> float:
    eq = np.asarray(equity, dtype=np.float64)
    if len(eq) == 0:
        return float("nan")
    peak = np.maximum.accumulate(eq)
    return float((1.0 - eq / peak).max())


def sharpe_daily(rets) -> float:
    r = np.asarray(rets, dtype=np.float64)
    if len(r) < 2:
        return float("nan")
    sd = r.std(ddof=1)
    if sd == 0.0 or not np.isfinite(sd):
        return float("nan")
    return float(r.mean() / sd * math.sqrt(TRADING_DAYS))


def trade_stats(P: Panel, X: Exits, sel: np.ndarray, cost: float = COST_PRIMARY) -> dict:
    gross = X.ret[sel].astype(np.float64)
    code = X.code[sel]
    k = X.k[sel]
    out = {"n_trades": int(len(sel))}
    if len(sel) == 0:
        out.update(pf=float("nan"), win_rate=float("nan"), mean_ret=float("nan"),
                   median_ret=float("nan"), mean_hold_days=float("nan"))
    else:
        net = gross - 2.0 * cost
        out.update(pf=profit_factor(net), win_rate=float((net > 0).mean()), mean_ret=float(net.mean()),
                   median_ret=float(np.median(net)), mean_hold_days=float(k.mean()))
    for cd, name in REASONS.items():
        out[f"exit_{name}"] = int((code == cd).sum())
    out["pf_cost005"] = profit_factor(gross - 2.0 * COST_SENS[0]) if len(sel) else float("nan")
    out["pf_cost020"] = profit_factor(gross - 2.0 * COST_SENS[1]) if len(sel) else float("nan")
    return out


# ---------------------------------------------------------------------------
# Portfolio simulation (SPEC §4)
# ---------------------------------------------------------------------------

def simulate_portfolio(P: Panel, sel: np.ndarray, X: Exits, day_lo: int, day_hi: int,
                       cost: float = COST_PRIMARY, pos_frac: float = POS_FRACTION,
                       max_pos: int = MAX_POSITIONS) -> dict:
    """Day-by-day equity simulation. `sel` = trades ordered by (entry day, largest drop first).

    Per calendar day d (entries only while d <= day_hi; afterwards the loop just lets open positions
    run off — no new entries, no forced close — and stops once nothing is open):
      1. exits scheduled for d are booked: cash += size * (1 + gross - 2*cost);
      2. remaining positions are marked at d's close: size * (1 + close[d]/close[entry] - 1 - cost)
         (the entry cost is booked immediately, the exit cost with the exit — so a position's value
         at exit equals size * (1 + net return), exactly the trade-level number);
      3. the day's signals fill the free slots (max `max_pos`), largest drop first, each sized at
         pos_frac of the equity from step 2 (capped by available cash: no leverage); extra signals
         are dropped. Entries are at the day's close, so they earn nothing until the next day.
    Equity starts at 1.0, cash earns 0."""
    n = len(sel)
    ncal = len(P.cal)
    ent = P.gidx[sel].tolist()
    kk = X.k[sel].astype(np.int64)
    ex = P.gidx[sel + kk].tolist()
    kk = kk.tolist()
    gross = X.ret[sel].astype(np.float64).tolist()
    sel_l = sel.tolist()
    close, gidx = P.close, P.gidx
    cash, e_prev, ptr = 1.0, 1.0, 0
    open_pos, exits_at, fills, rets = {}, {}, [], []
    d = day_lo
    while d < ncal:
        for pid in exits_at.pop(d, ()):
            st = open_pos.pop(pid)
            cash += st[0] * (1.0 + gross[pid] - 2.0 * cost)
        pos_val = 0.0
        for pid, st in open_pos.items():
            i, j, kmax = sel_l[pid], st[1], kk[pid]
            while j + 1 < kmax and gidx[i + j + 1] <= d:
                j += 1
            st[1] = j
            mark = (close[i + j] / close[i] - 1.0) if j > 0 else 0.0
            pos_val += st[0] * (1.0 + mark - cost)
        while ptr < n and ent[ptr] < d:      # defensive: entries before the first simulated day
            ptr += 1
        if d <= day_hi and ptr < n and ent[ptr] == d:
            e = ptr
            while e < n and ent[e] == d:
                e += 1
            equity_pre = cash + pos_val
            free = max_pos - len(open_pos)
            taken = 0
            for pid in range(ptr, e):
                if taken >= free:
                    break
                size = min(pos_frac * equity_pre, cash)
                if size <= 0.0:
                    break
                cash -= size
                open_pos[pid] = [size, 0]
                exits_at.setdefault(ex[pid], []).append(pid)
                pos_val += size * (1.0 - cost)
                fills.append(pid)
                taken += 1
            ptr = e
        equity = cash + pos_val
        rets.append(equity / e_prev - 1.0)
        e_prev = equity
        d += 1
        if d > day_hi and not open_pos:
            break
    rets_a = np.array(rets, dtype=np.float64)
    eq = np.concatenate([[1.0], np.cumprod(1.0 + rets_a)])
    return {"equity": eq, "rets": rets_a, "n_filled": len(fills), "n_skipped": n - len(fills),
            "fills": [sel_l[p] for p in fills], "total_return": float(eq[-1] - 1.0),
            "mdd": max_drawdown(eq), "sharpe": sharpe_daily(rets_a)}


def cagr_of(total_return: float, start_iso: str, end_iso: str) -> float:
    years = ((np.datetime64(end_iso, "D") - np.datetime64(start_iso, "D")).astype(int) + 1) / 365.25
    if years <= 0 or 1.0 + total_return <= 0:
        return float("nan")
    return float((1.0 + total_return) ** (1.0 / years) - 1.0)


def spy_buyhold_mdd(spy: pd.DataFrame | None, start_iso: str, end_iso: str) -> float:
    if spy is None:
        return float("nan")
    d = spy["date"].to_numpy().astype("datetime64[D]")
    m = (d >= np.datetime64(start_iso, "D")) & (d <= np.datetime64(end_iso, "D"))
    c = spy["close"].to_numpy(dtype=np.float64)[m]
    return max_drawdown(c) if len(c) else float("nan")


def evaluate_period(P: Panel, X: Exits, cand: np.ndarray, bounds: tuple, period_iso: tuple, spy) -> dict:
    """Everything one grid row needs for one (variant, period) except the placebo columns."""
    sel = select_trades(P, X, cand, bounds[0], bounds[1])
    row = trade_stats(P, X, sel, COST_PRIMARY)
    sim = simulate_portfolio(P, sel, X, bounds[0], bounds[1], COST_PRIMARY)
    row.update(total_return=sim["total_return"], cagr=cagr_of(sim["total_return"], *period_iso),
               mdd=sim["mdd"], sharpe=sim["sharpe"], n_filled=sim["n_filled"], n_skipped=sim["n_skipped"],
               spy_bh_mdd=spy_buyhold_mdd(spy, *period_iso))
    return row, sel


# ---------------------------------------------------------------------------
# Placebo (SPEC §6 (vii)/(viii))
# ---------------------------------------------------------------------------

def placebo_pool(P: Panel, hold: int, lo: int, hi: int) -> np.ndarray:
    """Valid (ticker, day) cells of a period: member at t, liquidity guard passes, split-day
    exclusion applied, exit determinable (depends on HOLD only). The real signals are a subset
    of this set (they additionally need the DROP condition)."""
    m = P.member & P.liq_ok & ~P.split_excl & P.fut_ok[hold] & (P.gidx >= lo) & (P.gidx <= hi)
    return np.flatnonzero(m)


def placebo_pfs(rng: np.random.Generator, pool: np.ndarray, ret: np.ndarray, n: int, cost: float,
                runs: int) -> np.ndarray:
    """PF of `runs` independent uniform samples (without replacement) of n cells from `pool`."""
    out = np.full(runs, np.nan)
    if n <= 0 or len(pool) == 0:
        return out
    n_eff = min(n, len(pool))
    for r in range(runs):
        cells = pool[rng.choice(len(pool), size=n_eff, replace=False)]
        out[r] = profit_factor(ret[cells].astype(np.float64) - 2.0 * cost)
    return out


def pct95(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float64)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan")
    v = np.minimum(v, PF_CAP)
    p = float(np.percentile(v, PLACEBO_PERCENTILE))
    return float("inf") if p >= PF_CAP else p


# ---------------------------------------------------------------------------
# PASS logic (SPEC §6)
# ---------------------------------------------------------------------------

def _ge(x, thr) -> bool:
    return bool(x is not None and not (isinstance(x, float) and math.isnan(x)) and x >= thr)


def _le(x, thr) -> bool:
    return bool(x is not None and not (isinstance(x, float) and math.isnan(x)) and x <= thr)


def pass_flags(is_row: dict, oos_row: dict, beats_is: bool, beats_oos: bool) -> dict:
    """SPEC §6 (i)-(vii). A NaN metric never passes."""
    spy = is_row["spy_bh_mdd"]
    mdd = is_row["mdd"]
    f = {
        "ok_i_trades": is_row["n_trades"] >= MIN_TRADES_IS,
        "ok_ii_pf_is": _ge(is_row["pf"], MIN_PF_IS),
        "ok_iii_mdd": _le(mdd, MAX_MDD_IS),
        "ok_iv_sharpe": _ge(is_row["sharpe"], MIN_SHARPE_IS),
        "ok_v_vs_spy": bool(not math.isnan(spy) and not math.isnan(mdd) and mdd < spy),
        "ok_vi_pf_oos": _ge(oos_row["pf"], MIN_PF_OOS),
        "ok_vii_placebo": bool(beats_is and beats_oos),
    }
    f["PASS"] = all(f.values())
    # SPEC §6 (viii): only the trade-level criteria (i), (ii), (vi) enter the multiple-testing count
    f["pass_trade_level"] = f["ok_i_trades"] and f["ok_ii_pf_is"] and f["ok_vi_pf_oos"]
    return f


def conclusion(n_pass: int, real_tl_count: int, p95_count: float) -> str:
    """SPEC §6 (viii): the overall verdict is NO EVIDENCE when the real trade-level PASS count does
    not exceed the placebo p95 count — even with individual PASS rows — and also when no single
    variant passes (i)-(vii) at all."""
    if n_pass >= 1 and real_tl_count > p95_count:
        return "EDGE"
    return "NO EVIDENCE"


# ---------------------------------------------------------------------------
# Study driver
# ---------------------------------------------------------------------------

def run_study(P: Panel, spy, periods: dict, placebo_n: int, seed: int, verbose: bool = True) -> dict:
    t0 = time.time()
    bounds = period_bounds(P, periods)
    variants = build_variants()
    cands = {d: candidate_cells(P, d) for d in DROPS}
    X = compute_all_exits(P)
    if verbose:
        log(f"exit arrays for {len(X)} (STOP,TARGET,HOLD) rules done in {time.time() - t0:.1f}s "
            f"(N cells={P.N:,})")

    rows = {"IS": {}, "OOS": {}}
    sels = {"IS": {}, "OOS": {}}
    for v in variants:
        drop, stop, target, hold = v
        for pname in ("IS", "OOS"):
            r, sel = evaluate_period(P, X[(stop, target, hold)], cands[drop], bounds[pname],
                                     periods[pname], spy)
            rows[pname][v] = r
            sels[pname][v] = sel
    if verbose:
        log(f"72 variants x (IS, OOS) evaluated incl. portfolio simulation in {time.time() - t0:.1f}s")

    # ---- placebo ----
    rng = np.random.default_rng(seed)
    pools = {(h, p): placebo_pool(P, h, *bounds[p]) for h in HOLDS for p in ("IS", "OOS")}
    plac = {"IS": {}, "OOS": {}}      # variant -> array of placebo PFs (len placebo_n)
    for v in variants:
        drop, stop, target, hold = v
        for pname in ("IS", "OOS"):
            plac[pname][v] = placebo_pfs(rng, pools[(hold, pname)], X[(stop, target, hold)].ret,
                                         rows[pname][v]["n_trades"], COST_PRIMARY, placebo_n)
    if verbose:
        log(f"placebo: {len(variants)} variants x 2 periods x {placebo_n} runs done in {time.time() - t0:.1f}s")

    flags, p95 = {}, {"IS": {}, "OOS": {}}
    beats = {"IS": {}, "OOS": {}}
    for v in variants:
        for pname in ("IS", "OOS"):
            p95[pname][v] = pct95(plac[pname][v])
            pf = rows[pname][v]["pf"]
            beats[pname][v] = bool(not math.isnan(pf) and not math.isnan(p95[pname][v]) and pf > p95[pname][v])
        flags[v] = pass_flags(rows["IS"][v], rows["OOS"][v], beats["IS"][v], beats["OOS"][v])

    # ---- (viii) multiple testing: trade-level pass counts, real vs each placebo run ----
    real_tl_count = int(sum(flags[v]["pass_trade_level"] for v in variants))
    plac_counts = []
    for r in range(placebo_n):
        cnt = 0
        for v in variants:
            pis, poos = plac["IS"][v][r], plac["OOS"][v][r]
            if (rows["IS"][v]["n_trades"] >= MIN_TRADES_IS and _ge(pis, MIN_PF_IS) and _ge(poos, MIN_PF_OOS)):
                cnt += 1
        plac_counts.append(cnt)
    p95_count = float(np.percentile(plac_counts, PLACEBO_PERCENTILE)) if plac_counts else float("nan")
    passers = [v for v in variants if flags[v]["PASS"]]
    verdict = conclusion(len(passers), real_tl_count, p95_count)

    # ---- HOLDOUT (SPEC §5): only if some variant passed (i)-(vii); only the best-IS-PF one; once ----
    holdout = None
    if passers:
        best = max(passers, key=lambda v: (rows["IS"][v]["pf"], -variants.index(v)))
        drop, stop, target, hold = best
        r, sel = evaluate_period(P, X[(stop, target, hold)], cands[drop], bounds["HOLDOUT"],
                                 periods["HOLDOUT"], spy)
        holdout = {"variant": {"drop": drop, "stop": stop_label(stop), "target": target, "hold": hold},
                   "variant_tuple": best, "period": list(periods["HOLDOUT"]), "metrics": r, "sel": sel,
                   "note": "SPEC has no pass threshold for HOLDOUT; reported for reference only."}
        if verbose:
            log(f"HOLDOUT computed once for {best}: n={r['n_trades']} pf={r['pf']:.3f}")
    elif verbose:
        log("HOLDOUT stays sealed: no variant passed SPEC §6 (i)-(vii)")

    return dict(P=P, X=X, cands=cands, bounds=bounds, periods=periods, variants=variants, rows=rows,
                sels=sels, plac=plac, p95=p95, beats=beats, flags=flags, placebo_n=placebo_n, seed=seed,
                real_tl_count=real_tl_count, plac_counts=plac_counts, p95_count=p95_count,
                passers=passers, verdict=verdict, holdout=holdout, runtime=time.time() - t0, spy=spy)


# ---------------------------------------------------------------------------
# Data hygiene diagnostics (counts only — no PnL)
# ---------------------------------------------------------------------------

def exclusion_counts(P: Panel, bounds_all: tuple) -> dict:
    """Signal days dropped by each SPEC §2 split rule, by DROP level, among days that would
    otherwise be signals (drop reached, member, liquid) inside the study window; plus how many
    tickers fell under which rule."""
    lo, hi = bounds_all
    win = (P.gidx >= lo) & (P.gidx <= hi)
    out = {"flagged_days_rule_A_split_file": int((P.excl_a & win).sum()),
           "flagged_days_rule_B_fallback": int((P.excl_b & win).sum()),
           "tickers_with_splits_file": int(sum(P.has_splits.values())),
           "tickers_using_fallback_rule": int(len(P.has_splits) - sum(P.has_splits.values())),
           "by_drop": {}}
    for d in DROPS:
        base = P.gap_ok & (P.ret1 <= -d + DROP_EPS) & P.member & P.liq_ok & win
        out["by_drop"][f"{d * 100:g}%"] = {
            "signals_before_exclusion": int(base.sum()),
            "excluded_rule_A_split_file": int((base & P.excl_a).sum()),
            "excluded_rule_B_fallback": int((base & P.excl_b).sum()),
        }
    return out


def extreme_drop_review(P: Panel, bounds_all: tuple, limit: int = 40) -> list:
    """Member+liquid closes of <= -40% that were NOT excluded as splits: listed for manual review
    (a raw forward split in a ticker without a splits file would look exactly like this)."""
    lo, hi = bounds_all
    m = (P.gap_ok & (P.ret1 <= -SPLIT_FALLBACK_DROP) & P.member & P.liq_ok & ~P.split_excl
         & (P.gidx >= lo) & (P.gidx <= hi))
    out = []
    for i in np.flatnonzero(m)[:limit]:
        out.append({"ticker": P.tickers[int(P.tick[i])], "date": P.iso(P.gidx[i]),
                    "close_to_close": round(float(P.ret1[i]), 4)})
    return out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

GRID_METRIC_COLS = ["n_trades", "pf", "pf_cost005", "pf_cost020", "win_rate", "mean_ret", "median_ret",
                    "mean_hold_days", "exit_target", "exit_stop", "exit_gap", "exit_time", "total_return",
                    "cagr", "mdd", "sharpe", "spy_bh_mdd"]
FLAG_COLS = ["ok_i_trades", "ok_ii_pf_is", "ok_iii_mdd", "ok_iv_sharpe", "ok_v_vs_spy", "ok_vi_pf_oos",
             "ok_vii_placebo", "PASS", "pass_trade_level"]


def grid_dataframe(res: dict, pname: str) -> pd.DataFrame:
    recs = []
    for v in res["variants"]:
        drop, stop, target, hold = v
        r = res["rows"][pname][v]
        rec = {"drop": drop, "stop": stop_label(stop), "target": target, "hold": hold}
        rec.update({c: r[c] for c in GRID_METRIC_COLS})
        rec["placebo_pf_p95"] = res["p95"][pname][v]
        rec["beats_placebo"] = res["beats"][pname][v]
        rec.update({c: res["flags"][v][c] for c in FLAG_COLS})
        rec["n_filled"] = r["n_filled"]
        rec["n_skipped"] = r["n_skipped"]
        recs.append(rec)
    return pd.DataFrame(recs)


def placebo_summary_dataframe(res: dict) -> pd.DataFrame:
    recs = []
    for pname in ("IS", "OOS"):
        for v in res["variants"]:
            drop, stop, target, hold = v
            arr = np.asarray(res["plac"][pname][v], dtype=np.float64)
            fin = np.minimum(arr[~np.isnan(arr)], PF_CAP)
            q = (lambda p: float(np.percentile(fin, p))) if len(fin) else (lambda p: float("nan"))
            recs.append({"stop": stop_label(stop), "target": target, "hold": hold, "drop": drop,
                         "period": pname, "n_trades_sampled": res["rows"][pname][v]["n_trades"],
                         "runs": int(len(arr)), "pf_mean": float(fin.mean()) if len(fin) else float("nan"),
                         "pf_p05": q(5), "pf_p25": q(25), "pf_median": q(50), "pf_p75": q(75),
                         "pf_p95": res["p95"][pname][v], "pf_max": float(arr[~np.isnan(arr)].max()) if len(fin) else float("nan"),
                         "real_pf": res["rows"][pname][v]["pf"], "real_beats_p95": res["beats"][pname][v]})
    df = pd.DataFrame(recs)
    return df.sort_values(["period", "stop", "target", "hold", "drop"], kind="stable").reset_index(drop=True)


def trades_dataframe(res: dict, variant: tuple, periods_to_write: list) -> pd.DataFrame:
    P, X = res["P"], res["X"]
    drop, stop, target, hold = variant
    Xv = X[(stop, target, hold)]
    recs = []
    for pname in periods_to_write:
        if pname == "HOLDOUT":
            sel = res["holdout"]["sel"]
        else:
            sel = res["sels"][pname][variant]
        for i in sel.tolist():
            code, k = int(Xv.code[i]), int(Xv.k[i])
            entry = float(P.close[i])
            xp = exit_price(P, i, code, k, stop, target)
            gross = xp / entry - 1.0
            recs.append({"ticker": P.tickers[int(P.tick[i])], "entry_date": P.iso(P.gidx[i]),
                         "entry_price": entry, "exit_date": P.iso(P.gidx[i + k]), "exit_price": xp,
                         "exit_reason": REASONS[code], "gross_ret": gross,
                         "net_ret": gross - 2.0 * COST_PRIMARY, "period": pname})
    return pd.DataFrame(recs, columns=["ticker", "entry_date", "entry_price", "exit_date", "exit_price",
                                       "exit_reason", "gross_ret", "net_ret", "period"])


def _clean_json(o):
    """json.dump-safe: NaN/inf become strings so the file is strict JSON."""
    if isinstance(o, dict):
        return {str(k): _clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean_json(v) for v in o]
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return float(f"{f:.10g}") if math.isfinite(f) else ("inf" if f > 0 else "-inf") if not math.isnan(f) else None
    if isinstance(o, np.ndarray):
        return _clean_json(o.tolist())
    return o


def _pf(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return "inf" if math.isinf(x) else f"{x:.2f}"


def _pc(x, d=2) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.{d}f}%"


def _num(x, d=2) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def vlabel(v) -> str:
    drop, stop, target, hold = v
    st = "없음" if stop is None else f"{stop * 100:+.0f}%"
    return f"DROP {drop * 100:g}% / STOP {st} / TARGET {target * 100:+.0f}% / HOLD {hold}일"


def write_summary(res: dict, out_dir: str, coverage: dict, diag: dict, load_info: dict, run_date: str,
                  synthetic: bool) -> None:
    rows, flags, variants = res["rows"], res["flags"], res["variants"]
    n_pass = len(res["passers"])
    verdict = res["verdict"]
    cnt = {k: sum(flags[v][k] for v in variants) for k in FLAG_COLS}
    L = []
    L.append(f"# panic_dip — 결과 요약 ({run_date})" + ("  [SYNTHETIC SELF-TEST — 실제 데이터 아님]" if synthetic else ""))
    L.append("")
    L.append("`SPEC.md`(2026-11-01 사전등록)를 `run_panic_dip.py`가 기계적으로 구현해 낸 결과다. "
             "기준·임계값은 명세 그대로이며 결과를 본 뒤 바꾼 것이 없다. 이 요약은 숫자만 옮긴 것이고 "
             "해석은 아래 '편향·한계'를 먼저 읽고 한다.")
    L.append("")
    L.append(f"## 전체 결론: **{verdict}**")
    L.append("")
    L.append(f"- 개별 PASS 변형 (SPEC §6 (i)~(vii) 전부 충족): **{n_pass}개** / 72")
    shown = sorted(res["passers"], key=lambda v: -rows["IS"][v]["pf"])[:25]
    for v in shown:
        r = rows["IS"][v]
        L.append(f"  - {vlabel(v)} — IS PF {_pf(r['pf'])}, OOS PF {_pf(rows['OOS'][v]['pf'])}, "
                 f"IS 거래 {r['n_trades']}")
    if n_pass > len(shown):
        L.append(f"  - … 외 {n_pass - len(shown)}개 (IS PF 순 상위 {len(shown)}개만 표시; 전체는 `grid_IS.csv`의 PASS 열)")
    L.append(f"- (viii) 다중검정: 거래 단위 기준 (i)·(ii)·(vi) PASS 수 — 실제 **{res['real_tl_count']}** vs "
             f"placebo {res['placebo_n']}회의 95퍼센타일 **{res['p95_count']:.2f}** "
             f"(placebo 평균 {np.mean(res['plac_counts']):.2f}, 최대 {max(res['plac_counts']) if res['plac_counts'] else 0}).")
    if verdict == "EDGE":
        L.append("  - 실제 PASS 수가 placebo 95퍼센타일을 넘고 개별 PASS 변형도 있으므로 명세상 EDGE.")
    elif n_pass == 0:
        L.append("  - 개별 PASS 변형이 하나도 없으므로 NO EVIDENCE.")
    else:
        L.append("  - 개별 PASS가 있어도 실제 PASS 수가 placebo 95퍼센타일 이하이므로 명세상 NO EVIDENCE.")
    L.append("")
    L.append("### 기준별 통과 변형 수 (72 중, 각 기준 단독)")
    L.append("")
    L.append("| (i) IS 거래≥100 | (ii) IS PF≥1.3 | (iii) IS MDD≤30% | (iv) IS Sharpe≥0.8 | (v) MDD<SPY | (vi) OOS PF≥1.1 | (vii) placebo 초과 | 전부(PASS) |")
    L.append("|---|---|---|---|---|---|---|---|")
    L.append(f"| {cnt['ok_i_trades']} | {cnt['ok_ii_pf_is']} | {cnt['ok_iii_mdd']} | {cnt['ok_iv_sharpe']} | "
             f"{cnt['ok_v_vs_spy']} | {cnt['ok_vi_pf_oos']} | {cnt['ok_vii_placebo']} | {cnt['PASS']} |")
    L.append("")

    # proposal variant
    pv = PROPOSAL
    L.append(f"## 원 제안 변형 — {vlabel(pv)}")
    L.append("")
    L.append("| 구간 | 거래 | PF (0.10%) | PF (0.05%) | PF (0.20%) | 승률 | 평균 | 중앙 | 평균 보유일 | 청산 target/stop/gap/time | "
             "총수익 | CAGR | MDD | Sharpe | SPY MDD | placebo p95 | 초과 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for pname in ("IS", "OOS"):
        r = rows[pname][pv]
        L.append(f"| {pname} | {r['n_trades']} | {_pf(r['pf'])} | {_pf(r['pf_cost005'])} | {_pf(r['pf_cost020'])} | "
                 f"{_pc(r['win_rate'], 1)} | {_pc(r['mean_ret'])} | {_pc(r['median_ret'])} | {_num(r['mean_hold_days'])} | "
                 f"{r['exit_target']}/{r['exit_stop']}/{r['exit_gap']}/{r['exit_time']} | {_pc(r['total_return'], 1)} | "
                 f"{_pc(r['cagr'], 1)} | {_pc(r['mdd'], 1)} | {_num(r['sharpe'])} | {_pc(r['spy_bh_mdd'], 1)} | "
                 f"{_pf(res['p95'][pname][pv])} | {'예' if res['beats'][pname][pv] else '아니오'} |")
    L.append("")
    L.append(f"PASS 여부: **{'PASS' if flags[pv]['PASS'] else 'FAIL'}** "
             f"(실패 기준: {', '.join(c for c in FLAG_COLS[:7] if not flags[pv][c]) or '없음'}). "
             f"거래 전체는 `trades_{PROPOSAL_SLUG}.csv`.")
    L.append("")

    # top by IS PF
    ranked = sorted(variants, key=lambda v: (-(rows['IS'][v]['pf'] if not math.isnan(rows['IS'][v]['pf']) else -1.0)))
    L.append("## IS PF 상위 10 변형 (참고용 — 선택 기준이 아님)")
    L.append("")
    L.append("| 변형 | IS 거래 | IS PF | IS MDD | IS Sharpe | OOS 거래 | OOS PF | IS placebo p95 | OOS placebo p95 | PASS |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for v in ranked[:10]:
        a, b = rows["IS"][v], rows["OOS"][v]
        L.append(f"| {vlabel(v)} | {a['n_trades']} | {_pf(a['pf'])} | {_pc(a['mdd'], 1)} | {_num(a['sharpe'])} | "
                 f"{b['n_trades']} | {_pf(b['pf'])} | {_pf(res['p95']['IS'][v])} | {_pf(res['p95']['OOS'][v])} | "
                 f"{'PASS' if flags[v]['PASS'] else '-'} |")
    L.append("")

    # holdout
    L.append("## HOLDOUT (2024-11-01 ~ 2026-10-31)")
    L.append("")
    ho = res["holdout"]
    if ho is None:
        L.append("봉인 유지 — SPEC §6 (i)~(vii)를 통과한 변형이 없어 holdout은 계산하지 않았다 (SPEC §5).")
    else:
        m = ho["metrics"]
        L.append(f"통과 변형 중 IS PF 최고 1개만, 단 한 번 계산: {vlabel(ho['variant_tuple'])} "
                 f"(IS PF {_pf(rows['IS'][ho['variant_tuple']]['pf'])}).")
        L.append("")
        L.append("| 거래 | PF (0.10%) | PF (0.05%) | PF (0.20%) | 승률 | 평균 | 총수익 | CAGR | MDD | Sharpe | SPY MDD |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        L.append(f"| {m['n_trades']} | {_pf(m['pf'])} | {_pf(m['pf_cost005'])} | {_pf(m['pf_cost020'])} | "
                 f"{_pc(m['win_rate'], 1)} | {_pc(m['mean_ret'])} | {_pc(m['total_return'], 1)} | {_pc(m['cagr'], 1)} | "
                 f"{_pc(m['mdd'], 1)} | {_num(m['sharpe'])} | {_pc(m['spy_bh_mdd'], 1)} |")
        L.append("")
        L.append("SPEC은 HOLDOUT에 별도 통과 임계값을 정하지 않았다. 숫자는 참고용이며 위 전체 결론을 바꾸지 않는다. "
                 "`holdout.json` 참고.")
    L.append("")

    # data coverage
    L.append("## 데이터 커버리지")
    L.append("")
    req, rec = coverage.get("requested"), coverage.get("received")
    L.append(f"- 요청 종목 수: {req} / 받은 종목 수: {rec} / 빠진 종목: {len(coverage.get('missing', []))}개"
             + (f" ({100 * len(coverage.get('missing', [])) / req:.1f}%)" if req else ""))
    miss = coverage.get("missing", [])
    if miss:
        L.append(f"- 빠진 종목 (처음 {min(len(miss), 60)}개): " + ", ".join(miss[:60]) + (" …" if len(miss) > 60 else "")
                 + " — 전체는 `data_coverage.json`.")
    L.append(f"- 이번 실행에서 실제 로드된 시리즈: {load_info['loaded']}개 (너무 짧아 제외 {load_info['skipped']}개). "
             f"SPY 벤치마크: {'있음' if res['spy'] is not None else '**없음 → 기준 (v) 판정 불가, 전부 불통과 처리**'}.")
    L.append(f"- split 이벤트 정보가 있는 종목: {coverage.get('tickers_with_split_events', 'n/a')}개 "
             f"(splits 파일 보유 {diag['tickers_with_splits_file']}개, 보조 규칙 적용 {diag['tickers_using_fallback_rule']}개).")
    L.append("")
    L.append("### 분할 의심 신호 제외 건수 (SPEC §2)")
    L.append("")
    L.append("| DROP | 제외 전 신호일 | 규칙 A (splits 이벤트) | 규칙 B (거래량 <0.5×중앙값 & −40%↓ 보조 규칙) |")
    L.append("|---|---|---|---|")
    for k, v in diag["by_drop"].items():
        L.append(f"| {k} | {v['signals_before_exclusion']} | {v['excluded_rule_A_split_file']} | {v['excluded_rule_B_fallback']} |")
    L.append("")
    if diag["extreme_review"]:
        L.append(f"제외되지 않은 −40% 이하 종가 (member·유동성 통과; 분할 의심 수동 검토용, 최대 {len(diag['extreme_review'])}건): "
                 + ", ".join(f"{e['ticker']} {e['date']} ({e['close_to_close'] * 100:.0f}%)" for e in diag["extreme_review"]))
        L.append("")

    # bias statement
    L.append("## 편향·한계 선언 (SPEC §2 사전 선언 + 구현 시 확인한 것)")
    L.append("")
    L.append("- **Survivorship bias**: Yahoo는 상장폐지·흡수된 종목의 과거를 주지 않는다. 빠진 종목은 대개 크게 떨어진 뒤 "
             "계속 떨어진 쪽이라 결과를 **좋은 쪽으로 부풀린다**. 따라서 PASS는 의심하고 FAIL은 더 믿는다.")
    L.append("- 폭락의 '이유'(실적 등)는 검사하지 않았다. 검사한 것은 이유가 무엇이든 X% 급락 뒤 반등 규칙이 통했나이다.")
    L.append("- 진입은 신호일 종가 '그 가격' 체결 가정이다. 실제 봇의 마감 15분 전 판단·지정가 차이는 편도 0.10% 비용으로 "
             "흡수했다고 가정했을 뿐 검증하지 않았다 (민감도 0.05%·0.20%는 PF 열에 있음).")
    L.append("- 거래 단위 통계는 사이징·슬롯과 무관하게 모든 신호를 거래로 센다. 같은 종목의 연속 급락일, 같은 날의 시장 전체 급락 "
             "(예: 2020-03) 신호는 서로 독립이 아니라 한 사건에 몰려 있으므로 거래 수 ≥100은 독립 표본 수가 아니다.")
    L.append("- 가격은 Yahoo chart API의 `open/high/low/close`다 (`adjclose` 아님). 알려진 바로는 이 값은 분할 조정은 되어 있고 배당 조정은 "
             "안 되어 있다 (이번 실행에서 확인한 것은 아님). 그렇다면 raw 분할일의 가짜 급락이 애초에 없어 분할일 제외 규칙은 대부분 "
             "no-op이다. 위 '제외되지 않은 −40% 이하 종가' 목록으로 점검할 것. SPY MDD도 배당 미반영 가격 기준이다.")
    L.append("- 일중 순서를 모르므로 stop과 target이 같은 날 모두 닿으면 stop 우선(보수적), 시가가 stop 아래면 시가 청산이다. "
             "target은 시가가 넘어도 target 가격에 체결된 것으로 센다 (보수적).")
    L.append("- 구성종목 기간은 [start_date, end_date) — end_date 당일은 제외 — 로 해석했다. 편입일 당일은 포함.")
    L.append(f"- placebo {res['placebo_n']}회, seed {res['seed']}. 변형별 거래 수를 맞춘 무작위 (날짜, 종목) 표본이며 "
             "거래 단위 PF만 비교한다 (placebo_summary.csv).")
    L.append("")
    L.append("## 실행 정보")
    L.append("")
    L.append(f"- 실행 시간 {res['runtime']:.1f}초, 패널 셀 {res['P'].N:,}개 ({len(res['P'].tickers)}종목).")
    L.append(f"- 구간: " + ", ".join(f"{k} {a}~{b}" for k, (a, b) in res["periods"].items()) + ".")
    L.append("- 산출물: `grid_IS.csv`, `grid_OOS.csv`, `placebo_summary.csv`, "
             f"`trades_{PROPOSAL_SLUG}.csv`, `data_coverage.json`, `summary.json`"
             + (", `holdout.json`" if ho is not None else "") + ".")
    L.append("")
    with open(os.path.join(out_dir, "SUMMARY.md"), "w") as f:
        f.write("\n".join(L) + "\n")


def write_outputs(res: dict, out_dir: str, coverage: dict, load_info: dict, run_date: str,
                  synthetic: bool = False) -> list:
    os.makedirs(out_dir, exist_ok=True)
    P = res["P"]
    written = []

    def w(name):
        written.append(name)
        return os.path.join(out_dir, name)

    grid_dataframe(res, "IS").to_csv(w("grid_IS.csv"), index=False, float_format="%.6g")
    grid_dataframe(res, "OOS").to_csv(w("grid_OOS.csv"), index=False, float_format="%.6g")
    placebo_summary_dataframe(res).to_csv(w("placebo_summary.csv"), index=False, float_format="%.6g")

    to_write = ["IS", "OOS"]
    if res["holdout"] is not None:
        hv = res["holdout"]["variant"]
        if (hv["drop"], hv["target"], hv["hold"]) == (PROPOSAL[0], PROPOSAL[2], PROPOSAL[3]) and hv["stop"] == stop_label(PROPOSAL[1]):
            to_write.append("HOLDOUT")
    trades_dataframe(res, PROPOSAL, to_write).to_csv(w(f"trades_{PROPOSAL_SLUG}.csv"), index=False)

    all_bounds = (res["bounds"]["IS"][0], res["bounds"]["HOLDOUT"][1])
    diag = exclusion_counts(P, all_bounds)
    diag["extreme_review"] = extreme_drop_review(P, all_bounds)

    cov_path = os.path.join(out_dir, "data_coverage.json")
    if not coverage.get("_from_fetch"):
        with open(w("data_coverage.json"), "w") as f:
            json.dump(_clean_json({k: v for k, v in coverage.items() if k != "_from_fetch"}), f, indent=2)
    else:
        written.append("data_coverage.json")  # fetch_prices.py's file, passed through untouched
        if not os.path.exists(cov_path):
            with open(cov_path, "w") as f:
                json.dump(_clean_json({k: v for k, v in coverage.items() if k != "_from_fetch"}), f, indent=2)

    if res["holdout"] is not None:
        ho = res["holdout"]
        with open(w("holdout.json"), "w") as f:
            json.dump(_clean_json({"variant": ho["variant"], "period": ho["period"], "metrics": ho["metrics"],
                                   "note": ho["note"]}), f, indent=2)

    summary = {
        "run_date": run_date, "synthetic": synthetic, "verdict": res["verdict"], "n_pass": len(res["passers"]),
        "passers": [list(map(lambda x: x if x is not None else "none", v)) for v in res["passers"]],
        "real_trade_level_pass_count": res["real_tl_count"], "placebo_pass_counts": res["plac_counts"],
        "placebo_pass_count_p95": res["p95_count"], "placebo_n": res["placebo_n"], "seed": res["seed"],
        "holdout_computed": res["holdout"] is not None, "periods": res["periods"],
        "n_tickers_in_panel": len(P.tickers), "n_cells": P.N, "load_info": load_info,
        "exclusions": diag, "runtime_seconds": round(res["runtime"], 1),
    }
    with open(w("summary.json"), "w") as f:
        json.dump(_clean_json(summary), f, indent=2)
    write_summary(res, out_dir, coverage, diag, load_info, run_date, synthetic)
    written.append("SUMMARY.md")
    return written


# ---------------------------------------------------------------------------
# Synthetic data (self-test, tests, timing)
# ---------------------------------------------------------------------------

def make_synthetic(n_tickers: int = 30, n_days: int = 1500, seed: int = 7, edge: bool = True,
                   start: str = DATA_START):
    """Random-walk OHLCV panel with frequent shock days (-9..-22%, ~1.2% of cells). With edge=True the day after a
    shock rebounds +3.5..6% (a planted, tradeable edge); with edge=False it does not. Returns
    (series, membership, splits, spy, periods). Ticker 0 carries a raw 2:1 split (split file ->
    rule A), ticker 1 a -50% low-volume day without a splits file (rule B), ticker 3 is illiquid."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    shape = (n_days, n_tickers)
    r = rng.normal(0.0003, 0.014, shape)
    shock = rng.random(shape) < 0.012
    shock[:30] = False
    shock[-8:] = False
    r = np.where(shock, -rng.uniform(0.09, 0.22, shape), r)
    after = np.zeros(shape, bool)
    after[1:] = shock[:-1] & ~shock[1:]
    if edge:
        r = np.where(after, rng.uniform(0.035, 0.06, shape), r)
    close = rng.uniform(40, 200, n_tickers) * np.cumprod(1.0 + r, axis=0)
    prev = np.empty_like(close)
    prev[0] = close[0] / (1.0 + r[0])
    prev[1:] = close[:-1]
    open_ = prev * (1.0 + rng.normal(0, 0.003, shape))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0, 0.004, shape)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0, 0.004, shape)))
    dollar = np.exp(rng.normal(np.log(4e8), 0.4, shape))
    if n_tickers > 3:
        dollar[:, 3] = 1e7                       # illiquid: the guard must veto every signal
    vol = dollar / close

    split_day = min(700, n_days // 2)
    splits = {}
    if n_tickers > 0:                            # ticker 0: raw forward split (prices before x2, volume /2)
        for a in (open_, high, low, close):
            a[:split_day, 0] *= 2.0
        vol[:split_day, 0] /= 2.0
        splits["T000"] = [str(dates[split_day].date())]
    if n_tickers > 1:                            # ticker 1: -50% close on very low volume, no splits file
        sd = min(900, n_days // 2 + 50)
        for a in (open_, high, low, close):
            a[sd:, 1] *= 0.5
        vol[sd, 1] *= 0.2
    series, membership = {}, {}
    names = [f"T{j:03d}" for j in range(n_tickers)]
    for j, t in enumerate(names):
        series[t] = pd.DataFrame({"date": dates, "open": open_[:, j], "high": high[:, j], "low": low[:, j],
                                  "close": close[:, j], "volume": vol[:, j]})
        splits.setdefault(t, None)
        if j % 10 == 5:      # joins late
            membership[t] = [(np.datetime64(str(dates[n_days // 3].date()), "D"), None)]
        elif j % 10 == 7:    # leaves early
            membership[t] = [(np.datetime64("2000-01-01", "D"), np.datetime64(str(dates[(2 * n_days) // 3].date()), "D"))]
        elif j % 10 == 9:    # two intervals
            membership[t] = [(np.datetime64("2000-01-01", "D"), np.datetime64(str(dates[n_days // 4].date()), "D")),
                             (np.datetime64(str(dates[n_days // 2].date()), "D"), None)]
        else:
            membership[t] = [(np.datetime64("2000-01-01", "D"), None)]
    sr = rng.normal(0.0004, 0.011, n_days)
    sr[n_days // 2: n_days // 2 + 25] -= 0.012      # a drawdown so SPY's MDD is non-trivial
    sc = 100 * np.cumprod(1 + sr)
    spy = pd.DataFrame({"date": dates, "open": sc, "high": sc * 1.003, "low": sc * 0.997, "close": sc,
                        "volume": 1e8})
    i0, i1, i2, i3 = 25, int(0.55 * n_days), int(0.80 * n_days), n_days - 1
    periods = {"IS": (str(dates[i0].date()), str(dates[i1].date())),
               "OOS": (str(dates[i1 + 1].date()), str(dates[i2].date())),
               "HOLDOUT": (str(dates[i2 + 1].date()), str(dates[i3].date()))}
    return series, membership, splits, spy, periods


def run_synthetic(n_tickers: int = 30, n_days: int = 1500, placebo_n: int = 5, out_dir: str | None = None) -> str:
    t0 = time.time()
    log(f"self-test: generating {n_tickers} tickers x {n_days} days ...")
    series, membership, splits, spy, periods = make_synthetic(n_tickers, n_days, seed=7, edge=True)
    P = build_panel(series, membership, splits)
    log(f"panel built: {len(P.tickers)} tickers, {P.N:,} cells, {time.time() - t0:.1f}s")
    res = run_study(P, spy, periods, placebo_n, SEED)
    own_tmp = out_dir is None
    out = out_dir or tempfile.mkdtemp(prefix="panic_dip_synth_")
    cov = {"requested": n_tickers, "received": len(P.tickers), "missing": [], "synthetic": True}
    load_info = {"loaded": len(P.tickers), "skipped": len(P.skipped)}
    written = write_outputs(res, out, cov, load_info, str(date.today()), synthetic=True)

    expected = ["grid_IS.csv", "grid_OOS.csv", "placebo_summary.csv", f"trades_{PROPOSAL_SLUG}.csv",
                "data_coverage.json", "summary.json", "SUMMARY.md"]
    for name in expected:
        assert os.path.exists(os.path.join(out, name)), f"missing output {name}"
    g_is = pd.read_csv(os.path.join(out, "grid_IS.csv"))
    g_oos = pd.read_csv(os.path.join(out, "grid_OOS.csv"))
    assert len(g_is) == 72 and len(g_oos) == 72, "grid must have 72 rows"
    assert len(pd.read_csv(os.path.join(out, "placebo_summary.csv"))) == 144
    assert g_is["n_trades"].sum() > 0, "synthetic data produced no trades"
    assert res["passers"], "planted rebound edge should yield at least one PASS variant"
    assert res["holdout"] is not None and os.path.exists(os.path.join(out, "holdout.json")), \
        "a PASS variant exists, so holdout.json must be written"
    assert res["verdict"] in ("EDGE", "NO EVIDENCE")
    log(f"self-test: verdict={res['verdict']} passers={len(res['passers'])} out={out} ({time.time() - t0:.1f}s)")
    if own_tmp:
        shutil.rmtree(out, ignore_errors=True)
    print("SYNTHETIC OK")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="default: data/panic_dip")
    ap.add_argument("--out-dir", default=None, help="default: research/panic_dip/results")
    ap.add_argument("--placebo-n", type=int, default=None, help=f"placebo runs per variant/period (default {PLACEBO_N})")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--synthetic", action="store_true", help="self-test on generated data (no files needed)")
    ap.add_argument("--synthetic-tickers", type=int, default=30, help="size knob for --synthetic (timing runs)")
    ap.add_argument("--synthetic-days", type=int, default=1500, help="size knob for --synthetic (timing runs)")
    args = ap.parse_args()

    if args.synthetic:
        run_synthetic(args.synthetic_tickers, args.synthetic_days,
                      placebo_n=args.placebo_n if args.placebo_n is not None else 5, out_dir=args.out_dir)
        return 0

    t0 = time.time()
    out_dir = args.out_dir or DEFAULT_OUT_DIR
    placebo_n = args.placebo_n if args.placebo_n is not None else PLACEBO_N
    membership = load_constituents()
    universe = sorted(membership)
    series, splits, missing = load_series(args.data_dir, universe)
    if not series:
        log(f"no data/*.csv under {args.data_dir} for any constituent — run fetch_prices.py first. "
            "Not treated as a failure.")
        return 0
    spy = load_spy(args.data_dir)
    log(f"universe {len(universe)} tickers, loaded {len(series)}, no file for {len(missing)}; "
        f"SPY {'loaded' if spy is not None else 'MISSING'} ({time.time() - t0:.1f}s)")
    P = build_panel(series, membership, splits)
    log(f"panel: {len(P.tickers)} tickers, {P.N:,} cells, {len(P.cal)} calendar days ({time.time() - t0:.1f}s)")

    cov_path = os.path.join(out_dir, "data_coverage.json")
    coverage = None
    if os.path.exists(cov_path):
        try:
            with open(cov_path) as f:
                coverage = json.load(f)
            coverage["_from_fetch"] = True
        except Exception as e:  # noqa: BLE001
            log(f"WARNING: could not read {cov_path}: {e}")
    if coverage is None:
        coverage = {"requested": len(universe), "received": len(series), "missing": missing,
                    "note": "written by run_panic_dip.py (no fetch_prices.py coverage file found)"}
    load_info = {"loaded": len(P.tickers), "skipped": len(P.skipped), "no_file": len(missing)}

    res = run_study(P, spy, PERIODS, placebo_n, args.seed)
    write_outputs(res, out_dir, coverage, load_info, str(date.today()))
    log(f"verdict: {res['verdict']} (PASS variants: {len(res['passers'])}, trade-level real {res['real_tl_count']} "
        f"vs placebo p95 {res['p95_count']:.2f}); total runtime {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
