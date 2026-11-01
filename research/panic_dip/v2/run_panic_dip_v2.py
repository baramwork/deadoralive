"""
run_panic_dip_v2.py — panic_dip v2 study (SPEC_v2.md, pre-registered 2026-11-01). Implements the spec
mechanically; no threshold in this file is a free parameter (every one is a SPEC_v2 / SPEC constant below).

Question (v2): on a MARKET-level panic day (SPY close-to-close <= -2% / -3% AND >= 50% of the S&P members
down >= 2%), does buying the close — (S) the members that fell about as much as the market, or (I) SPY itself —
and exiting by a fixed STOP / TARGET / HOLD rule beat the 9월 pass criteria AND three controls
(A same-day random members, B volatility-matched random cells, C SPY on non-panic days)?

v1 (research/panic_dip/run_panic_dip.py) is FROZEN and imported: Panel building, the exit-array kernel, PF / MDD /
Sharpe, trade statistics, CAGR and the SPY buy&hold MDD are v1's. Everything else here is v2.

  1. Series are loaded for the constituents, with SPEC_v2 §2 alias resolution: an old ticker's membership
     intervals are merged into its new ticker's series (one panel column per company -> a (series, date) cell
     exists once however many tickers of that company were members that day).
  2. SPY is a non-member column of the same panel (member = False), so its exit arrays use the same 12 rules and the
     same calendar.  r_spy, breadth and the panic flags (P2, P3) are computed per calendar day and ONLY for the
     days of the window being evaluated (IS+OOS first; HOLDOUT only if some variant passes).
  3. Mode S candidates, the 25-name liquidity cut, the 50%-of-equity equal-weight portfolio (SPEC §4), trade-level
     statistics, per-panic-day statistics, controls A/B/C (100 runs, seed 20261101), PASS flags (i)-(viii).

`--synthetic` is the self-test and needs no files.

Usage
    python research/panic_dip/v2/run_panic_dip_v2.py                  # full run (needs data/panic_dip_v2)
    python research/panic_dip/v2/run_panic_dip_v2.py --synthetic      # fast end-to-end self-test
    python research/panic_dip/v2/run_panic_dip_v2.py --control-n 20   # fewer control runs (smoke)
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
V1_DIR = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(V1_DIR))
if V1_DIR not in sys.path:
    sys.path.insert(0, V1_DIR)

import run_panic_dip as R  # noqa: E402  (v1, frozen)

DEFAULT_DATA_DIR = os.path.join(ROOT, "data", "panic_dip_v2")
DEFAULT_OUT_DIR = os.path.join(HERE, "results")
RESOLVED_FILE = "_aliases_resolved.json"
COVERAGE_FILE = "data_coverage_v2.json"

# ---------------------------------------------------------------------------
# Fixed parameters — SPEC_v2.md (periods, costs, PF/MDD/Sharpe thresholds are SPEC.md's, via R.*).
# ---------------------------------------------------------------------------
PERIODS = R.PERIODS                       # SPEC_v2 §5 = v1 §5
PANICS = ("P2", "P3")
MODES = ("S", "I")
PANIC_SPY_R = {"P2": -0.02, "P3": -0.03}  # SPEC_v2 §3.1: r_spy <= this
BREADTH_STOCK_R = -0.02                   # a member counts as "down" when its own return <= -2%
BREADTH_MIN = 0.5                         # breadth >= 50% (fixed)
S_MULT = 3.0                              # SPEC_v2 §3.2: r_spy >= r_i >= 3 x r_spy
HARD_CAP = -0.20                          # r_i <= -20% is always excluded
STOPS = (None, -0.05)                     # SPEC_v2 §3.4
TARGETS = (None, 0.03)
HOLDS = (1, 3, 5)
EXPOSURE = 0.5                            # SPEC_v2 §4: new exposure = 50% of equity (cash-capped)
MAX_NAMES = 25                            # per panic day, largest trailing-20d dollar volume first
MAX_POSITIONS = 50
MIN_PANIC_DAYS_IS = 15                    # SPEC_v2 §6 (i)
MIN_PANIC_DAYS_OOS = 6
CONTROL_N = 100                           # SPEC_v2 §6 (vii)
SEED = 20261101
CUMULATIVE_DENOM = 72 + 48                # SPEC_v2 §6 (viii): reported denominator (v1 72 + v2 48)
VOL_BINS = 10                             # control B: volatility deciles
VOL_WINDOW = 20

EPS = R.DROP_EPS                          # float tolerance on -x.000% thresholds (same convention as v1)
COST = R.COST_PRIMARY
REASONS = R.REASONS
PF_CAP = R.PF_CAP
SPY_NAME = "SPY"
CTRL_IDS = {"A": 1, "B": 2, "C": 3}


def log(msg: str) -> None:
    print(f"[panic_dip_v2] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Variant grid: 2 panic x 2 mode x 12 exits = 48
# ---------------------------------------------------------------------------

def combos() -> list:
    return [(s, t, h) for s in STOPS for t in TARGETS for h in HOLDS]


def build_variants() -> list:
    return [(p, m, s, t, h) for p in PANICS for m in MODES for (s, t, h) in combos()]


def variant_slug(v) -> str:
    p, m, s, t, h = v
    st = "none" if s is None else f"{abs(s) * 100:g}"
    tg = "none" if t is None else f"{t * 100:g}"
    return f"{p}_{m}_stop{st}_target{tg}_hold{h}"


def stop_label(s) -> str:
    return "none" if s is None else f"{s:.2f}"


def target_label(t) -> str:
    return "none" if t is None else f"{t:.2f}"


def vlabel(v) -> str:
    p, m, s, t, h = v
    st = "없음" if s is None else f"{s * 100:+.0f}%"
    tg = "없음" if t is None else f"{t * 100:+.0f}%"
    return f"{p} / 모드 {m} / STOP {st} / TARGET {tg} / HOLD {h}일"


# ---------------------------------------------------------------------------
# Loading (v1 loaders) + alias resolution
# ---------------------------------------------------------------------------

def load_resolved_aliases(data_dir: str) -> dict:
    p = os.path.join(data_dir, RESOLVED_FILE)
    if not os.path.exists(p):
        return {}
    try:
        with open(p) as f:
            d = json.load(f)
        return {str(k): str(v) for k, v in d.items()}
    except Exception as e:  # noqa: BLE001
        log(f"WARNING: could not read {p}: {e} (no aliases applied)")
        return {}


def resolve_name(t: str, resolved: dict) -> str:
    """Follow old -> new (a chain is followed to its end; a cycle stops at the first repeat)."""
    seen = set()
    while t in resolved and t not in seen:
        seen.add(t)
        t = resolved[t]
    return t


def apply_aliases(membership: dict, resolved: dict) -> dict:
    """Series-name -> merged membership intervals. Every ticker's intervals go to the series of its resolved name.
    The panel then has ONE column per company and `member` is the union of the intervals, so a date on which two
    tickers of the same company were both members is one cell (SPEC_v2 §2: counted once)."""
    out = {}
    for t, iv in membership.items():
        out.setdefault(resolve_name(t, resolved), []).extend(iv)
    return out


def add_extras(P) -> None:
    """Trailing-window arrays the v1 Panel lacks, both over bars t-20..t-1 of the same ticker (strictly before t):
    dv20 = mean(close*volume)  (the quantity v1's liquidity guard tests; ranks the 25-name cut),
    vol20 = std(daily return, ddof=1)  (control B's volatility; NaN until 20 valid returns exist)."""
    W = R.LIQ_WINDOW
    dv20 = np.full(P.N, np.nan)
    vol20 = np.full(P.N, np.nan)
    for j in range(len(P.tickers)):
        lo, hi = int(P.off[j]), int(P.off[j + 1])
        n = hi - lo
        if n <= W:
            continue
        dv = P.close[lo:hi] * P.volume[lo:hi]
        dv20[lo + W:hi] = sliding_window_view(dv, W).mean(axis=1)[: n - W]
        r = P.ret1[lo:hi]
        vol20[lo + W:hi] = sliding_window_view(r, W).std(axis=1, ddof=1)[: n - W]
    P.dv20, P.vol20 = dv20, vol20
    ncal = len(P.cal)
    spy_cell = np.full(ncal, -1, np.int64)
    P.spy_j = P.tickers.index(SPY_NAME) if SPY_NAME in P.tickers else -1
    if P.spy_j >= 0:
        lo, hi = int(P.off[P.spy_j]), int(P.off[P.spy_j + 1])
        cells = np.arange(lo, hi)
        spy_cell[P.gidx[lo:hi]] = cells
        P.spy_cells = cells
    else:
        P.spy_cells = np.empty(0, np.int64)
    P.spy_cell = spy_cell


def build_panel_v2(series: dict, membership: dict, splits: dict, spy_df) -> "R.Panel":
    """v1 build_panel + SPY as an extra, never-member column + v2 extras."""
    ser, mem, spl = dict(series), dict(membership), dict(splits)
    if spy_df is not None:
        ser[SPY_NAME], mem[SPY_NAME], spl[SPY_NAME] = spy_df, [], []
    P = R.build_panel(ser, mem, spl)
    add_extras(P)
    return P


# ---------------------------------------------------------------------------
# Market level: r_spy, breadth, panic days (SPEC_v2 §3.1)
# ---------------------------------------------------------------------------

def day_stats(P, lo: int, hi: int) -> dict:
    """Per calendar day (arrays of length ncal; only days lo..hi inclusive are filled, others NaN / False).
    breadth = share of member cells WITH a valid close-to-close return that day (previous bar <= 5 calendar days
    earlier, not a split day) whose own return is <= -2%."""
    ncal = len(P.cal)
    in_win = (P.gidx >= lo) & (P.gidx <= hi)
    valid = P.member & P.gap_ok & np.isfinite(P.ret1) & ~P.split_excl & in_win
    n_tot = np.bincount(P.gidx[valid], minlength=ncal)
    down = valid & (P.ret1 <= BREADTH_STOCK_R + EPS)
    n_down = np.bincount(P.gidx[down], minlength=ncal)
    breadth = np.full(ncal, np.nan)
    pos = n_tot > 0
    breadth[pos] = n_down[pos] / n_tot[pos]
    r_spy = np.full(ncal, np.nan)
    cell = P.spy_cell
    gs = np.arange(lo, hi + 1)
    c = cell[gs]
    ok = c >= 0
    ok[ok] = P.gap_ok[c[ok]] & np.isfinite(P.ret1[c[ok]])
    r_spy[gs[ok]] = P.ret1[c[ok]]
    wide = pos & (2 * n_down >= n_tot)            # breadth >= 50% in exact integer arithmetic
    out = {"lo": lo, "hi": hi, "n_member_valid": n_tot, "n_down": n_down, "breadth": breadth, "r_spy": r_spy}
    for name, thr in PANIC_SPY_R.items():
        out[name] = wide & (r_spy <= thr + EPS)   # NaN r_spy -> False
    return out


def spy_forward(P, g: int, k: int) -> float:
    """SPY close[t+k]/close[t]-1 for trading-day horizon k (NaN when SPY has no complete t+1..t+k)."""
    c = int(P.spy_cell[g])
    if c < 0 or not P.fut_ok[k][c]:
        return float("nan")
    return float(P.close[c + k] / P.close[c] - 1.0)


# ---------------------------------------------------------------------------
# Candidate sets
# ---------------------------------------------------------------------------

class CandSet:
    """cells: flat cell indices ordered by (entry day, larger trailing dollar volume first, flat index);
    g: calendar index of each; rank: 0-based position within its day (the SPEC_v2 §4 liquidity ordering)."""
    __slots__ = ("cells", "g", "rank")

    def __init__(self, cells, g, rank):
        self.cells, self.g, self.rank = cells, g, rank


def _ranks_within_day(g_sorted: np.ndarray) -> np.ndarray:
    n = len(g_sorted)
    if n == 0:
        return np.zeros(0, np.int64)
    start = np.empty(n, bool)
    start[0] = True
    start[1:] = g_sorted[1:] != g_sorted[:-1]
    first = np.flatnonzero(start)
    grp = np.cumsum(start) - 1
    return np.arange(n) - first[grp]


def _order_by_liquidity(P, cells: np.ndarray) -> CandSet:
    if len(cells) == 0:
        return CandSet(cells.astype(np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64))
    g = P.gidx[cells].astype(np.int64)
    order = np.lexsort((cells, -P.dv20[cells], g))      # primary: day; then larger dollar volume; then flat index
    cells, g = cells[order], g[order]
    return CandSet(cells, g, _ranks_within_day(g))


def mode_s_filter(P, ds: dict, panic: str, lo: int, hi: int):
    """SPEC_v2 §3.2 for the panic days of `panic` inside [lo, hi]. Returns (eligible cells, masks) where eligible =
    member & liquid & not a split day & valid return, on a panic day; masks = (ok_market, ok_3x, ok_cap) per eligible
    cell. A cell is a candidate iff all three hold: r_i <= r_spy, r_i >= 3 r_spy, r_i > -20%."""
    base = P.member & P.liq_ok & ~P.split_excl & P.gap_ok & np.isfinite(P.ret1)
    flag = ds[panic]
    cells = np.flatnonzero(base & flag[P.gidx] & (P.gidx >= lo) & (P.gidx <= hi))
    rs = ds["r_spy"][P.gidx[cells]]
    r = P.ret1[cells]
    ok_mkt = r <= rs + EPS
    ok_3x = r >= S_MULT * rs - EPS
    ok_cap = r > HARD_CAP + EPS
    return cells, (ok_mkt, ok_3x, ok_cap)


def candset_s(P, ds: dict, panic: str, lo: int, hi: int) -> CandSet:
    cells, (a, b, c) = mode_s_filter(P, ds, panic, lo, hi)
    return _order_by_liquidity(P, cells[a & b & c])


def candset_i(P, ds: dict, panic: str, lo: int, hi: int) -> CandSet:
    gs = np.flatnonzero(ds[panic])
    gs = gs[(gs >= lo) & (gs <= hi)]
    cells = P.spy_cell[gs]
    cells = cells[cells >= 0]
    g = P.gidx[cells].astype(np.int64)
    return CandSet(cells.astype(np.int64), g, np.zeros(len(cells), np.int64))


def build_candsets(P, ds: dict, bounds: dict, pnames) -> dict:
    out = {}
    for panic in PANICS:
        for pname in pnames:
            lo, hi = bounds[pname]
            out[(panic, "S", pname)] = candset_s(P, ds, panic, lo, hi)
            out[(panic, "I", pname)] = candset_i(P, ds, panic, lo, hi)
    return out


# ---------------------------------------------------------------------------
# Exits (v1 kernel; TARGET None = +inf)
# ---------------------------------------------------------------------------

def compute_exit_arrays_v2(P, stop, target, hold: int):
    """v1's compute_exit_arrays with TARGET None implemented as target = +inf (high >= inf is never true, so only
    gap / stop / time exits can occur). v1 is not modified."""
    return R.compute_exit_arrays(P, stop, math.inf if target is None else target, hold)


def compute_all_exits_v2(P) -> dict:
    return {c: compute_exit_arrays_v2(P, *c) for c in combos()}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def day_series(P, X, sel: np.ndarray, cost: float = COST):
    """Per panic day: mean NET return of that day's trades (all trades, trade-level). -> (days, means)."""
    if len(sel) == 0:
        return np.zeros(0, np.int64), np.zeros(0)
    net = X.ret[sel].astype(np.float64) - 2.0 * cost
    days, inv = np.unique(P.gidx[sel], return_inverse=True)
    return days, np.bincount(inv, weights=net) / np.bincount(inv)


def day_stats_row(days, means) -> dict:
    if len(means) == 0:
        return {"day_mean_ret": float("nan"), "day_median_ret": float("nan"), "day_frac_pos": float("nan")}
    return {"day_mean_ret": float(means.mean()), "day_median_ret": float(np.median(means)),
            "day_frac_pos": float((means > 0).mean())}


def simulate_portfolio_v2(P, sel: np.ndarray, rank: np.ndarray, X, day_lo: int, day_hi: int,
                          cost: float = COST, exposure: float = EXPOSURE, max_names: int = MAX_NAMES,
                          max_pos: int = MAX_POSITIONS) -> dict:
    """SPEC_v2 §4 day-by-day equity simulation (accounting identical to v1's simulate_portfolio).
    `sel` = trades ordered by (entry day, rank); `rank` = position within the day's liquidity ordering.
    Per calendar day d (entries only while d <= day_hi; afterwards open positions just run off):
      1. exits scheduled for d are booked: cash += size * (1 + gross - 2*cost);
      2. remaining positions are marked at d's close: size * (1 + close[d]/close[entry] - 1 - cost);
      3. the day's signals (rank < max_names; at most the free slots up to max_pos open positions) share
         total = min(exposure * equity_pre, cash) equally; extra signals are dropped.
    Equity starts at 1.0, cash earns 0, no leverage (total <= cash)."""
    keep = rank < max_names
    sel = sel[keep]
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
    open_pos, exits_at, fills, rets, entry_log = {}, {}, [], [], []
    min_cash = 1.0
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
            n_take = min(e - ptr, max_pos - len(open_pos))
            total = min(exposure * equity_pre, cash)
            if n_take > 0 and total > 0.0:
                size = total / n_take
                cash -= total
                min_cash = min(min_cash, cash)
                entry_log.append((d, total, n_take))
                for pid in range(ptr, ptr + n_take):
                    open_pos[pid] = [size, 0]
                    exits_at.setdefault(ex[pid], []).append(pid)
                    pos_val += size * (1.0 - cost)
                    fills.append(pid)
            ptr = e
        equity = cash + pos_val
        rets.append(equity / e_prev - 1.0)
        e_prev = equity
        d += 1
        if d > day_hi and not open_pos:
            break
    rets_a = np.array(rets, dtype=np.float64)
    eq = np.concatenate([[1.0], np.cumprod(1.0 + rets_a)])
    return {"equity": eq, "rets": rets_a, "n_filled": len(fills), "fills": [sel_l[p] for p in fills],
            "entry_log": entry_log, "min_cash": min_cash,
            "total_return": float(eq[-1] - 1.0), "mdd": R.max_drawdown(eq), "sharpe": R.sharpe_daily(rets_a)}


def evaluate_period(P, X, cs: CandSet, bounds: tuple, period_iso: tuple, spy_df, n_panic_days: int,
                    cost: float = COST):
    """Everything one grid row needs for one (variant, period) except the control columns."""
    ok = X.code[cs.cells] >= 0                 # exit determinable (enough future bars)
    sel, rk = cs.cells[ok], cs.rank[ok]
    row = R.trade_stats(P, X, sel, cost)
    days, means = day_series(P, X, sel, cost)
    row.update(day_stats_row(days, means))
    row.update(n_panic_days=int(n_panic_days), n_trade_days=int(len(days)))
    sim = simulate_portfolio_v2(P, sel, rk, X, bounds[0], bounds[1], cost)
    row.update(total_return=sim["total_return"], cagr=R.cagr_of(sim["total_return"], *period_iso),
               mdd=sim["mdd"], sharpe=sim["sharpe"], n_filled=sim["n_filled"], n_skipped=int(len(sel) - sim["n_filled"]),
               spy_bh_mdd=R.spy_buyhold_mdd(spy_df, *period_iso))
    return row, sel, rk, sim


# ---------------------------------------------------------------------------
# Controls (SPEC_v2 §6 (vii)) — trade-level PF only
# ---------------------------------------------------------------------------

def pctl(values, q: float) -> float:
    v = np.asarray(values, dtype=np.float64)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan")
    p = float(np.percentile(np.minimum(v, PF_CAP), q))
    return float("inf") if p >= PF_CAP else p


class Pools:
    """Per (HOLD, period) pools of valid (day, member) cells for controls A and B, cached.
    A pool cell is a member, passes the liquidity guard, is not a split day and has a determinable exit — i.e. every
    v1-style placebo-eligible cell; NO own-drop / market-relative filter."""

    def __init__(self, P, bounds: dict):
        self.P, self.bounds = P, bounds
        self.base = P.member & P.liq_ok & ~P.split_excl
        self._day, self._bin = {}, {}

    def _mask(self, hold: int, pname: str):
        lo, hi = self.bounds[pname]
        return self.base & self.P.fut_ok[hold] & (self.P.gidx >= lo) & (self.P.gidx <= hi)

    def by_day(self, hold: int, pname: str):
        key = (hold, pname)
        if key not in self._day:
            cells = np.flatnonzero(self._mask(hold, pname))
            cells = cells[np.argsort(self.P.gidx[cells], kind="stable")]
            self._day[key] = (cells, self.P.gidx[cells].astype(np.int64))
        return self._day[key]

    def by_vol_bin(self, hold: int, pname: str):
        """(edges, [cells of decile 0..9]) — deciles of the 20-day volatility over the period's valid cells."""
        key = (hold, pname)
        if key not in self._bin:
            cells = np.flatnonzero(self._mask(hold, pname) & np.isfinite(self.P.vol20))
            if len(cells) == 0:
                self._bin[key] = None
            else:
                v = self.P.vol20[cells]
                edges = np.quantile(v, np.linspace(0.0, 1.0, VOL_BINS + 1)[1:-1])
                b = np.searchsorted(edges, v, side="right")
                self._bin[key] = (edges, [cells[b == k] for k in range(VOL_BINS)])
        return self._bin[key]


def control_a_spec(P, pools: Pools, hold: int, pname: str, sel: np.ndarray):
    """Same panic days, same number of trades per day. -> (spec [(start, stop, k)], pool cells ordered by day)."""
    cells, g = pools.by_day(hold, pname)
    days, counts = np.unique(P.gidx[sel], return_counts=True)
    starts = np.searchsorted(g, days, side="left")
    stops = np.searchsorted(g, days, side="right")
    return list(zip(starts.tolist(), stops.tolist(), counts.tolist())), cells


def control_a_draw(rng, spec, cells) -> np.ndarray:
    parts = []
    for a, b, k in spec:
        n = b - a
        if n <= 0 or k <= 0:
            continue
        parts.append(cells[a:b][rng.choice(n, size=min(k, n), replace=False)])
    return np.concatenate(parts) if parts else np.empty(0, np.int64)


def control_b_spec(P, pools: Pools, hold: int, pname: str, sel: np.ndarray):
    """Decile histogram of the real trades' 20-day volatility (deciles of the period's valid cells)."""
    bp = pools.by_vol_bin(hold, pname)
    if bp is None:
        return None
    edges, bin_cells = bp
    v = P.vol20[sel]
    v = v[np.isfinite(v)]
    counts = np.bincount(np.searchsorted(edges, v, side="right"), minlength=VOL_BINS)
    return counts, bin_cells


def control_b_draw(rng, spec) -> np.ndarray:
    if spec is None:
        return np.empty(0, np.int64)
    counts, bin_cells = spec
    parts = []
    for k in range(VOL_BINS):
        n, want = len(bin_cells[k]), int(counts[k])
        if n > 0 and want > 0:
            parts.append(bin_cells[k][rng.choice(n, size=min(want, n), replace=False)])
    return np.concatenate(parts) if parts else np.empty(0, np.int64)


def control_c_spec(P, hold: int, bounds: tuple, panic_flag: np.ndarray, n: int):
    """SPY on non-panic days of the period (determinable exit), `n` draws."""
    c = P.spy_cells
    if len(c) == 0:
        return None
    g = P.gidx[c]
    m = (g >= bounds[0]) & (g <= bounds[1]) & P.fut_ok[hold][c] & ~panic_flag[g]
    return c[m], int(n)


def control_c_draw(rng, spec) -> np.ndarray:
    if spec is None:
        return np.empty(0, np.int64)
    pool, n = spec
    if len(pool) == 0 or n <= 0:
        return np.empty(0, np.int64)
    return pool[rng.choice(len(pool), size=min(n, len(pool)), replace=False)]


def control_pfs(draw, ret: np.ndarray, runs: int, rng, cost: float = COST) -> np.ndarray:
    out = np.full(runs, np.nan)
    for r in range(runs):
        cells = draw(rng)
        if len(cells):
            out[r] = R.profit_factor(ret[cells].astype(np.float64) - 2.0 * cost)
    return out


# ---------------------------------------------------------------------------
# PASS logic (SPEC_v2 §6)
# ---------------------------------------------------------------------------

def _ge(x, thr) -> bool:
    return R._ge(x, thr)


def _le(x, thr) -> bool:
    return R._le(x, thr)


def ok_i(mode: str, is_row: dict, oos_row: dict) -> bool:
    """(i) IS trades >= 100 (mode S only) AND IS panic days >= 15 AND OOS panic days >= 6.
    A 'panic day' here is a panic day that produced at least one evaluable trade (n_trade_days) — the conservative
    reading: a panic day with no tradable candidate is not an independent sample."""
    days = is_row["n_trade_days"] >= MIN_PANIC_DAYS_IS and oos_row["n_trade_days"] >= MIN_PANIC_DAYS_OOS
    trades = True if mode == "I" else is_row["n_trades"] >= R.MIN_TRADES_IS
    return bool(days and trades)


def pass_flags_v2(v, is_row: dict, oos_row: dict, beats: dict) -> dict:
    """beats = {"A": {"IS": bool, "OOS": bool}, "B": {...}, "C": {...}} (only the controls that apply to the mode)."""
    mode = v[1]
    spy, mdd = is_row["spy_bh_mdd"], is_row["mdd"]
    f = {
        "ok_i": ok_i(mode, is_row, oos_row),
        "ok_ii_pf_is": _ge(is_row["pf"], R.MIN_PF_IS),
        "ok_iii_mdd": _le(mdd, R.MAX_MDD_IS),
        "ok_iv_sharpe": _ge(is_row["sharpe"], R.MIN_SHARPE_IS),
        "ok_v_vs_spy": bool(not math.isnan(spy) and not math.isnan(mdd) and mdd < spy),
        "ok_vi_pf_oos": _ge(oos_row["pf"], R.MIN_PF_OOS),
    }
    need = ("A", "B") if mode == "S" else ("C",)
    f["ok_vii_controls"] = bool(all(beats.get(c, {}).get(p, False) for c in need for p in ("IS", "OOS")))
    f["PASS"] = all(f.values())
    f["pass_trade_level"] = bool(f["ok_i"] and f["ok_ii_pf_is"] and f["ok_vi_pf_oos"])     # SPEC_v2 §6 (viii)
    return f


def conclusion(n_pass: int, real_count: int, p95_count: float) -> str:
    """SPEC_v2 §6 (viii): EDGE only if (a) some variant PASSes AND (b) the real trade-level PASS count exceeds the
    95th percentile of the control runs' counts. Otherwise NO EVIDENCE."""
    return "EDGE" if (n_pass >= 1 and real_count > p95_count) else "NO EVIDENCE"


# ---------------------------------------------------------------------------
# Study driver
# ---------------------------------------------------------------------------

def run_study_v2(P, spy_df, periods: dict, control_n: int, seed: int, verbose: bool = True) -> dict:
    t0 = time.time()
    bounds = R.period_bounds(P, periods)
    variants = build_variants()
    X = compute_all_exits_v2(P)
    if verbose:
        log(f"exit arrays for {len(X)} (STOP,TARGET,HOLD) rules done in {time.time() - t0:.1f}s (N cells={P.N:,})")

    windows = []                                   # every [lo, hi] calendar window market stats were computed for
    lo_all, hi_all = bounds["IS"][0], bounds["OOS"][1]
    ds = day_stats(P, lo_all, hi_all)
    windows.append((lo_all, hi_all))
    pnames = ("IS", "OOS")
    cs = build_candsets(P, ds, bounds, pnames)
    n_panic = {(p, pn): int(ds[p][bounds[pn][0]: bounds[pn][1] + 1].sum()) for p in PANICS for pn in pnames}
    if verbose:
        log("panic days IS/OOS: " + ", ".join(f"{p} {n_panic[(p, 'IS')]}/{n_panic[(p, 'OOS')]}" for p in PANICS))

    rows = {pn: {} for pn in pnames}
    sels = {pn: {} for pn in pnames}
    ranks = {pn: {} for pn in pnames}
    sims = {pn: {} for pn in pnames}
    for v in variants:
        panic, mode, stop, target, hold = v
        Xv = X[(stop, target, hold)]
        for pn in pnames:
            r, sel, rk, sim = evaluate_period(P, Xv, cs[(panic, mode, pn)], bounds[pn], periods[pn], spy_df,
                                              n_panic[(panic, pn)])
            rows[pn][v], sels[pn][v], ranks[pn][v], sims[pn][v] = r, sel, rk, sim
    if verbose:
        log(f"{len(variants)} variants x (IS, OOS) evaluated incl. portfolio simulation in {time.time() - t0:.1f}s")

    # ---- controls ----
    pools = Pools(P, bounds)
    ctrl = {pn: {} for pn in pnames}               # ctrl[pn][v] = {"A": arr, "B": arr} | {"C": arr}
    for vi, v in enumerate(variants):
        panic, mode, stop, target, hold = v
        ret = X[(stop, target, hold)].ret
        for pi, pn in enumerate(pnames):
            sel = sels[pn][v]
            d = {}
            if mode == "S":
                spec_a, cells_a = control_a_spec(P, pools, hold, pn, sel)
                rng = np.random.default_rng([seed, CTRL_IDS["A"], vi, pi])
                d["A"] = control_pfs(lambda g_, s=spec_a, c=cells_a: control_a_draw(g_, s, c), ret, control_n, rng)
                spec_b = control_b_spec(P, pools, hold, pn, sel)
                rng = np.random.default_rng([seed, CTRL_IDS["B"], vi, pi])
                d["B"] = control_pfs(lambda g_, s=spec_b: control_b_draw(g_, s), ret, control_n, rng)
            else:
                spec_c = control_c_spec(P, hold, bounds[pn], ds[panic], len(sel))
                rng = np.random.default_rng([seed, CTRL_IDS["C"], vi, pi])
                d["C"] = control_pfs(lambda g_, s=spec_c: control_c_draw(g_, s), ret, control_n, rng)
            ctrl[pn][v] = d
    if verbose:
        log(f"controls: {len(variants)} variants x 2 periods x {control_n} runs done in {time.time() - t0:.1f}s")

    p95 = {pn: {v: {c: pctl(a, 95) for c, a in ctrl[pn][v].items()} for v in variants} for pn in pnames}
    beats = {pn: {v: {} for v in variants} for pn in pnames}
    for pn in pnames:
        for v in variants:
            pf = rows[pn][v]["pf"]
            for c in ctrl[pn][v]:
                q = p95[pn][v][c]
                beats[pn][v][c] = bool(not math.isnan(pf) and not math.isnan(q) and pf > q)
    flags = {}
    for v in variants:
        b = {c: {pn: beats[pn][v][c] for pn in pnames} for c in ctrl["IS"][v]}
        flags[v] = pass_flags_v2(v, rows["IS"][v], rows["OOS"][v], b)

    # ---- (viii): real trade-level PASS count vs the same count in each control run ----
    real_count = int(sum(flags[v]["pass_trade_level"] for v in variants))
    ctrl_counts = []
    for r in range(control_n):
        cnt = 0
        for v in variants:
            cc = "A" if v[1] == "S" else "C"
            if (flags[v]["ok_i"] and _ge(ctrl["IS"][v][cc][r], R.MIN_PF_IS) and _ge(ctrl["OOS"][v][cc][r], R.MIN_PF_OOS)):
                cnt += 1
        ctrl_counts.append(cnt)
    p95_count = float(np.percentile(ctrl_counts, R.PLACEBO_PERCENTILE)) if ctrl_counts else float("nan")
    passers = [v for v in variants if flags[v]["PASS"]]
    verdict = conclusion(len(passers), real_count, p95_count)

    # ---- HOLDOUT (SPEC_v2 §5): only if some variant passed; best IS PF; once; numbers only ----
    holdout = None
    if passers:
        best = max(passers, key=lambda v: (min(rows["IS"][v]["pf"], PF_CAP), -variants.index(v)))
        panic, mode, stop, target, hold = best
        lo_h, hi_h = bounds["HOLDOUT"]
        ds_h = day_stats(P, lo_h, hi_h)
        windows.append((lo_h, hi_h))
        cs_h = candset_s(P, ds_h, panic, lo_h, hi_h) if mode == "S" else candset_i(P, ds_h, panic, lo_h, hi_h)
        nph = int(ds_h[panic][lo_h: hi_h + 1].sum())
        r, sel, rk, sim = evaluate_period(P, X[(stop, target, hold)], cs_h, bounds["HOLDOUT"], periods["HOLDOUT"],
                                          spy_df, nph)
        holdout = {"variant": list(best), "variant_slug": variant_slug(best), "period": list(periods["HOLDOUT"]),
                   "metrics": r, "sel": sel, "rank": rk, "sim": sim,
                   "note": "SPEC_v2 has no pass threshold for HOLDOUT; reported for reference only."}
        if verbose:
            log(f"HOLDOUT computed once for {variant_slug(best)}: n={r['n_trades']} pf={r['pf']:.3f}")
    elif verbose:
        log("HOLDOUT stays sealed: no variant passed SPEC_v2 §6 (i)-(vii)")

    return dict(P=P, X=X, ds=ds, cs=cs, bounds=bounds, periods=periods, variants=variants, rows=rows, sels=sels,
                ranks=ranks, sims=sims, ctrl=ctrl, p95=p95, beats=beats, flags=flags, control_n=control_n, seed=seed,
                n_panic=n_panic, real_count=real_count, ctrl_counts=ctrl_counts, p95_count=p95_count,
                passers=passers, verdict=verdict, holdout=holdout, windows=windows, spy=spy_df,
                runtime=time.time() - t0)


# ---------------------------------------------------------------------------
# panic_days.csv (SPEC_v2 §7): every P2 day of IS + OOS
# ---------------------------------------------------------------------------

PANIC_DAY_COLS = ["date", "period", "r_spy", "breadth", "n_member_valid", "is_P3", "n_eligible", "n_candidates_S",
                  "n_excluded_less_than_market", "n_excluded_over_3x", "n_excluded_cap20",
                  "spy_fwd_1", "spy_fwd_3", "spy_fwd_5", "mean_1d_net_ret_of_candidates"]


def panic_day_table(res: dict) -> pd.DataFrame:
    """One row per P2 day in IS+OOS (never HOLDOUT). Exclusion counts partition the ELIGIBLE cells (member, liquid, not
    a split day, valid return) of that day: candidate + less_than_market + over_3x + cap20 = eligible, attributed in
    this order: (1) r_i > r_spy; (2) r_i < 3 r_spy; (3) r_i <= -20% (only cells that passed 1 and 2)."""
    P, ds, bounds = res["P"], res["ds"], res["bounds"]
    lo, hi = bounds["IS"][0], bounds["OOS"][1]
    cells, (a, b, c) = mode_s_filter(P, ds, "P2", lo, hi)
    g = P.gidx[cells].astype(np.int64)
    ncal = len(P.cal)
    cnt = lambda m: np.bincount(g[m], minlength=ncal)  # noqa: E731
    n_el = cnt(np.ones(len(cells), bool))
    n_less = cnt(~a)
    n_3x = cnt(a & ~b)
    n_cap = cnt(a & b & ~c)
    cand = a & b & c
    n_cand = cnt(cand)
    X1 = res["X"][(None, None, 1)]
    ok1 = cand & (X1.code[cells] >= 0)
    net1 = X1.ret[cells].astype(np.float64) - 2.0 * COST
    s1 = np.bincount(g[ok1], weights=net1[ok1], minlength=ncal)
    c1 = np.bincount(g[ok1], minlength=ncal)
    recs = []
    for gi in np.flatnonzero(ds["P2"]):
        if gi < lo or gi > hi:
            continue
        recs.append({
            "date": P.iso(gi), "period": "IS" if gi <= bounds["IS"][1] else "OOS",
            "r_spy": float(ds["r_spy"][gi]), "breadth": float(ds["breadth"][gi]),
            "n_member_valid": int(ds["n_member_valid"][gi]), "is_P3": bool(ds["P3"][gi]),
            "n_eligible": int(n_el[gi]), "n_candidates_S": int(n_cand[gi]),
            "n_excluded_less_than_market": int(n_less[gi]), "n_excluded_over_3x": int(n_3x[gi]),
            "n_excluded_cap20": int(n_cap[gi]),
            "spy_fwd_1": spy_forward(P, int(gi), 1), "spy_fwd_3": spy_forward(P, int(gi), 3),
            "spy_fwd_5": spy_forward(P, int(gi), 5),
            "mean_1d_net_ret_of_candidates": float(s1[gi] / c1[gi]) if c1[gi] > 0 else float("nan"),
        })
    return pd.DataFrame(recs, columns=PANIC_DAY_COLS)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

GRID_METRIC_COLS = ["n_trades", "n_panic_days", "n_trade_days", "pf", "pf_cost005", "pf_cost020", "win_rate",
                    "mean_ret", "median_ret", "mean_hold_days", "exit_target", "exit_stop", "exit_gap", "exit_time",
                    "day_mean_ret", "day_median_ret", "day_frac_pos", "total_return", "cagr", "mdd", "sharpe",
                    "spy_bh_mdd"]
FLAG_COLS = ["ok_i", "ok_ii_pf_is", "ok_iii_mdd", "ok_iv_sharpe", "ok_v_vs_spy", "ok_vi_pf_oos", "ok_vii_controls",
             "PASS", "pass_trade_level"]


def grid_dataframe(res: dict, pname: str) -> pd.DataFrame:
    recs = []
    for v in res["variants"]:
        panic, mode, stop, target, hold = v
        r = res["rows"][pname][v]
        rec = {"variant": variant_slug(v), "panic": panic, "mode": mode, "stop": stop_label(stop),
               "target": target_label(target), "hold": hold}
        rec.update({c: r[c] for c in GRID_METRIC_COLS})
        for c in ("A", "B", "C"):
            rec[f"ctrl{c}_pf_p95"] = res["p95"][pname][v].get(c, float("nan"))
        for c in ("A", "B", "C"):
            rec[f"beats_{c}"] = res["beats"][pname][v].get(c, np.nan)
        rec.update({c: res["flags"][v][c] for c in FLAG_COLS})
        rec["n_filled"] = r["n_filled"]
        rec["n_skipped"] = r["n_skipped"]
        recs.append(rec)
    return pd.DataFrame(recs)


def controls_dataframe(res: dict) -> pd.DataFrame:
    recs = []
    for pn in ("IS", "OOS"):
        for v in res["variants"]:
            panic, mode, stop, target, hold = v
            for c, arr in res["ctrl"][pn][v].items():
                a = np.asarray(arr, dtype=np.float64)
                fin = a[~np.isnan(a)]
                recs.append({"variant": variant_slug(v), "panic": panic, "mode": mode, "stop": stop_label(stop),
                             "target": target_label(target), "hold": hold, "period": pn, "control": c,
                             "n_trades": res["rows"][pn][v]["n_trades"], "runs": int(len(a)),
                             "pf_p05": pctl(fin, 5), "pf_p50": pctl(fin, 50), "pf_p95": res["p95"][pn][v][c],
                             "real_pf": res["rows"][pn][v]["pf"], "real_beats_p95": res["beats"][pn][v][c]})
    return pd.DataFrame(recs)


def best_is_variant(res: dict):
    rows, variants = res["rows"]["IS"], res["variants"]
    key = lambda v: (-(min(rows[v]["pf"], PF_CAP) if not math.isnan(rows[v]["pf"]) else -1.0), variants.index(v))  # noqa: E731
    return min(variants, key=key)


TRADE_COLS = ["ticker", "entry_date", "entry_price", "exit_date", "exit_price", "exit_reason", "gross_ret", "net_ret",
              "period", "r_i", "r_spy", "dollar_vol20", "rank_dv", "in_portfolio"]


def trades_dataframe(res: dict, v, periods_to_write: list) -> pd.DataFrame:
    P, X = res["P"], res["X"]
    panic, mode, stop, target, hold = v
    Xv = X[(stop, target, hold)]
    recs = []
    for pn in periods_to_write:
        if pn == "HOLDOUT":
            sel, rk, filled = res["holdout"]["sel"], res["holdout"]["rank"], set(res["holdout"]["sim"]["fills"])
        else:
            sel, rk, filled = res["sels"][pn][v], res["ranks"][pn][v], set(res["sims"][pn][v]["fills"])
        for i, rank in zip(sel.tolist(), rk.tolist()):
            code, k = int(Xv.code[i]), int(Xv.k[i])
            entry = float(P.close[i])
            xp = R.exit_price(P, i, code, k, stop, 0.0 if target is None else target)
            gross = xp / entry - 1.0
            g = int(P.gidx[i])
            recs.append({"ticker": P.tickers[int(P.tick[i])], "entry_date": P.iso(g), "entry_price": entry,
                         "exit_date": P.iso(P.gidx[i + k]), "exit_price": xp, "exit_reason": REASONS[code],
                         "gross_ret": gross, "net_ret": gross - 2.0 * COST, "period": pn,
                         "r_i": float(P.ret1[i]), "r_spy": _r_spy_cell(P, g),
                         "dollar_vol20": float(P.dv20[i]), "rank_dv": int(rank), "in_portfolio": i in filled})
    return pd.DataFrame(recs, columns=TRADE_COLS)


def _r_spy_cell(P, g: int) -> float:
    c = int(P.spy_cell[g])
    return float(P.ret1[c]) if c >= 0 else float("nan")


def _clean_json(o):
    return R._clean_json(o)


def _pf(x) -> str:
    return R._pf(x)


def _pc(x, d=2) -> str:
    return R._pc(x, d)


def _num(x, d=2) -> str:
    return R._num(x, d)


def _isnan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


# ---------------------------------------------------------------------------
# SUMMARY_v2.md (Korean; technical terms stay in English)
# ---------------------------------------------------------------------------

def _coverage_section(cov: dict, load_info: dict, resolved: dict) -> list:
    L = ["## 데이터 복구 결과 (SPEC_v2 §2)", ""]
    if not cov:
        L.append("`data_coverage_v2.json`이 없어 fetch 결과를 인용하지 못했다.")
        L.append("")
        return L
    if "recovered_via_alias" not in cov:
        L.append("- fetch의 복구 통계(`data_coverage_v2.json`의 recovered_via_alias / still_missing)가 없다 — 이번 실행에서 알 수 있는 것만 적는다.")
        L.append(f"- 이번 실행에서 실제 로드된 시리즈: {load_info.get('loaded')}개 (너무 짧아 제외 {load_info.get('skipped')}개), "
                 f"alias로 합쳐진 옛 티커 {len(resolved)}개. SPY 벤치마크: 있음.")
        L.append("- 남는 생존 편향은 v1과 같은 방향 — 결과를 **좋게 부풀리는** 쪽 — 이다. PASS는 의심하고 FAIL은 더 믿는다.")
        L.append("")
        return L
    sm = cov.get("still_missing_counts", {})
    rec = cov.get("recovered_via_alias", [])
    retr = cov.get("retries", {})
    L.append(f"- 요청 종목 {cov.get('requested')}개 · 직접 수신 {cov.get('received_direct')}개.")
    L.append(f"- **개명 복구 (alias rename)**: {len(rec)}개 — 옛 티커 구간에 새 티커 이력을 붙였다"
             + (": " + ", ".join(f"{r['old']}→{r['new']}" for r in rec[:60]) + (" …" if len(rec) > 60 else "") if rec else "."))
    L.append(f"- 재시도로 직접 복구: 404 재시도 {len(retr.get('missing_retried', []))}개 중 "
             f"{len(retr.get('missing_recovered_on_retry', []))}개 복구; short 재시도 {len(retr.get('short_retried', []))}개 중 "
             f"{len(retr.get('short_recovered_on_retry', []))}개 복구.")
    L.append(f"- **복구되지 않은 종목**: 개명인데 실패(rename-failed) {sm.get('rename-failed', 0)}개 · "
             f"인수/파산/상폐(acquired) {sm.get('acquired', 0)}개 · 원인 불명(unknown) {sm.get('unknown', 0)}개.")
    sm_lists = cov.get("still_missing", {})
    for k in ("rename-failed", "acquired", "unknown"):
        lst = sm_lists.get(k, [])
        if lst:
            L.append(f"  - {k}: " + ", ".join(lst[:80]) + (" …" if len(lst) > 80 else ""))
    if cov.get("short"):
        L.append("- 여전히 short(<50행): " + ", ".join(f"{s['ticker']}({s['rows']})" for s in cov["short"]))
    ls = cov.get("late_start_informational", [])
    if ls:
        L.append(f"- 참고(복구에 쓰지 않음): 구성 시작일보다 60일 넘게 늦게 시작하는 데이터 {len(ls)}개 "
                 "(예: 티커가 재사용된 경우) — " + ", ".join(f"{x['ticker']}(+{x['gap_days']}d)" for x in ls[:30]))
    L.append(f"- 이번 실행에서 실제 로드된 시리즈: {load_info.get('loaded')}개 (너무 짧아 제외 {load_info.get('skipped')}개), "
             f"alias로 합쳐진 옛 티커 {len(resolved)}개. SPY 벤치마크: 있음.")
    L.append("- 남는 생존 편향(acquired/unknown 누락)은 v1과 같은 방향 — 결과를 **좋게 부풀리는** 쪽 — 이다. PASS는 의심하고 FAIL은 더 믿는다.")
    L.append("")
    return L


def write_summary(res: dict, out_dir: str, cov: dict, load_info: dict, resolved: dict, run_date: str,
                  synthetic: bool, panic_df: pd.DataFrame) -> None:
    rows, flags, variants = res["rows"], res["flags"], res["variants"]
    n_pass = len(res["passers"])
    cnt = {k: sum(flags[v][k] for v in variants) for k in FLAG_COLS}
    L = []
    L.append(f"# panic_dip v2 — 결과 요약 ({run_date})" + ("  [SYNTHETIC SELF-TEST — 실제 데이터 아님]" if synthetic else ""))
    L.append("")
    L.append("`SPEC_v2.md`(2026-11-01 사전등록)를 `run_panic_dip_v2.py`가 기계적으로 구현해 낸 결과다. 기준·임계값은 명세 그대로이며 "
             "결과를 본 뒤 바꾼 것이 없다. v1(NO EVIDENCE)을 본 뒤 설계한 두 번째 그물이므로 다중검정 비용을 §6 (viii)에 반영했다 "
             "(분모 = 누적 **120**개 = v1 72 + v2 48).")
    L.append("")
    L.append(f"## 전체 결론: **{res['verdict']}**")
    L.append("")
    L.append(f"- 개별 PASS 변형 (SPEC_v2 §6 (i)~(vii) 전부 충족): **{n_pass}개** / v2 48개 (누적 그물 {CUMULATIVE_DENOM}개 중 {n_pass}개)")
    shown = sorted(res["passers"], key=lambda v: -min(rows["IS"][v]["pf"], PF_CAP))[:25]
    for v in shown:
        r = rows["IS"][v]
        L.append(f"  - {vlabel(v)} — IS PF {_pf(r['pf'])}, OOS PF {_pf(rows['OOS'][v]['pf'])}, IS 거래 {r['n_trades']}, "
                 f"IS 패닉일 {r['n_trade_days']}, OOS 패닉일 {rows['OOS'][v]['n_trade_days']}")
    if n_pass > len(shown):
        L.append(f"  - … 외 {n_pass - len(shown)}개 (전체는 `grid_IS.csv`의 PASS 열)")
    cc = res["ctrl_counts"]
    L.append(f"- (viii) 다중검정: 거래 단위 (i)·(ii)·(vi) PASS 수 — 실제 **{res['real_count']}** vs 대조군 {res['control_n']}회의 "
             f"95퍼센타일 **{res['p95_count']:.2f}** (대조군 평균 {np.mean(cc) if cc else float('nan'):.2f}, 최대 {max(cc) if cc else 0}). "
             f"보고 분모는 {CUMULATIVE_DENOM}이지만 대조군 비교 자체는 v2의 48개 변형 위에서 계산했다 "
             "(대조군 실행 r마다: 모드 S는 대조군 A, 모드 I는 대조군 C의 PF로 (ii)·(vi)를 판정하고 실제 (i)이 성립하는 변형만 센다).")
    if res["verdict"] == "EDGE":
        L.append("  - 개별 PASS가 있고 실제 PASS 수가 대조군 95퍼센타일을 넘으므로 명세상 EDGE.")
    elif n_pass == 0:
        L.append("  - 개별 PASS 변형이 하나도 없으므로 NO EVIDENCE.")
    else:
        L.append("  - 개별 PASS가 있어도 실제 PASS 수가 대조군 95퍼센타일 이하이므로 명세상 NO EVIDENCE.")
    L.append("")
    L.append("### 기준별 통과 변형 수 (48 중, 각 기준 단독)")
    L.append("")
    L.append("| (i) 거래·패닉일 수 | (ii) IS PF≥1.3 | (iii) IS MDD≤30% | (iv) IS Sharpe≥0.8 | (v) MDD<SPY | (vi) OOS PF≥1.1 | (vii) 대조군 초과 | 전부(PASS) | 거래 단위 (i)(ii)(vi) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    L.append(f"| {cnt['ok_i']} | {cnt['ok_ii_pf_is']} | {cnt['ok_iii_mdd']} | {cnt['ok_iv_sharpe']} | {cnt['ok_v_vs_spy']} | "
             f"{cnt['ok_vi_pf_oos']} | {cnt['ok_vii_controls']} | {cnt['PASS']} | {cnt['pass_trade_level']} |")
    L.append("")

    # ---- panic days ----
    P = res["P"]
    n2 = {pn: res["n_panic"][("P2", pn)] for pn in ("IS", "OOS")}
    n3 = {pn: res["n_panic"][("P3", pn)] for pn in ("IS", "OOS")}
    L.append("## 시장 패닉일 (SPEC_v2 §3.1)")
    L.append("")
    L.append(f"- P2 (SPY ≤ −2% 그리고 breadth ≥ 50%): IS {n2['IS']}일 · OOS {n2['OOS']}일. "
             f"P3 (SPY ≤ −3% 그리고 breadth ≥ 50%): IS {n3['IS']}일 · OOS {n3['OOS']}일. (HOLDOUT 날짜는 이 표·산출물에 넣지 않는다.)")
    if len(panic_df):
        ym = pd.to_datetime(panic_df["date"]).dt.to_period("M").value_counts()
        top = ", ".join(f"{k} {v}일" for k, v in ym.head(5).items())
        L.append(f"- 패닉일은 뭉쳐 있다: 가장 많은 달 — {top}. 독립 표본은 거래가 아니라 이 날들이다.")
    L.append("")
    L.append("### 전체 P2 패닉일 (IS+OOS) — 사람이 읽는 핵심 표")
    L.append("")
    L.append("| 날짜 | 구간 | SPY | breadth | P3 | 후보 수(S) | 제외: 시장보다 덜 빠짐 | 제외: 3배 초과 | 제외: −20% 캡 | SPY +1d | +3d | +5d | 후보 1일 평균 net |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in panic_df.itertuples():
        L.append(f"| {r.date} | {r.period} | {_pc(r.r_spy)} | {_pc(r.breadth, 0)} | {'예' if r.is_P3 else ''} | {r.n_candidates_S} | "
                 f"{r.n_excluded_less_than_market} | {r.n_excluded_over_3x} | {r.n_excluded_cap20} | {_pc(r.spy_fwd_1)} | "
                 f"{_pc(r.spy_fwd_3)} | {_pc(r.spy_fwd_5)} | {_pc(r.mean_1d_net_ret_of_candidates)} |")
    L.append("")

    # ---- S vs I ----
    L.append("## 모드 S (종목 필터) vs 모드 I (SPY) — 같은 청산 규칙")
    L.append("")
    L.append("S에서 종목을 고르는 층이 I(SPY 그대로)보다 나아야 의미가 있다. PF는 거래 단위(비용 0.10%), "
             "'일 평균'은 패닉일 단위 평균 net 수익이다. 모드 I의 거래 수 = 패닉일 수.")
    L.append("")
    L.append("| 패닉 | STOP | TARGET | HOLD | S IS PF (거래/패닉일) | I IS PF (패닉일) | S OOS PF (거래/패닉일) | I OOS PF (패닉일) | S IS 일평균 | I IS 일평균 | S OOS 일평균 | I OOS 일평균 | S PASS | I PASS |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for panic in PANICS:
        for (s, t, h) in combos():
            vs, vi_ = (panic, "S", s, t, h), (panic, "I", s, t, h)
            a, b, c, d = rows["IS"][vs], rows["IS"][vi_], rows["OOS"][vs], rows["OOS"][vi_]
            L.append(f"| {panic} | {'없음' if s is None else f'{s * 100:+.0f}%'} | {'없음' if t is None else f'{t * 100:+.0f}%'} | {h} | "
                     f"{_pf(a['pf'])} ({a['n_trades']}/{a['n_trade_days']}) | {_pf(b['pf'])} ({b['n_trade_days']}) | "
                     f"{_pf(c['pf'])} ({c['n_trades']}/{c['n_trade_days']}) | {_pf(d['pf'])} ({d['n_trade_days']}) | "
                     f"{_pc(a['day_mean_ret'])} | {_pc(b['day_mean_ret'])} | {_pc(c['day_mean_ret'])} | {_pc(d['day_mean_ret'])} | "
                     f"{'PASS' if flags[vs]['PASS'] else '-'} | {'PASS' if flags[vi_]['PASS'] else '-'} |")
    L.append("")

    # ---- top 10 ----
    ranked = sorted(variants, key=lambda v: -(min(rows['IS'][v]['pf'], PF_CAP) if not _isnan(rows['IS'][v]['pf']) else -1.0))
    L.append("## IS PF 상위 10 변형 (참고용 — 선택 기준이 아님)")
    L.append("")
    L.append("| 변형 | IS 거래 | IS 패닉일 | IS PF | IS MDD | IS Sharpe | OOS 거래 | OOS 패닉일 | OOS PF | 대조군 p95 (IS) | 대조군 p95 (OOS) | PASS |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")

    def cp(pn, v):
        d = res["p95"][pn][v]
        return " / ".join(f"{c}:{_pf(d[c])}" for c in sorted(d))
    for v in ranked[:10]:
        a, b = rows["IS"][v], rows["OOS"][v]
        L.append(f"| {vlabel(v)} | {a['n_trades']} | {a['n_trade_days']} | {_pf(a['pf'])} | {_pc(a['mdd'], 1)} | {_num(a['sharpe'])} | "
                 f"{b['n_trades']} | {b['n_trade_days']} | {_pf(b['pf'])} | {cp('IS', v)} | {cp('OOS', v)} | {'PASS' if flags[v]['PASS'] else '-'} |")
    L.append("")

    # ---- controls ----
    L.append("## 대조군 요약")
    L.append("")
    L.append("실제 PF가 대조군 100회 PF의 95퍼센타일을 IS·OOS 둘 다에서 넘은 변형 수:")
    L.append("")
    both = {c: sum(1 for v in variants if c in res["beats"]["IS"][v] and res["beats"]["IS"][v][c] and res["beats"]["OOS"][v][c])
            for c in ("A", "B", "C")}
    tot = {"A": 24, "B": 24, "C": 24}
    L.append(f"- 대조군 A (같은 패닉일, 종목 필터 없는 무작위 종목): {both['A']} / {tot['A']} (모드 S)")
    L.append(f"- 대조군 B (변동성 10분위를 맞춘 전 기간 무작위 (날, 종목)): {both['B']} / {tot['B']} (모드 S)")
    L.append(f"- 대조군 C (패닉이 아닌 날의 SPY): {both['C']} / {tot['C']} (모드 I)")
    L.append("")

    # ---- holdout ----
    L.append("## HOLDOUT (2024-11-01 ~ 2026-10-31)")
    L.append("")
    ho = res["holdout"]
    if ho is None:
        L.append("봉인 유지 — SPEC_v2 §6 (i)~(vii)를 통과한 변형이 없어 holdout은 계산하지 않았다 (SPEC_v2 §5).")
    else:
        m = ho["metrics"]
        hv = tuple(ho["variant"])
        L.append(f"통과 변형 중 IS PF 최고 1개만, 단 한 번 계산: {vlabel(hv)} (IS PF {_pf(rows['IS'][hv]['pf'])}).")
        L.append("")
        L.append("| 거래 | 패닉일 | PF (0.10%) | PF (0.05%) | PF (0.20%) | 승률 | 평균 | 일 평균 | 총수익 | CAGR | MDD | Sharpe | SPY MDD |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        L.append(f"| {m['n_trades']} | {m['n_trade_days']} | {_pf(m['pf'])} | {_pf(m['pf_cost005'])} | {_pf(m['pf_cost020'])} | "
                 f"{_pc(m['win_rate'], 1)} | {_pc(m['mean_ret'])} | {_pc(m['day_mean_ret'])} | {_pc(m['total_return'], 1)} | "
                 f"{_pc(m['cagr'], 1)} | {_pc(m['mdd'], 1)} | {_num(m['sharpe'])} | {_pc(m['spy_bh_mdd'], 1)} |")
        L.append("")
        L.append("SPEC_v2는 HOLDOUT에 별도 통과 임계값을 정하지 않았다. 숫자만 보고하며 위 전체 결론을 바꾸지 않는다 (`holdout.json`).")
    L.append("")

    L += _coverage_section(cov, load_info, resolved)

    L.append("## 편향·한계 선언 (SPEC_v2 §0 + 구현 시 확인한 것)")
    L.append("")
    L.append("- **Survivorship bias**: 개명은 복구했지만 인수·파산·상폐 종목은 Yahoo에 없다. 빠진 종목은 대개 크게 떨어진 뒤 계속 떨어진 쪽이라 "
             "결과를 **좋은 쪽으로 부풀린다**. PASS는 의심하고 FAIL은 더 믿는다.")
    L.append("- 개별 기업 악재는 걸러내지 못한다(뉴스·실적 캘린더 없음). 쓴 대용물은 '모두가 떨어진 날, 시장 낙폭의 1~3배 범위'뿐이다.")
    L.append("- 패닉일은 드물고 2020-03 등에 뭉친다. 거래 수는 독립 표본이 아니다 — (i)에 패닉일 수 기준을 넣은 이유다. "
             "'패닉일'은 평가 가능한 거래가 1건 이상 있었던 패닉일로 센다.")
    L.append("- 이 데이터로 던진 그물은 v1 72 + v2 48 = 누적 120개다. v1 결과를 본 뒤 설계한 가설이라는 사실 자체가 다중검정 비용이다.")
    L.append("- 진입은 신호일 종가 '그 가격' 체결 가정(편도 0.10%로 흡수, 민감도 0.05%·0.20%는 PF 열). 일중 순서를 모르므로 stop·target 동시면 stop 우선, "
             "시가가 stop 아래면 시가 청산(v1과 동일). TARGET 없음은 +∞로 구현(손절·갭·시간 청산만 발생).")
    L.append("- 구성종목 기간은 [start_date, end_date). breadth는 분할 의심일을 뺀 구성종목 중 그날 유효한 수익률이 있는 종목 기준. "
             "모드 S의 25종목 제한은 직전 20일 평균 거래대금 순(낙폭 순 아님), 거래 단위 통계는 모든 후보를 센다. "
             "SPY MDD는 배당 미반영 가격 기준.")
    L.append(f"- 대조군 {res['control_n']}회, seed {res['seed']}, 거래 단위 PF만 비교(대조군 B의 변동성 = 신호일 직전 20일 일수익률 표준편차).")
    L.append("")
    L.append("## 실행 정보")
    L.append("")
    L.append(f"- 실행 시간 {res['runtime']:.1f}초, 패널 셀 {P.N:,}개 ({len(P.tickers)}개 시리즈, SPY 포함).")
    L.append("- 구간: " + ", ".join(f"{k} {a}~{b}" for k, (a, b) in res["periods"].items()) + ".")
    L.append("- 시장 통계를 계산한 구간: " + ", ".join(f"{P.iso(a)}~{P.iso(b)}" for a, b in res["windows"]) + ".")
    best = best_is_variant(res)
    L.append(f"- 산출물: `grid_IS.csv`, `grid_OOS.csv`, `panic_days.csv`, `controls_summary.csv`, `trades_{variant_slug(best)}.csv`, "
             "`data_coverage_v2.json`, `summary_v2.json`" + (", `holdout.json`" if ho is not None else "") + ".")
    L.append("")
    with open(os.path.join(out_dir, "SUMMARY_v2.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def write_outputs(res: dict, out_dir: str, cov: dict, load_info: dict, resolved: dict, run_date: str,
                  synthetic: bool = False) -> list:
    os.makedirs(out_dir, exist_ok=True)
    written = []

    def w(name):
        written.append(name)
        return os.path.join(out_dir, name)

    grid_dataframe(res, "IS").to_csv(w("grid_IS.csv"), index=False, float_format="%.6g")
    grid_dataframe(res, "OOS").to_csv(w("grid_OOS.csv"), index=False, float_format="%.6g")
    panic_df = panic_day_table(res)
    panic_df.to_csv(w("panic_days.csv"), index=False, float_format="%.6g")
    controls_dataframe(res).to_csv(w("controls_summary.csv"), index=False, float_format="%.6g")

    best = best_is_variant(res)
    to_write = ["IS", "OOS"]
    if res["holdout"] is not None and tuple(res["holdout"]["variant"]) == best:
        to_write.append("HOLDOUT")
    trades_dataframe(res, best, to_write).to_csv(w(f"trades_{variant_slug(best)}.csv"), index=False)

    cov_path = os.path.join(out_dir, COVERAGE_FILE)
    if cov and not cov.get("_from_fetch"):
        with open(w(COVERAGE_FILE), "w") as f:
            json.dump(_clean_json({k: v for k, v in cov.items() if k != "_from_fetch"}), f, indent=2)
    else:
        written.append(COVERAGE_FILE)             # fetch_prices_v2.py's file, passed through untouched
        if not os.path.exists(cov_path):
            with open(cov_path, "w") as f:
                json.dump(_clean_json({k: v for k, v in (cov or {}).items() if k != "_from_fetch"}), f, indent=2)

    if res["holdout"] is not None:
        ho = res["holdout"]
        with open(w("holdout.json"), "w") as f:
            json.dump(_clean_json({"variant": ho["variant"], "slug": ho["variant_slug"], "period": ho["period"],
                                   "metrics": ho["metrics"], "note": ho["note"]}), f, indent=2)

    P = res["P"]
    summary = {
        "run_date": run_date, "synthetic": synthetic, "verdict": res["verdict"], "n_pass": len(res["passers"]),
        "passers": [variant_slug(v) for v in res["passers"]], "cumulative_denominator": CUMULATIVE_DENOM,
        "real_trade_level_pass_count": res["real_count"], "control_pass_counts": res["ctrl_counts"],
        "control_pass_count_p95": res["p95_count"], "control_n": res["control_n"], "seed": res["seed"],
        "panic_days": {f"{p}_{pn}": res["n_panic"][(p, pn)] for p in PANICS for pn in ("IS", "OOS")},
        "holdout_computed": res["holdout"] is not None, "periods": res["periods"],
        "market_windows_computed": [[P.iso(a), P.iso(b)] for a, b in res["windows"]],
        "n_series_in_panel": len(P.tickers), "n_cells": P.N, "load_info": load_info,
        "aliases_applied": len(resolved), "runtime_seconds": round(res["runtime"], 1),
    }
    with open(w("summary_v2.json"), "w") as f:
        json.dump(_clean_json(summary), f, indent=2)
    write_summary(res, out_dir, cov if cov else {}, load_info, resolved, run_date, synthetic, panic_df)
    written.append("SUMMARY_v2.md")
    return written


# ---------------------------------------------------------------------------
# Synthetic data (self-test, tests, timing)
# ---------------------------------------------------------------------------

def make_synthetic_v2(n_tickers: int = 40, n_days: int = 1500, seed: int = 11, edge: bool = True,
                      start: str = R.DATA_START):
    """Market-factor random-walk panel. Returns (series, membership, splits, spy_df, periods, resolved).
    * market return m_t ~ N(0.04%, 0.7%); ~max(40, n_days/35) PANIC days with m = -2.5%..-6% (a few are 'narrow': the index falls
      but the stocks barely move, so breadth stays < 50% and no panic is flagged);
    * stock return = beta_i m_t + idio (1.2%); on panic days ~8% of stocks get an idiosyncratic -22..-40% (hard-cap /
      3x-rule exclusions) and ~8% an upward idiosyncratic move (less-than-market exclusion);
    * edge=True: the day after a panic day, stocks that fell at least as much as the market rebound by an extra
      +2..3.5% (a planted stock-selection edge) and SPY by +0.5%; edge=False: nothing is planted.
    Ticker T000 carries a raw 2:1 split (splits file), T001 a -50% low-volume day without a splits file, T003 is
    illiquid. OLD00 has membership but NO series; `resolved` maps it to T010 (whose own membership overlaps)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    nt = n_tickers
    m = rng.normal(0.0004, 0.007, n_days)
    n_pan = max(40, n_days // 35)
    cand_days = np.arange(40, n_days - 12)
    pan = np.sort(rng.choice(cand_days, size=n_pan, replace=False))
    narrow = set(pan[rng.random(len(pan)) < 0.12].tolist())
    beta = rng.uniform(0.7, 1.3, nt)
    idio = rng.normal(0.0, 0.012, (n_days, nt))
    r = beta[None, :] * m[:, None] + idio
    pan_set = set(pan.tolist())
    for p in pan:
        mp = -rng.uniform(0.025, 0.06)
        m[p] = mp
        if p in narrow:
            r[p] = 0.1 * mp + idio[p]
        else:
            r[p] = beta * mp + idio[p]
            u = rng.random(nt)
            big = u < 0.08
            r[p, big] = -rng.uniform(0.22, 0.40, int(big.sum()))
            up = (u >= 0.08) & (u < 0.16)
            r[p, up] = rng.uniform(0.0, 0.02, int(up.sum()))
    if edge:
        for p in pan:
            q = p + 1
            if q >= n_days or q in pan_set:
                continue
            fell = r[p] <= m[p] + 1e-12
            m[q] = max(m[q], 0.0)
            r[q] = beta * m[q] + idio[q] + np.where(fell, rng.uniform(0.02, 0.035, nt), 0.0)
    close = rng.uniform(40, 200, nt) * np.cumprod(1.0 + r, axis=0)
    prev = np.empty_like(close)
    prev[0] = close[0] / (1.0 + r[0])
    prev[1:] = close[:-1]
    shape = (n_days, nt)
    open_ = prev * (1.0 + rng.normal(0, 0.003, shape))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0, 0.004, shape)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0, 0.004, shape)))
    dollar = np.exp(rng.normal(np.log(4e8), 0.4, shape))
    if nt > 3:
        dollar[:, 3] = 1e7
    vol = dollar / close
    split_day = min(700, n_days // 2)
    splits = {}
    if nt > 0:
        for a in (open_, high, low, close):
            a[:split_day, 0] *= 2.0
        vol[:split_day, 0] /= 2.0
        splits["T000"] = [str(dates[split_day].date())]
    if nt > 1:
        sd = min(900, n_days // 2 + 50)
        for a in (open_, high, low, close):
            a[sd:, 1] *= 0.5
        vol[sd, 1] *= 0.2
    names = [f"T{j:03d}" for j in range(nt)]
    series, membership = {}, {}
    for j, t in enumerate(names):
        series[t] = pd.DataFrame({"date": dates, "open": open_[:, j], "high": high[:, j], "low": low[:, j],
                                  "close": close[:, j], "volume": vol[:, j]})
        splits.setdefault(t, None)
        always = [(np.datetime64("2000-01-01", "D"), None)]
        if j % 10 == 5:
            membership[t] = [(np.datetime64(str(dates[n_days // 3].date()), "D"), None)]
        elif j % 10 == 7:
            membership[t] = [(np.datetime64("2000-01-01", "D"), np.datetime64(str(dates[(2 * n_days) // 3].date()), "D"))]
        elif j % 10 == 9:
            membership[t] = [(np.datetime64("2000-01-01", "D"), np.datetime64(str(dates[n_days // 4].date()), "D")),
                             (np.datetime64(str(dates[n_days // 2].date()), "D"), None)]
        else:
            membership[t] = always
    resolved = {}
    if nt > 15:
        # OLD00 (a renamed ticker, no series of its own) was a member until 10 days AFTER T010 joined: overlap days
        membership["OLD00"] = [(np.datetime64("2000-01-01", "D"), np.datetime64(str(dates[n_days // 3 + 10].date()), "D"))]
        resolved["OLD00"] = "T010"
        membership["T010"] = [(np.datetime64(str(dates[n_days // 3].date()), "D"), None)]
    spy_ret = m + rng.normal(0, 0.0005, n_days)
    if edge:
        for p in pan:
            if p + 1 < n_days and (p + 1) not in pan_set:
                spy_ret[p + 1] += 0.005
    sc = 100 * np.cumprod(1 + spy_ret)
    spy = pd.DataFrame({"date": dates, "open": sc, "high": sc * 1.003, "low": sc * 0.997, "close": sc, "volume": 1e8})
    i0, i1, i2, i3 = 25, int(0.55 * n_days), int(0.80 * n_days), n_days - 1
    periods = {"IS": (str(dates[i0].date()), str(dates[i1].date())),
               "OOS": (str(dates[i1 + 1].date()), str(dates[i2].date())),
               "HOLDOUT": (str(dates[i2 + 1].date()), str(dates[i3].date()))}
    return series, membership, splits, spy, periods, resolved


def run_synthetic(n_tickers: int = 40, n_days: int = 1500, control_n: int = 5, out_dir: str | None = None,
                  seed: int = 11) -> str:
    t0 = time.time()
    log(f"self-test: generating {n_tickers} tickers x {n_days} days ...")
    series, membership, splits, spy, periods, resolved = make_synthetic_v2(n_tickers, n_days, seed=seed, edge=True)
    mem2 = apply_aliases(membership, resolved)
    P = build_panel_v2(series, mem2, splits, spy)
    log(f"panel built: {len(P.tickers)} series, {P.N:,} cells, {time.time() - t0:.1f}s")
    res = run_study_v2(P, spy, periods, control_n, SEED)
    own_tmp = out_dir is None
    out = out_dir or tempfile.mkdtemp(prefix="panic_dip_v2_synth_")
    cov = {"requested": n_tickers, "received_direct": len(P.tickers), "synthetic": True}
    load_info = {"loaded": len(P.tickers), "skipped": len(P.skipped)}
    write_outputs(res, out, cov, load_info, resolved, str(date.today()), synthetic=True)
    best = best_is_variant(res)
    expected = ["grid_IS.csv", "grid_OOS.csv", "panic_days.csv", "controls_summary.csv", "summary_v2.json",
                "SUMMARY_v2.md", COVERAGE_FILE, f"trades_{variant_slug(best)}.csv"]
    for name in expected:
        assert os.path.exists(os.path.join(out, name)), f"missing output {name}"
    g_is = pd.read_csv(os.path.join(out, "grid_IS.csv"))
    g_oos = pd.read_csv(os.path.join(out, "grid_OOS.csv"))
    assert len(g_is) == 48 and len(g_oos) == 48, "grid must have 48 rows"
    assert len(pd.read_csv(os.path.join(out, "controls_summary.csv"))) == 2 * (24 * 2 + 24 * 1)
    assert g_is["n_trades"].sum() > 0, "synthetic data produced no trades"
    pdays = pd.read_csv(os.path.join(out, "panic_days.csv"))
    assert len(pdays) > 0, "synthetic data produced no panic days"
    assert res["passers"], "planted stock-selection edge should yield at least one PASS variant"
    assert res["holdout"] is not None and os.path.exists(os.path.join(out, "holdout.json")), \
        "a PASS variant exists, so holdout.json must be written"
    assert res["verdict"] in ("EDGE", "NO EVIDENCE")
    log(f"self-test: verdict={res['verdict']} passers={len(res['passers'])} out={out} ({time.time() - t0:.1f}s)")
    if own_tmp:
        shutil.rmtree(out, ignore_errors=True)
    print("SYNTHETIC V2 OK")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def write_no_spy_summary(out_dir: str, cov: dict, run_date: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    txt = (f"# panic_dip v2 — 실행 불가 ({run_date})\n\nSPY 데이터(`SPY.csv`)가 없어 패닉일을 정의할 수 없다. "
           "Yahoo fetch가 SPY를 받지 못한 것으로 보인다 — `fetch.log`·`data_coverage_v2.json`을 확인할 것. "
           "결론 없음(NO RESULT).\n")
    with open(os.path.join(out_dir, "SUMMARY_v2.md"), "w", encoding="utf-8") as f:
        f.write(txt)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="default: data/panic_dip_v2")
    ap.add_argument("--out-dir", default=None, help="default: research/panic_dip/v2/results")
    ap.add_argument("--control-n", type=int, default=None, help=f"control runs per variant/period (default {CONTROL_N})")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--synthetic", action="store_true", help="self-test on generated data (no files needed)")
    ap.add_argument("--synthetic-tickers", type=int, default=40, help="size knob for --synthetic (timing runs)")
    ap.add_argument("--synthetic-days", type=int, default=1500, help="size knob for --synthetic (timing runs)")
    args = ap.parse_args()

    if args.synthetic:
        run_synthetic(args.synthetic_tickers, args.synthetic_days,
                      control_n=args.control_n if args.control_n is not None else 5, out_dir=args.out_dir)
        return 0

    t0 = time.time()
    out_dir = args.out_dir or DEFAULT_OUT_DIR
    control_n = args.control_n if args.control_n is not None else CONTROL_N
    membership = R.load_constituents()
    resolved = load_resolved_aliases(args.data_dir)
    membership2 = apply_aliases(membership, resolved)
    names = sorted(membership2)
    series, splits, missing = R.load_series(args.data_dir, names)
    if not series:
        log(f"no data under {args.data_dir} for any constituent — run fetch_prices_v2.py first. Not treated as a failure.")
        return 0
    spy = R.load_spy(args.data_dir)
    cov_path = os.path.join(out_dir, COVERAGE_FILE)
    cov = None
    if os.path.exists(cov_path):
        try:
            with open(cov_path) as f:
                cov = json.load(f)
            cov["_from_fetch"] = True
        except Exception as e:  # noqa: BLE001
            log(f"WARNING: could not read {cov_path}: {e}")
    if spy is None:
        log("ERROR: SPY.csv missing — panic days cannot be defined; writing a stub summary and stopping.")
        write_no_spy_summary(out_dir, cov or {}, str(date.today()))
        return 0
    log(f"universe {len(membership)} tickers -> {len(names)} series after {len(resolved)} alias(es); loaded {len(series)}, "
        f"no file for {len(missing)}; SPY loaded ({time.time() - t0:.1f}s)")
    P = build_panel_v2(series, membership2, splits, spy)
    log(f"panel: {len(P.tickers)} series, {P.N:,} cells, {len(P.cal)} calendar days ({time.time() - t0:.1f}s)")
    if cov is None:
        cov = {"requested": len(membership), "received_direct": len(series), "missing": missing,
               "note": "written by run_panic_dip_v2.py (no fetch_prices_v2.py coverage file found)"}
    load_info = {"loaded": len(P.tickers) - (1 if SPY_NAME in P.tickers else 0), "skipped": len(P.skipped),
                 "no_file": len(missing)}
    res = run_study_v2(P, spy, PERIODS, control_n, args.seed)
    write_outputs(res, out_dir, cov, load_info, resolved, str(date.today()))
    log(f"verdict: {res['verdict']} (PASS variants: {len(res['passers'])}, trade-level real {res['real_count']} "
        f"vs control p95 {res['p95_count']:.2f}); total runtime {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
