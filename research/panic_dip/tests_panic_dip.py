"""
tests_panic_dip.py — SPEC.md §8 (+ the building blocks around it). Plain asserts, no pytest:

    python research/panic_dip/tests_panic_dip.py

Covered
  1.  lookahead: truncating the series anywhere changes no signal / liquidity flag / determinable
      exit up to the cut; garbage written AFTER t changes nothing at t.
  2.  synthetic (a): every drop is followed by +4%  -> (10%,-1%,+3%,1d) PF >> 1, every exit "target".
  3.  synthetic (b): every drop is followed by -4%  -> every exit stop/gap, PF < 1.
  4.  same-day stop+target tie -> stop wins; open gapping through the stop -> exit at the OPEN.
  5.  split-day exclusion: rule A (splits file) and rule B (fallback) fire exactly where specified,
      with per-rule counts.
  6.  point-in-time membership: signals before start / on-or-after end are ignored, start inclusive,
      several intervals.
  7.  slot overflow: >10 signals on a day -> the 10 largest drops are filled; open positions block.
  8.  vectorised exits == a literal scalar reading of SPEC §3 on wild random data, all 24 rules.
  9.  liquidity guard window (bars t-20..t-1 only), PF / MDD / Sharpe / cost arithmetic.
  10. portfolio accounting, placebo mechanics, PASS / (viii) logic, HOLDOUT seal, determinism.
  11. fetch_prices.py helpers against fake HTTP responses (the sandbox cannot reach Yahoo).
"""
from __future__ import annotations
import json
import math
import os
import sys
import tempfile
import traceback

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run_panic_dip as R  # noqa: E402
import fetch_prices as F  # noqa: E402

FAILURES = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append((name, detail))


# ---------------------------------------------------------------------------
# Hand-made series builder
# ---------------------------------------------------------------------------

class S:
    """Flat 100.0 series (volume 1e7 shares => $1e9/day) that tests overwrite bar by bar."""

    def __init__(self, n=200, start="2016-01-04", price=100.0, vol=1e7):
        self.dates = pd.bdate_range(start, periods=n)
        self.o = np.full(n, price)
        self.h = np.full(n, price)
        self.l = np.full(n, price)
        self.c = np.full(n, price)
        self.v = np.full(n, vol)

    def bar(self, i, c, o=None, h=None, l=None, v=None):
        o = c if o is None else o
        self.o[i], self.c[i] = o, c
        self.h[i] = max(o, c) if h is None else h
        self.l[i] = min(o, c) if l is None else l
        if v is not None:
            self.v[i] = v
        return self

    def iso(self, i):
        return str(self.dates[i].date())

    def df(self):
        return pd.DataFrame({"date": self.dates, "open": self.o, "high": self.h, "low": self.l,
                             "close": self.c, "volume": self.v})


ALWAYS = [(np.datetime64("2000-01-01", "D"), None)]


def mk_panel(series: dict, membership=None, splits=None):
    mem = {t: ALWAYS for t in series} if membership is None else membership
    spl = {t: None for t in series} if splits is None else {t: splits.get(t) for t in series}
    return R.build_panel({t: s.df() for t, s in series.items()}, mem, spl)


def run_variant(P, drop, stop, target, hold, cost=R.COST_PRIMARY):
    X = R.compute_exit_arrays(P, stop, target, hold)
    cand = R.candidate_cells(P, drop)
    sel = R.select_trades(P, X, cand, 0, len(P.cal) - 1)
    return X, sel, R.trade_stats(P, X, sel, cost)


PROP = dict(drop=0.10, stop=-0.01, target=0.03, hold=1)


def cycle_series(next_day, n=200, first=25, every=10):
    """Flat 100; every `every` days: close -12% (o=100,c=88), a next-day bar set by next_day(s, i),
    then back to 100 (an up day: never a signal)."""
    s = S(n)
    for d in range(first, n - 6, every):
        s.bar(d, c=88.0, o=100.0)
        next_day(s, d + 1)
        s.bar(d + 2, c=100.0, o=100.0)
    return s


# ---------------------------------------------------------------------------
# 2./3. SPEC §8 synthetic series (a) and (b)
# ---------------------------------------------------------------------------

def test_synthetic_a_rebound():
    s = cycle_series(lambda s, i: s.bar(i, c=88 * 1.04, o=88.0, h=88 * 1.045, l=88.0))
    P = mk_panel({"AAA": s})
    X, sel, st = run_variant(P, **PROP)
    check("(a) rebound: trades found", st["n_trades"] >= 10, str(st["n_trades"]))
    check("(a) rebound: every exit is 'target'", st["exit_target"] == st["n_trades"] and st["exit_stop"] == 0
          and st["exit_gap"] == 0 and st["exit_time"] == 0, str(st))
    check("(a) rebound: PF >> 1 (loss-free => inf)", st["pf"] == float("inf") or st["pf"] > 100, str(st["pf"]))
    check("(a) rebound: gross +3.0% / net +2.8% per trade", abs(st["mean_ret"] - 0.028) < 1e-6, str(st["mean_ret"]))
    check("(a) rebound: held exactly 1 day", st["mean_hold_days"] == 1.0)


def test_synthetic_b_continued_fall():
    s = cycle_series(lambda s, i: s.bar(i, c=88 * 0.96, o=88 * 0.999, h=88.0, l=88 * 0.955))
    P = mk_panel({"AAA": s})
    X, sel, st = run_variant(P, **PROP)
    check("(b) fall (open above stop): trades found", st["n_trades"] >= 10)
    check("(b) fall: every exit is stop/gap", st["exit_stop"] + st["exit_gap"] == st["n_trades"], str(st))
    check("(b) fall: PF < 1", st["pf"] < 1.0, str(st["pf"]))
    s2 = cycle_series(lambda s, i: s.bar(i, c=88 * 0.96, o=88 * 0.97, h=88 * 0.975, l=88 * 0.955))
    X2, sel2, st2 = run_variant(mk_panel({"AAA": s2}), **PROP)
    check("(b) fall (open gaps below stop): every exit is gap", st2["exit_gap"] == st2["n_trades"], str(st2))
    check("(b) gap: PF < 1", st2["pf"] < 1.0)


# ---------------------------------------------------------------------------
# 4. tie-break and gap-through
# ---------------------------------------------------------------------------

def _one_drop(next_bar, hold=1, stop=-0.01, target=0.03, extra=None):
    s = S(80)
    s.bar(30, c=88.0, o=100.0)
    next_bar(s)
    if extra:
        extra(s)
    P = mk_panel({"AAA": s})
    X = R.compute_exit_arrays(P, stop, target, hold)
    i = P.cell("AAA", s.iso(30))
    return P, X, i, s


def test_exit_rules():
    # stop and target both reachable the same day (open above stop) -> stop wins, at the stop price
    P, X, i, s = _one_drop(lambda s: s.bar(31, c=88.0, o=88.0, h=88 * 1.05, l=88 * 0.985))
    check("tie: stop wins over target", int(X.code[i]) == 1, f"code={X.code[i]}")
    check("tie: exit at the stop price (-1.0%)", abs(float(X.ret[i]) + 0.01) < 1e-6, str(X.ret[i]))

    # open already below the stop -> exit at the OPEN (worse than the stop), even if high >= target
    P, X, i, s = _one_drop(lambda s: s.bar(31, c=88 * 0.99, o=88 * 0.97, h=88 * 1.06, l=88 * 0.95))
    check("gap-through: code is gap", int(X.code[i]) == 2, f"code={X.code[i]}")
    check("gap-through: exit at the open (-3.0%), not the stop (-1.0%)", abs(float(X.ret[i]) + 0.03) < 1e-6,
          str(X.ret[i]))
    check("gap-through: gap beats a same-day target", int(X.code[i]) != 0)

    # open exactly AT the stop counts as a gap (SPEC: open <= stop)
    P, X, i, s = _one_drop(lambda s: s.bar(31, c=88.0, o=88 * 0.99, h=88.0, l=88 * 0.98))
    check("open == stop (<=) -> gap exit at open", int(X.code[i]) == 2 and abs(float(X.ret[i]) + 0.01) < 1e-6)

    # low exactly at the stop counts as a stop; low just above does not
    P, X, i, s = _one_drop(lambda s: s.bar(31, c=88.0, o=88.0, h=88.5, l=88 * 0.99 - 1e-9))
    check("low <= stop -> stop", int(X.code[i]) == 1)
    P, X, i, s = _one_drop(lambda s: s.bar(31, c=88.1, o=88.0, h=88.5, l=88 * 0.99 + 1e-6))
    check("low > stop, no target -> time exit at close", int(X.code[i]) == 3 and abs(float(X.ret[i]) - (88.1 / 88 - 1)) < 1e-6)

    # target only
    P, X, i, s = _one_drop(lambda s: s.bar(31, c=88.2, o=88.0, h=88 * 1.03 + 1e-6, l=88.0))
    check("high >= target -> target price", int(X.code[i]) == 0 and abs(float(X.ret[i]) - 0.03) < 1e-6)

    # multi-day: day t+1 quiet, day t+2 hits the target -> k = 2, ret = +3%; HOLD=3
    def quiet_then_hit(s):
        s.bar(31, c=88.2, o=88.0, h=88.4, l=87.8)
        s.bar(32, c=89.0, o=88.2, h=88 * 1.031, l=88.1)
    P, X, i, s = _one_drop(quiet_then_hit, hold=3)
    check("HOLD=3: exit on day 2 with target", int(X.code[i]) == 0 and int(X.k[i]) == 2)

    # time exit at HOLD=3 uses close[t+3]
    def drift(s):
        s.bar(31, c=88.2, o=88.0, h=88.4, l=87.8)
        s.bar(32, c=88.3, o=88.2, h=88.5, l=88.0)
        s.bar(33, c=88.8, o=88.3, h=89.0, l=88.1)
    P, X, i, s = _one_drop(drift, hold=3)
    check("HOLD=3 time exit at close[t+3]", int(X.code[i]) == 3 and int(X.k[i]) == 3
          and abs(float(X.ret[i]) - (88.8 / 88 - 1)) < 1e-6)

    # STOP none: a deep intraday dip is ignored, the target on day 2 is taken
    def deep_dip(s):
        s.bar(31, c=86.0, o=88.0, h=88.2, l=70.0)
        s.bar(32, c=90.0, o=86.0, h=88 * 1.05, l=85.0)
    P, X, i, s = _one_drop(deep_dip, hold=3, stop=None)
    check("STOP none: never stops, takes target on day 2", int(X.code[i]) == 0 and int(X.k[i]) == 2)
    # ... and with a stop the same path is stopped on day 1 (low 70 <= stop)
    P, X, i, s = _one_drop(deep_dip, hold=3, stop=-0.05)
    check("STOP -5%: same path stops on day 1", int(X.code[i]) == 1 and int(X.k[i]) == 1)

    # day t itself is never tested: drop day with a huge high must not exit at t
    s = S(80)
    s.bar(30, c=88.0, o=100.0, h=140.0, l=60.0)
    s.bar(31, c=88.2, o=88.0, h=88.4, l=87.9)
    P = mk_panel({"AAA": s})
    X = R.compute_exit_arrays(P, -0.01, 0.03, 1)
    i = P.cell("AAA", s.iso(30))
    check("day t is not tested for stop/target", int(X.code[i]) == 3 and int(X.k[i]) == 1)

    # insufficient future bars at the end of the data -> -1 / nan
    last = len(P.close) - 1
    X5 = R.compute_exit_arrays(P, -0.01, 0.03, 5)
    check("insufficient future bars -> code -1 and NaN return",
          all(int(X5.code[last - j]) == -1 and np.isnan(X5.ret[last - j]) for j in range(5))
          and int(X5.code[last - 5]) >= 0)


# ---------------------------------------------------------------------------
# 5. split-day exclusion
# ---------------------------------------------------------------------------

def test_split_exclusion():
    def make(vol_on_drop=None):
        s = S(100)
        s.bar(40, c=50.0, o=100.0, v=vol_on_drop)
        s.bar(41, c=100.0, o=50.0)
        return s
    series = {"SPL": make(), "LOW": make(1e7 * 0.3), "NRM": make(), "OTH": make(1e7 * 0.3)}
    splits = {"SPL": [series["SPL"].iso(40)], "OTH": [series["OTH"].iso(60)]}   # LOW, NRM: no file
    P = mk_panel(series, splits=splits)
    cand = set(R.candidate_cells(P, 0.15).tolist())
    day = series["SPL"].iso(40)
    check("rule A: split date in the splits file -> no signal", P.cell("SPL", day) not in cand)
    check("rule B: no file, -50% on 0.3x volume -> excluded", P.cell("LOW", day) not in cand)
    check("no file, -50% on normal volume -> NOT excluded", P.cell("NRM", day) in cand)
    check("file exists (other date): rule B not applied -> signal kept", P.cell("OTH", day) in cand)
    check("flag counts: A=2 flagged days (SPL@40, OTH@60), B=1 (LOW)",
          int(P.excl_a.sum()) == 2 and int(P.excl_b.sum()) == 1, f"{P.excl_a.sum()} {P.excl_b.sum()}")
    d = R.exclusion_counts(P, (0, len(P.cal) - 1))["by_drop"]["15%"]
    check("exclusion report by rule at DROP 15%",
          d["signals_before_exclusion"] == 4 and d["excluded_rule_A_split_file"] == 1
          and d["excluded_rule_B_fallback"] == 1, str(d))
    # a plain -12% drop is untouched by both rules
    s = S(100)
    s.bar(40, c=88.0, o=100.0)
    P2 = mk_panel({"X": s})
    check("-12% is not a split suspect", P2.cell("X", s.iso(40)) in set(R.candidate_cells(P2, 0.10).tolist()))
    # splits file that is an empty list (events present, no splits) -> nothing excluded by A, B not used
    P3 = mk_panel({"X": s}, splits={"X": []})
    check("empty splits list: nothing flagged", int(P3.split_excl.sum()) == 0 and P3.has_splits["X"])


# ---------------------------------------------------------------------------
# 6. point-in-time membership
# ---------------------------------------------------------------------------

def test_membership_pit():
    dip_days = ["2016-02-29", "2016-03-01", "2016-05-31", "2016-06-01", "2016-06-02", "2016-07-05"]
    series = {}
    for n, d in enumerate(dip_days):          # one ticker per dip day: the dips never touch each other
        s = S(140)
        i = int(np.flatnonzero(s.dates == pd.Timestamp(d))[0])
        s.bar(i, c=88.0, o=100.0)
        s.bar(i + 1, c=100.0, o=88.0)
        series[f"D{n}"] = s
    # interval [2016-03-01, 2016-06-01), then a second interval from 2016-07-01 (same for every ticker)
    iv = [(np.datetime64("2016-03-01", "D"), np.datetime64("2016-06-01", "D")),
          (np.datetime64("2016-07-01", "D"), None)]
    P = mk_panel(series, membership={t: iv for t in series})
    got = {str(P.cal[P.gidx[i]]) for i in R.candidate_cells(P, 0.10)}
    check("signal before start_date ignored (2016-02-29)", "2016-02-29" not in got, str(sorted(got)))
    check("start_date itself is a member day (2016-03-01)", "2016-03-01" in got, str(sorted(got)))
    check("last day before end_date counts (2016-05-31)", "2016-05-31" in got)
    check("end_date itself is NOT a member day (2016-06-01)", "2016-06-01" not in got)
    check("signal after end_date ignored (2016-06-02)", "2016-06-02" not in got)
    check("second interval counts (2016-07-05)", "2016-07-05" in got)
    check("exactly the 3 in-membership dips signal", got == {"2016-03-01", "2016-05-31", "2016-07-05"}, str(sorted(got)))
    check("ticker with no membership rows never signals",
          len(R.candidate_cells(mk_panel(series, membership={}), 0.10)) == 0)


# ---------------------------------------------------------------------------
# 7. slots / portfolio
# ---------------------------------------------------------------------------

def _overflow_panel(n_tick=15, day=30, hold_flat=True, rebound=True):
    series = {}
    for j in range(n_tick):
        s = S(70)
        c30 = 100.0 * (1 - (0.10 + 0.01 * j))
        s.bar(day, c=c30, o=100.0)
        if rebound:
            s.bar(day + 1, c=c30 * 1.04, o=c30, h=c30 * 1.045, l=c30)
            s.bar(day + 2, c=100.0, o=100.0)
        else:
            for k in range(day + 1, 70):
                s.bar(k, c=c30, o=c30)
        series[f"T{j:02d}"] = s
    return mk_panel(series)


def test_slot_overflow():
    P = _overflow_panel()
    X = R.compute_exit_arrays(P, -0.01, 0.03, 1)
    cand = R.candidate_cells(P, 0.08)
    sel = R.select_trades(P, X, cand, 0, len(P.cal) - 1)
    check("15 signals on one day", len(sel) == 15, str(len(sel)))
    drops = P.ret1[sel]
    check("signals ordered largest drop first", bool(np.all(np.diff(drops) >= 0)), str(drops))
    sim = R.simulate_portfolio(P, sel, X, 0, len(P.cal) - 1, R.COST_PRIMARY)
    check("only 10 filled, 5 skipped", sim["n_filled"] == 10 and sim["n_skipped"] == 5, str(sim["n_filled"]))
    filled_names = {P.tickers[int(P.tick[i])] for i in sim["fills"]}
    expect = {f"T{j:02d}" for j in range(5, 15)}          # j=14 is the deepest drop
    check("the 10 largest drops are the ones filled", filled_names == expect, str(sorted(filled_names)))
    check("each fill is 10% of equity: end equity = 1 + 10 x 0.1 x (3% - 2 x 0.10%)",
          abs(sim["total_return"] - 10 * 0.1 * (0.03 - 0.002)) < 1e-6, str(sim["total_return"]))

    # open positions block new entries: HOLD=3, flat afterwards (time exits), a second group a day later
    series = {}
    for j in range(15):
        s = S(70)
        c = 100.0 * (1 - (0.10 + 0.01 * j))
        d0 = 30 if j < 10 else 31
        s.bar(d0, c=c, o=100.0)
        for k in range(d0 + 1, 70):
            s.bar(k, c=c, o=c)
        series[f"T{j:02d}"] = s
    P2 = mk_panel(series)
    X2 = R.compute_exit_arrays(P2, None, 0.05, 3)
    sel2 = R.select_trades(P2, X2, R.candidate_cells(P2, 0.08), 0, len(P2.cal) - 1)
    sim2 = R.simulate_portfolio(P2, sel2, X2, 0, len(P2.cal) - 1, R.COST_PRIMARY)
    names2 = {P2.tickers[int(P2.tick[i])] for i in sim2["fills"]}
    check("10 open positions block next-day signals (no waiting list)",
          sim2["n_filled"] == 10 and names2 == {f"T{j:02d}" for j in range(10)}, str(names2))
    check("time exit with flat price loses exactly the round-trip cost",
          abs(sim2["total_return"] - 10 * 0.1 * (-0.002)) < 1e-9, str(sim2["total_return"]))
    check("no leverage: equity never negative, MDD sane", sim2["mdd"] < 0.01 and sim2["equity"].min() > 0.99)


def test_portfolio_accounting():
    s = S(80)
    s.bar(30, c=88.0, o=100.0)
    s.bar(31, c=88.2, o=88.0, h=88 * 1.04, l=88.0)      # target
    s.bar(32, c=100.0, o=100.0)
    s.bar(50, c=88.0, o=100.0)
    s.bar(51, c=88.2, o=88.0, h=88 * 1.04, l=88.0)
    s.bar(52, c=100.0, o=100.0)
    P = mk_panel({"A": s})
    X = R.compute_exit_arrays(P, -0.01, 0.03, 1)
    sel = R.select_trades(P, X, R.candidate_cells(P, 0.10), 0, len(P.cal) - 1)
    sim = R.simulate_portfolio(P, sel, X, 0, len(P.cal) - 1, R.COST_PRIMARY)
    e1 = 1.0 + 0.1 * (0.03 - 0.002)
    e2 = e1 + 0.1 * e1 * (0.03 - 0.002)
    check("two sequential trades: 2nd sized at 10% of the CURRENT equity", abs(sim["equity"][-1] - e2) < 1e-6,
          f"{sim['equity'][-1]} vs {e2}")
    check("MDD of an always-winning book is 0 (entry cost shows up only as a one-day dip)", sim["mdd"] < 0.002)
    # mark-to-market: a 3-day hold that sinks then exits at time -> MDD reflects the dip at the close
    s2 = S(80)
    s2.bar(30, c=88.0, o=100.0)
    s2.bar(31, c=80.0, o=88.0, h=88.0, l=80.0)           # -9.1% close, STOP none
    s2.bar(32, c=88.0, o=80.0, h=88.5, l=80.0)
    s2.bar(33, c=88.0, o=88.0)
    P2 = mk_panel({"A": s2})
    X2 = R.compute_exit_arrays(P2, None, 0.05, 3)
    sel2 = R.select_trades(P2, X2, R.candidate_cells(P2, 0.10), 0, len(P2.cal) - 1)
    sim2 = R.simulate_portfolio(P2, sel2, X2, 0, len(P2.cal) - 1, R.COST_PRIMARY)
    check("MDD uses daily mark-to-market inside the hold",
          sim2["mdd"] > 0.08 * 0.1, f"mdd={sim2['mdd']}")
    X3 = R.compute_exit_arrays(P2, -0.01, 0.03, 1)
    sim3 = R.simulate_portfolio(P2, np.array([], dtype=np.int64), X3, 0, 40, R.COST_PRIMARY)
    check("empty trade list: total_return 0, mdd 0, sharpe nan",
          sim3["total_return"] == 0.0 and sim3["mdd"] == 0.0 and math.isnan(sim3["sharpe"]))


# ---------------------------------------------------------------------------
# 1. lookahead / truncation
# ---------------------------------------------------------------------------

def _flat_frame(P, X, drop):
    cand = np.zeros(P.N, bool)
    cand[R.candidate_cells(P, drop)] = True
    names = np.array(P.tickers)[P.tick]
    return pd.DataFrame({"t": names, "d": P.dnum, "sig": cand, "liq": P.liq_ok, "mem": P.member,
                         "sx": P.split_excl, "ret": X.ret.astype(np.float64), "code": X.code, "k": X.k})


def test_lookahead():
    series, mem, spl, spy, periods = R.make_synthetic(12, 420, seed=11, edge=True)
    full = R.build_panel(series, mem, spl)
    Xf = R.compute_exit_arrays(full, -0.01, 0.03, 5)
    ff = {d: _flat_frame(full, Xf, d) for d in R.DROPS}
    n_sig = int(ff[0.08]["sig"].sum())
    check("lookahead: synthetic panel has signals to test with", n_sig >= 20, str(n_sig))
    for cut in (150, 230, 333):
        sub = {t: df.iloc[:cut] for t, df in series.items()}
        cp = R.build_panel(sub, mem, spl)
        Xc = R.compute_exit_arrays(cp, -0.01, 0.03, 5)
        ok_sig = ok_flags = ok_exit = True
        for d in R.DROPS:
            m = _flat_frame(cp, Xc, d).merge(ff[d], on=["t", "d"], suffixes=("_c", "_f"))
            ok_sig &= (len(m) == cp.N) and bool((m["sig_c"] == m["sig_f"]).all())
            ok_flags &= bool((m["liq_c"] == m["liq_f"]).all() and (m["mem_c"] == m["mem_f"]).all()
                             and (m["sx_c"] == m["sx_f"]).all())
            det = m[m["code_c"] >= 0]
            ok_exit &= bool((det["code_c"] == det["code_f"]).all() and (det["k_c"] == det["k_f"]).all()
                             and np.allclose(det["ret_c"], det["ret_f"], atol=1e-7))
            # cells undeterminable in the cut but determinable in the full series: only the last 5 bars
            und = m[(m["code_c"] < 0) & (m["code_f"] >= 0)]
            ok_exit &= len(und) <= 5 * len(series)
        check(f"truncation at bar {cut}: signals identical up to the cut", ok_sig)
        check(f"truncation at bar {cut}: liquidity/membership/split flags identical", ok_flags)
        check(f"truncation at bar {cut}: every exit determinable in the cut equals the full-series exit", ok_exit)

    # garbage after bar m must not change anything at or before m (flags and signals)
    m_cut = 250
    rng = np.random.default_rng(5)
    garbage = {}
    for t, df in series.items():
        g = df.copy()
        for col in ("open", "high", "low", "close"):
            g.loc[g.index[m_cut + 1:], col] = rng.uniform(1, 500, len(g) - m_cut - 1)
        g.loc[g.index[m_cut + 1:], "volume"] = rng.uniform(1, 1e9, len(g) - m_cut - 1)
        garbage[t] = g
    gp = R.build_panel(garbage, mem, spl)
    Xg = R.compute_exit_arrays(gp, -0.01, 0.03, 5)
    cutoff = int(np.datetime64(str(series["T000"]["date"].iloc[m_cut].date()), "D").astype("int64"))
    ok = True
    for d in R.DROPS:
        m = _flat_frame(gp, Xg, d).merge(ff[d], on=["t", "d"], suffixes=("_g", "_f"))
        m = m[m["d"] <= cutoff]
        ok &= bool((m["sig_g"] == m["sig_f"]).all() and (m["liq_g"] == m["liq_f"]).all())
    check("future garbage after t leaves signal and liquidity flag at t unchanged", ok)


# ---------------------------------------------------------------------------
# 8. vectorised exits == literal SPEC §3 on wild data
# ---------------------------------------------------------------------------

def _wild_series(seed, n=300):
    rng = np.random.default_rng(seed)
    c = 100 * np.cumprod(1 + rng.normal(0, 0.02, n))
    prev = np.concatenate([[c[0]], c[:-1]])
    o = prev * (1 + rng.normal(0, 0.015, n))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.01, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.01, n)))
    s = S(n)
    s.o, s.h, s.l, s.c = o, h, l, c
    return s


def _ref_exit(P, i, stop, target, hold):
    end = int(P.off[int(P.tick[i]) + 1])
    if i + hold >= end:
        return float("nan"), -1, 0
    c0 = P.close[i]
    stop_px = None if stop is None else c0 * (1 + stop)
    tgt_px = c0 * (1 + target)
    for k in range(1, hold + 1):
        o, h, l, c = P.open[i + k], P.high[i + k], P.low[i + k], P.close[i + k]
        if stop_px is not None and o <= stop_px:
            return o / c0 - 1, 2, k
        if stop_px is not None and l <= stop_px:
            return stop, 1, k
        if h >= tgt_px:
            return target, 0, k
        if k == hold:
            return c / c0 - 1, 3, k
    raise AssertionError


def test_exits_match_reference():
    P = mk_panel({f"W{j}": _wild_series(100 + j) for j in range(6)})
    all_ok, seen = True, set()
    for (stop, target, hold) in R.combos():
        X = R.compute_exit_arrays(P, stop, target, hold)
        for i in range(P.N):
            r, cd, k = _ref_exit(P, i, stop, target, hold)
            seen.add(cd)
            good = (int(X.code[i]) == cd and int(X.k[i]) == k
                    and ((math.isnan(r) and math.isnan(float(X.ret[i]))) or abs(float(X.ret[i]) - r) < 1e-6))
            if not good:
                all_ok = False
                print(f"   mismatch combo={(stop, target, hold)} cell={i}: vec={(X.code[i], X.k[i], X.ret[i])} ref={(cd, k, r)}")
                break
    check("vectorised exits == literal SPEC §3 reading, all 24 rules x all cells", all_ok)
    check("the wild data exercised every exit reason (target, stop, gap, time, -1)", seen == {0, 1, 2, 3, -1}, str(seen))
    # candidate cells == pandas pct_change reference
    ok = True
    for t, s in {f"W{j}": _wild_series(100 + j) for j in range(6)}.items():
        pass
    series = {f"W{j}": _wild_series(100 + j) for j in range(6)}
    for d in (0.01, 0.02):    # wild data: small thresholds just to have events
        P2 = mk_panel(series)
        got = {(P2.tickers[int(P2.tick[i])], int(P2.dnum[i])) for i in R.candidate_cells(P2, d)}
        want = set()
        for t, s in series.items():
            ret = pd.Series(s.c).pct_change().to_numpy()
            dn = s.dates.values.astype("datetime64[D]").astype("int64")
            liq = pd.Series(s.c * s.v).shift(1).rolling(20).mean().to_numpy() >= R.LIQ_MIN_DOLLAR_VOL
            for q in np.flatnonzero((ret <= -d + 1e-9) & liq):
                want.add((t, int(dn[q])))
        ok &= got == want
    check("signal cells == pandas pct_change/rolling reference", ok)


# ---------------------------------------------------------------------------
# 9. liquidity guard + statistics helpers
# ---------------------------------------------------------------------------

def test_liquidity_and_stats():
    def with_dv(vals_by_idx, base_dv):
        s = S(80)
        s.v = np.full(80, base_dv / 100.0)
        for i, dv in vals_by_idx.items():
            s.v[i] = dv / 100.0
        s.bar(40, c=88.0, o=100.0)       # volume on the drop day left at base
        return s

    def signalled(s):
        P = mk_panel({"A": s})
        return P.cell("A", s.iso(40)) in set(R.candidate_cells(P, 0.10).tolist())

    check("liquidity: $1e6/day average -> vetoed", not signalled(with_dv({}, 1e6)))
    check("liquidity: exactly $50M average passes (>=)", signalled(with_dv({}, 5e7)))
    check("liquidity: $49.9M average vetoed", not signalled(with_dv({}, 4.99e7)))
    s = with_dv({40: 1e12}, 1e6)          # huge volume on day t itself must not rescue it
    s.v[40] = 1e12 / 88.0
    check("liquidity: day t's own volume is excluded from the guard", not signalled(s))
    check("liquidity: bar t-20 is inside the window", signalled(with_dv({20: 1e9}, 1e6)))
    check("liquidity: bar t-21 is outside the window", not signalled(with_dv({19: 1e9}, 1e6)))
    check("liquidity: bar t-1 is inside the window", signalled(with_dv({39: 1e9}, 1e6)))

    check("PF = sum(gains)/|sum(losses)|", abs(R.profit_factor(np.array([0.10, -0.05, 0.02])) - 0.12 / 0.05) < 1e-12)
    check("PF loss-free = inf, empty = nan", R.profit_factor(np.array([0.1, 0.2])) == float("inf")
          and math.isnan(R.profit_factor(np.array([]))))
    check("MDD of [1, 1.2, 0.9, 1.1] = 25%", abs(R.max_drawdown([1, 1.2, 0.9, 1.1]) - 0.25) < 1e-12)
    r = np.array([0.01, -0.02, 0.015, 0.0, 0.007])
    check("Sharpe = mean/std(ddof=1) x sqrt(252)", abs(R.sharpe_daily(r) - r.mean() / r.std(ddof=1) * math.sqrt(252)) < 1e-12)
    check("Sharpe of a constant series is NaN", math.isnan(R.sharpe_daily([0.0, 0.0, 0.0])))
    spy = pd.DataFrame({"date": pd.bdate_range("2016-01-04", periods=6), "close": [100, 120, 90, 95, 130, 100.0]})
    check("SPY B&H MDD over a sub-period", abs(R.spy_buyhold_mdd(spy, "2016-01-04", "2016-01-11") - 0.25) < 1e-12
          and math.isnan(R.spy_buyhold_mdd(None, "2016-01-04", "2016-01-11")))

    # net = gross - 2 x cost, PF sensitivity columns
    s = cycle_series(lambda s, i: s.bar(i, c=88 * 1.04, o=88.0, h=88 * 1.045, l=88.0))
    s.bar(75, c=88.0, o=100.0)
    s.bar(76, c=88 * 0.96, o=88.0, h=88.0, l=88 * 0.95)      # one loser (-1% stop)
    s.bar(77, c=100.0, o=100.0)
    P = mk_panel({"A": s})
    X, sel, st = run_variant(P, **PROP)
    n = st["n_trades"]
    gains, loss = (n - 1) * (0.03 - 0.002), 0.01 + 0.002
    check("trade stats: PF at 0.10% one-way", abs(st["pf"] - gains / loss) < 1e-4, f"{st['pf']} vs {gains / loss}")
    check("trade stats: PF at 0.05% / 0.20% one-way",
          abs(st["pf_cost005"] - (n - 1) * 0.029 / 0.011) < 1e-4 and abs(st["pf_cost020"] - (n - 1) * 0.026 / 0.014) < 1e-4)
    check("trade stats: win rate and exit counts", abs(st["win_rate"] - (n - 1) / n) < 1e-12
          and st["exit_stop"] == 1 and st["exit_target"] == n - 1)


# ---------------------------------------------------------------------------
# calendar / data hygiene
# ---------------------------------------------------------------------------

def test_data_hygiene():
    series = {f"T{j}": S(120) for j in range(10)}
    stray = series["T3"].df()
    extra = pd.DataFrame({"date": [pd.Timestamp("2016-02-27")], "open": [100.0], "high": [100.0],
                          "low": [100.0], "close": [100.0], "volume": [1e7]})     # a Saturday
    dfs = {t: s.df() for t, s in series.items()}
    dfs["T3"] = pd.concat([stray, extra]).sort_values("date").reset_index(drop=True)
    P = R.build_panel(dfs, {t: ALWAYS for t in dfs}, {t: None for t in dfs})
    check("a stray bar on a non-trading day (1 of 10 tickers) is dropped from calendar and series",
          np.datetime64("2016-02-27", "D") not in set(P.cal) and P.N == 10 * 120)

    # data hole: a ticker missing 2 weeks of bars -> the first bar back is no valid "previous close"
    a, b = S(120), S(120)
    b.bar(60, c=88.0, o=100.0)           # would be a signal, but follows a hole
    df_b = b.df().drop(index=range(50, 60)).reset_index(drop=True)
    P = R.build_panel({"A": a.df(), "B": df_b}, {"A": ALWAYS, "B": ALWAYS}, {"A": None, "B": None})
    check("signal right after a >5-calendar-day data hole is ignored",
          P.cell("B", b.iso(60)) not in set(R.candidate_cells(P, 0.10).tolist()))
    X = R.compute_exit_arrays(P, -0.01, 0.03, 5)
    ib = P.cell("B", b.iso(48))
    check("exit window crossing a data hole -> insufficient (-1)", int(X.code[ib]) == -1)
    dup = S(60).df()
    dup = pd.concat([dup, dup.iloc[[10]]]).sort_values("date").reset_index(drop=True)
    dup.loc[20, "close"] = 0.0
    P = R.build_panel({"D": dup}, {"D": ALWAYS}, {"D": None})
    check("duplicate/zero-price rows removed", P.N == 59, str(P.N))


# ---------------------------------------------------------------------------
# 10. placebo, PASS logic, HOLDOUT seal, determinism, outputs
# ---------------------------------------------------------------------------

def test_placebo_mechanics():
    rng = np.random.default_rng(1)
    pool = np.arange(100, 130)
    ret = np.zeros(300, dtype=np.float32)
    ret[pool] = np.linspace(-0.02, 0.05, len(pool))
    full_pf = R.profit_factor(ret[pool].astype(np.float64) - 0.002)
    pfs = R.placebo_pfs(rng, pool, ret, len(pool), 0.001, 10)
    check("placebo: n == pool size (without replacement) reproduces the pool's PF every run",
          bool(np.allclose(pfs, full_pf)), str(pfs[:3]))
    pfs2 = R.placebo_pfs(np.random.default_rng(1), pool, ret, 12, 0.001, 50)
    pfs3 = R.placebo_pfs(np.random.default_rng(1), pool, ret, 12, 0.001, 50)
    check("placebo: same seed -> identical draws", bool(np.array_equal(pfs2, pfs3)))
    check("placebo: draws vary across runs", len(set(np.round(pfs2, 9))) > 10)
    check("placebo: n larger than the pool is clipped, n=0 gives NaN",
          bool(np.allclose(R.placebo_pfs(rng, pool, ret, 999, 0.001, 3), full_pf))
          and bool(np.isnan(R.placebo_pfs(rng, pool, ret, 0, 0.001, 3)).all()))
    check("pct95 handles inf and empty", R.pct95(np.array([1.0, 2.0, np.inf])) > 2.0 and math.isnan(R.pct95(np.array([np.nan]))))

    series, mem, spl, spy, periods = R.make_synthetic(12, 420, seed=3, edge=True)
    P = R.build_panel(series, mem, spl)
    b = R.period_bounds(P, periods)
    pool = R.placebo_pool(P, 3, *b["IS"])
    sel = R.candidate_cells(P, 0.08)
    sel = sel[(P.gidx[sel] >= b["IS"][0]) & (P.gidx[sel] <= b["IS"][1]) & P.fut_ok[3][sel]]
    check("placebo pool contains every real signal of that hold/period (signals are a subset)",
          set(sel.tolist()) <= set(pool.tolist()) and len(pool) > len(sel))
    ok = bool(np.all(P.member[pool] & P.liq_ok[pool] & ~P.split_excl[pool] & P.fut_ok[3][pool]))
    check("placebo pool cells are member + liquid + not split-excluded + exit determinable", ok)


def test_pass_logic():
    base_is = dict(n_trades=150, pf=1.5, mdd=0.2, sharpe=1.0, spy_bh_mdd=0.3)
    oos = dict(pf=1.2)
    f = R.pass_flags(base_is, oos, True, True)
    check("PASS when every criterion holds", f["PASS"] and f["pass_trade_level"])
    for key, bad in [("n_trades", 99), ("pf", 1.29), ("mdd", 0.31), ("sharpe", 0.79), ("spy_bh_mdd", 0.2)]:
        row = dict(base_is)
        row[key] = bad
        f = R.pass_flags(row, oos, True, True)
        check(f"not PASS when {key}={bad}", not f["PASS"])
    check("boundary values pass: n=100, PF=1.3, MDD=30%, Sharpe=0.8",
          R.pass_flags(dict(n_trades=100, pf=1.3, mdd=0.30, sharpe=0.8, spy_bh_mdd=0.31), dict(pf=1.1), True, True)["PASS"])
    check("MDD == SPY MDD fails (strict <)", not R.pass_flags(dict(base_is, mdd=0.3, spy_bh_mdd=0.3), oos, True, True)["PASS"])
    check("not PASS when OOS PF < 1.1", not R.pass_flags(base_is, dict(pf=1.09), True, True)["PASS"])
    check("not PASS when placebo beaten in IS only / OOS only",
          not R.pass_flags(base_is, oos, True, False)["PASS"] and not R.pass_flags(base_is, oos, False, True)["PASS"])
    check("NaN metrics never pass", not R.pass_flags(dict(base_is, pf=float("nan")), oos, True, True)["PASS"]
          and not R.pass_flags(dict(base_is, spy_bh_mdd=float("nan")), oos, True, True)["PASS"]
          and not R.pass_flags(base_is, dict(pf=float("nan")), True, True)["PASS"])
    check("PF = inf passes the PF thresholds", R.pass_flags(dict(base_is, pf=float("inf")), dict(pf=float("inf")), True, True)["PASS"])
    f = R.pass_flags(dict(base_is, mdd=0.5), oos, False, False)
    check("pass_trade_level uses only (i),(ii),(vi) — MDD / placebo failures do not matter", f["pass_trade_level"] and not f["PASS"])
    check("(viii): real count <= placebo p95 -> NO EVIDENCE even with individual PASS", R.conclusion(3, 5, 5.0) == "NO EVIDENCE")
    check("(viii): real count > placebo p95 and a PASS -> EDGE", R.conclusion(3, 6, 5.0) == "EDGE")
    check("(viii): no individual PASS -> NO EVIDENCE", R.conclusion(0, 10, 0.0) == "NO EVIDENCE")


def _study(edge, seed=7, n_tickers=30, n_days=1500, placebo_n=20):
    series, mem, spl, spy, periods = R.make_synthetic(n_tickers, n_days, seed=seed, edge=edge)
    P = R.build_panel(series, mem, spl)
    return P, spy, periods


def test_study_end_to_end():
    # planted edge -> EDGE + holdout computed exactly once, for the highest-IS-PF passer
    P, spy, periods = _study(True)
    calls = []
    orig = R.evaluate_period

    def spy_eval(P_, X_, cand_, bounds_, period_iso_, spy_):
        calls.append(tuple(period_iso_))
        return orig(P_, X_, cand_, bounds_, period_iso_, spy_)

    R.evaluate_period = spy_eval
    try:
        res = R.run_study(P, spy, periods, 20, R.SEED, verbose=False)
    finally:
        R.evaluate_period = orig
    ho_calls = [c for c in calls if c == tuple(periods["HOLDOUT"])]
    check("planted edge: verdict EDGE", res["verdict"] == "EDGE", res["verdict"])
    check("HOLDOUT evaluated exactly once when a variant passes", len(ho_calls) == 1, str(len(ho_calls)))
    best = max(res["passers"], key=lambda v: res["rows"]["IS"][v]["pf"])
    check("HOLDOUT variant = passer with the highest IS PF", res["holdout"]["variant_tuple"] == best
          or res["rows"]["IS"][res["holdout"]["variant_tuple"]]["pf"] == res["rows"]["IS"][best]["pf"])
    check("(viii) bookkeeping: one placebo PASS count per run", len(res["plac_counts"]) == 20)

    # pure noise -> no HOLDOUT evaluation at all
    Pn, spyn, periodsn = _study(False)
    calls.clear()

    def spy_eval2(P_, X_, cand_, bounds_, period_iso_, spy_):
        calls.append(tuple(period_iso_))
        return orig(P_, X_, cand_, bounds_, period_iso_, spy_)

    R.evaluate_period = spy_eval2
    try:
        resn = R.run_study(Pn, spyn, periodsn, 100, R.SEED, verbose=False)
    finally:
        R.evaluate_period = orig
    check("noise: verdict NO EVIDENCE", resn["verdict"] == "NO EVIDENCE",
          f"{resn['verdict']} passers={len(resn['passers'])} real={resn['real_tl_count']} p95={resn['p95_count']}")
    check("noise: no PASS variant => HOLDOUT never computed (sealed)",
          resn["holdout"] is None and tuple(periodsn["HOLDOUT"]) not in calls, str(calls[-3:]))
    check("every real-signal period evaluated is IS or OOS only", set(calls) <= {tuple(periodsn["IS"]), tuple(periodsn["OOS"])})

    # determinism
    r1 = R.run_study(Pn, spyn, periodsn, 15, 123, verbose=False)
    r2 = R.run_study(Pn, spyn, periodsn, 15, 123, verbose=False)
    r3 = R.run_study(Pn, spyn, periodsn, 15, 124, verbose=False)
    g1, g2, g3 = (R.grid_dataframe(r, "IS") for r in (r1, r2, r3))
    check("same seed -> identical grid", g1.equals(g2) and r1["plac_counts"] == r2["plac_counts"])
    check("different seed -> different placebo thresholds", not g1["placebo_pf_p95"].equals(g3["placebo_pf_p95"]))
    check("real columns do not depend on the seed", g1["pf"].equals(g3["pf"]) and g1["n_trades"].equals(g3["n_trades"]))

    # outputs
    with tempfile.TemporaryDirectory() as d:
        written = R.write_outputs(res, d, {"requested": 30, "received": 30, "missing": []},
                                  {"loaded": 30, "skipped": 0}, "2026-11-01", synthetic=True)
        for name in ("grid_IS.csv", "grid_OOS.csv", "placebo_summary.csv", "SUMMARY.md", "data_coverage.json",
                     f"trades_{R.PROPOSAL_SLUG}.csv", "holdout.json", "summary.json"):
            check(f"output written: {name}", os.path.exists(os.path.join(d, name)))
        g = pd.read_csv(os.path.join(d, "grid_IS.csv"))
        need = ["drop", "stop", "target", "hold", "n_trades", "pf", "pf_cost005", "pf_cost020", "win_rate", "mean_ret",
                "median_ret", "mean_hold_days", "exit_target", "exit_stop", "exit_gap", "exit_time", "total_return",
                "cagr", "mdd", "sharpe", "spy_bh_mdd", "placebo_pf_p95", "beats_placebo", "PASS"]
        check("grid_IS.csv: 72 rows and every SPEC §7 column", len(g) == 72 and all(c in g.columns for c in need),
              str([c for c in need if c not in g.columns]))
        check("grid axes are exactly the SPEC grid",
              set(g["drop"]) == set(R.DROPS) and set(g["hold"]) == set(R.HOLDS) and set(g["target"]) == set(R.TARGETS)
              and set(g["stop"].astype(str)) == {"-0.01", "-0.03", "-0.05", "none"})
        ps = pd.read_csv(os.path.join(d, "placebo_summary.csv"))
        check("placebo_summary.csv: 72 variants x {IS,OOS}, 24 (STOP,TARGET,HOLD) rules", len(ps) == 144
              and len(ps[["stop", "target", "hold"]].drop_duplicates()) == 24)
        t = pd.read_csv(os.path.join(d, f"trades_{R.PROPOSAL_SLUG}.csv"))
        check("trades file columns", list(t.columns) == ["ticker", "entry_date", "entry_price", "exit_date", "exit_price",
                                                         "exit_reason", "gross_ret", "net_ret", "period"])
        check("trades file: only IS/OOS rows (holdout variant differs from the proposal)",
              set(t["period"]) <= {"IS", "OOS"} and len(t) == res["rows"]["IS"][R.PROPOSAL]["n_trades"] + res["rows"]["OOS"][R.PROPOSAL]["n_trades"])
        check("trades file: net = gross - 0.20% and exit_price = entry x (1+gross)",
              bool(np.allclose(t["net_ret"], t["gross_ret"] - 0.002)) and bool(np.allclose(t["exit_price"], t["entry_price"] * (1 + t["gross_ret"]))))
        check("trades file: exit after entry, reasons valid", bool((pd.to_datetime(t["exit_date"]) > pd.to_datetime(t["entry_date"])).all())
              and set(t["exit_reason"]) <= {"target", "stop", "gap", "time"})
        # re-derive every proposal trade from the raw series with the literal reference
        ok = True
        for row in t.head(60).itertuples():
            i = P.cell(row.ticker, row.entry_date)
            r, cd, k = _ref_exit(P, i, PROP["stop"], PROP["target"], PROP["hold"])
            ok &= abs(row.gross_ret - r) < 1e-6 and R.REASONS[cd] == row.exit_reason
        check("trades file matches the literal reference on the first 60 trades", ok)
        txt = open(os.path.join(d, "SUMMARY.md"), encoding="utf-8").read()
        check("SUMMARY.md is Korean and states the verdict, coverage and bias declaration",
              "전체 결론" in txt and "데이터 커버리지" in txt and "Survivorship bias" in txt and res["verdict"] in txt)
        ok_json = True
        for jn in ("summary.json", "holdout.json", "data_coverage.json"):
            try:
                json.loads(open(os.path.join(d, jn)).read(), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
            except ValueError:
                ok_json = False
        check("JSON outputs are strict JSON (no NaN / Infinity tokens)", ok_json)

    # a PASS-free run writes no holdout.json and says the seal held
    with tempfile.TemporaryDirectory() as d:
        R.write_outputs(resn, d, {"requested": 30, "received": 30, "missing": []}, {"loaded": 30, "skipped": 0},
                        "2026-11-01", synthetic=True)
        check("no PASS => no holdout.json", not os.path.exists(os.path.join(d, "holdout.json")))
        check("no PASS => SUMMARY says holdout stays sealed", "봉인 유지" in open(os.path.join(d, "SUMMARY.md"), encoding="utf-8").read())


def test_loaders():
    mem = R.load_constituents()
    check("constituents.csv: 751 distinct tickers, BRK.B / BF.B keep their dots",
          len(mem) == 751 and "BRK.B" in mem and "BF.B" in mem)
    check("constituents.csv: multi-interval ticker parsed (FSLR has 2)", len(mem["FSLR"]) == 2)
    check("constituents.csv: AAL interval [2015-03-23, 2024-09-23)",
          mem["AAL"] == [(np.datetime64("2015-03-23", "D"), np.datetime64("2024-09-23", "D"))])
    with tempfile.TemporaryDirectory() as d:
        s = S(60)
        s.df().to_csv(os.path.join(d, "BRK.B.csv"), index=False, date_format="%Y-%m-%d")
        json.dump(["2016-02-01"], open(os.path.join(d, "BRK.B.splits.json"), "w"))
        s.df().to_csv(os.path.join(d, "AAA.csv"), index=False, date_format="%Y-%m-%d")
        s.df().to_csv(os.path.join(d, "SPY.csv"), index=False, date_format="%Y-%m-%d")
        series, splits, missing = R.load_series(d, ["BRK.B", "AAA", "ZZZ"])
        check("load_series: original ticker spelling is the file name; splits file optional; missing reported",
              set(series) == {"BRK.B", "AAA"} and splits["BRK.B"] == ["2016-02-01"] and splits["AAA"] is None
              and missing == ["ZZZ"])
        check("load_spy reads SPY.csv; absent -> None", R.load_spy(d) is not None and R.load_spy(os.path.join(d, "nope")) is None)


def test_main_real_path():
    """The non-synthetic path: real constituents.csv + CSV/splits files on disk + fetch_prices'
    data_coverage.json passed through untouched."""
    mem = R.load_constituents()
    names = [t for t, iv in sorted(mem.items())
             if len(iv) == 1 and iv[0][1] is None and iv[0][0] <= np.datetime64("2015-01-01", "D")][:40]
    series, _, spl, spy, _ = R.make_synthetic(len(names), 2800, seed=5, edge=True)
    with tempfile.TemporaryDirectory() as d:
        data, out = os.path.join(d, "data"), os.path.join(d, "out")
        os.makedirs(data)
        for j, t in enumerate(names):
            series[f"T{j:03d}"].to_csv(os.path.join(data, f"{t}.csv"), index=False, date_format="%Y-%m-%d")
        json.dump(spl["T000"], open(os.path.join(data, f"{names[0]}.splits.json"), "w"))
        spy.to_csv(os.path.join(data, "SPY.csv"), index=False, date_format="%Y-%m-%d")
        old_argv = sys.argv
        try:
            sys.argv = ["run_panic_dip.py", "--data-dir", data, "--out-dir", out, "--placebo-n", "4"]
            rc = R.main()
        finally:
            sys.argv = old_argv
        check("main(): real path returns 0 and writes the SPEC §7 files",
              rc == 0 and all(os.path.exists(os.path.join(out, n)) for n in
                              ("grid_IS.csv", "grid_OOS.csv", "placebo_summary.csv", "SUMMARY.md", "data_coverage.json")))
        cov = json.load(open(os.path.join(out, "data_coverage.json")))
        check("main(): coverage written by the study when fetch_prices left none",
              cov["requested"] == 751 and cov["received"] == len(names) and len(cov["missing"]) == 751 - len(names))
        txt = open(os.path.join(out, "SUMMARY.md"), encoding="utf-8").read()
        check("main(): real PERIODS used and SPY found", "IS 2015-11-01~2021-10-31" in txt and "SPY 벤치마크: 있음" in txt)
        # fetch_prices' coverage file is passed through untouched
        fake = {"requested": 751, "received": 740, "missing": ["ZZZ1", "ZZZ2"], "tickers_with_split_events": 77}
        json.dump(fake, open(os.path.join(out, "data_coverage.json"), "w"))
        try:
            sys.argv = ["run_panic_dip.py", "--data-dir", data, "--out-dir", out, "--placebo-n", "4"]
            R.main()
        finally:
            sys.argv = old_argv
        check("main(): existing data_coverage.json (from fetch) is passed through unchanged",
              json.load(open(os.path.join(out, "data_coverage.json"))) == fake)
        txt = open(os.path.join(out, "SUMMARY.md"), encoding="utf-8").read()
        check("main(): SUMMARY quotes the fetch coverage numbers", "받은 종목 수: 740" in txt and "ZZZ1" in txt and "77개" in txt)
        # empty data dir -> graceful no-op (workflow must not fail on a missing fetch)
        try:
            sys.argv = ["run_panic_dip.py", "--data-dir", os.path.join(d, "empty"), "--out-dir", os.path.join(d, "o2")]
            check("main(): no data -> returns 0 without writing results", R.main() == 0 and not os.path.exists(os.path.join(d, "o2")))
        finally:
            sys.argv = old_argv


# ---------------------------------------------------------------------------
# 11. fetch_prices.py (no network)
# ---------------------------------------------------------------------------

class FakeResp:
    def __init__(self, status, payload=None):
        self.status_code = status
        self._p = payload

    def json(self):
        if self._p is None:
            raise ValueError("no json")
        return self._p


def _payload(with_events=True, splits=None, n=5):
    ts = [1451917800 + 86400 * i for i in range(n)]       # 2016-01-04 13:30 UTC onwards
    res = {"timestamp": ts, "indicators": {"quote": [{
        "open": [10.0 + i for i in range(n)], "high": [11.0 + i for i in range(n)], "low": [9.0 + i for i in range(n)],
        "close": [10.5 + i for i in range(n)], "volume": [1000 + i for i in range(n)]}]}}
    if with_events:
        res["events"] = {"splits": {str(t): {"date": t, "numerator": 2, "denominator": 1}
                                    for t in (splits or [])}}
    return {"chart": {"result": [res], "error": None}}


def test_fetch_prices():
    check("yahoo_symbol: BRK.B -> BRK-B, BF.B -> BF-B, plain unchanged",
          F.yahoo_symbol("BRK.B") == "BRK-B" and F.yahoo_symbol("BF.B") == "BF-B" and F.yahoo_symbol("AAPL") == "AAPL")
    check("period1 = 2015-06-01 UTC", F.period1_epoch() == 1433116800, str(F.period1_epoch()))
    uni = F.read_universe()
    check("read_universe: 751 distinct tickers, first-appearance order", len(uni) == 751 and uni[0] == "A" and len(set(uni)) == 751)
    url = F.CHART_URL.format(symbol="BRK-B", period1=F.period1_epoch(), period2=1)
    check("URL carries interval=1d and events=splits", "interval=1d" in url and "events=splits" in url and "BRK-B" in url)

    split_ts = 1451917800 + 86400 * 2
    check("parse_split_dates: ISO date from events.splits",
          F.parse_split_dates(_payload(splits=[split_ts])) == [pd.Timestamp(split_ts, unit="s").strftime("%Y-%m-%d")])
    check("parse_split_dates: events present but empty -> []", F.parse_split_dates(_payload(with_events=True)) == [])
    check("parse_split_dates: no events key -> None", F.parse_split_dates(_payload(with_events=False)) is None)
    check("parse_split_dates: garbage -> None, never raises",
          F.parse_split_dates({}) is None and F.parse_split_dates(None) is None and F.parse_split_dates({"chart": {"result": None}}) is None)

    # backoff: 429, 5xx, network error are retried with 2,4,8,16 s; 404 is not
    sleeps = []
    seq = [FakeResp(429), FakeResp(503), FakeResp(200, _payload())]
    resp, err = F.request_with_backoff("u", get=lambda *a, **k: seq.pop(0), sleep=sleeps.append)
    check("backoff: 429 then 503 then 200 -> sleeps [2, 4], returns the 200", sleeps == [2, 4] and resp.status_code == 200)
    sleeps.clear()
    n_calls = []
    resp, err = F.request_with_backoff("u", get=lambda *a, **k: (n_calls.append(1), FakeResp(500))[1], sleep=sleeps.append)
    check("backoff: persistent 500 -> 4 retries (5 calls) with sleeps [2,4,8,16]",
          sleeps == [2, 4, 8, 16] and len(n_calls) == 5 and resp.status_code == 500, f"{sleeps} {len(n_calls)}")
    sleeps.clear()
    n_calls.clear()
    resp, err = F.request_with_backoff("u", get=lambda *a, **k: (n_calls.append(1), FakeResp(404))[1], sleep=sleeps.append)
    check("backoff: 404 is final immediately", sleeps == [] and len(n_calls) == 1)

    import requests

    def boom(*a, **k):
        raise requests.exceptions.ConnectionError("down")
    sleeps.clear()
    resp, err = F.request_with_backoff("u", get=boom, sleep=sleeps.append)
    check("backoff: network errors retried, then reported (resp None)", resp is None and sleeps == [2, 4, 8, 16] and "ConnectionError" in err)

    r = F.fetch_symbol("X", 1, get=lambda *a, **k: FakeResp(404), sleep=lambda s: None)
    check("fetch_symbol: 404 -> missing", r["status"] == "missing" and r["reason"] == "HTTP 404")
    r = F.fetch_symbol("X", 1, get=lambda *a, **k: FakeResp(200, {"chart": {"result": None}}), sleep=lambda s: None)
    check("fetch_symbol: empty data -> missing", r["status"] == "missing" and r["reason"] == "empty data")
    r = F.fetch_symbol("X", 1, get=lambda *a, **k: FakeResp(200, _payload(splits=[split_ts])), sleep=lambda s: None)
    check("fetch_symbol: ok with split list", r["status"] == "ok" and len(r["df"]) == 5 and len(r["splits"]) == 1)

    with tempfile.TemporaryDirectory() as d:
        res_ev = {"status": "ok", "df": r["df"], "splits": ["2016-01-06"]}
        F.write_ticker(d, "BRK.B", res_ev)
        csv_df = pd.read_csv(os.path.join(d, "BRK.B.csv"))
        check("write_ticker: raw OHLCV columns, file named with the original ticker",
              list(csv_df.columns) == ["date", "open", "high", "low", "close", "volume"] and len(csv_df) == 5
              and csv_df["date"].iloc[0] == "2016-01-04")
        check("write_ticker: splits.json written when events exist", json.load(open(os.path.join(d, "BRK.B.splits.json"))) == ["2016-01-06"])
        F.write_ticker(d, "BRK.B", {"status": "ok", "df": r["df"], "splits": None})
        check("write_ticker: no events key -> no splits file (stale one removed)", not os.path.exists(os.path.join(d, "BRK.B.splits.json")))

        # full run() with fake HTTP: 3 tickers (A, AABA, AAL) + SPY
        def fake_get(url, **k):
            if "/AABA?" in url:
                return FakeResp(404)
            if "/AAL?" in url:
                return FakeResp(200, _payload(with_events=False))
            return FakeResp(200, _payload(splits=[split_ts]))
        out, resd = os.path.join(d, "px"), os.path.join(d, "res")
        cov = F.run(3, out, resd, get=fake_get, sleep=lambda s: None)
        on_disk = json.load(open(os.path.join(resd, "data_coverage.json")))
        check("run(): coverage counts (requested 3, received 2, missing AABA)",
              on_disk["requested"] == 3 and on_disk["received"] == 2 and on_disk["missing"] == ["AABA"], str(on_disk["missing"]))
        check("run(): SPY fetched explicitly, kept out of the universe counts",
              "SPY" in on_disk["benchmark"] and os.path.exists(os.path.join(out, "SPY.csv")) and "SPY" not in on_disk["per_ticker"])
        check("run(): split-event ticker count and per-ticker first/last/rows",
              on_disk["tickers_with_split_events"] == 1 and on_disk["per_ticker"]["A"]["rows"] == 5
              and on_disk["per_ticker"]["A"]["first_date"] == "2016-01-04")
        check("run(): AAL (no events key) has CSV but no splits file; A has both",
              os.path.exists(os.path.join(out, "AAL.csv")) and not os.path.exists(os.path.join(out, "AAL.splits.json"))
              and os.path.exists(os.path.join(out, "A.splits.json")))
        # the study can read what fetch wrote
        series, splits, missing = R.load_series(out, ["A", "AABA", "AAL"])
        check("run_panic_dip.load_series reads fetch_prices output", set(series) == {"A", "AAL"} and missing == ["AABA"]
              and splits["A"] is not None and splits["AAL"] is None)


# ---------------------------------------------------------------------------

def test_cli_self_test():
    with tempfile.TemporaryDirectory() as d:
        out = R.run_synthetic(n_tickers=30, n_days=1500, placebo_n=3, out_dir=d)
        check("--synthetic path runs end-to-end and writes outputs", os.path.exists(os.path.join(d, "SUMMARY.md")))
    check("variant grid is 3 x 4 x 2 x 3 = 72 and the proposal cell is in it",
          len(R.build_variants()) == 72 and R.PROPOSAL in R.build_variants() and len(R.combos()) == 24
          and R.variant_slug(*R.PROPOSAL) == R.PROPOSAL_SLUG)


def main():
    tests = [test_synthetic_a_rebound, test_synthetic_b_continued_fall, test_exit_rules, test_split_exclusion,
             test_membership_pit, test_slot_overflow, test_portfolio_accounting, test_lookahead,
             test_exits_match_reference, test_liquidity_and_stats, test_data_hygiene, test_placebo_mechanics,
             test_pass_logic, test_study_end_to_end, test_loaders, test_main_real_path, test_fetch_prices,
             test_cli_self_test]
    for t in tests:
        try:
            t()
        except Exception as e:  # noqa: BLE001 — a crashing test is a failing test, not an abort
            traceback.print_exc()
            check(f"{t.__name__} ran without raising", False, f"{type(e).__name__}: {e}")
    print()
    if FAILURES:
        print(f"{len(FAILURES)} TEST(S) FAILED:")
        for name, detail in FAILURES:
            print(f" - {name}: {detail}")
        return 1
    print("ALL PANIC_DIP TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
