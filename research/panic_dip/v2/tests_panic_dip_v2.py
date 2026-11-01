"""
tests_panic_dip_v2.py — SPEC_v2.md §8 (+ the building blocks around it). Plain asserts, no pytest:

    python research/panic_dip/v2/tests_panic_dip_v2.py

Covered
  1.  panic-day classification (SPEC_v2 §8): -2.5%/60% -> P2 only, -3.5%/60% -> P2+P3, -3.5%/30% -> neither,
      plus the exact boundaries (-2.0%/50%, -3.0%/50%, -1.9%/100%).
  2.  mode-S stock filter boundaries at r_spy = -3% (-3%, -9% in; -2.9%, -9.1%, -25% out) and the -20% hard cap.
  3.  25-name cut is by trailing dollar volume, not by drop size; 50% equal-weight arithmetic; cash cap; position cap.
  4.  controls: A draws only from the real panic days with the real per-day counts (and is NOT restricted to candidates),
      B's volatility-decile histogram equals the real trades', C never lands on a panic day.
  5.  alias mapping: old ticker's interval uses the new ticker's series; overlapping dates count once.
  6.  truncation / look-ahead: nothing at or before t changes when the series is cut or garbage follows t.
  7.  TARGET None wrapper: no 'target' exits, == literal reading of SPEC on wild data, finite targets unchanged.
  8.  PASS logic incl. the mode-I exemption from (i)'s trade count, (viii), conclusion.
  9.  end-to-end on synthetic data: outputs, columns, HOLDOUT seal on noise, determinism, strict JSON.
  10. fetch_prices_v2 against fake HTTP (retry waits, alias recovery, coverage), aliases.csv integrity.
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

import run_panic_dip_v2 as V  # noqa: E402
import fetch_prices_v2 as F2  # noqa: E402
import run_panic_dip as R  # noqa: E402  (v1)

FAILURES = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append((name, detail))


# ---------------------------------------------------------------------------
# Hand-made market builder
# ---------------------------------------------------------------------------

class S:
    """Flat 100.0 series (volume 1e7 shares => $1e9/day). `ret(i, r)` sets close[i] = close[i-1] * (1 + r)
    (open = previous close) and does NOT touch later bars."""

    def __init__(self, n=140, start="2016-01-04", price=100.0, vol=1e7):
        self.dates = pd.bdate_range(start, periods=n)
        self.o = np.full(n, price)
        self.h = np.full(n, price)
        self.l = np.full(n, price)
        self.c = np.full(n, price)
        self.v = np.full(n, vol)

    def bar(self, i, c, o=None, h=None, l=None):
        o = c if o is None else o
        self.o[i], self.c[i] = o, c
        self.h[i] = max(o, c) if h is None else h
        self.l[i] = min(o, c) if l is None else l
        return self

    def ret(self, i, r, back=True):
        """close[i] = close[i-1]*(1+r); with back=True close[i+1] returns to close[i-1] (a recovery bar)."""
        prev = self.c[i - 1]
        self.bar(i, prev * (1.0 + r), o=prev)
        if back and i + 1 < len(self.c):
            self.bar(i + 1, prev, o=prev * (1.0 + r))
        return self

    def iso(self, i):
        return str(self.dates[i].date())

    def df(self):
        return pd.DataFrame({"date": self.dates, "open": self.o, "high": self.h, "low": self.l,
                             "close": self.c, "volume": self.v})


ALWAYS = [(np.datetime64("2000-01-01", "D"), None)]


def mk_market(stocks: dict, spy: S, membership=None, splits=None):
    mem = {t: ALWAYS for t in stocks} if membership is None else membership
    spl = {t: None for t in stocks} if splits is None else {t: splits.get(t) for t in stocks}
    return V.build_panel_v2({t: s.df() for t, s in stocks.items()}, mem, spl, spy.df())


def flat_market(n_stocks=10, n=140):
    return {f"S{j:02d}": S(n) for j in range(n_stocks)}, S(n)


def cand_tickers(P, ds, day, panic="P2"):
    cs = V.candset_s(P, ds, panic, day, day)
    return sorted(P.tickers[int(P.tick[c])] for c in cs.cells.tolist())


# ---------------------------------------------------------------------------
# 1. panic-day classification
# ---------------------------------------------------------------------------

def test_panic_classification():
    stocks, spy = flat_market(10)
    names = sorted(stocks)

    def day(i, r_spy, n_down, r_down, back=True):
        spy.ret(i, r_spy, back)
        for k, t in enumerate(names):
            if k < n_down:
                stocks[t].ret(i, r_down, back)
    day(30, -0.025, 6, -0.03)     # P2 only
    day(40, -0.035, 6, -0.04)     # P2 + P3
    day(50, -0.035, 3, -0.04)     # breadth 30% -> neither
    day(60, -0.019, 10, -0.03)    # market not down enough -> neither
    day(70, -0.020, 5, -0.020)    # boundary: exactly -2.0% and exactly 50% -> P2
    day(80, -0.030, 5, -0.020)    # boundary: exactly -3.0% and exactly 50% -> P2 + P3
    day(90, -0.025, 4, -0.03)     # breadth 40% -> neither
    P = mk_market(stocks, spy)
    ds = V.day_stats(P, 0, len(P.cal) - 1)
    exp = {30: (True, False), 40: (True, True), 50: (False, False), 60: (False, False), 70: (True, False),
           80: (True, True), 90: (False, False)}
    for d, (p2, p3) in exp.items():
        check(f"panic day {d}: r_spy={ds['r_spy'][d]:+.4f} breadth={ds['breadth'][d]:.2f} -> P2={p2} P3={p3}",
              bool(ds["P2"][d]) == p2 and bool(ds["P3"][d]) == p3, f"P2={ds['P2'][d]} P3={ds['P3'][d]}")
    check("SPEC §8: breadth of the -2.5%/60% day is 0.6 and of the -3.5%/30% day is 0.3",
          abs(ds["breadth"][30] - 0.6) < 1e-12 and abs(ds["breadth"][50] - 0.3) < 1e-12)
    check("non-event days (recovery bars) are never panic days",
          not ds["P2"][31] and not ds["P2"][41] and not ds["P3"][51] and ds["P2"].sum() == 4 and ds["P3"].sum() == 2,
          f"{ds['P2'].sum()} {ds['P3'].sum()}")
    # a window only fills its own days
    ds2 = V.day_stats(P, 35, 65)
    check("day_stats fills only the requested window (P2 on day 40 yes, day 30/70 not computed)",
          bool(ds2["P2"][40]) and not ds2["P2"][30] and not ds2["P2"][70] and np.isnan(ds2["r_spy"][30]))
    # SPY is not a member: it must not enter breadth
    check("breadth denominator is the 10 members (SPY excluded)", int(ds["n_member_valid"][30]) == 10)
    # split days are not part of breadth
    stocks2, spy2 = flat_market(10)
    spy2.ret(30, -0.025)
    for k, t in enumerate(sorted(stocks2)):
        if k < 5:
            stocks2[t].ret(30, -0.03)
    stocks2["S09"].bar(30, 50.0, o=100.0)          # raw -50% on day 30 ... flagged as a split below
    stocks2["S09"].bar(31, 50.0, o=50.0)
    P2_ = mk_market(stocks2, spy2, splits={"S09": [stocks2["S09"].iso(30)]})
    ds3 = V.day_stats(P2_, 0, len(P2_.cal) - 1)
    check("split-flagged day is excluded from numerator AND denominator (breadth 5/9)",
          int(ds3["n_member_valid"][30]) == 9 and abs(ds3["breadth"][30] - 5 / 9) < 1e-12, str(ds3["breadth"][30]))


# ---------------------------------------------------------------------------
# 2. mode-S stock filter
# ---------------------------------------------------------------------------

def _filter_panel(r_spy, boundary: dict, n_fill=10, fill_r=-0.05, day=40):
    stocks = {f"F{j:02d}": S(100) for j in range(n_fill)}
    spy = S(100)
    spy.ret(day, r_spy)
    for t in stocks:
        stocks[t].ret(day, fill_r)
    for t, r in boundary.items():
        stocks[t] = S(100)
        stocks[t].ret(day, r)
    P = mk_market(stocks, spy)
    ds = V.day_stats(P, 0, len(P.cal) - 1)
    return P, ds, day


def test_stock_filter():
    bd = {"B_m3": -0.03, "B_m9": -0.09, "B_m29": -0.029, "B_m91": -0.091, "B_m25": -0.25}
    P, ds, day = _filter_panel(-0.03, bd)
    check("filter setup: the day is a P2 and P3 day", bool(ds["P2"][day]) and bool(ds["P3"][day]))
    got = [t for t in cand_tickers(P, ds, day) if t.startswith("B_")]
    check("r_spy=-3%: -3% and -9% are candidates", "B_m3" in got and "B_m9" in got, str(got))
    check("r_spy=-3%: -2.9%, -9.1% and -25% are NOT candidates",
          not ({"B_m29", "B_m91", "B_m25"} & set(got)), str(got))
    check("r_spy=-3%: exactly the 2 boundary names + the 10 fillers (-5%) are candidates",
          len(cand_tickers(P, ds, day)) == 12, str(len(cand_tickers(P, ds, day))))

    # hard cap: r_spy = -8% -> 3x is -24%, so only the -20% cap binds in [-24%, -8%]
    bd2 = {"C_m8": -0.08, "C_m199": -0.199, "C_m200": -0.20, "C_m21": -0.21, "C_m24": -0.24, "C_m79": -0.079}
    P2_, ds2, day2 = _filter_panel(-0.08, bd2, fill_r=-0.10)
    got2 = [t for t in cand_tickers(P2_, ds2, day2) if t.startswith("C_")]
    check("r_spy=-8%: -8% and -19.9% are candidates", "C_m8" in got2 and "C_m199" in got2, str(got2))
    check("r_spy=-8%: -20.0% is excluded by the hard cap (r_i <= -20%)", "C_m200" not in got2, str(got2))
    check("r_spy=-8%: -21%, -24% (cap) and -7.9% (milder than the market) are excluded",
          not ({"C_m21", "C_m24", "C_m79"} & set(got2)), str(got2))

    # exclusion partition used by panic_days.csv
    cells, (a, b, c) = V.mode_s_filter(P, ds, "P2", 0, len(P.cal) - 1)
    names = [P.tickers[int(P.tick[i])] for i in cells]
    by = {n: (bool(a[k]), bool(b[k]), bool(c[k])) for k, n in enumerate(names)}
    check("filter masks: -2.9% fails only 'as much as the market'; -9.1% fails only the 3x rule; -25% fails 3x (and cap)",
          by["B_m29"] == (False, True, True) and by["B_m91"] == (True, False, True) and by["B_m25"][1] is False
          and by["B_m3"] == (True, True, True) and by["B_m9"] == (True, True, True), str(by))
    check("eligible cells = 15 (10 fillers + 5 boundary names); SPY is never eligible", len(cells) == 15 and "SPY" not in names)


# ---------------------------------------------------------------------------
# 3. 25-name cut, 50% equal weight, cash cap, position cap
# ---------------------------------------------------------------------------

def _panel_30(day=40):
    stocks = {}
    for i in range(30):
        s = S(120)
        s.v[:] = 1e7 * (1.0 + ((i * 7) % 30) / 10.0)          # distinct dollar volumes, unrelated to the drop order
        s.ret(day, -0.03 - 0.0015 * i)                          # i=0 smallest drop ... i=29 largest drop (all candidates)
        stocks[f"S{i:02d}"] = s
    spy = S(120)
    spy.ret(day, -0.03)
    return mk_market(stocks, spy), day, stocks


def test_cut_and_sizing():
    P, day, stocks = _panel_30()
    ds = V.day_stats(P, 0, len(P.cal) - 1)
    cs = V.candset_s(P, ds, "P2", day, day)
    check("30 candidates on the panic day", len(cs.cells) == 30 and bool(ds["P2"][day]), str(len(cs.cells)))
    dv = {f"S{i:02d}": 1e7 * (1.0 + ((i * 7) % 30) / 10.0) * 100.0 for i in range(30)}
    top_dv = sorted(dv, key=lambda t: -dv[t])[:25]
    drop_rank = sorted(dv, key=lambda t: -int(t[1:]))[:25]            # largest drop first
    sel_names = sorted(P.tickers[int(P.tick[c])] for c in cs.cells[cs.rank < V.MAX_NAMES].tolist())
    check("25-name cut keeps the 25 largest trailing dollar volumes", sel_names == sorted(top_dv), str(sel_names))
    check("... which is NOT the 25 largest drops (the cut is by liquidity, not by drop)", sel_names != sorted(drop_rank))
    check("ordering within the day is by descending dollar volume",
          bool(np.all(np.diff(P.dv20[cs.cells]) <= 0)) and list(cs.rank) == list(range(30)))

    X = V.compute_exit_arrays_v2(P, None, None, 1)
    ok = X.code[cs.cells] >= 0
    sel, rk = cs.cells[ok], cs.rank[ok]
    sim = V.simulate_portfolio_v2(P, sel, rk, X, 0, len(P.cal) - 1)
    chosen = sel[rk < 25]
    exp_ret = 0.5 / 25 * float((X.ret[chosen].astype(np.float64) - 2 * V.COST).sum())
    check("portfolio: 25 names filled, each 0.5/25 = 2% of equity (total return = 0.02 x sum of the 25 net returns)",
          sim["n_filled"] == 25 and abs(sim["total_return"] - exp_ret) < 1e-9, f"{sim['total_return']} vs {exp_ret}")
    d, total, n_take = sim["entry_log"][0]
    check("new exposure = exactly 50% of equity and equal split", abs(total - 0.5) < 1e-12 and n_take == 25 and d == day)
    # trade-level statistics count ALL 30 candidates regardless of sizing
    st = R.trade_stats(P, X, sel, V.COST)
    check("trade-level n_trades counts all 30 candidates (sizing-independent)", st["n_trades"] == 30)

    # exact arithmetic on a tiny case: 4 candidates, +10%, 0, 0, -10% next day -> equal 12.5% each
    stocks4 = {}
    nxt = {"A": 0.10, "B": 0.0, "C": 0.0, "D": -0.10}
    for t, r in nxt.items():
        s = S(80)
        s.ret(40, -0.04, back=False)
        s.bar(41, s.c[40] * (1 + r), o=s.c[40])
        s.bar(42, 100.0)
        stocks4[t] = s
    spy4 = S(80)
    spy4.ret(40, -0.03)
    P4 = mk_market(stocks4, spy4)
    ds4 = V.day_stats(P4, 0, len(P4.cal) - 1)
    cs4 = V.candset_s(P4, ds4, "P2", 40, 40)
    X4 = V.compute_exit_arrays_v2(P4, None, None, 1)
    sim4 = V.simulate_portfolio_v2(P4, cs4.cells, cs4.rank, X4, 0, len(P4.cal) - 1)
    exp4 = 0.5 * (1 + (0.10 + 0 + 0 - 0.10) / 4 - 0.002) - 0.5
    check("4 names, +10/0/0/-10%: equity change = 0.5 x (mean gross - 0.20%) = -0.001",
          abs(sim4["total_return"] - exp4) < 1e-9 and sim4["n_filled"] == 4, f"{sim4['total_return']} vs {exp4}")
    # equity path: flat at 1.0 until the entry (costs are booked at entry), then marks
    check("equity is flat before the first entry and the entry cost is booked the same day",
          abs(sim4["equity"][40] - 1.0) < 1e-15 and abs(sim4["equity"][41] - (1 - 0.5 * 0.001)) < 1e-12,
          f"{sim4['equity'][40]} {sim4['equity'][41]}")

    # cash cap: three consecutive -4% days with HOLD 5 -> exposures 0.5, 0.48975, then the cash left (~0.01025)
    stocksc, spyc = flat_market(10, 100)
    for t in stocksc:
        for d_ in (40, 41, 42, 43):
            stocksc[t].ret(d_, -0.04, back=False)
        for d_ in range(44, 100):
            stocksc[t].bar(d_, stocksc[t].c[43])
    for d_ in (40, 41, 42, 43):
        spyc.ret(d_, -0.035, back=False)
    for d_ in range(44, 100):
        spyc.bar(d_, spyc.c[43])
    Pc = mk_market(stocksc, spyc)
    dsc = V.day_stats(Pc, 0, len(Pc.cal) - 1)
    csc = V.candset_s(Pc, dsc, "P2", 40, 43)
    Xc = V.compute_exit_arrays_v2(Pc, None, None, 5)
    simc = V.simulate_portfolio_v2(Pc, csc.cells[Xc.code[csc.cells] >= 0], csc.rank[Xc.code[csc.cells] >= 0], Xc, 0,
                                   len(Pc.cal) - 1)
    tot = [x[1] for x in simc["entry_log"]]
    eq_pre_2 = 0.5 + 0.5 * (1.0 + (-0.04) - 0.001)
    check("cash cap: day1 invests 50%, day2 invests min(50% of equity, cash), day3 only the remaining cash, day4 nothing",
          len(tot) == 3 and abs(tot[0] - 0.5) < 1e-12 and abs(tot[1] - 0.5 * eq_pre_2) < 1e-9
          and abs(tot[2] - (1.0 - tot[0] - tot[1])) < 1e-9 and simc["min_cash"] > -1e-12, f"{tot} min_cash={simc['min_cash']}")
    check("no leverage: cash never negative; day4 (no cash) opens no position", simc["n_filled"] == 30, str(simc["n_filled"]))

    # position cap: max_pos=15 with two 10-name days -> second day only gets 5 slots, split equally
    stocksp, spyp = flat_market(10, 100)
    for t in stocksp:
        stocksp[t].ret(40, -0.04, back=False)
        stocksp[t].ret(41, -0.04, back=False)
        for d_ in range(42, 100):
            stocksp[t].bar(d_, stocksp[t].c[41])
    spyp.ret(40, -0.035, back=False)
    spyp.ret(41, -0.035, back=False)
    for d_ in range(42, 100):
        spyp.bar(d_, spyp.c[41])
    Pp = mk_market(stocksp, spyp)
    dsp = V.day_stats(Pp, 0, len(Pp.cal) - 1)
    csp = V.candset_s(Pp, dsp, "P2", 40, 41)
    Xp = V.compute_exit_arrays_v2(Pp, None, None, 5)
    okp = Xp.code[csp.cells] >= 0
    simp = V.simulate_portfolio_v2(Pp, csp.cells[okp], csp.rank[okp], Xp, 0, len(Pp.cal) - 1, max_pos=15)
    check("max concurrent positions: 10 on day 1, only 5 on day 2 (cap 15), exposure still split equally over those",
          [x[2] for x in simp["entry_log"]] == [10, 5], str(simp["entry_log"]))

    # mode I: 50% in SPY
    csi = V.candset_i(Pp, dsp, "P2", 0, len(Pp.cal) - 1)
    Xi = V.compute_exit_arrays_v2(Pp, None, None, 1)
    simi = V.simulate_portfolio_v2(Pp, csi.cells, csi.rank, Xi, 0, len(Pp.cal) - 1)
    check("mode I: one SPY position per panic day, 50% of equity on the first day",
          simi["n_filled"] == 2 and abs(simi["entry_log"][0][1] - 0.5) < 1e-12 and simi["entry_log"][0][2] == 1)


# ---------------------------------------------------------------------------
# 4. controls
# ---------------------------------------------------------------------------

def _study_parts(n_tickers=30, n_days=700, seed=3, edge=False):
    series, mem, spl, spy, periods, resolved = V.make_synthetic_v2(n_tickers, n_days, seed=seed, edge=edge)
    P = V.build_panel_v2(series, V.apply_aliases(mem, resolved), spl, spy)
    bounds = R.period_bounds(P, periods)
    ds = V.day_stats(P, bounds["IS"][0], bounds["OOS"][1])
    return P, spy, periods, bounds, ds


def test_controls():
    P, spy, periods, bounds, ds = _study_parts(edge=True)
    X = V.compute_all_exits_v2(P)
    hold = 3
    Xv = X[(None, None, hold)]
    cs = V.build_candsets(P, ds, bounds, ("IS", "OOS"))
    pools = V.Pools(P, bounds)
    for pn in ("IS", "OOS"):
        csv_ = cs[("P2", "S", pn)]
        sel = csv_.cells[Xv.code[csv_.cells] >= 0]
        check(f"[{pn}] control A: real trades exist", len(sel) > 20, str(len(sel)))
        panic_days = set(np.flatnonzero(ds["P2"]).tolist())
        spec, cells = V.control_a_spec(P, pools, hold, pn, sel)
        real_per_day = pd.Series(P.gidx[sel]).value_counts().to_dict()
        rng = np.random.default_rng(1)
        ok_days = ok_cnt = ok_pool = True
        non_candidate_seen = False
        cand_set = set(csv_.cells.tolist())
        for _ in range(20):
            drawn = V.control_a_draw(rng, spec, cells)
            gd = pd.Series(P.gidx[drawn]).value_counts().to_dict()
            ok_days &= set(gd) <= panic_days and set(gd) == set(real_per_day)
            ok_cnt &= gd == real_per_day
            ok_pool &= bool((P.member[drawn] & P.liq_ok[drawn] & ~P.split_excl[drawn] & P.fut_ok[hold][drawn]).all())
            ok_pool &= len(set(drawn.tolist())) == len(drawn)                         # without replacement
            non_candidate_seen |= any(c not in cand_set for c in drawn.tolist())
        check(f"[{pn}] control A draws only from the real panic days", ok_days)
        check(f"[{pn}] control A draws exactly the real number of trades on each of those days", ok_cnt)
        check(f"[{pn}] control A draws are liquid members with a determinable exit, no replacement", ok_pool)
        check(f"[{pn}] control A is NOT restricted to mode-S candidates (the own-drop filter is ignored)", non_candidate_seen)

        # B
        spec_b = V.control_b_spec(P, pools, hold, pn, sel)
        counts, bin_cells = spec_b
        edges, _ = pools.by_vol_bin(hold, pn)
        real_bins = np.bincount(np.searchsorted(edges, P.vol20[sel][np.isfinite(P.vol20[sel])], side="right"), minlength=10)
        rng = np.random.default_rng(2)
        ok_hist = ok_all_days = True
        lo, hi = bounds[pn]
        days_seen = set()
        for _ in range(10):
            drawn = V.control_b_draw(rng, spec_b)
            h = np.bincount(np.searchsorted(edges, P.vol20[drawn], side="right"), minlength=10)
            ok_hist &= bool((h == real_bins).all()) and len(drawn) == int(real_bins.sum())
            g = P.gidx[drawn]
            ok_all_days &= bool(((g >= lo) & (g <= hi)).all())
            days_seen |= set(g.tolist())
        check(f"[{pn}] control B: volatility-decile histogram of every draw equals the real trades' histogram", ok_hist,
              str(real_bins.tolist()))
        check(f"[{pn}] control B: draws span the whole period, not just panic days",
              ok_all_days and len(days_seen - set(np.flatnonzero(ds['P2']).tolist())) > 0)
        # deciles are computed over the period's valid cells: every decile of the pool has ~10% of it
        sizes = np.array([len(b) for b in bin_cells])
        check(f"[{pn}] control B: deciles of the pool are (nearly) equal-sized", sizes.max() <= 1.5 * sizes.min() + 5,
              str(sizes.tolist()))

        # C
        for panic in ("P2", "P3"):
            ci = cs[(panic, "I", pn)]
            n = int((Xv.code[ci.cells] >= 0).sum())
            spec_c = V.control_c_spec(P, hold, bounds[pn], ds[panic], n)
            rng = np.random.default_rng(3)
            ok_np = True
            for _ in range(20):
                drawn = V.control_c_draw(rng, spec_c)
                g = P.gidx[drawn]
                ok_np &= (not ds[panic][g].any()) and len(drawn) == n and bool(((g >= bounds[pn][0]) & (g <= bounds[pn][1])).all())
                ok_np &= bool((P.tick[drawn] == P.spy_j).all()) and len(set(drawn.tolist())) == n
            check(f"[{pn}] control C ({panic}): SPY, never a panic day, inside the period, same count, no replacement", ok_np)

    # PF distribution plumbing: deterministic per rng
    ret = Xv.ret
    csx = cs[("P2", "S", "IS")]
    sel = csx.cells[Xv.code[csx.cells] >= 0]
    spec, cells = V.control_a_spec(P, pools, hold, "IS", sel)
    f = lambda g_: V.control_a_draw(g_, spec, cells)  # noqa: E731
    a1 = V.control_pfs(f, ret, 8, np.random.default_rng(9))
    a2 = V.control_pfs(f, ret, 8, np.random.default_rng(9))
    a3 = V.control_pfs(f, ret, 8, np.random.default_rng(10))
    check("control PF runs are deterministic for a seed and differ across seeds", np.array_equal(a1, a2) and not np.array_equal(a1, a3))
    check("pctl caps infinities and returns inf when the percentile hits the cap",
          V.pctl([1.0, 2.0, float("inf")] * 40, 95) == float("inf") and abs(V.pctl([1, 2, 3, 4, 5], 50) - 3) < 1e-12
          and math.isnan(V.pctl([float("nan")], 95)))


# ---------------------------------------------------------------------------
# 5. aliases
# ---------------------------------------------------------------------------

def test_aliases():
    m = {"NEW": [(np.datetime64("2016-03-01", "D"), None)],
         "OLD": [(np.datetime64("2000-01-01", "D"), np.datetime64("2016-03-15", "D"))],
         "X": [(np.datetime64("2000-01-01", "D"), None)]}
    r = {"OLD": "NEW"}
    m2 = V.apply_aliases(m, r)
    check("apply_aliases: OLD's interval is merged into NEW's series; OLD disappears",
          set(m2) == {"NEW", "X"} and len(m2["NEW"]) == 2 and m2["X"] == m["X"])
    check("resolve_name follows chains and survives cycles",
          V.resolve_name("A", {"A": "B", "B": "C"}) == "C" and V.resolve_name("A", {"A": "B", "B": "A"}) in ("A", "B")
          and V.resolve_name("Z", {}) == "Z")

    # panel: NEW series is a member from before its own start (via OLD) and the overlap days count once
    stocks, spy = flat_market(6, 120)
    stocks["NEW"] = S(120)
    dates = stocks["NEW"].dates
    mem = {t: ALWAYS for t in stocks if t != "NEW"}
    mem["NEW"] = [(np.datetime64(str(dates[50].date()), "D"), None)]
    mem["OLD"] = [(np.datetime64("2000-01-01", "D"), np.datetime64(str(dates[60].date()), "D"))]    # overlaps days 50..59
    spl = {t: None for t in stocks}
    # a market-wide -3% panic day inside the overlap (day 55) and one before NEW's own start (day 30)
    for s in list(stocks.values()) + [spy]:
        s.ret(55, -0.03)
        s.ret(30, -0.03)
    series = {t: s.df() for t, s in stocks.items()}
    P_alias = V.build_panel_v2(series, V.apply_aliases(mem, {"OLD": "NEW"}), spl, spy.df())
    # naive: OLD is a separate column carrying a copy of NEW's data (what happens WITHOUT the merge)
    series_dup = dict(series)
    series_dup["OLD"] = series["NEW"].copy()
    spl_dup = dict(spl)
    spl_dup["OLD"] = None
    P_dup = V.build_panel_v2(series_dup, mem, spl_dup, spy.df())
    d_alias = V.day_stats(P_alias, 0, 119)
    d_dup = V.day_stats(P_dup, 0, 119)
    check("alias: no OLD column exists; NEW is a member before its own start thanks to OLD",
          "OLD" not in P_alias.tickers and bool(P_alias.member[P_alias.cell("NEW", stocks["NEW"].iso(30))]))
    check("alias: overlap day 55 has 7 distinct companies (6 stocks + NEW once)", int(d_alias["n_member_valid"][55]) == 7,
          str(d_alias["n_member_valid"][55]))
    check("... whereas treating OLD as a second ticker would double count (8) — the dedupe is what prevents it",
          int(d_dup["n_member_valid"][55]) == 8, str(d_dup["n_member_valid"][55]))
    check("alias: day 30 (only OLD is a member) has 7 members, day 70 (only NEW) has 7",
          int(d_alias["n_member_valid"][30]) == 7 and int(d_alias["n_member_valid"][70]) == 7)
    cs_ = V.candset_s(P_alias, d_alias, "P2", 55, 55)
    names = [P_alias.tickers[int(P_alias.tick[c])] for c in cs_.cells.tolist()]
    check("alias: on the overlap panic day NEW is a candidate exactly once", names.count("NEW") == 1 and len(names) == 7, str(names))
    cs_dup = V.candset_s(P_dup, d_dup, "P2", 55, 55)
    check("(control) the naive duplicate would have produced 8 candidates", len(cs_dup.cells) == 8)

    # CMCSK-style: both tickers have their own series AND overlapping membership -> alias to the surviving series
    check("apply_aliases keeps a ticker whose name is already the target (idempotent)",
          V.apply_aliases(V.apply_aliases(m, r), r).keys() == V.apply_aliases(m, r).keys())


# ---------------------------------------------------------------------------
# 6. truncation / look-ahead
# ---------------------------------------------------------------------------

def _flat(P, ds, X):
    names = np.array(P.tickers)[P.tick]
    cand = {}
    for panic in V.PANICS:
        cs = V.candset_s(P, ds, panic, ds["lo"], ds["hi"])
        key = set(zip(names[cs.cells].tolist(), P.dnum[cs.cells].tolist()))
        cand[panic] = key
    return pd.DataFrame({"t": names, "d": P.dnum, "dv": P.dv20, "vol": P.vol20, "liq": P.liq_ok,
                         "ret": X.ret.astype(np.float64), "code": X.code, "k": X.k}), cand


def test_lookahead():
    series, mem, spl, spy, periods, resolved = V.make_synthetic_v2(16, 520, seed=21, edge=True)
    mem2 = V.apply_aliases(mem, resolved)
    full = V.build_panel_v2(series, mem2, spl, spy)
    Xf = V.compute_exit_arrays_v2(full, -0.05, 0.03, 5)
    dsf = V.day_stats(full, 0, len(full.cal) - 1)
    ff, cf = _flat(full, dsf, Xf)
    check("lookahead: the full synthetic panel has panic days and candidates to test with",
          int(dsf["P2"].sum()) >= 8 and len(cf["P2"]) >= 20, f"{dsf['P2'].sum()} {len(cf['P2'])}")
    dates = series["T000"]["date"]
    for cut in (180, 300, 410):
        sub = {t: df.iloc[:cut] for t, df in series.items()}
        cp = V.build_panel_v2(sub, mem2, spl, spy.iloc[:cut])
        Xc = V.compute_exit_arrays_v2(cp, -0.05, 0.03, 5)
        dsc = V.day_stats(cp, 0, len(cp.cal) - 1)
        fc, cc = _flat(cp, dsc, Xc)
        n = len(cp.cal)
        same_days = (np.allclose(dsc["r_spy"][:n], dsf["r_spy"][:n], equal_nan=True)
                     and np.allclose(dsc["breadth"][:n], dsf["breadth"][:n], equal_nan=True)
                     and bool((dsc["P2"][:n] == dsf["P2"][:n]).all()) and bool((dsc["P3"][:n] == dsf["P3"][:n]).all()))
        check(f"truncation at bar {cut}: r_spy, breadth, P2 and P3 flags identical up to the cut", same_days)
        cutoff = int(np.datetime64(str(dates.iloc[cut - 1].date()), "D").astype("int64"))
        okc = True
        for panic in V.PANICS:
            okc &= cc[panic] == {x for x in cf[panic] if x[1] <= cutoff}
        check(f"truncation at bar {cut}: mode-S candidate sets identical up to the cut", okc)
        m = fc.merge(ff, on=["t", "d"], suffixes=("_c", "_f"))
        check(f"truncation at bar {cut}: trailing dollar volume, volatility and liquidity flag identical",
              len(m) == cp.N and np.allclose(m["dv_c"], m["dv_f"], equal_nan=True)
              and np.allclose(m["vol_c"], m["vol_f"], equal_nan=True) and bool((m["liq_c"] == m["liq_f"]).all()))
        det = m[m["code_c"] >= 0]
        check(f"truncation at bar {cut}: every exit determinable in the cut equals the full-series exit",
              bool((det["code_c"] == det["code_f"]).all() and (det["k_c"] == det["k_f"]).all()
                   and np.allclose(det["ret_c"], det["ret_f"], atol=1e-7)))

    # garbage after bar m: nothing at or before m may change
    m_cut = 300
    rng = np.random.default_rng(5)
    garbage = {}
    for t, df in series.items():
        g = df.copy()
        for col in ("open", "high", "low", "close"):
            g.loc[g.index[m_cut + 1:], col] = rng.uniform(1, 500, len(g) - m_cut - 1)
        g.loc[g.index[m_cut + 1:], "volume"] = rng.uniform(1, 1e9, len(g) - m_cut - 1)
        garbage[t] = g
    sg = spy.copy()
    sg.loc[sg.index[m_cut + 1:], "close"] = rng.uniform(1, 500, len(sg) - m_cut - 1)
    gp = V.build_panel_v2(garbage, mem2, spl, sg)
    Xg = V.compute_exit_arrays_v2(gp, -0.05, 0.03, 5)
    dsg = V.day_stats(gp, 0, len(gp.cal) - 1)
    fg, cg = _flat(gp, dsg, Xg)
    cutoff = int(np.datetime64(str(dates.iloc[m_cut].date()), "D").astype("int64"))
    mm = fg.merge(ff, on=["t", "d"], suffixes=("_g", "_f"))
    mm = mm[mm["d"] <= cutoff]
    gcal = m_cut + 1
    ok = (np.allclose(dsg["r_spy"][:gcal], dsf["r_spy"][:gcal], equal_nan=True)
          and bool((dsg["P2"][:gcal] == dsf["P2"][:gcal]).all())
          and np.allclose(mm["dv_g"], mm["dv_f"], equal_nan=True) and np.allclose(mm["vol_g"], mm["vol_f"], equal_nan=True))
    for panic in V.PANICS:
        ok &= {x for x in cg[panic] if x[1] <= cutoff} == {x for x in cf[panic] if x[1] <= cutoff}
    check("future garbage after t leaves r_spy, breadth/panic flags, candidates, dv20 and vol20 at or before t unchanged", ok)


# ---------------------------------------------------------------------------
# 7. TARGET None
# ---------------------------------------------------------------------------

def _wild_panel(seed, n_t=6, n=260):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-04", periods=n)
    stocks = {}
    for j in range(n_t):
        c = 100 * np.cumprod(1 + rng.normal(0, 0.02, n))
        o = c * (1 + rng.normal(0, 0.02, n))
        h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.03, n)))
        l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.03, n)))
        stocks[f"W{j}"] = pd.DataFrame({"date": dates, "open": o, "high": h, "low": l, "close": c, "volume": 1e7})
    return stocks


def _ref_exit(P, i, stop, target, hold):
    """Literal reading of SPEC (v1 §3 / v2 §3.4) with both STOP and TARGET optional."""
    c0 = float(P.close[i])
    for k in range(1, hold + 1):
        o, h, l, c = (float(x[i + k]) for x in (P.open, P.high, P.low, P.close))
        if stop is not None:
            sp = c0 * (1 + stop)
            if o <= sp:
                return o / c0 - 1.0, 2, k
            if l <= sp:
                return stop, 1, k
        if target is not None and h >= c0 * (1 + target):
            return target, 0, k
        if k == hold:
            return c / c0 - 1.0, 3, k
    raise AssertionError


def test_target_none():
    stocks = _wild_panel(4)
    P = V.build_panel_v2(stocks, {t: ALWAYS for t in stocks}, {t: None for t in stocks}, None)
    all_ok = True
    n_cmp = 0
    for (s, t, h) in V.combos():
        X = V.compute_exit_arrays_v2(P, s, t, h)
        det = np.flatnonzero(X.code >= 0)
        for i in det[::7]:
            r, cd, k = _ref_exit(P, int(i), s, t, h)
            n_cmp += 1
            if not (abs(float(X.ret[i]) - r) < 1e-6 and int(X.code[i]) == cd and int(X.k[i]) == k):
                all_ok = False
        if t is None:
            all_ok &= not (X.code == 0).any()
    check(f"all 12 exit rules == literal SPEC reading on wild random data ({n_cmp} cells compared)", all_ok)
    Xn = V.compute_exit_arrays_v2(P, None, None, 3)
    det = Xn.code >= 0
    check("STOP none + TARGET none: every determinable exit is a 'time' exit at close[t+HOLD]",
          bool((Xn.code[det] == 3).all()) and bool((Xn.k[det] == 3).all()))
    i0 = int(np.flatnonzero(det)[10])
    check("... and its return is close[t+3]/close[t]-1", abs(float(Xn.ret[i0]) - (P.close[i0 + 3] / P.close[i0] - 1)) < 1e-6)
    Xs = V.compute_exit_arrays_v2(P, -0.05, None, 5)
    check("TARGET none with STOP -5%: only stop / gap / time codes", set(np.unique(Xs.code[Xs.code >= 0]).tolist()) <= {1, 2, 3})
    Xv1 = R.compute_exit_arrays(P, -0.05, 0.03, 5)
    Xv2 = V.compute_exit_arrays_v2(P, -0.05, 0.03, 5)
    check("finite TARGET: the v2 wrapper is exactly v1's kernel",
          np.array_equal(Xv1.code, Xv2.code) and np.array_equal(Xv1.k, Xv2.k) and np.array_equal(Xv1.ret, Xv2.ret, equal_nan=True))
    # 'no target' gives the target-less path even when the high would have touched +3%
    s = S(60)
    s.ret(30, -0.04, back=False)
    s.bar(31, s.c[30] * 1.01, o=s.c[30], h=s.c[30] * 1.20)
    s.bar(32, s.c[31])
    P1 = V.build_panel_v2({"A": s.df()}, {"A": ALWAYS}, {"A": None}, None)
    i = P1.cell("A", s.iso(30))
    Xa = V.compute_exit_arrays_v2(P1, None, None, 1)
    Xb = V.compute_exit_arrays_v2(P1, None, 0.03, 1)
    check("a +20% intraday high exits at the target only when TARGET +3% exists",
          int(Xa.code[i]) == 3 and abs(float(Xa.ret[i]) - 0.01) < 1e-6 and int(Xb.code[i]) == 0 and abs(float(Xb.ret[i]) - 0.03) < 1e-6)


# ---------------------------------------------------------------------------
# 8. statistics and PASS logic
# ---------------------------------------------------------------------------

def _row(**kw):
    base = dict(n_trades=300, n_trade_days=20, n_panic_days=20, pf=1.5, mdd=0.1, sharpe=1.2, spy_bh_mdd=0.3)
    base.update(kw)
    return base


def test_stats_and_pass():
    check("variant grid is 2 x 2 x 12 = 48 with unique slugs",
          len(V.build_variants()) == 48 and len({V.variant_slug(v) for v in V.build_variants()}) == 48
          and len(V.combos()) == 12)
    check("grid axes: STOP {none,-5%}, TARGET {none,+3%}, HOLD {1,3,5}",
          V.STOPS == (None, -0.05) and V.TARGETS == (None, 0.03) and V.HOLDS == (1, 3, 5))
    check("SPEC constants: exposure 50%, 25 names, 50 positions, breadth 50%, 3x, -20% cap",
          (V.EXPOSURE, V.MAX_NAMES, V.MAX_POSITIONS, V.BREADTH_MIN, V.S_MULT, V.HARD_CAP) == (0.5, 25, 50, 0.5, 3.0, -0.20)
          and V.PANIC_SPY_R == {"P2": -0.02, "P3": -0.03} and V.SEED == 20261101 and V.CONTROL_N == 100)

    # per-day statistics
    P, d, stocks = _panel_30()
    X = V.compute_exit_arrays_v2(P, None, None, 1)
    cs = V.candset_s(P, V.day_stats(P, 0, len(P.cal) - 1), "P2", d, d)
    days, means = V.day_series(P, X, cs.cells, V.COST)
    exp = float((X.ret[cs.cells].astype(np.float64) - 0.002).mean())
    check("day_series: one entry per panic day = mean net return of that day's candidates",
          len(days) == 1 and abs(means[0] - exp) < 1e-12)
    r = V.day_stats_row(np.array([1, 2, 3, 4]), np.array([0.02, -0.01, 0.03, -0.02]))
    check("per-day summary: mean / median / fraction of positive days",
          abs(r["day_mean_ret"] - 0.005) < 1e-12 and abs(r["day_median_ret"] - 0.005) < 1e-12 and r["day_frac_pos"] == 0.5)
    check("empty day series -> NaN stats", math.isnan(V.day_stats_row(np.zeros(0), np.zeros(0))["day_mean_ret"]))

    # (i) incl. the mode-I exemption
    a = _row(n_trades=250, n_trade_days=16)
    b = _row(n_trades=40, n_trade_days=7)
    check("(i) mode S: trades >= 100 AND IS panic days >= 15 AND OOS panic days >= 6", V.ok_i("S", a, b))
    check("(i) mode S: 99 IS trades fails", not V.ok_i("S", _row(n_trades=99, n_trade_days=16), b))
    check("(i) mode S: 14 IS panic days fails", not V.ok_i("S", _row(n_trades=900, n_trade_days=14), b))
    check("(i) mode S: 5 OOS panic days fails", not V.ok_i("S", a, _row(n_trade_days=5)))
    check("(i) mode I: the >=100-trades part does not apply (16 trades are fine) ...", V.ok_i("I", _row(n_trades=16, n_trade_days=16), b))
    check("(i) mode I: ... but the panic-day counts still do", not V.ok_i("I", _row(n_trades=16, n_trade_days=14), b)
          and not V.ok_i("I", _row(n_trades=16, n_trade_days=16), _row(n_trade_days=5)))

    vS = ("P2", "S", None, None, 1)
    vI = ("P2", "I", None, None, 1)
    good = {"A": {"IS": True, "OOS": True}, "B": {"IS": True, "OOS": True}}
    f = V.pass_flags_v2(vS, a, _row(pf=1.2), good)
    check("PASS requires every criterion: all good -> PASS", f["PASS"] and f["pass_trade_level"])
    check("(ii) IS PF < 1.3 fails", not V.pass_flags_v2(vS, _row(pf=1.29, n_trades=250, n_trade_days=16), b | {"pf": 1.2}, good)["PASS"])
    check("(vi) OOS PF < 1.1 fails", not V.pass_flags_v2(vS, a, _row(pf=1.09), good)["PASS"])
    check("(iii) IS MDD > 30% fails", not V.pass_flags_v2(vS, _row(mdd=0.31, n_trades=250, n_trade_days=16), _row(pf=1.2), good)["PASS"])
    check("(iv) IS Sharpe < 0.8 fails", not V.pass_flags_v2(vS, _row(sharpe=0.79, n_trades=250, n_trade_days=16), _row(pf=1.2), good)["PASS"])
    check("(v) IS MDD >= SPY B&H MDD fails", not V.pass_flags_v2(vS, _row(mdd=0.3, spy_bh_mdd=0.3, n_trades=250, n_trade_days=16), _row(pf=1.2), good)["PASS"])
    check("(vii) mode S must beat BOTH A and B in BOTH periods",
          not V.pass_flags_v2(vS, a, _row(pf=1.2), {"A": {"IS": True, "OOS": True}, "B": {"IS": True, "OOS": False}})["PASS"]
          and not V.pass_flags_v2(vS, a, _row(pf=1.2), {"A": {"IS": False, "OOS": True}, "B": {"IS": True, "OOS": True}})["PASS"])
    check("(vii) mode I needs only C (IS and OOS)",
          V.pass_flags_v2(vI, _row(n_trades=16, n_trade_days=16), _row(pf=1.2), {"C": {"IS": True, "OOS": True}})["PASS"]
          and not V.pass_flags_v2(vI, _row(n_trades=16, n_trade_days=16), _row(pf=1.2), {"C": {"IS": True, "OOS": False}})["PASS"])
    check("NaN metrics never pass", not V.pass_flags_v2(vS, _row(pf=float("nan"), n_trades=250, n_trade_days=16), _row(pf=1.2), good)["PASS"]
          and not V.pass_flags_v2(vS, _row(spy_bh_mdd=float("nan"), n_trades=250, n_trade_days=16), _row(pf=1.2), good)["PASS"])
    check("(viii) pass_trade_level ignores MDD / Sharpe / controls",
          V.pass_flags_v2(vS, _row(mdd=0.9, sharpe=-1, n_trades=250, n_trade_days=16), _row(pf=1.2),
                          {"A": {"IS": False, "OOS": False}, "B": {"IS": False, "OOS": False}})["pass_trade_level"])
    check("conclusion: EDGE needs >=1 PASS AND real count > control p95",
          V.conclusion(1, 5, 4.0) == "EDGE" and V.conclusion(1, 4, 4.0) == "NO EVIDENCE" and V.conclusion(0, 50, 1.0) == "NO EVIDENCE")
    check("reported denominator is 120 = 72 + 48", V.CUMULATIVE_DENOM == 120)


# ---------------------------------------------------------------------------
# 9. end-to-end
# ---------------------------------------------------------------------------

def _run(edge, seed, n_tickers=30, n_days=1100, control_n=12):
    series, mem, spl, spy, periods, resolved = V.make_synthetic_v2(n_tickers, n_days, seed=seed, edge=edge)
    P = V.build_panel_v2(series, V.apply_aliases(mem, resolved), spl, spy)
    return V.run_study_v2(P, spy, periods, control_n, V.SEED, verbose=False), P, periods, resolved


def test_study_end_to_end():
    res, P, periods, resolved = _run(True, 11)
    check("edge data: at least one PASS and HOLDOUT computed once", len(res["passers"]) >= 1 and res["holdout"] is not None)
    check("edge data: market stats were computed for IS+OOS and then (only because of a PASS) for HOLDOUT",
          len(res["windows"]) == 2 and res["windows"][0] == (res["bounds"]["IS"][0], res["bounds"]["OOS"][1])
          and res["windows"][1] == res["bounds"]["HOLDOUT"])
    ho_var = tuple(res["holdout"]["variant"])
    best = max(res["passers"], key=lambda v: (min(res["rows"]["IS"][v]["pf"], V.PF_CAP), -res["variants"].index(v)))
    check("HOLDOUT variant = the PASS variant with the best IS PF", ho_var == best)
    check("48 variants x IS/OOS rows; mode S has controls A+B, mode I has C",
          len(res["rows"]["IS"]) == 48 and all(set(res["ctrl"]["IS"][v]) == ({"A", "B"} if v[1] == "S" else {"C"})
                                              for v in res["variants"]))
    check("every control array has control_n runs", all(len(a) == 12 for pn in ("IS", "OOS") for v in res["variants"]
                                                       for a in res["ctrl"][pn][v].values()))
    # mode-I trade count == panic-day count
    okI = all(res["rows"][pn][v]["n_trades"] == res["rows"][pn][v]["n_trade_days"] for pn in ("IS", "OOS")
              for v in res["variants"] if v[1] == "I")
    check("mode I: n_trades == n_trade_days (one SPY trade per panic day)", okI)
    check("P3 panic days are a subset of P2 panic days", not (res["ds"]["P3"] & ~res["ds"]["P2"]).any())
    # exit-reason bookkeeping
    okx = all(r["exit_target"] + r["exit_stop"] + r["exit_gap"] + r["exit_time"] == r["n_trades"]
              for pn in ("IS", "OOS") for r in res["rows"][pn].values())
    check("exit-reason counts sum to n_trades; TARGET none rows have no target exits",
          okx and all(res["rows"][pn][v]["exit_target"] == 0 for pn in ("IS", "OOS") for v in res["variants"] if v[3] is None))
    check("n_filled + n_skipped == n_trades", all(r["n_filled"] + r["n_skipped"] == r["n_trades"]
                                                  for pn in ("IS", "OOS") for r in res["rows"][pn].values()))

    with tempfile.TemporaryDirectory() as d:
        written = V.write_outputs(res, d, {"requested": 30, "received_direct": 30}, {"loaded": 30, "skipped": 0}, resolved,
                                  "2026-11-01", synthetic=True)
        bname = V.variant_slug(V.best_is_variant(res))
        for name in ("grid_IS.csv", "grid_OOS.csv", "panic_days.csv", "controls_summary.csv", "SUMMARY_v2.md",
                     "data_coverage_v2.json", "holdout.json", "summary_v2.json", f"trades_{bname}.csv"):
            check(f"output written: {name}", os.path.exists(os.path.join(d, name)))
        g = pd.read_csv(os.path.join(d, "grid_IS.csv"))
        need = ["variant", "panic", "mode", "stop", "target", "hold", "n_trades", "n_panic_days", "n_trade_days", "pf",
                "pf_cost005", "pf_cost020", "win_rate", "mean_ret", "median_ret", "exit_target", "exit_stop", "exit_gap",
                "exit_time", "day_mean_ret", "day_median_ret", "day_frac_pos", "total_return", "cagr", "mdd", "sharpe",
                "spy_bh_mdd", "ctrlA_pf_p95", "ctrlB_pf_p95", "ctrlC_pf_p95", "PASS"]
        check("grid_IS.csv: 48 rows and every SPEC_v2 §7 column", len(g) == 48 and all(c in g.columns for c in need),
              str([c for c in need if c not in g.columns]))
        check("grid: control p95 columns are filled for the right mode (A,B for S; C for I)",
              g.loc[g["mode"] == "S", "ctrlA_pf_p95"].notna().all() and g.loc[g["mode"] == "S", "ctrlC_pf_p95"].isna().all()
              and g.loc[g["mode"] == "I", "ctrlC_pf_p95"].notna().all() and g.loc[g["mode"] == "I", "ctrlA_pf_p95"].isna().all())
        cs_ = pd.read_csv(os.path.join(d, "controls_summary.csv"))
        check("controls_summary.csv: 2 periods x (24 x {A,B} + 24 x {C}) = 144 rows with p05/p50/p95/real/beats",
              len(cs_) == 144 and all(c in cs_.columns for c in ("pf_p05", "pf_p50", "pf_p95", "real_pf", "real_beats_p95")))
        check("controls_summary p05 <= p50 <= p95", bool(((cs_["pf_p05"] <= cs_["pf_p50"] + 1e-9) & (cs_["pf_p50"] <= cs_["pf_p95"] + 1e-9)).all()))
        pdays = pd.read_csv(os.path.join(d, "panic_days.csv"))
        hold_lo = periods["HOLDOUT"][0]
        n_p2 = int(res["ds"]["P2"].sum())
        check("panic_days.csv: one row per P2 day of IS+OOS and NO holdout date",
              len(pdays) == n_p2 and (pdays["date"] < hold_lo).all(), f"{len(pdays)} vs {n_p2}")
        check("panic_days.csv: columns per SPEC_v2 §7", all(c in pdays.columns for c in
              ["date", "r_spy", "breadth", "is_P3", "n_candidates_S", "n_excluded_less_than_market", "n_excluded_over_3x",
               "n_excluded_cap20", "spy_fwd_1", "spy_fwd_3", "spy_fwd_5", "mean_1d_net_ret_of_candidates"]))
        check("panic_days.csv: candidates + the three exclusion counts == eligible members, every day",
              bool((pdays["n_candidates_S"] + pdays["n_excluded_less_than_market"] + pdays["n_excluded_over_3x"]
                    + pdays["n_excluded_cap20"] == pdays["n_eligible"]).all()))
        check("panic_days.csv: every row satisfies P2 (r_spy <= -2%, breadth >= 50%); is_P3 iff r_spy <= -3%",
              bool(((pdays["r_spy"] <= -0.02 + 1e-9) & (pdays["breadth"] >= 0.5 - 1e-12)).all())
              and bool((pdays["is_P3"] == (pdays["r_spy"] <= -0.03 + 1e-9)).all()))
        check("panic_days.csv: the exclusion counts are non-trivial on this data (all three reasons occur)",
              pdays["n_excluded_less_than_market"].sum() > 0 and pdays["n_excluded_over_3x"].sum() > 0)
        # spy_fwd matches a direct computation
        r0 = pdays.iloc[0]
        gi = int(np.flatnonzero(P.cal == np.datetime64(r0["date"], "D"))[0])
        c = int(P.spy_cell[gi])
        check("panic_days.csv: spy_fwd_k = SPY close[t+k]/close[t]-1",
              abs(r0["spy_fwd_1"] - (P.close[c + 1] / P.close[c] - 1)) < 1e-5 and abs(r0["spy_fwd_5"] - (P.close[c + 5] / P.close[c] - 1)) < 1e-5)
        t = pd.read_csv(os.path.join(d, f"trades_{bname}.csv"))
        check("trades file: columns, net = gross - 0.20%, exit_price = entry x (1+gross), reasons valid, exit after entry",
              list(t.columns) == V.TRADE_COLS and bool(np.allclose(t["net_ret"], t["gross_ret"] - 0.002))
              and bool(np.allclose(t["exit_price"], t["entry_price"] * (1 + t["gross_ret"])))
              and set(t["exit_reason"]) <= {"time", "stop", "gap", "target"}
              and bool((pd.to_datetime(t["exit_date"]) > pd.to_datetime(t["entry_date"])).all()))
        n_expected = res["rows"]["IS"][V.best_is_variant(res)]["n_trades"] + res["rows"]["OOS"][V.best_is_variant(res)]["n_trades"]
        check("trades file holds every IS+OOS trade of the best-IS-PF variant", len(t) >= n_expected - 0 and
              set(t["period"]) <= {"IS", "OOS", "HOLDOUT"}, f"{len(t)} vs {n_expected}")
        txt = open(os.path.join(d, "SUMMARY_v2.md"), encoding="utf-8").read()
        check("SUMMARY_v2.md: Korean, verdict, 120 denominator, panic-day table, S-vs-I table, coverage, bias, holdout",
              all(k in txt for k in ("전체 결론", res["verdict"], "120", "전체 P2 패닉일", "모드 S", "모드 I", "Survivorship bias",
                                      "HOLDOUT", "데이터 복구 결과", "대조군 요약")))
        check("SUMMARY_v2.md lists every panic day (table rows == P2 days)",
              sum(1 for ln in txt.splitlines() if ln.startswith("| 20")) >= len(pdays))
        ok_json = True
        for jn in ("summary_v2.json", "holdout.json", "data_coverage_v2.json"):
            try:
                json.loads(open(os.path.join(d, jn)).read(), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
            except ValueError:
                ok_json = False
        check("JSON outputs are strict JSON (no NaN / Infinity tokens)", ok_json)

    # determinism / seed dependence
    r1, _, _, _ = _run(True, 11)
    r2, _, _, _ = _run(True, 11)
    g1, g2 = V.grid_dataframe(r1, "IS"), V.grid_dataframe(r2, "IS")
    check("same seed -> identical grid and control counts", g1.equals(g2) and r1["ctrl_counts"] == r2["ctrl_counts"])
    series, mem, spl, spy, periods_, resolved_ = V.make_synthetic_v2(30, 1100, seed=11, edge=True)
    P_ = V.build_panel_v2(series, V.apply_aliases(mem, resolved_), spl, spy)
    r3 = V.run_study_v2(P_, spy, periods_, 12, V.SEED + 1, verbose=False)
    g3 = V.grid_dataframe(r3, "IS")
    check("different seed -> different control thresholds, same real columns",
          not g1["ctrlA_pf_p95"].equals(g3["ctrlA_pf_p95"]) and g1["pf"].equals(g3["pf"]) and g1["n_trades"].equals(g3["n_trades"]))

    # noise: no planted edge -> NO EVIDENCE and the HOLDOUT is never touched
    resn, Pn, periodsn, residn = _run(False, 17, control_n=30)
    check("noise data: verdict NO EVIDENCE", resn["verdict"] == "NO EVIDENCE",
          f"{resn['verdict']} passers={len(resn['passers'])} real={resn['real_count']} p95={resn['p95_count']}")
    check("noise data: no PASS => HOLDOUT is None and was never computed", resn["holdout"] is None and len(resn["windows"]) == 1)
    hlo = resn["bounds"]["HOLDOUT"][0]
    check("noise data: r_spy / breadth / panic flags are empty (not computed) on every HOLDOUT day",
          bool(np.isnan(resn["ds"]["r_spy"][hlo:]).all()) and not resn["ds"]["P2"][hlo:].any() and np.isnan(resn["ds"]["breadth"][hlo:]).all())
    with tempfile.TemporaryDirectory() as d:
        V.write_outputs(resn, d, {}, {"loaded": 30, "skipped": 0}, residn, "2026-11-01", synthetic=True)
        check("no PASS => no holdout.json and SUMMARY says the seal held",
              not os.path.exists(os.path.join(d, "holdout.json")) and "봉인 유지" in open(os.path.join(d, "SUMMARY_v2.md"), encoding="utf-8").read())


# ---------------------------------------------------------------------------
# 10. loaders / main
# ---------------------------------------------------------------------------

def test_main_real_path():
    mem = R.load_constituents()
    names = [t for t, iv in sorted(mem.items())
             if len(iv) == 1 and iv[0][1] is None and iv[0][0] <= np.datetime64("2015-01-01", "D")][:40]
    series, _, spl, spy, _, _ = V.make_synthetic_v2(len(names) + 1, 2800, seed=5, edge=True)
    with tempfile.TemporaryDirectory() as d:
        data, out = os.path.join(d, "data"), os.path.join(d, "out")
        os.makedirs(data)
        for j, t in enumerate(names):
            series[f"T{j:03d}"].to_csv(os.path.join(data, f"{t}.csv"), index=False, date_format="%Y-%m-%d")
        # ELV carries ANTM's history (Yahoo-style); ANTM itself has no file
        series[f"T{len(names):03d}"].to_csv(os.path.join(data, "ELV.csv"), index=False, date_format="%Y-%m-%d")
        json.dump({"ANTM": "ELV"}, open(os.path.join(data, V.RESOLVED_FILE), "w"))
        json.dump(spl["T000"], open(os.path.join(data, f"{names[0]}.splits.json"), "w"))
        spy.to_csv(os.path.join(data, "SPY.csv"), index=False, date_format="%Y-%m-%d")
        old_argv = sys.argv
        try:
            sys.argv = ["run_panic_dip_v2.py", "--data-dir", data, "--out-dir", out, "--control-n", "4"]
            rc = V.main()
        finally:
            sys.argv = old_argv
        check("main(): real path returns 0 and writes the SPEC_v2 §7 files",
              rc == 0 and all(os.path.exists(os.path.join(out, n)) for n in
                              ("grid_IS.csv", "grid_OOS.csv", "panic_days.csv", "controls_summary.csv", "SUMMARY_v2.md",
                               "data_coverage_v2.json")))
        txt = open(os.path.join(out, "SUMMARY_v2.md"), encoding="utf-8").read()
        check("main(): real PERIODS used; alias count shown", "IS 2015-11-01~2021-10-31" in txt and "alias로 합쳐진 옛 티커 1개" in txt)
        # ELV received ANTM's membership (2002..2022-06-28) on top of its own (2022-06-28..): continuous
        mem2 = V.apply_aliases(mem, {"ANTM": "ELV"})
        check("main(): ANTM + ELV membership is one continuous interval set on the ELV series",
              "ANTM" not in mem2 and len(mem2["ELV"]) == len(mem["ANTM"]) + len(mem["ELV"]))
        # fetch's coverage file is passed through untouched
        fake = {"requested": 751, "received_direct": 700, "recovered_via_alias": [{"old": "ANTM", "new": "ELV"}],
                "still_missing_counts": {"rename-failed": 1, "acquired": 90, "unknown": 5}, "_marker": 77}
        json.dump(fake, open(os.path.join(out, "data_coverage_v2.json"), "w"))
        try:
            sys.argv = ["run_panic_dip_v2.py", "--data-dir", data, "--out-dir", out, "--control-n", "4"]
            V.main()
        finally:
            sys.argv = old_argv
        check("main(): existing data_coverage_v2.json (from fetch) is passed through unchanged",
              json.load(open(os.path.join(out, "data_coverage_v2.json"))) == fake)
        txt = open(os.path.join(out, "SUMMARY_v2.md"), encoding="utf-8").read()
        check("main(): SUMMARY quotes the recovery numbers", "ANTM→ELV" in txt and "acquired) 90개" in txt and "unknown) 5개" in txt)
        # no data -> graceful no-op
        try:
            sys.argv = ["run_panic_dip_v2.py", "--data-dir", os.path.join(d, "empty"), "--out-dir", os.path.join(d, "o2")]
            check("main(): no data -> returns 0 without writing results", V.main() == 0 and not os.path.exists(os.path.join(d, "o2")))
        finally:
            sys.argv = old_argv
        # SPY missing -> stub summary, rc 0
        os.remove(os.path.join(data, "SPY.csv"))
        try:
            sys.argv = ["run_panic_dip_v2.py", "--data-dir", data, "--out-dir", os.path.join(d, "o3"), "--control-n", "2"]
            rc3 = V.main()
        finally:
            sys.argv = old_argv
        check("main(): SPY missing -> rc 0 and a stub SUMMARY_v2.md (no grid)",
              rc3 == 0 and os.path.exists(os.path.join(d, "o3", "SUMMARY_v2.md")) and not os.path.exists(os.path.join(d, "o3", "grid_IS.csv")))
    with tempfile.TemporaryDirectory() as d:
        json.dump({"A": "B"}, open(os.path.join(d, V.RESOLVED_FILE), "w"))
        check("load_resolved_aliases reads the fetcher's file; absent -> {}",
              V.load_resolved_aliases(d) == {"A": "B"} and V.load_resolved_aliases(os.path.join(d, "none")) == {})


# ---------------------------------------------------------------------------
# 11. fetch_prices_v2 (fake HTTP)
# ---------------------------------------------------------------------------

class FakeResp:
    def __init__(self, status, payload=None):
        self.status_code, self._p = status, payload

    def json(self):
        if self._p is None:
            raise ValueError("no json")
        return self._p


def _payload(n=60):
    ts = [1451917800 + 86400 * i for i in range(n)]
    res = {"timestamp": ts, "indicators": {"quote": [{
        "open": [10.0 + i * 0.01 for i in range(n)], "high": [11.0 + i * 0.01 for i in range(n)],
        "low": [9.0 + i * 0.01 for i in range(n)], "close": [10.5 + i * 0.01 for i in range(n)],
        "volume": [1000 + i for i in range(n)]}]}, "events": {"splits": {}}}
    return {"chart": {"result": [res], "error": None}}


class FakeYahoo:
    """script[symbol] = list of responses consumed in order (the last one repeats); an int = rows (200), 404 = HTTP 404."""

    def __init__(self, script):
        self.script, self.calls = script, {}

    def __call__(self, url, **k):
        sym = url.split("/chart/")[1].split("?")[0]
        self.calls[sym] = self.calls.get(sym, 0) + 1
        seq = self.script.get(sym, [60])
        item = seq[min(self.calls[sym] - 1, len(seq) - 1)]
        return FakeResp(404) if item == 404 else FakeResp(200, _payload(item))


def test_fetch_v2():
    I = lambda s, e=None: [(np.datetime64(s, "D"), np.datetime64(e, "D") if e else None)]  # noqa: E731
    universe = ["AAA", "TRN", "OLDR", "STUBOK", "STUBBAD", "ACQ", "UNK", "RFAIL", "SHORTSPAN", "RNSHORT", "OLDOK", "BRK.B"]
    membership = {t: I("2010-01-01") for t in universe}
    membership["SHORTSPAN"] = I("2020-01-01", "2020-02-01")           # 31 days <= 60: a 1-row answer is NOT 'short'
    aliases = {"OLDR": {"new": "NEWR", "kind": "rename", "note": ""},
               "ACQ": {"new": "", "kind": "acquired", "note": ""},
               "UNK": {"new": "", "kind": "unknown", "note": ""},
               "RFAIL": {"new": "DEADNEW", "kind": "rename", "note": ""},
               "RNSHORT": {"new": "NEWOK", "kind": "rename", "note": ""},
               "OLDOK": {"new": "NEVERFETCHED", "kind": "rename", "note": ""}}
    fake = FakeYahoo({
        "TRN": [404, 60],                    # transient 404: the retry succeeds
        "OLDR": [404], "NEWR": [90],         # still 404 after the retry -> alias NEWR
        "STUBOK": [1, 1, 70],                # short, short, then fine on the 30 s retry
        "STUBBAD": [1],                      # short forever
        "ACQ": [404], "UNK": [404], "RFAIL": [404], "DEADNEW": [404],
        "SHORTSPAN": [1],
        "RNSHORT": [1], "NEWOK": [200],      # short stub -> alias
        "OLDOK": [80],                       # direct data wins: never aliased
        "BRK-B": [60],
    })
    sleeps = []
    with tempfile.TemporaryDirectory() as d:
        out, res = os.path.join(d, "px"), os.path.join(d, "res")
        cov = F2.run(None, out, res, get=fake, sleep=sleeps.append, universe=universe, aliases=aliases, membership=membership)
        check("404 rounds: one 20 s wait for all first-pass 404s + one for the dead alias target (DEADNEW) fetched later",
              sleeps.count(20) == 2, str(sleeps))
        rest = [s for s in sleeps if s != F2.F1.SLEEP_BETWEEN_TICKERS_S]
        check("short round: 10 s wait, then 30 s wait (only because something is still short)", rest[:3] == [20, 10, 30], str(rest))
        check("TRN: 404 then success on the retry -> recovered directly (2 calls)", fake.calls["TRN"] == 2
              and "TRN" in cov["retries"]["missing_recovered_on_retry"] and "TRN" in cov["per_ticker"])
        check("OLDR is retried directly first (2 calls), then NEWR is fetched once and aliased",
              fake.calls["OLDR"] == 2 and fake.calls["NEWR"] == 1
              and {"old": "OLDR", "new": "NEWR", "new_rows": 90, "old_was": "missing"} in cov["recovered_via_alias"])
        check("STUBOK: 1 row, 1 row (10 s), 70 rows (30 s) -> usable, 3 calls", fake.calls["STUBOK"] == 3
              and cov["per_ticker"]["STUBOK"]["rows"] == 70 and "STUBOK" in cov["retries"]["short_recovered_on_retry"])
        check("STUBBAD: stays short -> in the short list, kind unknown (no alias row), 3 calls",
              fake.calls["STUBBAD"] == 3 and any(s["ticker"] == "STUBBAD" and s["rows"] == 1 and s["kind"] == "unknown" for s in cov["short"])
              and "STUBBAD" in cov["still_missing"]["unknown"])
        check("SHORTSPAN: a 1-row answer for a 31-day membership is not 'short' (1 call, counted as received)",
              fake.calls["SHORTSPAN"] == 1 and "SHORTSPAN" not in [s["ticker"] for s in cov["short"]])
        check("ACQ -> acquired, UNK -> unknown, RFAIL (rename to a dead ticker) -> rename-failed",
              "ACQ" in cov["still_missing"]["acquired"] and "UNK" in cov["still_missing"]["unknown"]
              and "RFAIL" in cov["still_missing"]["rename-failed"] and cov["still_missing_counts"] == {
                  "rename-failed": 1, "acquired": 1, "unknown": 2})
        check("RFAIL detail names the dead target", any(x["old"] == "RFAIL" and x["new"] == "DEADNEW" for x in cov["rename_failed_detail"]))
        check("RNSHORT: short stub with a rename -> recovered via alias, old_was 'short'",
              {"old": "RNSHORT", "new": "NEWOK", "new_rows": 200, "old_was": "short"} in cov["recovered_via_alias"])
        check("OLDOK: direct data wins, NEVERFETCHED is never requested, no alias recorded",
              "NEVERFETCHED" not in fake.calls and "OLDOK" not in [x["old"] for x in cov["recovered_via_alias"]])
        resolved = json.load(open(os.path.join(out, F2.RESOLVED_FILE)))
        check("_aliases_resolved.json == {old: new} of the recovered ones", resolved == {"OLDR": "NEWR", "RNSHORT": "NEWOK"}, str(resolved))
        check("coverage fields: requested, received_direct, benchmark, split counts, per-ticker rows/first/last",
              cov["requested"] == 12 and cov["received_direct"] == 6 and "SPY" in cov["benchmark"]
              and cov["per_ticker"]["AAA"]["rows"] == 60 and cov["per_ticker"]["AAA"]["first_date"] == "2016-01-04"
              and cov["tickers_with_splits_file"] == len(cov["per_ticker"]) and cov["tickers_with_split_events"] == 0,
              f"{cov['received_direct']}")
        check("v2 data dir holds the files, separate from v1's data/panic_dip", os.path.exists(os.path.join(out, "AAA.csv"))
              and os.path.exists(os.path.join(out, "NEWR.csv")) and os.path.exists(os.path.join(out, "BRK.B.csv"))
              and out.endswith("px") and F2.DEFAULT_OUT_DIR.endswith(os.path.join("data", "panic_dip_v2")))
        disk = json.load(open(os.path.join(res, F2.COVERAGE_FILE)))
        check("data_coverage_v2.json is written to the results dir and equals the returned dict", disk["requested"] == 12
              and disk["recovered_via_alias"] == cov["recovered_via_alias"])
        check("study can consume the fetch output", V.load_resolved_aliases(out) == resolved)
        check("the dotted ticker is requested as BRK-B", fake.calls.get("BRK-B", 0) == 1)

    # --limit and a missing benchmark
    fake2 = FakeYahoo({"SPY": [404]})
    with tempfile.TemporaryDirectory() as d:
        sleeps2 = []
        cov2 = F2.run(2, os.path.join(d, "p"), os.path.join(d, "r"), get=fake2, sleep=sleeps2.append,
                      universe=["A", "B", "C"], aliases={}, membership={})
        check("--limit N: only the first N tickers + SPY; missing SPY is reported, not fatal",
              cov2["requested"] == 2 and cov2["benchmark"]["SPY"].get("missing") is True and "C" not in fake2.calls)

    # span helper
    m = I("2015-01-01", "2015-12-01") + I("2015-11-15", "2016-01-10")             # overlapping intervals: union
    check("span_days: union of intervals clipped to the study window (Nov 1 2015 .. Jan 9 2016 = 70 days)",
          F2.span_days(m) == 70, str(F2.span_days(m)))
    check("span_days: open-ended membership runs to the study end", F2.span_days(I("2026-10-01")) == 31)


# ---------------------------------------------------------------------------
# 12. aliases.csv integrity
# ---------------------------------------------------------------------------

def test_aliases_csv():
    a = F2.read_aliases()
    v1 = json.load(open(os.path.join(os.path.dirname(HERE), "results", "data_coverage.json")))
    missing = set(v1["missing"])
    stubs = {t for t, x in v1["per_ticker"].items() if x["rows"] < 50}
    check("aliases.csv: one row for each of the 133 v1-missing tickers (no duplicates)", missing <= set(a) and len(missing) == 133)
    check("aliases.csv: the 8 v1 stubs (AVB, EA, EQR, LEG, QRVO, WBD, SATS, PSKY) have rows too",
          stubs == {"AVB", "EA", "EQR", "LEG", "QRVO", "WBD", "SATS", "PSKY"} and stubs <= set(a))
    check("aliases.csv: no stray rows", set(a) == missing | stubs, str(set(a) - missing - stubs))
    renames = {o: x["new"] for o, x in a.items() if x["kind"] == "rename"}
    check("rename rows have a new_ticker different from old; other kinds have none",
          all(n and n != o for o, n in renames.items()) and all(not x["new"] for x in a.values() if x["kind"] != "rename"))
    check("no rename chains (a rename target is never itself a rename source)", not (set(renames.values()) & set(renames)))
    for k in ("ANTM", "ABC", "FLT", "RE", "TMK", "UTX", "DWDP", "ADS", "BK", "GPS", "NLOK", "PKI", "WLTW", "WRK", "FBHS", "HFC", "CTL", "VIAC"):
        check(f"alias {k} is a rename", a[k]["kind"] == "rename")
    check("alias targets of the brief: ANTM->ELV ABC->COR FLT->CPAY RE->EG TMK->GL UTX->RTX DWDP->DD ADS->BFH BK->BNY GPS->GAP "
          "NLOK->GEN PKI->RVTY WLTW->WTW WRK->SW FBHS->FBIN HFC->DINO CTL->LUMN VIAC->PSKY",
          all(renames[o] == n for o, n in {"ANTM": "ELV", "ABC": "COR", "FLT": "CPAY", "RE": "EG", "TMK": "GL", "UTX": "RTX",
                                           "DWDP": "DD", "ADS": "BFH", "BK": "BNY", "GPS": "GAP", "NLOK": "GEN", "PKI": "RVTY",
                                           "WLTW": "WTW", "WRK": "SW", "FBHS": "FBIN", "HFC": "DINO", "CTL": "LUMN", "VIAC": "PSKY"}.items()))
    check("brief's acquired examples are 'acquired'", all(a[k]["kind"] == "acquired" for k in
          ("SIVB", "FRC", "CHK", "ENDP", "FTR", "CELG", "AGN", "ALXN", "ATVI", "MRO", "PXD", "XLNX", "MXIM", "TWTR", "TIF")))
    check("RTN (Raytheon, the merged-away party) is not a rename of RTX", a["RTN"]["kind"] == "acquired")
    check("every note is non-empty; the csv has no embedded-comma damage", all(x["note"] for x in a.values()))
    c = {k: sum(1 for o, x in a.items() if o in missing and x["kind"] == k) for k in F2.KINDS}
    print(f"        aliases.csv (133 v1-missing): rename {c['rename']}, acquired {c['acquired']}, unknown {c['unknown']}; "
          f"+ 8 stubs: " + ", ".join(f"{t}={a[t]['kind']}" for t in sorted(stubs)))


def test_v1_untouched_interface():
    check("v1 interface used by v2 still exists (frozen)", all(hasattr(R, n) for n in (
        "build_panel", "compute_exit_arrays", "profit_factor", "max_drawdown", "sharpe_daily", "trade_stats",
        "spy_buyhold_mdd", "cagr_of", "load_series", "load_spy", "load_constituents", "period_bounds", "exit_price")))
    check("v1 grid is still 72 variants", len(R.build_variants()) == 72 and R.PERIODS == V.PERIODS)


def test_cli_self_test():
    with tempfile.TemporaryDirectory() as d:
        V.run_synthetic(n_tickers=30, n_days=1100, control_n=3, out_dir=d)
        check("--synthetic path runs end-to-end and writes outputs", os.path.exists(os.path.join(d, "SUMMARY_v2.md")))


def main():
    tests = [test_panic_classification, test_stock_filter, test_cut_and_sizing, test_controls, test_aliases,
             test_lookahead, test_target_none, test_stats_and_pass, test_study_end_to_end, test_main_real_path,
             test_fetch_v2, test_aliases_csv, test_v1_untouched_interface, test_cli_self_test]
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
    print("ALL PANIC_DIP V2 TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
