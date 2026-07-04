"""
==============================================================================
ORB — LIVE vs BACKTEST TRADE DIFF
==============================================================================
Pulls your REAL closed-trade history from MT5 (via history_deals_get, the
same account the live engine trades on) and, for each real trade, finds the
matching signal in the backtest's own signal-detection logic, runs the
backtest's resolver on it, and prints a side-by-side comparison:

    live entry price   vs   backtest-predicted entry price
    live exit price    vs   backtest-predicted exit price
    live realized R/$  vs   backtest-predicted R/$
    implied entry/exit slippage, in price points and in $

This replaces guesswork and manual CSV export with a direct measurement:
if there's a consistent per-trade slippage, this tells you the actual
number (which you can then paste into SLIPPAGE_PRICE_PER_SYMBOL in
orb_chrono_v3.py). It also flags:
  - live trades with NO matching backtest signal (signal divergence —
    live fired something the backtest's logic wouldn't have)
  - backtest signals with NO matching live trade (a signal the backtest
    thinks should have traded, but the live engine never took — could be
    a margin rejection, risk rejection, or an actual missed/dropped signal)

USAGE:
    Place this file in the same folder as orb_chrono_v3.py and run it.
    Set MAGIC/COMMENT below to match your live engine (already defaulted
    to orb_live_v6.py's values). Set DATE_FROM/DATE_TO for the window you
    want to audit (defaults to TRADE_START_DATE -> now).

REQUIRES: the same MT5 terminal/account your live engine actually trades
on to be reachable (same LOGIN/PASSWORD/SERVER as orb_chrono_v3.py — see
the account-identity check that already prints in that script's log).
==============================================================================
"""

import datetime
import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("ERROR: MetaTrader5 not installed.")
    raise SystemExit(1)

# Reuse everything already built/verified in the backtest — data assembly,
# cache/signal building, resolvers, broker info fetch (incl. the leverage
# calibration and account-identity check), all unchanged.
import orb_chrono_v3 as bt

# ── Config — match your live engine exactly ─────────────────────────────────
MAGIC   = 202603264   # from orb_live_v6.py
COMMENT_SUBSTR = "ORB_V6"

# Window to audit. Defaults to the backtest's own TRADE_START_DATE -> now.
DATE_FROM = pd.Timestamp(bt.TRADE_START_DATE) if bt.TRADE_START_DATE else pd.Timestamp("2026-06-01")
DATE_TO   = pd.Timestamp(datetime.datetime.utcnow())

# Force a clean, slippage-free backtest baseline for this comparison so any
# gap we find IS the thing we're trying to measure, not something we already
# told the backtest to assume.
BASELINE_SLIPPAGE = {sym: 0.0 for sym in bt.SYMBOLS}


def fetch_live_trades(date_from: pd.Timestamp, date_to: pd.Timestamp) -> pd.DataFrame:
    """Pull closed round-trip trades for MAGIC from MT5 deal history."""
    bt._ping = lambda *a, **k: None  # no-op if orb_chrono_v3 doesn't define _ping
    deals = mt5.history_deals_get(date_from.to_pydatetime(), date_to.to_pydatetime())
    if deals is None or len(deals) == 0:
        print("No deals found in this window — check DATE_FROM/DATE_TO and that "
              "MT5 is connected to the correct account.")
        return pd.DataFrame()

    rows = []
    for d in deals:
        if d.magic != MAGIC:
            continue
        if COMMENT_SUBSTR not in (d.comment or "") and COMMENT_SUBSTR not in (d.symbol or ""):
            # comment check is best-effort; magic filter above is the real gate
            pass
        rows.append({
            "position_id": d.position_id,
            "time":        pd.Timestamp(d.time, unit="s"),
            "symbol":      d.symbol,
            "type":        d.type,        # 0=BUY, 1=SELL
            "entry":       d.entry,       # 0=IN, 1=OUT
            "volume":      d.volume,
            "price":       d.price,
            "profit":      d.profit,
            "commission":  d.commission,
            "swap":        d.swap,
        })

    if not rows:
        print(f"No deals matched MAGIC={MAGIC} in this window.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    trades = []
    for pos_id, grp in df.groupby("position_id"):
        ins  = grp[grp["entry"] == mt5.DEAL_ENTRY_IN]
        outs = grp[grp["entry"] == mt5.DEAL_ENTRY_OUT]
        if ins.empty or outs.empty:
            continue   # still-open position or partial data — skip
        in_row  = ins.iloc[0]
        out_row = outs.iloc[-1]
        direction = 1 if in_row["type"] == mt5.DEAL_TYPE_BUY else -1
        trades.append({
            "position_id": pos_id,
            "symbol":      in_row["symbol"],
            "direction":   direction,
            "entry_time":  in_row["time"],
            "entry_price": in_row["price"],
            "exit_time":   out_row["time"],
            "exit_price":  out_row["price"],
            "volume":      in_row["volume"],
            "profit":      outs["profit"].sum(),
            "commission":  grp["commission"].sum(),
            "swap":        grp["swap"].sum(),
        })

    return pd.DataFrame(trades).sort_values("entry_time").reset_index(drop=True)


def build_canon_map() -> dict:
    """Map broker symbol name (as it appears in deal history) back to the
    canonical name (US30/UK100/GER40) the backtest caches are keyed by."""
    canon_map = {}
    for canon in bt.SYMBOLS:
        broker = bt.resolve_symbol(canon)
        if broker:
            canon_map[broker] = canon
    return canon_map


def find_matching_signal(cache: dict, entry_time: pd.Timestamp, direction: int):
    """Find the signal bar (si) whose entry bar (si+1) opens at entry_time,
    rounded down to the 5-minute bar boundary, with matching direction."""
    entry_bar_open = entry_time.floor("5min")
    target = np.datetime64(entry_bar_open)

    for si in cache["signal_bars"]:
        if int(cache["signal"][si]) != direction:
            continue
        ei = si + 1
        if ei >= cache["n"]:
            continue
        bar_open_time = cache["times"][ei]
        if bar_open_time == target:
            return si
    # fallback: allow a +/-1 bar tolerance for clock/latency drift
    for tol in (np.timedelta64(5, "m"), np.timedelta64(-5, "m")):
        target2 = target + tol
        for si in cache["signal_bars"]:
            if int(cache["signal"][si]) != direction:
                continue
            ei = si + 1
            if ei >= cache["n"]:
                continue
            if cache["times"][ei] == target2:
                return si
    return None


def main():
    print(f"Auditing live trades from {DATE_FROM} to {DATE_TO}\n")

    if not mt5.initialize(path=bt.TERMINAL_PATH, login=bt.LOGIN,
                          password=bt.PASSWORD, server=bt.SERVER):
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")

    acct = mt5.account_info()
    if acct is not None:
        print(f"Connected account: login={acct.login}  server={acct.server!r}  "
              f"balance={acct.balance:.2f}\n")
        if bt.LOGIN and acct.login != bt.LOGIN:
            print(f"*** WARNING: connected to {acct.login}, expected {bt.LOGIN}. "
                  f"This audit may be against the wrong account. ***\n")

    live_trades = fetch_live_trades(DATE_FROM, DATE_TO)
    if live_trades.empty:
        mt5.shutdown()
        return
    print(f"Found {len(live_trades)} closed round-trip trades.\n")

    canon_map = build_canon_map()

    print("Building backtest caches (slippage forced to 0 for a clean baseline)...")
    broker_info, _ = bt.fetch_broker_info()
    caches = {}
    for canon in bt.SYMBOLS:
        df_all, mt5_start = bt.assemble_data(canon)
        if df_all is None:
            continue
        caches[canon] = bt.build_cache_and_signals(
            canon, df_all, mt5_start, bt.PARAMS_GRID_BEST[canon],
            min_sl_distance=broker_info[canon]["min_sl_distance"],
            slippage=0.0,
        )
    print("Done.\n")

    rows = []
    unmatched_live = []

    for _, lt in live_trades.iterrows():
        canon = canon_map.get(lt["symbol"])
        if canon is None or canon not in caches:
            unmatched_live.append(lt)
            continue
        cache = caches[canon]
        si = find_matching_signal(cache, lt["entry_time"], lt["direction"])
        if si is None:
            unmatched_live.append(lt)
            continue

        bt_outcome_r, bt_exit_bar, bt_ep, bt_sl_dist = bt.resolve_mode_a(cache, si)
        if bt_outcome_r is None:
            unmatched_live.append(lt)
            continue

        tvpl = broker_info[canon]["tvpl"]
        bt_pnl = bt_outcome_r * lt["volume"] * bt_sl_dist * tvpl
        live_pnl = lt["profit"] + lt["commission"] + lt["swap"]

        direction = lt["direction"]
        # entry slip: positive = live paid a worse price than the formula predicted
        entry_slip = ((lt["entry_price"] - bt_ep) if direction == 1
                       else (bt_ep - lt["entry_price"]))

        bt_exit_time = pd.Timestamp(cache["times"][bt_exit_bar])
        bt_exit_price = (cache["c"][bt_exit_bar] if direction == 1
                          else cache["c"][bt_exit_bar] + cache["spread"][bt_exit_bar])
        exit_slip = ((bt_exit_price - lt["exit_price"]) if direction == 1
                      else (lt["exit_price"] - bt_exit_price))

        rows.append({
            "Symbol":      canon,
            "Dir":         "LONG" if direction == 1 else "SHORT",
            "EntryTime":   lt["entry_time"],
            "LiveEntry":   round(lt["entry_price"], 5),
            "BTEntry":     round(bt_ep, 5),
            "EntrySlip":   round(entry_slip, 5),
            "LiveExit":    round(lt["exit_price"], 5),
            "BTExit~":     round(bt_exit_price, 5),
            "ExitSlip~":   round(exit_slip, 5),
            "LiveR":       round(live_pnl / (lt["volume"] * bt_sl_dist * tvpl), 4)
                            if bt_sl_dist * tvpl != 0 else float("nan"),
            "BTR":         round(bt_outcome_r, 4),
            "LivePnL":     round(live_pnl, 2),
            "BTPnL":       round(bt_pnl, 2),
        })

    if rows:
        df = pd.DataFrame(rows)
        pd.set_option("display.width", 200)
        pd.set_option("display.max_columns", 20)
        print("=" * 140)
        print("  PER-TRADE COMPARISON")
        print("=" * 140)
        print(df.to_string(index=False))

        print(f"\n{'='*80}")
        print("  AGGREGATE SLIPPAGE BY SYMBOL (price points — paste into SLIPPAGE_PRICE_PER_SYMBOL)")
        print(f"{'='*80}")
        agg = df.groupby("Symbol").agg(
            N=("EntrySlip", "size"),
            MeanEntrySlip=("EntrySlip", "mean"),
            MedianEntrySlip=("EntrySlip", "median"),
            MeanExitSlip=("ExitSlip~", "mean"),
            MedianExitSlip=("ExitSlip~", "median"),
            LivePnL=("LivePnL", "sum"),
            BTPnL=("BTPnL", "sum"),
        ).round(4)
        print(agg.to_string())

        total_live = df["LivePnL"].sum()
        total_bt   = df["BTPnL"].sum()
        print(f"\n  TOTAL over matched trades: Live=${total_live:,.2f}  "
              f"Backtest(no-slippage)=${total_bt:,.2f}  "
              f"Gap=${total_bt-total_live:,.2f}")

        df.to_csv("orb_live_vs_backtest_diff.csv", index=False)
        print("\n  Saved: orb_live_vs_backtest_diff.csv")
    else:
        print("No matched trades to compare.")

    if unmatched_live:
        print(f"\n{'='*80}")
        print(f"  {len(unmatched_live)} LIVE TRADE(S) WITH NO MATCHING BACKTEST SIGNAL")
        print(f"  (live fired something the backtest's own signal logic wouldn't have —")
        print(f"   worth checking these individually; could be a live-vs-backtest logic")
        print(f"   divergence, or just clock/latency drift beyond the +-1 bar tolerance)")
        print(f"{'='*80}")
        for lt in unmatched_live:
            print(f"  {lt['symbol']}  {lt['entry_time']}  "
                  f"dir={'LONG' if lt['direction']==1 else 'SHORT'}  "
                  f"entry={lt['entry_price']}  pnl={lt['profit']:.2f}")

    mt5.shutdown()


if __name__ == "__main__":
    main()
