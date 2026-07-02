"""
==============================================================================
ORB  —  CHRONOLOGICAL MULTI-SYMBOL BACKTEST  v3
         LEAN LOGGING  +  EXIT-MODE COMPARISON
==============================================================================

EXIT MODES (compared per combo, no re-gridding):
  A  CURRENT   — BE at 1R then ATR trailing stop (original logic)
  B  FIXED_RR  — Fixed SL (same sl_dist), exit at first RR target hit
                 Tested at RR 1, 2, 3, 4, 5. Best by expectancy reported.
  C  WIDE_RR   — SL doubled (2× sl_dist), same RR ladder.
                 Rationale: wider SL → fewer stop-outs, larger targets.

Logging changes vs v2:
  • Per-trade inline log REMOVED (verbose noise)
  • Trade-by-trade header block REMOVED
  • Summary R/loss-cluster stats still printed per combo per mode
  • Clean 3-way comparison table printed at end
==============================================================================
"""

import os, sys, io, logging, bisect, datetime, collections
import numpy as np
import pandas as pd
from logging.handlers import RotatingFileHandler

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

# ── Logging ────────────────────────────────────────────────────────────────
logger = logging.getLogger("ORB_CHRONO")
logger.setLevel(logging.INFO)
_fh = RotatingFileHandler("orb_chrono_compare.log", maxBytes=15_000_000,
                           backupCount=3, encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
logger.addHandler(_fh)
_sh = logging.StreamHandler(
    io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace"))
_sh.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_sh)

# ── MT5 connection ─────────────────────────────────────────────────────────
TERMINAL_PATH = os.environ.get("MT5_TERMINAL_PATH",
                                r"C:\Program Files\MetaTrader 5\terminal64.exe")
LOGIN    = int(os.environ.get("MT5_LOGIN",    0))
PASSWORD =     os.environ.get("MT5_PASSWORD", "")
SERVER   =     os.environ.get("MT5_SERVER",   "")

CSV_DIR = os.environ.get("ORB_CSV_DIR", r"C:\Users\Administrator\Documents")

# ── Broker constants ────────────────────────────────────────────────────────
STARTING_BALANCE  = 25_000.0
RISK_PER_TRADE    = 0.005
VOL_MIN           = 0.10
VOL_STEP          = 0.01
VOL_MAX           = 250.0
MAX_RISK_MULTIPLE = 2.0
FIXED_LOT         = 0.10

TICK_VALUE_FALLBACK = {
    "US30":  1.0,
    "US500": 1.0,
    "UK100": 1.0,
    "GER40": 1.0,
}

# ── Strategy constants ──────────────────────────────────────────────────────
FETCH_BARS_M5  = 140_000
M5_SECONDS     = 300
WARMUP_M5      = 200
ATR_PERIOD     = 14
ATR_PCT_THRESH = 0.30
MAX_HOLD       = 48

OR_BARS = {15: 3, 30: 6, 60: 12}

SESSION = {
    "US30":  {"open_h": 13, "open_m": 30, "close_h": 20},
    "UK100": {"open_h":  8, "open_m":  0, "close_h": 16},
    "GER40": {"open_h":  8, "open_m":  0, "close_h": 17},
}

PARAMS_GRID_BEST = {
    "US30":  {"or_minutes": 15, "sl_range_mult": 0.5, "trail_atr_mult": 0.5,
              "min_break_atr": 0.0, "max_trades_day": 1, "cooldown_bars": 3},
    "UK100": {"or_minutes": 15, "sl_range_mult": 0.5, "trail_atr_mult": 0.5,
              "min_break_atr": 0.0, "max_trades_day": 1, "cooldown_bars": 3},
    "GER40": {"or_minutes": 15, "sl_range_mult": 0.5, "trail_atr_mult": 0.5,
              "min_break_atr": 0.0, "max_trades_day": 2, "cooldown_bars": 3},
}

SYMBOL_ALIASES = {
    "US30":  ["US30C",  "US30.cash", "US30",  "DJI30",  "DJIA",  "WS30",  "DOW30",  "US30Cash"],
    "US500": ["US500",  "SPX500",    "SP500", "SPX",   "US500Cash"],
    "UK100": ["UK100",  "FTSE100",   "FTSE",  "UK100Cash", "UKX"],
    "GER40": ["DE40C",  "GER40.cash","GER40", "DAX40", "DAX",   "GER30", "DE40",   "GER40Cash"],
}

CSV_STEMS = {
    "US30":  ["US30.cash", "US30C",  "US30",  "DJ30"],
    "US500": ["US500",     "SP500",  "SPX500"],
    "UK100": ["UK100.cash", "UK100", "FTSE",  "FTSE100"],
    "GER40": ["GER40.cash","DE40C",  "GER40", "DAX40", "DAX"],
}

SYMBOLS            = list(SESSION.keys())
_BAR_DURATION      = np.timedelta64(5, "m")
SPREAD_PTS_PER_PT  = 100.0
ROLLING_R_DAYS     = 7

# Exit-mode RR ladder for modes B and C
RR_TARGETS = [1, 2, 3, 4, 5]

# SL multiplier for mode C
WIDE_SL_MULT = 2.0

DAILY_LOSS_CAP_PCT = 0.0475
DAILY_LOSS_BUDGET  = STARTING_BALANCE * DAILY_LOSS_CAP_PCT


# ==============================================================================
#  SECTION 0 — R STATS & LOSS CLUSTER HELPERS  (lean — no per-trade logging)
# ==============================================================================

class RTracker:
    """
    Accumulates trade records silently; prints a summary block on request.
    Per-trade inline logging removed.
    """

    def __init__(self, combo_name: str, mode_label: str,
                 rolling_days: int = ROLLING_R_DAYS):
        self.combo        = combo_name
        self.mode         = mode_label
        self.rolling_days = rolling_days
        self.trades: list[dict] = []
        self._streak          = 0
        self._cur_streak_type = None
        self._loss_streaks: list[int] = []
        self._win_streaks:  list[int] = []

    def record(self, *, sym: str, trade_date, entry_hour: int,
               direction: int, outcome_r: float, pnl: float,
               lot: float, balance_after: float):

        cutoff = pd.Timestamp(trade_date) - pd.Timedelta(days=self.rolling_days)
        recent = [t["outcome_r"] for t in self.trades
                  if pd.Timestamp(t["trade_date"]) > cutoff]
        roll_r = float(np.mean(recent)) if recent else float("nan")

        result = "W" if outcome_r > 0 else "L"
        if result == self._cur_streak_type:
            self._streak = (abs(self._streak) + 1) * (1 if result == "W" else -1)
        else:
            if self._cur_streak_type == "L" and self._streak < 0:
                self._loss_streaks.append(abs(self._streak))
            elif self._cur_streak_type == "W" and self._streak > 0:
                self._win_streaks.append(self._streak)
            self._streak = 1 if result == "W" else -1
            self._cur_streak_type = result

        self.trades.append({
            "combo":          self.combo,
            "mode":           self.mode,
            "sym":            sym,
            "trade_date":     str(trade_date),
            "entry_hour_utc": entry_hour,
            "direction":      direction,
            "outcome_r":      round(outcome_r, 4),
            "rolling_7d_r":   round(roll_r, 4) if not np.isnan(roll_r) else None,
            "pnl":            round(pnl, 2),
            "lot":            lot,
            "balance_after":  round(balance_after, 2),
            "result":         result,
            "cur_streak":     self._streak,
        })

    def _flush_streaks(self):
        if self._cur_streak_type == "L" and self._streak < 0:
            self._loss_streaks.append(abs(self._streak))
        elif self._cur_streak_type == "W" and self._streak > 0:
            self._win_streaks.append(self._streak)

    def print_stats(self):
        self._flush_streaks()
        trades = self.trades
        if not trades:
            logger.info(f"  [{self.combo} | {self.mode}] No trades recorded.")
            return

        df     = pd.DataFrame(trades)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        r      = df["outcome_r"].values
        wins   = df[df["result"] == "W"]
        losses = df[df["result"] == "L"]
        sep    = "─" * 78

        logger.info(f"\n{'='*78}")
        logger.info(f"  R STATS — {self.combo}  [{self.mode}]")
        logger.info(f"{'='*78}")
        logger.info(f"  {sep}")
        logger.info(f"  CORE R SUMMARY")
        logger.info(f"  {sep}")
        logger.info(f"  Trades          : {len(r):,}")
        logger.info(f"  Avg R/trade     : {r.mean():+.4f}")
        logger.info(f"  Median R        : {np.median(r):+.4f}")
        logger.info(f"  Std Dev R       : {r.std():.4f}")
        logger.info(f"  Best / Worst    : {r.max():+.4f}R  /  {r.min():+.4f}R")
        logger.info(f"  Win rate        : {len(wins)/len(r):.1%}")
        if len(wins):
            logger.info(f"  Avg win R       : {wins['outcome_r'].mean():+.4f}")
        if len(losses):
            logger.info(f"  Avg loss R      : {losses['outcome_r'].mean():+.4f}")
        if len(wins) and len(losses):
            rr = abs(wins["outcome_r"].mean() / losses["outcome_r"].mean())
            logger.info(f"  Win/Loss R ratio: {rr:.3f}")

        # Rolling 7d
        roll = df["rolling_7d_r"].dropna()
        if len(roll):
            logger.info(f"\n  {sep}")
            logger.info(f"  ROLLING {self.rolling_days}-DAY AVG R")
            logger.info(f"  {sep}")
            logger.info(f"  Avg of rolling values : {roll.mean():+.4f}")
            logger.info(f"  Range                 : {roll.min():+.4f}  to  {roll.max():+.4f}")

        # Loss streaks
        logger.info(f"\n  {sep}")
        logger.info(f"  LOSS CLUSTER ANALYSIS")
        logger.info(f"  {sep}")
        if self._loss_streaks:
            ls = np.array(self._loss_streaks)
            logger.info(f"  Completed loss streaks : {len(ls)}")
            logger.info(f"  Longest                : {ls.max()}")
            logger.info(f"  Avg length             : {ls.mean():.2f}")
            logger.info(f"  Streaks ≥3             : {(ls >= 3).sum()}")
            logger.info(f"  Streaks ≥5             : {(ls >= 5).sum()}")
        else:
            logger.info("  No completed loss streaks detected.")

        # Hour heatmap (compact)
        logger.info(f"\n  {sep}")
        logger.info(f"  LOSS HEATMAP — HOUR (UTC)")
        logger.info(f"  {sep}")
        hour_total = df.groupby("entry_hour_utc").size()
        hour_loss  = losses.groupby("entry_hour_utc").size()
        logger.info(f"  {'Hr':>3}  {'Tot':>5}  {'Los':>5}  {'LR':>7}  {'AvgR':>7}")
        for h in sorted(set(hour_total.index) | set(hour_loss.index)):
            tot  = hour_total.get(h, 0)
            lss  = hour_loss.get(h, 0)
            rate = lss / tot if tot else 0.0
            avgr = df[df["entry_hour_utc"] == h]["outcome_r"].mean()
            logger.info(f"  {h:>3}h  {tot:>5}  {lss:>5}  {rate:>6.1%}  {avgr:>+6.3f}")

        # Monthly
        logger.info(f"\n  {sep}")
        logger.info(f"  MONTHLY BREAKDOWN")
        logger.info(f"  {sep}")
        df["ym"]     = df["trade_date"].dt.to_period("M")
        mo_total     = df.groupby("ym").size()
        mo_loss      = df[df["result"] == "L"].groupby("ym").size()
        mo_avgr      = df.groupby("ym")["outcome_r"].mean()
        mo_cumr      = df.groupby("ym")["outcome_r"].sum()
        logger.info(f"  {'Month':<9}  {'Trd':>5}  {'Los':>5}  {'LR':>7}  {'AvgR':>7}  {'SumR':>8}")
        for m in sorted(mo_total.index):
            tot  = mo_total.get(m, 0)
            lss  = mo_loss.get(m, 0)
            rate = lss / tot if tot else 0.0
            avgr = mo_avgr.get(m, float("nan"))
            cumr = mo_cumr.get(m, float("nan"))
            flag = "  ← worst" if not np.isnan(avgr) and avgr == mo_avgr.min() else ""
            logger.info(f"  {str(m):<9}  {tot:>5}  {lss:>5}  {rate:>6.1%}  "
                        f"{avgr:>+6.3f}  {cumr:>+7.3f}{flag}")

        logger.info(f"\n{'='*78}\n")


# ==============================================================================
#  SECTION 1 — CSV LOADING
# ==============================================================================

def _csv_path(canon: str):
    if not os.path.isdir(CSV_DIR):
        return None
    for stem in CSV_STEMS.get(canon, [canon]):
        for ext in (".csv", ".CSV", ".cvs", ".CVS"):
            p = os.path.join(CSV_DIR, stem + ext)
            if os.path.isfile(p):
                return p
    return None


def _load_csv(canon: str) -> pd.DataFrame:
    path = _csv_path(canon)
    if path is None:
        logger.info(f"  [{canon}] No CSV in {CSV_DIR} — ATR will use MT5 data only")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            first_line = f.readline()
        sep = "\t" if "\t" in first_line else ","
        df  = pd.read_csv(path, sep=sep, engine="python")

        def _clean_col(c):
            c = c.strip().lower().replace("<", "").replace(">", "")
            if (c.startswith("t") and
                    c[1:] in ("date","time","open","high","low",
                               "close","tickvol","vol","spread")):
                c = c[1:]
            return c

        df.columns = [_clean_col(c) for c in df.columns]

        if "date" in df.columns and "time" in df.columns:
            combined = (df["date"].astype(str).str.strip()
                        + " " + df["time"].astype(str).str.strip())
            fixed = combined.str.replace(
                r"(\d{4})\.(\d{2})\.(\d{2})",
                lambda m: f"{m.group(1)}-{m.group(2)}-{m.group(3)}", regex=True)
            df["time_utc"] = pd.to_datetime(fixed)
        elif "time_utc" in df.columns:
            df["time_utc"] = pd.to_datetime(df["time_utc"].astype(str))
        elif "time" in df.columns:
            if pd.api.types.is_numeric_dtype(df["time"]):
                df["time_utc"] = pd.to_datetime(df["time"].astype(np.int64), unit="s")
            else:
                sample = str(df["time"].iloc[0]).strip()
                if "." in sample.split(" ")[0]:
                    fixed = df["time"].astype(str).str.replace(
                        r"(\d{4})\.(\d{2})\.(\d{2})",
                        lambda m: f"{m.group(1)}-{m.group(2)}-{m.group(3)}", regex=True)
                    df["time_utc"] = pd.to_datetime(fixed)
                else:
                    df["time_utc"] = pd.to_datetime(df["time"].astype(str))
        else:
            raise ValueError(f"No usable time column. Found: {list(df.columns)}")

        df = df[["time_utc","open","high","low","close"]].copy()
        for col in ("open","high","low","close"):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float64)
        df.dropna(inplace=True)
        df.sort_values("time_utc", inplace=True)
        df.drop_duplicates(subset="time_utc", keep="last", inplace=True)
        df.reset_index(drop=True, inplace=True)

        logger.info(f"  [{canon}] CSV ({os.path.basename(path)}): "
                    f"{len(df):,} bars  "
                    f"{df['time_utc'].iloc[0].date()} -> "
                    f"{df['time_utc'].iloc[-1].strftime('%Y-%m-%d %H:%M')}")
        return df
    except Exception as e:
        logger.error(f"  [{canon}] CSV load failed: {e}", exc_info=True)
        return None


# ==============================================================================
#  SECTION 2 — MT5 HELPERS
# ==============================================================================

def resolve_symbol(canonical: str):
    all_broker = {s.name.upper(): s.name for s in (mt5.symbols_get() or [])}
    for alias in SYMBOL_ALIASES[canonical]:
        info = mt5.symbol_info(alias)
        if info is not None:
            if not info.visible:
                mt5.symbol_select(alias, True)
            return alias
        for up, name in all_broker.items():
            if up.startswith(alias.upper()):
                mt5.symbol_select(name, True)
                return name
    return None


def fetch_tick_values() -> dict:
    tvpl = {}
    for canon in SYMBOLS:
        fallback = TICK_VALUE_FALLBACK.get(canon, 1.0)
        if not MT5_AVAILABLE:
            tvpl[canon] = fallback
            continue
        broker = resolve_symbol(canon)
        if broker is None:
            tvpl[canon] = fallback
            logger.warning(f"  [{canon}] not found — fallback tv={fallback}")
            continue
        info = mt5.symbol_info(broker)
        if info is None or info.trade_tick_size <= 0:
            tvpl[canon] = fallback
            logger.warning(f"  [{canon}] info invalid — fallback tv={fallback}")
            continue
        tv = info.trade_tick_value / info.trade_tick_size
        tvpl[canon] = tv
        logger.info(f"  [{canon}] broker={broker} tick_val/lot={tv:.6f}")
    return tvpl


def _last_closed_bar_open_time() -> datetime.datetime:
    now  = datetime.datetime.utcnow()
    secs = now.hour * 3600 + now.minute * 60 + now.second
    rem  = secs % M5_SECONDS
    forming_open = now - datetime.timedelta(seconds=rem)
    return forming_open - datetime.timedelta(seconds=M5_SECONDS)


def fetch_m5_from_mt5(canonical: str) -> pd.DataFrame:
    broker = resolve_symbol(canonical)
    if broker is None:
        return None

    cols = ["time","open","high","low","close",
            "tick_volume","spread","real_volume"]
    raw  = None
    for attempt in [FETCH_BARS_M5, FETCH_BARS_M5 // 2, 5_000]:
        r = mt5.copy_rates_from_pos(broker, mt5.TIMEFRAME_M5, 0, attempt)
        if r is not None and len(r) >= 500:
            raw = pd.DataFrame(r, columns=cols)
            break

    if raw is None:
        logger.warning(f"  [{canonical}] MT5 fetch failed")
        return None

    df = raw[["time","open","high","low","close","spread"]].copy()
    df["time_utc"]   = pd.to_datetime(df["time"].astype(np.int64), unit="s")
    df["spread_pts"] = df["spread"].values.astype(np.float64)
    df.drop(columns=["time","spread"], inplace=True)

    last_closed = pd.Timestamp(_last_closed_bar_open_time())
    df = df[df["time_utc"] <= last_closed].copy()
    df.sort_values("time_utc", inplace=True)
    df.drop_duplicates(subset="time_utc", keep="last", inplace=True)
    df.reset_index(drop=True, inplace=True)

    if df["spread_pts"].max() == 0:
        logger.warning(f"  [{canonical}] spread all zeros")

    logger.info(
        f"  [{canonical}] MT5 signal window: {len(df):,} bars  "
        f"{df['time_utc'].iloc[0].strftime('%Y-%m-%d')} -> "
        f"{df['time_utc'].iloc[-1].strftime('%Y-%m-%d %H:%M')}  "
        f"spread mean={df['spread_pts'].mean():.1f} max={df['spread_pts'].max():.0f}"
    )
    return df


# ==============================================================================
#  SECTION 3 — DATA ASSEMBLY
# ==============================================================================

def assemble_data(canonical: str) -> tuple:
    df_mt5 = fetch_m5_from_mt5(canonical)
    if df_mt5 is None or len(df_mt5) == 0:
        logger.error(f"  [{canonical}] No MT5 data — cannot run backtest")
        return None, None

    mt5_start = df_mt5["time_utc"].iloc[0]
    df_csv    = _load_csv(canonical)

    if df_csv is not None and len(df_csv) > 0:
        df_csv_pre = df_csv[df_csv["time_utc"] < mt5_start].copy()
        df_csv_pre["spread_pts"] = 0.0
        if len(df_csv_pre) == 0:
            df_csv_pre = None
        else:
            logger.info(
                f"  [{canonical}] CSV ATR prefix: {len(df_csv_pre):,} bars  "
                f"({df_csv_pre['time_utc'].iloc[0].date()} -> "
                f"{df_csv_pre['time_utc'].iloc[-1].strftime('%Y-%m-%d %H:%M')})"
                f"  [ATR warmup only]"
            )
    else:
        df_csv_pre = None

    frames = []
    if df_csv_pre is not None:
        frames.append(df_csv_pre[["time_utc","open","high","low","close","spread_pts"]])
    frames.append(df_mt5[["time_utc","open","high","low","close","spread_pts"]])

    df_all = pd.concat(frames, ignore_index=True)
    df_all.sort_values("time_utc", inplace=True)
    df_all.drop_duplicates(subset="time_utc", keep="last", inplace=True)
    df_all.reset_index(drop=True, inplace=True)

    n_pre    = int((df_all["time_utc"] < mt5_start).sum())
    n_signal = int((df_all["time_utc"] >= mt5_start).sum())
    logger.info(
        f"  [{canonical}] ASSEMBLED: {len(df_all):,} bars  "
        f"| ATR-prefix={n_pre:,}  signal={n_signal:,}  "
        f"| {df_all['time_utc'].iloc[0].date()} -> "
        f"{df_all['time_utc'].iloc[-1].strftime('%Y-%m-%d %H:%M')}"
    )
    return df_all, mt5_start


# ==============================================================================
#  SECTION 4 — LOT SIZING
# ==============================================================================

def compute_lot_aware(balance, sl_dist, tick_value_per_lot, vol_max_cap=VOL_MAX):
    if sl_dist < 1e-9 or tick_value_per_lot <= 0:
        return None, 0.0, 0.0, 0.0, True
    intended_risk = balance * RISK_PER_TRADE
    raw_lot       = intended_risk / (sl_dist * tick_value_per_lot)
    lot = max(VOL_MIN, min(vol_max_cap, round(raw_lot / VOL_STEP) * VOL_STEP))
    lot = round(lot, 8)
    actual_loss   = lot * sl_dist * tick_value_per_lot
    risk_multiple = actual_loss / intended_risk if intended_risk > 0 else float("inf")
    rejected      = risk_multiple > MAX_RISK_MULTIPLE
    return lot, intended_risk, actual_loss, risk_multiple, rejected


# ==============================================================================
#  SECTION 5 — INDICATORS
# ==============================================================================

def atr_wilder(h, l, c):
    n  = len(h)
    tr = np.empty(n)
    tr[0]  = h[0] - l[0]
    tr[1:] = np.maximum(
        h[1:] - l[1:],
        np.maximum(np.abs(h[1:] - c[:-1]),
                   np.abs(l[1:] - c[:-1])))
    out = np.full(n, np.nan)
    if n < ATR_PERIOD:
        return out
    out[ATR_PERIOD - 1] = tr[:ATR_PERIOD].mean()
    k = 1.0 / ATR_PERIOD
    for i in range(ATR_PERIOD, n):
        out[i] = out[i - 1] * (1.0 - k) + tr[i] * k
    return out


def expanding_pct_rank(arr):
    n    = len(arr)
    out  = np.full(n, np.nan)
    hist = []
    for i in range(WARMUP_M5, n):
        v = arr[i]
        if np.isnan(v):
            continue
        if hist:
            out[i] = bisect.bisect_left(hist, v) / len(hist)
        bisect.insort(hist, v)
    return out


# ==============================================================================
#  SECTION 6 — CACHE + SIGNAL PRE-COMPUTATION
# ==============================================================================

def build_cache_and_signals(canonical: str, df_all: pd.DataFrame,
                             mt5_start: pd.Timestamp, params: dict) -> dict:
    cfg = SESSION[canonical]

    o  = df_all["open"].values.astype(np.float64)
    h  = df_all["high"].values.astype(np.float64)
    l  = df_all["low"].values.astype(np.float64)
    c  = df_all["close"].values.astype(np.float64)
    n  = len(c)

    spread_pts = df_all["spread_pts"].values.astype(np.float64) \
                 if "spread_pts" in df_all.columns else np.zeros(n)
    spread = spread_pts / SPREAD_PTS_PER_PT

    times = df_all["time_utc"].values
    utc_h = df_all["time_utc"].dt.hour.values.astype(np.int32)
    utc_m = df_all["time_utc"].dt.minute.values.astype(np.int32)
    dates = df_all["time_utc"].dt.date.values

    atr14   = atr_wilder(h, l, c)
    atr_pct = expanding_pct_rank(atr14)

    in_signal_window = df_all["time_utc"].values >= np.datetime64(mt5_start)

    in_session = np.array([
        (utc_h[i] > cfg["open_h"] or
         (utc_h[i] == cfg["open_h"] and utc_m[i] >= cfg["open_m"]))
        and utc_h[i] < cfg["close_h"]
        for i in range(n)
    ])
    is_open_bar = (utc_h == cfg["open_h"]) & (utc_m == cfg["open_m"])

    or_bars   = OR_BARS[params["or_minutes"]]
    day_start = {}
    for i in range(n):
        if is_open_bar[i]:
            d = dates[i]
            if d not in day_start:
                day_start[d] = i

    day_or = {}
    for d, si in day_start.items():
        ei = si + or_bars
        if ei <= n:
            day_or[d] = (h[si:ei].max(), l[si:ei].min())

    or_high = np.full(n, np.nan)
    or_low  = np.full(n, np.nan)
    for i in range(n):
        if not in_session[i]:
            continue
        d = dates[i]
        if d not in day_or or d not in day_start:
            continue
        if i < day_start[d] + or_bars:
            continue
        or_high[i], or_low[i] = day_or[d]

    cooldown      = params["cooldown_bars"]
    max_t         = params["max_trades_day"]
    min_break_atr = params["min_break_atr"]

    atr_ok = (~np.isnan(atr_pct)) & (atr_pct >= ATR_PCT_THRESH)
    valid  = np.zeros(n, dtype=bool)
    valid[WARMUP_M5:n - 1] = True

    base = (in_session & atr_ok & valid
            & ~np.isnan(or_high)
            & in_signal_window)
    body = np.abs(c - o)

    breaks_up   = base & (c > or_high)
    breaks_down = base & (c < or_low)

    if min_break_atr > 0:
        strong      = ~np.isnan(atr14) & (body >= min_break_atr * atr14)
        breaks_up   = breaks_up   & strong
        breaks_down = breaks_down & strong

    signal    = np.zeros(n, dtype=np.int8)
    last_sig  = -9999
    day_count = {}
    candidates = sorted(
        [(i,  1) for i in np.where(breaks_up)[0]] +
        [(i, -1) for i in np.where(breaks_down)[0]]
    )
    for i, d in candidates:
        if i - last_sig < cooldown:
            continue
        day = dates[i]
        if day_count.get(day, 0) >= max_t:
            continue
        signal[i] = d
        last_sig   = i
        day_count[day] = day_count.get(day, 0) + 1

    signal_bars = np.where(signal != 0)[0]
    signal_bars = signal_bars[signal_bars + 1 < n]

    n_csv_pre = int((~in_signal_window).sum())
    n_mt5     = int(in_signal_window.sum())
    logger.info(
        f"  [{canonical}] signals={len(signal_bars):,}  "
        f"ATR-prefix={n_csv_pre:,}  MT5-window={n_mt5:,}  "
        f"n_weeks={n_mt5 / (12 * 24 * 5):.1f}"
    )

    return {
        "sym":              canonical,
        "n":                n,
        "o": o, "h": h, "l": l, "c": c,
        "atr14":            atr14,
        "or_high":          or_high,
        "or_low":           or_low,
        "signal":           signal,
        "signal_bars":      signal_bars,
        "utc_h":            utc_h,
        "dates":            dates,
        "times":            times,
        "in_session":       in_session,
        "in_signal_window": in_signal_window,
        "spread":           spread,
        "cfg":              cfg,
        "n_weeks":          n_mt5 / (12 * 24 * 5),
        "params":           params,
    }


# ==============================================================================
#  SECTION 7 — TIMELINE BUILDER
# ==============================================================================

def build_master_timeline(caches: dict) -> list:
    events = []
    for sym, cache in caches.items():
        for si in cache["signal_bars"]:
            bar_close_ts = cache["times"][si] + _BAR_DURATION
            events.append((bar_close_ts, sym, si))
    events.sort(key=lambda x: x[0])
    total_sigs = len(events)
    syms_repr  = {sym: sum(1 for _, s, _ in events if s == sym)
                  for sym in caches}
    logger.info(f"  Timeline: {total_sigs:,} events  |  "
                + "  ".join(f"{s}={v}" for s, v in syms_repr.items()))
    return events


# ==============================================================================
#  SECTION 8 — TRADE RESOLVERS
# ==============================================================================

def _entry_params(cache: dict, si: int):
    """Shared entry price, sl_dist, direction for all modes."""
    o      = cache["o"]
    h      = cache["h"]
    l      = cache["l"]
    atr14  = cache["atr14"]
    params = cache["params"]
    signal = cache["signal"]
    spread = cache["spread"]

    ei        = si + 1
    direction = int(signal[si])
    sp_entry  = spread[ei]
    ep        = o[ei] + sp_entry if direction == 1 else o[ei]
    atr       = atr14[si]

    if np.isnan(atr) or atr <= 0:
        return None

    or_size = cache["or_high"][si] - cache["or_low"][si]
    if np.isnan(or_size) or or_size <= 0:
        return None

    sl_mult = params["sl_range_mult"]
    sl_dist = max(sl_mult * or_size, atr * 0.05)
    if sl_dist < 0.05 * atr:
        sl_dist = 0.05 * atr
    if sl_dist <= 0:
        return None

    return {
        "ei": ei, "direction": direction,
        "ep": ep, "sl_dist": sl_dist,
        "atr": atr,
    }


# ── MODE A: BE + ATR trail (original) ─────────────────────────────────────

def resolve_mode_a(cache: dict, si: int):
    """Original: BE at 1R then ATR trailing stop."""
    ep_info = _entry_params(cache, si)
    if ep_info is None:
        return None, 0, 0.0, 0.0

    n, o, h, l, c = (cache["n"], cache["o"], cache["h"],
                     cache["l"], cache["c"])
    atr14  = cache["atr14"]
    dates  = cache["dates"]
    utc_h  = cache["utc_h"]
    cfg    = cache["cfg"]
    params = cache["params"]
    spread = cache["spread"]

    ei        = ep_info["ei"]
    direction = ep_info["direction"]
    ep        = ep_info["ep"]
    sl_dist   = ep_info["sl_dist"]
    atr       = ep_info["atr"]

    trail_mult = params["trail_atr_mult"]
    one_r_tgt  = ep + direction * sl_dist

    cur_sl    = ep - direction * sl_dist
    be_active = False
    outcome_r = 0.0
    closed    = False
    exit_bar  = ei
    entry_date = dates[ei]

    for k in range(1, MAX_HOLD + 1):
        bi = ei + k
        if bi >= n:
            break
        if dates[bi] != entry_date or utc_h[bi] >= cfg["close_h"]:
            exit_price = o[bi] if direction == 1 else o[bi] + spread[bi]
            outcome_r  = direction * (exit_price - ep) / sl_dist
            exit_bar   = bi
            closed     = True
            break

        bh, bl = h[bi], l[bi]
        sp_bi  = spread[bi]

        if direction == 1 and bl <= cur_sl:
            outcome_r = (cur_sl - ep) / sl_dist
            exit_bar  = bi; closed = True; break
        if direction == -1 and (bh + sp_bi) >= cur_sl:
            outcome_r = (ep - cur_sl) / sl_dist
            exit_bar  = bi; closed = True; break

        if not be_active:
            if direction == 1  and bh >= one_r_tgt:
                be_active = True; cur_sl = ep
            if direction == -1 and bl <= one_r_tgt:
                be_active = True; cur_sl = ep

        if be_active:
            ta = atr14[bi] if not np.isnan(atr14[bi]) else atr
            if direction == 1:
                cur_sl = max(cur_sl, bh - trail_mult * ta)
            else:
                cur_sl = min(cur_sl, bl + trail_mult * ta)

    if not closed:
        exit_bar   = min(ei + MAX_HOLD, n - 1)
        exit_price = (c[exit_bar] if direction == 1
                      else c[exit_bar] + spread[exit_bar])
        outcome_r  = direction * (exit_price - ep) / sl_dist

    return outcome_r, exit_bar, ep, sl_dist


# ── MODE B: Fixed SL, fixed RR target ─────────────────────────────────────

def resolve_mode_b(cache: dict, si: int, rr_target: float,
                   sl_multiplier: float = 1.0):
    """
    Fixed stop, single RR exit target.
    sl_multiplier=1.0  →  Mode B (normal SL)
    sl_multiplier=2.0  →  Mode C (wide SL)
    """
    ep_info = _entry_params(cache, si)
    if ep_info is None:
        return None, 0, 0.0, 0.0

    n, o, h, l, c = (cache["n"], cache["o"], cache["h"],
                     cache["l"], cache["c"])
    dates  = cache["dates"]
    utc_h  = cache["utc_h"]
    cfg    = cache["cfg"]
    spread = cache["spread"]

    ei        = ep_info["ei"]
    direction = ep_info["direction"]
    ep        = ep_info["ep"]
    sl_dist   = ep_info["sl_dist"] * sl_multiplier

    cur_sl    = ep - direction * sl_dist
    tp        = ep + direction * sl_dist * rr_target
    outcome_r = 0.0
    closed    = False
    exit_bar  = ei
    entry_date = dates[ei]

    for k in range(1, MAX_HOLD + 1):
        bi = ei + k
        if bi >= n:
            break
        if dates[bi] != entry_date or utc_h[bi] >= cfg["close_h"]:
            exit_price = o[bi] if direction == 1 else o[bi] + spread[bi]
            outcome_r  = direction * (exit_price - ep) / sl_dist
            exit_bar   = bi; closed = True; break

        bh, bl = h[bi], l[bi]
        sp_bi  = spread[bi]

        # TP hit
        if direction == 1  and bh >= tp:
            outcome_r = rr_target
            exit_bar  = bi; closed = True; break
        if direction == -1 and bl <= tp:
            outcome_r = rr_target
            exit_bar  = bi; closed = True; break

        # SL hit
        if direction == 1  and bl <= cur_sl:
            outcome_r = -1.0
            exit_bar  = bi; closed = True; break
        if direction == -1 and (bh + sp_bi) >= cur_sl:
            outcome_r = -1.0
            exit_bar  = bi; closed = True; break

    if not closed:
        exit_bar   = min(ei + MAX_HOLD, n - 1)
        exit_price = (c[exit_bar] if direction == 1
                      else c[exit_bar] + spread[exit_bar])
        outcome_r  = direction * (exit_price - ep) / sl_dist

    return outcome_r, exit_bar, ep, sl_dist


# ==============================================================================
#  SECTION 9 — SIMULATION CORE (shared across modes)
# ==============================================================================

SYMBOL_COMBOS = [
    ("US30",),
    ("UK100",),
    ("GER40",),
    ("US30", "UK100"),
    ("US30", "GER40"),
    ("UK100", "GER40"),
    ("US30", "UK100", "GER40"),
]


def compute_vol_max_cap(sl_dist, tick_value_per_lot, max_trades_per_day_combo):
    if sl_dist < 1e-9 or tick_value_per_lot <= 0:
        return VOL_MIN
    per_trade = DAILY_LOSS_BUDGET / max_trades_per_day_combo
    raw = per_trade / (sl_dist * tick_value_per_lot)
    cap = max(VOL_MIN, min(VOL_MAX, round(raw / VOL_STEP) * VOL_STEP))
    return round(cap, 8)


# ── FIX: chronological entry/exit resolution ───────────────────────────────
#
# BUG (pre-fix): the old loop below processed signals in ENTRY order and
# applied each trade's realized PnL to `balance` immediately, in that same
# entry-order pass. That means position sizing for a later-ENTERED trade
# could reflect the PnL of an earlier-ENTERED trade that, in real time,
# hadn't actually EXITED yet (ORB holds can run up to MAX_HOLD=48 bars =
# 4 hours, so overlap across symbols in multi-symbol combos is common).
# The R-multiple of each trade (outcome_r) is unaffected — it only depends
# on price action — so expectancy/win-rate/profit-factor stats computed
# from per-trade R are unchanged. What's wrong is everything downstream of
# lot sizing: final balance, drawdown, and (via lot-dependent daily-loss
# breaches) the FTMO eval pass rate.
#
# FIX: resolve every trade's outcome up front (deterministic, balance-
# independent), then replay a single merged timeline of ENTRY and EXIT
# events in true chronological order. Lot size is computed at ENTRY time
# using only the balance that has actually been realized (i.e. reflects
# only trades that have already EXITED by that timestamp). PnL is applied
# to `balance` only at EXIT time.

def _resolve_all_trades(symbols: list, caches: dict, resolver_fn) -> list:
    """Pre-resolve every signal's outcome (price-action only, no balance
    dependency) and attach its entry/exit timestamps for chronological replay."""
    trades = []
    for sym in symbols:
        cache = caches[sym]
        for si in cache["signal_bars"]:
            outcome_r, exit_bar, ep, sl_dist = resolver_fn(cache, si)
            if outcome_r is None:
                continue
            entry_time = cache["times"][si] + _BAR_DURATION
            exit_time  = cache["times"][exit_bar] + _BAR_DURATION
            trades.append({
                "sym": sym, "si": si, "cache": cache,
                "entry_time": entry_time, "exit_time": exit_time,
                "outcome_r": outcome_r, "sl_dist": sl_dist,
            })
    trades.sort(key=lambda t: t["entry_time"])
    return trades


def _build_chrono_events(trades: list) -> list:
    """Merge entries and exits into one chronologically sorted event list.
    On an exact timestamp tie, EXIT is processed before ENTRY so that a
    trade closing at the same instant another opens never lets the new
    entry see PnL from a trade that, mechanically, hadn't closed yet."""
    events = []
    for idx, t in enumerate(trades):
        events.append((t["entry_time"], 1, idx))  # 1 = ENTRY
        events.append((t["exit_time"],  0, idx))  # 0 = EXIT  (sorts first on tie)
    events.sort(key=lambda e: (e[0], e[1]))
    return events


def _run_simulation_core(combo_name: str, mode_label: str,
                          symbols: list, all_caches: dict,
                          tick_values: dict,
                          resolver_fn) -> dict:
    """
    Generic simulation loop. resolver_fn(cache, si) → (outcome_r, exit_bar, ep, sl_dist).
    Returns a results dict + populated RTracker.
    """
    param_set = {sym: PARAMS_GRID_BEST[sym] for sym in symbols}
    max_trades_per_day_combo = sum(
        param_set[sym]["max_trades_day"] for sym in symbols
    )

    caches = {sym: all_caches[sym] for sym in symbols}
    # kept for the log line / event count parity with previous behaviour
    build_master_timeline(caches)

    trades = _resolve_all_trades(symbols, caches, resolver_fn)
    chrono = _build_chrono_events(trades)

    balance  = STARTING_BALANCE
    peak_bal = STARTING_BALANCE
    max_dd   = 0.0
    day_pnl: dict = {}
    max_day_loss  = 0.0

    per_sym = {sym: {"r": [], "pnl": [], "rejected": 0} for sym in symbols}
    r_tracker = RTracker(combo_name, mode_label)
    open_state: dict = {}   # idx -> {"lot": float, "rejected": bool}

    for t_time, etype, idx in chrono:
        t     = trades[idx]
        sym   = t["sym"]
        cache = t["cache"]
        tvpl  = tick_values[sym]
        sl_dist = t["sl_dist"]

        if etype == 1:  # ENTRY — size using balance realized so far
            vol_max_cap = compute_vol_max_cap(sl_dist, tvpl, max_trades_per_day_combo)
            lot, intended, actual_loss, risk_mult, rejected = \
                compute_lot_aware(balance, sl_dist, tvpl, vol_max_cap)
            open_state[idx] = {"lot": lot, "rejected": rejected}
            if rejected:
                per_sym[sym]["rejected"] += 1
            continue

        # etype == 0: EXIT — realize PnL into balance now
        st = open_state.get(idx)
        if st is None or st["rejected"]:
            continue
        lot = st["lot"]
        outcome_r = t["outcome_r"]

        pnl      = outcome_r * lot * sl_dist * tvpl
        balance += pnl
        peak_bal = max(peak_bal, balance)
        dd       = (peak_bal - balance) / peak_bal if peak_bal > 0 else 0.0
        max_dd   = max(max_dd, dd)

        si = t["si"]
        trade_date = cache["dates"][si + 1]
        entry_hour = int(cache["utc_h"][si + 1])

        day_pnl[str(trade_date)] = day_pnl.get(str(trade_date), 0.0) + pnl
        if day_pnl[str(trade_date)] < 0:
            max_day_loss = max(max_day_loss,
                               abs(day_pnl[str(trade_date)]))

        per_sym[sym]["r"].append(outcome_r)
        per_sym[sym]["pnl"].append(pnl)

        r_tracker.record(
            sym=sym, trade_date=trade_date, entry_hour=entry_hour,
            direction=int(cache["signal"][si]),
            outcome_r=outcome_r, pnl=pnl, lot=lot, balance_after=balance,
        )

    all_r   = np.array([v for s in symbols for v in per_sym[s]["r"]])
    total_rej = sum(per_sym[s]["rejected"] for s in symbols)
    nt      = len(all_r)
    wr      = float((all_r > 0).sum() / nt)   if nt > 0 else 0.0
    ex      = float(all_r.mean())              if nt > 0 else 0.0
    pos     = all_r[all_r > 0]
    neg     = all_r[all_r < 0]
    pf      = float(pos.sum() / -neg.sum()) \
              if len(neg) and neg.sum() != 0 else 0.0
    ret     = (balance - STARTING_BALANCE) / STARTING_BALANCE

    return {
        "combo":           combo_name,
        "mode":            mode_label,
        "symbols":         "+".join(symbols),
        "trades":          nt,
        "rejected":        total_rej,
        "win_rate":        round(wr, 4),
        "expectancy_r":    round(ex, 4),
        "profit_factor":   round(pf, 4),
        "mdd_pct":         round(max_dd * 100, 2),
        "max_day_loss_pct": round(max_day_loss / STARTING_BALANCE * 100, 2),
        "return_pct":      round(ret * 100, 2),
        "final_balance":   round(balance, 2),
    }, r_tracker


# ==============================================================================
#  SECTION 10 — COMBO SWEEP WITH MODE COMPARISON
# ==============================================================================

def run_combo_all_modes(combo_name: str, symbols: list,
                         all_caches: dict, tick_values: dict,
                         print_detailed_stats: bool = True) -> dict:
    """
    Runs Mode A, Mode B×RR_TARGETS, Mode C×RR_TARGETS for one combo.
    Returns a summary dict with the 3-way comparison.
    """
    results_by_mode = {}

    # ── Mode A: current (BE + trail) ─────────────────────────────────────
    res_a, tracker_a = _run_simulation_core(
        combo_name, "A:BE+TRAIL", symbols, all_caches, tick_values,
        lambda cache, si: resolve_mode_a(cache, si)
    )
    results_by_mode["A:BE+TRAIL"] = res_a
    if print_detailed_stats:
        tracker_a.print_stats()

    # ── Mode B: fixed SL + RR ladder ─────────────────────────────────────
    best_b = None
    for rr in RR_TARGETS:
        label = f"B:RR{rr}"
        res, tracker = _run_simulation_core(
            combo_name, label, symbols, all_caches, tick_values,
            lambda cache, si, _rr=rr: resolve_mode_b(cache, si, _rr,
                                                      sl_multiplier=1.0)
        )
        results_by_mode[label] = res
        if print_detailed_stats:
            tracker.print_stats()
        if best_b is None or res["expectancy_r"] > best_b["expectancy_r"]:
            best_b = res

    # ── Mode C: wide SL (×2) + RR ladder ─────────────────────────────────
    best_c = None
    for rr in RR_TARGETS:
        label = f"C:WIDE_RR{rr}"
        res, tracker = _run_simulation_core(
            combo_name, label, symbols, all_caches, tick_values,
            lambda cache, si, _rr=rr: resolve_mode_b(cache, si, _rr,
                                                      sl_multiplier=WIDE_SL_MULT)
        )
        results_by_mode[label] = res
        if print_detailed_stats:
            tracker.print_stats()
        if best_c is None or res["expectancy_r"] > best_c["expectancy_r"]:
            best_c = res

    return {
        "combo":          combo_name,
        "mode_A":         res_a,
        "best_B":         best_b,
        "best_C":         best_c,
        "all_modes":      results_by_mode,
    }


# ==============================================================================
#  SECTION 11 — COMPARISON PRINTER
# ==============================================================================

def print_mode_comparison(all_combo_results: list) -> None:
    sep = "=" * 118
    logger.info(f"\n{sep}")
    logger.info("  EXIT MODE COMPARISON — ALL COMBOS")
    logger.info(f"  Mode A : BE at 1R then ATR trailing stop  (current)")
    logger.info(f"  Mode B : Fixed SL, best RR target from {RR_TARGETS}  (normal SL)")
    logger.info(f"  Mode C : Fixed SL ×{WIDE_SL_MULT}, best RR target from {RR_TARGETS}  (wide SL)")
    logger.info(sep)

    hdr = (f"  {'Combo':<22}  {'Mode':<14}  {'Trd':>5}  {'WR':>6}  "
           f"{'AvgR':>8}  {'PF':>6}  {'MDD%':>6}  {'DayLoss%':>9}  "
           f"{'Ret%':>7}  {'FinalBal':>12}")
    logger.info(hdr)
    logger.info(f"  {'-'*114}")

    def _row(combo_name, mode_res, marker=""):
        mode  = mode_res["mode"]
        trades = mode_res["trades"]
        wr    = mode_res["win_rate"]
        ex    = mode_res["expectancy_r"]
        pf    = mode_res["profit_factor"]
        mdd   = mode_res["mdd_pct"]
        dl    = mode_res["max_day_loss_pct"]
        ret   = mode_res["return_pct"]
        bal   = mode_res["final_balance"]
        mdd_flag = " ✓" if mdd < 10.0 else "  "
        logger.info(
            f"  {combo_name:<22}  {mode:<14}  {trades:>5}  {wr:>6.1%}  "
            f"{ex:>+8.4f}  {pf:>6.3f}  {mdd:>5.2f}%  {dl:>8.2f}%  "
            f"{ret:>+6.1f}%  ${bal:>11,.2f}{mdd_flag}{marker}"
        )

    for cr in all_combo_results:
        combo_name = cr["combo"]
        _row(combo_name, cr["mode_A"])
        _row(combo_name, cr["best_B"], marker="  ← best B")
        _row(combo_name, cr["best_C"], marker="  ← best C")
        logger.info(f"  {'·'*114}")

    # ── Winner per combo ──────────────────────────────────────────────────
    logger.info(f"\n{sep}")
    logger.info("  WINNER PER COMBO  (by expectancy R, then MDD tiebreak)")
    logger.info(sep)
    logger.info(f"  {'Combo':<22}  {'Winner':<14}  {'AvgR':>8}  "
                f"{'WR':>6}  {'MDD%':>6}  {'Ret%':>7}  {'FinalBal':>12}")
    logger.info(f"  {'-'*80}")

    for cr in all_combo_results:
        candidates = [cr["mode_A"], cr["best_B"], cr["best_C"]]
        winner = max(candidates, key=lambda x: (x["expectancy_r"], -x["mdd_pct"]))
        logger.info(
            f"  {cr['combo']:<22}  {winner['mode']:<14}  "
            f"{winner['expectancy_r']:>+8.4f}  {winner['win_rate']:>6.1%}  "
            f"{winner['mdd_pct']:>5.2f}%  {winner['return_pct']:>+6.1f}%  "
            f"${winner['final_balance']:>11,.2f}"
        )

    # ── Full RR breakdown per combo ───────────────────────────────────────
    logger.info(f"\n{sep}")
    logger.info("  RR LADDER DETAIL — EXPECTANCY R BY COMBO × MODE")
    logger.info(sep)
    header = f"  {'Combo':<22}  {'Mode A':>10}"
    for rr in RR_TARGETS:
        header += f"  {'B:RR'+str(rr):>10}"
    for rr in RR_TARGETS:
        header += f"  {'C:RR'+str(rr):>11}"
    logger.info(header)
    logger.info(f"  {'-'*116}")

    for cr in all_combo_results:
        am    = cr["all_modes"]
        row   = f"  {cr['combo']:<22}  {am['A:BE+TRAIL']['expectancy_r']:>+10.4f}"
        for rr in RR_TARGETS:
            row += f"  {am[f'B:RR{rr}']['expectancy_r']:>+10.4f}"
        for rr in RR_TARGETS:
            row += f"  {am[f'C:WIDE_RR{rr}']['expectancy_r']:>+11.4f}"
        logger.info(row)

    logger.info(sep)


# ==============================================================================
#  SECTION 12 — FTMO EVAL SIMULATOR
# ==============================================================================

EVAL_P1_TARGET    = 0.10
EVAL_P2_TARGET    = 0.05
EVAL_DAILY_LIMIT  = 0.0475
EVAL_TOTAL_DD_LIM = 0.10
EVAL_MIN_DAYS     = 4


def run_eval_simulation(combo_name: str, symbols: list, events: list,
                         caches: dict, tick_values: dict,
                         resolver_fn=None) -> list:
    """
    resolver_fn(cache, si) -> (outcome_r, exit_bar, ep, sl_dist).
    Defaults to resolve_mode_a (BE+trail) if not supplied.

    FIX: same chronological entry/exit issue as _run_simulation_core (see
    comment above that function). `events` here is entry-ordered only; we
    pre-resolve every trade and replay a merged entry/exit timeline so lot
    sizing at ENTRY only ever reflects balance that has actually been
    realized at EXIT, never PnL from trades still open in real time.
    `open_state` persists across phases/cycles so a trade opened near a
    phase boundary still resolves its lot correctly when it exits later.
    """
    if resolver_fn is None:
        resolver_fn = resolve_mode_a

    max_trades_per_day = sum(
        PARAMS_GRID_BEST[s]["max_trades_day"] for s in symbols
    )

    trades = _resolve_all_trades(symbols, caches, resolver_fn)
    chrono = _build_chrono_events(trades)
    open_state: dict = {}   # idx -> {"lot": float, "rejected": bool}  (persists across phases)

    def run_phase(target_pct, event_idx):
        nonlocal balance
        phase_start_bal = balance
        phase_target    = phase_start_bal * (1 + target_pct)
        dd_floor        = phase_start_bal * (1 - EVAL_TOTAL_DD_LIM)
        day_pnl: dict   = {}
        trading_days    = set()
        trade_count     = 0
        start_date      = None
        end_date        = None

        while event_idx[0] < len(chrono):
            t_time, etype, idx = chrono[event_idx[0]]
            event_idx[0] += 1
            t     = trades[idx]
            sym   = t["sym"]
            cache = t["cache"]
            tvpl  = tick_values[sym]
            sl_dist = t["sl_dist"]

            if etype == 1:  # ENTRY — size using balance realized so far
                vol_max_cap = compute_vol_max_cap(sl_dist, tvpl, max_trades_per_day)
                lot, _, _, _, rejected = compute_lot_aware(
                    balance, sl_dist, tvpl, vol_max_cap)
                open_state[idx] = {"lot": lot, "rejected": rejected}
                if start_date is None:
                    start_date = cache["dates"][t["si"] + 1]
                continue

            # etype == 0: EXIT — realize PnL now
            st = open_state.get(idx)
            if st is None or st["rejected"]:
                continue
            lot = st["lot"]
            outcome_r = t["outcome_r"]

            pnl        = outcome_r * lot * sl_dist * tvpl
            trade_date = cache["dates"][t["si"] + 1]
            end_date   = trade_date

            trading_days.add(trade_date)
            day_pnl[trade_date] = day_pnl.get(trade_date, 0.0) + pnl
            balance    += pnl
            trade_count += 1

            if day_pnl[trade_date] < -(phase_start_bal * EVAL_DAILY_LIMIT):
                return (False, len(trading_days), trade_count,
                        start_date, end_date, "DAILY_BREACH")

            if balance < dd_floor:
                return (False, len(trading_days), trade_count,
                        start_date, end_date, "DD_BREACH")

            if balance >= phase_target and len(trading_days) >= EVAL_MIN_DAYS:
                return (True, len(trading_days), trade_count,
                        start_date, end_date, "PASSED")

        return (False, len(trading_days), trade_count,
                start_date, end_date, "DATA_END")

    balance   = STARTING_BALANCE
    event_idx = [0]
    cycles    = []

    while event_idx[0] < len(chrono):
        balance = STARTING_BALANCE

        p1_pass, p1_days, p1_trades, p1_start, p1_end, p1_reason = \
            run_phase(EVAL_P1_TARGET, event_idx)

        if not p1_pass:
            cycles.append({
                "attempt":    len(cycles) + 1,
                "result":     "FAIL",
                "failed_at":  "P1",
                "reason":     p1_reason,
                "p1_days":    p1_days,
                "p1_trades":  p1_trades,
                "p2_days":    0,
                "p2_trades":  0,
                "total_days": p1_days,
                "start_date": str(p1_start),
                "end_date":   str(p1_end),
            })
            continue

        p2_pass, p2_days, p2_trades, p2_start, p2_end, p2_reason = \
            run_phase(EVAL_P2_TARGET, event_idx)

        cycles.append({
            "attempt":    len(cycles) + 1,
            "result":     "PASS" if p2_pass else "FAIL",
            "failed_at":  "—"    if p2_pass else "P2",
            "reason":     p2_reason,
            "p1_days":    p1_days,
            "p1_trades":  p1_trades,
            "p2_days":    p2_days,
            "p2_trades":  p2_trades,
            "total_days": p1_days + p2_days,
            "start_date": str(p1_start),
            "end_date":   str(p2_end) if p2_end else str(p1_end),
        })

    return cycles


def print_eval_results(combo_name: str, cycles: list,
                        mode_label: str = "A:BE+TRAIL") -> None:
    if not cycles:
        logger.info(f"  [{combo_name} | {mode_label}] No eval cycles completed.")
        return

    total     = len(cycles)
    passes    = sum(1 for c in cycles if c["result"] == "PASS")
    fails     = total - passes
    pass_rate = passes / total * 100

    p1_fails = sum(1 for c in cycles if c["failed_at"] == "P1")
    p2_fails = sum(1 for c in cycles if c["failed_at"] == "P2")
    daily_br = sum(1 for c in cycles if c["reason"] == "DAILY_BREACH")
    dd_br    = sum(1 for c in cycles if c["reason"] == "DD_BREACH")

    pass_days = [c["total_days"] for c in cycles if c["result"] == "PASS"]
    avg_days  = np.mean(pass_days) if pass_days else 0

    sep = "=" * 90
    logger.info(f"\n{sep}")
    logger.info(f"  FTMO EVAL SIM — {combo_name}  [{mode_label}]")
    logger.info(f"  Total attempts : {total}")
    logger.info(f"  Passed         : {passes}  ({pass_rate:.1f}%)")
    logger.info(f"  Failed         : {fails}  "
                f"(P1={p1_fails}  P2={p2_fails}  "
                f"DailyBreach={daily_br}  DDBreach={dd_br})")
    logger.info(f"  Avg days to pass (passes only): {avg_days:.1f} trading days")
    logger.info(f"\n  {'#':>4}  {'Result':<6}  {'FailAt':<6}  {'Reason':<14}"
                f"  {'P1d':>4}  {'P1t':>4}  {'P2d':>4}  {'P2t':>4}"
                f"  {'TotDays':>7}  Start        End")
    logger.info(f"  {'-'*88}")
    for c in cycles:
        logger.info(
            f"  {c['attempt']:>4}  {c['result']:<6}  {c['failed_at']:<6}  "
            f"{c['reason']:<14}  {c['p1_days']:>4}  {c['p1_trades']:>4}  "
            f"{c['p2_days']:>4}  {c['p2_trades']:>4}  "
            f"{c['total_days']:>7}  {c['start_date']}  {c['end_date']}"
        )
    logger.info(sep)


# ==============================================================================
#  SECTION 13 — MAIN
# ==============================================================================

_GLOBAL_TRADE_LOG: list = []


def main():
    if MT5_AVAILABLE:
        if not mt5.initialize(path=TERMINAL_PATH, login=LOGIN,
                              password=PASSWORD, server=SERVER):
            raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")
        logger.info("MT5 connected")
    else:
        logger.warning("MT5 not available — using fallback tick values")

    logger.info("\n=== TICK VALUES ===")
    tick_values = fetch_tick_values()
    logger.info("=== END TICK VALUES ===\n")

    if not MT5_AVAILABLE:
        logger.error("MT5 required for signal-window data. Exiting.")
        return

    logger.info(f"\n{'='*80}")
    logger.info(f"  DATA ASSEMBLY")
    logger.info(f"  CSV dir   : {CSV_DIR}  (ATR warmup only)")
    logger.info(f"  MT5 fetch : up to {FETCH_BARS_M5:,} bars")
    logger.info(f"{'='*80}\n")

    all_caches: dict = {}
    for sym in SYMBOLS:
        logger.info(f"[{sym}] assembling...")
        df_all, mt5_start = assemble_data(sym)
        if df_all is None:
            logger.warning(f"  [{sym}] skipped — no data")
            continue
        cache = build_cache_and_signals(sym, df_all, mt5_start,
                                         PARAMS_GRID_BEST[sym])
        all_caches[sym] = cache

    mt5.shutdown()

    if not all_caches:
        logger.error("No symbol data. Exiting.")
        return

    logger.info(f"\n{'='*80}")
    logger.info(f"  EXIT MODE SWEEP — {len(SYMBOL_COMBOS)} combos × "
                f"{1 + 2*len(RR_TARGETS)} modes each")
    logger.info(f"  Modes: A (current BE+trail)  |  "
                f"B (fixed SL, RR {RR_TARGETS})  |  "
                f"C (SL×{WIDE_SL_MULT}, RR {RR_TARGETS})")
    logger.info(f"{'='*80}\n")

    all_combo_results = []
    flat_rows         = []

    for combo in SYMBOL_COMBOS:
        if not all(s in all_caches for s in combo):
            continue
        combo_name = "+".join(combo)
        logger.info(f"\n{'─'*60}")
        logger.info(f"  COMBO: {combo_name}")
        logger.info(f"{'─'*60}")

        cr = run_combo_all_modes(
            combo_name, list(combo), all_caches, tick_values,
            print_detailed_stats=True,   # set False to suppress per-mode stat blocks
        )
        all_combo_results.append(cr)

        for mode_label, res in cr["all_modes"].items():
            flat_rows.append(res)

    # ── Comparison table ──────────────────────────────────────────────────
    print_mode_comparison(all_combo_results)

    # ── CSV outputs ───────────────────────────────────────────────────────
    pd.DataFrame(flat_rows).to_csv("orb_combo_sweep.csv", index=False)
    logger.info("\n  orb_combo_sweep.csv written")

    # ── FTMO eval — all combos × all exit modes ───────────────────────────
    eval_combos = [cr["combo"].split("+") for cr in all_combo_results]

    # Build the full resolver map: label -> fn
    eval_modes: dict = {"A:BE+TRAIL": resolve_mode_a}
    for rr in RR_TARGETS:
        eval_modes[f"B:RR{rr}"]      = (lambda _rr=rr:
            lambda cache, si: resolve_mode_b(cache, si, _rr, sl_multiplier=1.0))()
        eval_modes[f"C:WIDE_RR{rr}"] = (lambda _rr=rr:
            lambda cache, si: resolve_mode_b(cache, si, _rr, sl_multiplier=WIDE_SL_MULT))()

    logger.info(f"\n{'='*90}")
    logger.info(
        f"  FTMO EVAL SIMULATION — {len(eval_combos)} combo(s) × "
        f"{len(eval_modes)} modes"
    )
    logger.info(
        f"  Modes: {', '.join(eval_modes.keys())}"
    )
    logger.info(
        f"  P1 target=+10%  P2 target=+5%  "
        f"DailyLimit={EVAL_DAILY_LIMIT:.2%}  TotalDD={EVAL_TOTAL_DD_LIM:.0%}  "
        f"MinDays={EVAL_MIN_DAYS}"
    )
    logger.info(f"{'='*90}")

    all_eval_cycles = []
    # eval_summary[combo_name][mode_label] = pass_rate
    eval_summary: dict = {}

    for symbols in eval_combos:
        combo_name = "+".join(symbols)
        missing    = [s for s in symbols if s not in all_caches]
        if missing:
            logger.warning(f"  [{combo_name}] missing {missing} — skipping")
            continue

        caches = {sym: all_caches[sym] for sym in symbols}
        events = build_master_timeline(caches)
        logger.info(f"\n  [{combo_name}] {len(events):,} signal events")

        eval_summary[combo_name] = {}

        for mode_label, resolver_fn in eval_modes.items():
            cycles = run_eval_simulation(
                combo_name, symbols, events, caches, tick_values,
                resolver_fn=resolver_fn,
            )
            print_eval_results(combo_name, cycles, mode_label=mode_label)

            total  = len(cycles)
            passes = sum(1 for c in cycles if c["result"] == "PASS")
            eval_summary[combo_name][mode_label] = (
                round(passes / total * 100, 1) if total else 0.0
            )

            for c in cycles:
                c["combo"] = combo_name
                c["mode"]  = mode_label
            all_eval_cycles.extend(cycles)

    pd.DataFrame(all_eval_cycles).to_csv("orb_eval_cycles.csv", index=False)
    logger.info("\n  orb_eval_cycles.csv written")

    # ── FTMO eval pass-rate summary table ─────────────────────────────────
    mode_labels = list(eval_modes.keys())
    sep = "=" * (24 + 9 * len(mode_labels))
    logger.info(f"\n{sep}")
    logger.info("  FTMO EVAL PASS RATE SUMMARY  (%)")
    logger.info(sep)
    hdr = f"  {'Combo':<22}" + "".join(f"  {m:>10}" for m in mode_labels)
    logger.info(hdr)
    logger.info(f"  {'-'*(22 + 12*len(mode_labels))}")
    for combo_name, mode_rates in eval_summary.items():
        row = f"  {combo_name:<22}"
        best_rate = max(mode_rates.values()) if mode_rates else 0
        for m in mode_labels:
            rate = mode_rates.get(m, 0.0)
            flag = " ←" if rate == best_rate and best_rate > 0 else "  "
            row += f"  {rate:>8.1f}%{flag[0]}"
        logger.info(row)
    logger.info(sep)
    logger.info("  ← marks the highest pass rate per combo")

    logger.info(
        "\nOutputs saved:\n"
        "  orb_combo_sweep.csv\n"
        "  orb_eval_cycles.csv\n"
        "  orb_chrono_compare.log\n"
        "Done."
    )


if __name__ == "__main__":
    main()
