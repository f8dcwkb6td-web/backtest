"""
==============================================================================
ORB  —  CHRONOLOGICAL MULTI-SYMBOL BACKTEST  v3
         LEAN LOGGING  +  EXIT-MODE COMPARISON
==============================================================================

EXIT MODES (compared per combo, no re-gridding):
  A  CURRENT   — BE at 1R then ATR trailing stop (original logic)
  A_LONG       — Same as A, but LONG signals only (shorts ignored)
  B  FIXED_RR  — Fixed SL (same sl_dist), exit at first RR target hit
                 Tested at RR 1, 2, 3, 4, 5. Best by expectancy reported.
  C  WIDE_RR   — SL doubled (2× sl_dist), same RR ladder.
                 Rationale: wider SL → fewer stop-outs, larger targets.

Logging changes vs v2:
  • Per-trade inline log REMOVED (verbose noise)
  • Trade-by-trade header block REMOVED
  • Per-mode detailed R-stat dumps REMOVED (was 60+ lines x 7 combos x 12
    modes ≈ thousands of lines)
  • Per-FTMO-cycle attempt logging REMOVED
  • Replaced with 6 consolidated summary tables printed once at the end
==============================================================================
"""

import os, sys, io, logging, bisect, datetime, traceback
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

# Round-turn commission in USD per 1.00 lot. The backtest previously modeled
# ZERO transaction cost beyond spread. If live results run cooler than the
# backtest (e.g. backtest +19% vs live ~breakeven for the same month), an
# unmodeled commission is one of the most common causes — it doesn't show up
# in win rate or R-multiples at all, it just quietly eats the edge trade by
# trade. Set this to your broker's actual round-turn commission per lot for
# each symbol (defaults to 0.0 = unchanged/backtest-optimistic behavior).
COMMISSION_PER_LOT = {
    "US30":  0.0,
    "US500": 0.0,
    "UK100": 0.0,
    "GER40": 0.0,
}

TICK_VALUE_FALLBACK = {
    "US30":  1.0,
    "US500": 1.0,
    "UK100": 1.0,
    "GER40": 1.0,
}

# Fallbacks used only when MT5 / symbol_info is unavailable at fetch time.
MIN_SL_DISTANCE_FALLBACK_PTS = 5.0     # broker "stops level" fallback
CONTRACT_SIZE_FALLBACK       = 1.0
ACCOUNT_LEVERAGE_FALLBACK    = 100.0

# Extra ADVERSE slippage (price units, same scale as spread) applied on top
# of quoted spread, modeling real market/stop-order execution slippage that
# a bar-level backtest can never see directly — this is what MT5's
# `deviation=20` tolerance in the live engine's send_market_order() actually
# allows to happen, and what its retcode-10016 retry path (price moved past
# SL between signal and send) is evidence of happening in practice. Sweep
# this per symbol to see how much unmodeled slippage would be needed to
# explain a backtest-vs-live gap, given real trade counts you can read off
# Table 1's "Trades" column after a run. 0.0 = off (original behavior).
SLIPPAGE_PRICE_PER_SYMBOL = {
    "US30":  0.0,
    "US500": 0.0,
    "UK100": 0.0,
    "GER40": 0.0,
}


def _apply_entry_slippage(direction: int, price: float, slip: float) -> float:
    """Entering LONG = buying -> worse fill = higher price.
    Entering SHORT = selling -> worse fill = lower price."""
    return price + slip if direction == 1 else price - slip


def _apply_exit_slippage(direction: int, price: float, slip: float) -> float:
    """Exiting a LONG = selling -> worse fill = lower price.
    Exiting a SHORT = buying -> worse fill = higher price."""
    return price - slip if direction == 1 else price + slip

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

# ── Trade-start cutoff ───────────────────────────────────────────────────────
TRADE_START_DATE = "2026-06-04"

# ── Live-engine identity, for the live-vs-backtest diff (SECTION 12B) ───────
# Must match orb_live_v6.py exactly, or the diff will find zero deals.
LIVE_MAGIC   = 202603264
LIVE_COMMENT_SUBSTR = "ORB_V6"

# Toggle the live-vs-backtest trade diff on/off. Requires MT5 connected to
# the SAME account the live engine actually trades on (see the account-
# identity check printed during MT5 connect in main()).
RUN_LIVE_VS_BACKTEST_DIFF = True
LIVE_DIFF_DATE_FROM = TRADE_START_DATE   # None = same as TRADE_START_DATE


# ==============================================================================
#  SECTION 0 — R STATS (lean — collects trades, no verbose per-mode dumps)
# ==============================================================================

class RTracker:
    def __init__(self, combo_name: str, mode_label: str,
                 rolling_days: int = ROLLING_R_DAYS):
        self.combo        = combo_name
        self.mode         = mode_label
        self.rolling_days = rolling_days
        self.trades: list[dict] = []

    def record(self, *, sym: str, trade_date, entry_hour: int,
               direction: int, outcome_r: float, pnl: float,
               lot: float, balance_after: float):
        self.trades.append({
            "combo":          self.combo,
            "mode":           self.mode,
            "sym":            sym,
            "trade_date":     str(trade_date),
            "entry_hour_utc": entry_hour,
            "direction":      direction,
            "outcome_r":      round(outcome_r, 4),
            "pnl":            round(pnl, 2),
            "lot":            lot,
            "balance_after":  round(balance_after, 2),
            "result":         "W" if outcome_r > 0 else "L",
        })


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


def _calibrate_effective_leverage(broker_sym: str, contract_size: float):
    """
    account_info().leverage is a nominal/max account figure. Many brokers
    (this one included, per the user: 1:20 for indices, 1:200 for FX on the
    SAME account) apply a lower effective leverage per instrument class on
    top of it. Using the flat account leverage for margin math on an index
    understates required margin — exactly the bug that made the backtest's
    margin-downsize never fire while the live engine's real margin math did.

    Calibrate the REAL effective leverage the same way live effectively
    experiences it: call order_calc_margin() (the same function
    execute_entry() calls on every live order) once at the current price for
    a reference 1.0 lot, then back out the leverage implied by that margin.
    This is assumed stable over the backtest window (leverage tiers don't
    change bar to bar).
    """
    tick = mt5.symbol_info_tick(broker_sym)
    if tick is None or tick.ask <= 0:
        return None
    price = tick.ask
    test_lot = 1.0
    margin = mt5.order_calc_margin(mt5.ORDER_TYPE_BUY, broker_sym, test_lot, price)
    if margin is None or margin <= 0:
        return None
    return (test_lot * contract_size * price) / margin


def fetch_broker_info() -> tuple:
    """
    Mirrors the live engine's `_tick_info` build-up in run_live() plus its
    account leverage, so the backtest can apply the SAME broker-minimum-stop
    clamp (clamp_sl / get_min_sl_distance) and the SAME margin-based lot
    downsize (execute_entry's MD-1 patch) that the live engine actually
    enforces. Without this, the backtest silently assumes zero stop-distance
    floor and infinite margin — both optimistic relative to live.

    Leverage is calibrated PER SYMBOL via order_calc_margin (see
    _calibrate_effective_leverage) rather than using the flat account-wide
    leverage, since instrument classes (indices vs FX) can have different
    effective leverage on the same account.
    """
    info_out: dict = {}
    account_leverage = ACCOUNT_LEVERAGE_FALLBACK

    if MT5_AVAILABLE:
        acct = mt5.account_info()
        if acct is not None and acct.leverage:
            account_leverage = float(acct.leverage)

    for canon in SYMBOLS:
        entry = {
            "tvpl":            TICK_VALUE_FALLBACK.get(canon, 1.0),
            "min_sl_distance": 0.0,
            "contract_size":   CONTRACT_SIZE_FALLBACK,
            "leverage":        account_leverage,   # placeholder until calibrated below
        }
        if MT5_AVAILABLE:
            broker = resolve_symbol(canon)
            if broker is None:
                logger.warning(f"  [{canon}] not found — using fallback broker info")
            else:
                sinfo = mt5.symbol_info(broker)
                if sinfo is None or sinfo.trade_tick_size <= 0:
                    logger.warning(f"  [{canon}] symbol_info invalid — using fallback")
                else:
                    entry["tvpl"] = sinfo.trade_tick_value / sinfo.trade_tick_size
                    point   = sinfo.point if sinfo.point > 0 else 0.0001
                    sl_lvl  = int(sinfo.trade_stops_level or 0)
                    fallback_dist = MIN_SL_DISTANCE_FALLBACK_PTS * point
                    entry["min_sl_distance"] = (max(sl_lvl * point, fallback_dist)
                                                 if sl_lvl > 0 else fallback_dist)
                    entry["contract_size"] = (sinfo.trade_contract_size
                                               if sinfo.trade_contract_size > 0
                                               else CONTRACT_SIZE_FALLBACK)

                    calibrated = _calibrate_effective_leverage(broker, entry["contract_size"])
                    if calibrated is not None and calibrated > 0:
                        entry["leverage"] = calibrated
                        if abs(calibrated - account_leverage) > 1.0:
                            logger.warning(
                                f"  [{canon}] EFFECTIVE leverage 1:{calibrated:.1f} "
                                f"differs from account leverage 1:{account_leverage:.0f} "
                                f"— using the calibrated per-symbol value for margin math"
                            )
                    else:
                        logger.warning(f"  [{canon}] leverage calibration failed "
                                       f"(no tick/margin calc) — using account "
                                       f"leverage 1:{account_leverage:.0f} as fallback")
        info_out[canon] = entry
        logger.info(f"  [{canon}] tvpl={entry['tvpl']:.6f}  "
                    f"min_sl_dist={entry['min_sl_distance']:.5f}  "
                    f"contract_size={entry['contract_size']:.2f}  "
                    f"leverage=1:{entry['leverage']:.1f}")

    logger.info(f"  Account-wide leverage (nominal, NOT used for margin math): "
                f"1:{account_leverage:.0f}")
    return info_out, account_leverage


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
                             mt5_start: pd.Timestamp, params: dict,
                             min_sl_distance: float = 0.0,
                             slippage: float = 0.0) -> dict:
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

    if TRADE_START_DATE is not None:
        trade_start_ts = np.datetime64(pd.Timestamp(TRADE_START_DATE))
        in_trade_window = df_all["time_utc"].values >= trade_start_ts
    else:
        in_trade_window = np.ones(n, dtype=bool)

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
            & in_signal_window
            & in_trade_window)
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
    trade_window_note = (
        f"  trade-start={TRADE_START_DATE}"
        if TRADE_START_DATE is not None else ""
    )
    logger.info(
        f"  [{canonical}] signals={len(signal_bars):,}  "
        f"ATR-prefix={n_csv_pre:,}  MT5-window={n_mt5:,}  "
        f"n_weeks={n_mt5 / (12 * 24 * 5):.1f}{trade_window_note}"
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
        "min_sl_distance":  min_sl_distance,
        "slippage":         slippage,
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
    return events


# ==============================================================================
#  SECTION 8 — TRADE RESOLVERS
# ==============================================================================

def _entry_params(cache: dict, si: int):
    """Shared entry price, sl_dist, direction for all modes.

    SL CLAMP (matches live's clamp_sl / get_min_sl_distance): the live
    engine will never place a stop closer to price than the broker's
    `trade_stops_level`; if the strategy's computed sl_dist is tighter than
    that, live widens it before sending the order. The backtest previously
    had no equivalent, so it could implicitly assume tighter stops (and
    therefore a different realized R-multiple, and different BE/trail
    timing) than what actually executes live. `cache["min_sl_distance"]`
    is populated at build time from fetch_broker_info().
    """
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
    ep        = _apply_entry_slippage(direction, ep, cache.get("slippage", 0.0))
    atr       = atr14[si]

    if np.isnan(atr) or atr <= 0:
        return None

    or_size = cache["or_high"][si] - cache["or_low"][si]
    if np.isnan(or_size) or or_size <= 0:
        return None

    sl_mult = params["sl_range_mult"]
    sl_dist = max(sl_mult * or_size, atr * 0.05)

    min_sl_distance = cache.get("min_sl_distance", 0.0)
    if sl_dist < min_sl_distance:
        sl_dist = min_sl_distance   # widen to broker's minimum, same as live's clamp_sl

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
    slip       = cache.get("slippage", 0.0)

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
            exit_price = _apply_exit_slippage(direction, exit_price, slip)
            outcome_r  = direction * (exit_price - ep) / sl_dist
            exit_bar   = bi
            closed     = True
            break

        bh, bl = h[bi], l[bi]
        sp_bi  = spread[bi]

        if direction == 1 and bl <= cur_sl:
            fill_px   = _apply_exit_slippage(1, cur_sl, slip)
            outcome_r = (fill_px - ep) / sl_dist
            exit_bar  = bi; closed = True; break
        if direction == -1 and (bh + sp_bi) >= cur_sl:
            fill_px   = _apply_exit_slippage(-1, cur_sl, slip)
            outcome_r = (ep - fill_px) / sl_dist
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
        exit_price = _apply_exit_slippage(direction, exit_price, slip)
        outcome_r  = direction * (exit_price - ep) / sl_dist

    return outcome_r, exit_bar, ep, sl_dist


# ── MODE B: Fixed SL, fixed RR target ─────────────────────────────────────

def resolve_mode_b(cache: dict, si: int, rr_target: float,
                   sl_multiplier: float = 1.0):
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
    slip      = cache.get("slippage", 0.0)
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
            exit_price = _apply_exit_slippage(direction, exit_price, slip)
            outcome_r  = direction * (exit_price - ep) / sl_dist
            exit_bar   = bi; closed = True; break

        bh, bl = h[bi], l[bi]
        sp_bi  = spread[bi]

        if direction == 1  and bh >= tp:
            fill_px   = _apply_exit_slippage(1, tp, slip)
            outcome_r = (fill_px - ep) / sl_dist
            exit_bar  = bi; closed = True; break
        if direction == -1 and (bl + sp_bi) <= tp:
            fill_px   = _apply_exit_slippage(-1, tp, slip)
            outcome_r = (ep - fill_px) / sl_dist
            exit_bar  = bi; closed = True; break

        if direction == 1  and bl <= cur_sl:
            fill_px   = _apply_exit_slippage(1, cur_sl, slip)
            outcome_r = (fill_px - ep) / sl_dist
            exit_bar  = bi; closed = True; break
        if direction == -1 and (bh + sp_bi) >= cur_sl:
            fill_px   = _apply_exit_slippage(-1, cur_sl, slip)
            outcome_r = (ep - fill_px) / sl_dist
            exit_bar  = bi; closed = True; break

    if not closed:
        exit_bar   = min(ei + MAX_HOLD, n - 1)
        exit_price = (c[exit_bar] if direction == 1
                      else c[exit_bar] + spread[exit_bar])
        exit_price = _apply_exit_slippage(direction, exit_price, slip)
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


def _resolve_all_trades(symbols: list, caches: dict, resolver_fn,
                         direction_filter: int = None) -> list:
    trades = []
    for sym in symbols:
        cache = caches[sym]
        for si in cache["signal_bars"]:
            if direction_filter is not None and int(cache["signal"][si]) != direction_filter:
                continue
            outcome_r, exit_bar, ep, sl_dist = resolver_fn(cache, si)
            if outcome_r is None:
                continue
            entry_time = cache["times"][si] + _BAR_DURATION
            exit_time  = cache["times"][exit_bar] + _BAR_DURATION
            trades.append({
                "sym": sym, "si": si, "cache": cache,
                "entry_time": entry_time, "exit_time": exit_time,
                "outcome_r": outcome_r, "sl_dist": sl_dist, "ep": ep,
            })
    trades.sort(key=lambda t: t["entry_time"])
    return trades


def _build_chrono_events(trades: list) -> list:
    events = []
    for idx, t in enumerate(trades):
        events.append((t["entry_time"], 1, idx))  # 1 = ENTRY
        events.append((t["exit_time"],  0, idx))  # 0 = EXIT  (sorts first on tie)
    events.sort(key=lambda e: (e[0], e[1]))
    return events


def _check_margin_and_downsize(lot: float, sl_dist: float, ep: float,
                                contract_size: float, leverage: float,
                                free_margin: float, vol_max_cap: float):
    """
    Mirrors execute_entry()'s MD-1 margin-downsize patch from the live
    engine exactly:
      1. required_margin = lot * contract_size * price / leverage
      2. if free_margin < required_margin * 1.10: try to downsize to the
         largest lot that fits in 90% of free_margin.
      3. reject if the downsized lot is still < VOL_MIN or still doesn't fit.
    Returns (final_lot_or_None, required_margin, was_downsized: bool).
    """
    if leverage <= 0 or contract_size <= 0 or ep <= 0:
        return lot, 0.0, False

    required_margin = lot * contract_size * ep / leverage
    required_with_buffer = required_margin * 1.10

    if free_margin >= required_with_buffer:
        return lot, required_margin, False

    # MD-1: attempt downsize
    available_margin_with_buffer = free_margin * 0.90
    if required_margin <= 0:
        return None, 0.0, False
    max_lot = lot * (available_margin_with_buffer / required_margin)
    max_lot = max(VOL_MIN, min(max_lot, vol_max_cap))
    max_lot = (int(max_lot / VOL_STEP) * VOL_STEP)
    max_lot = round(max_lot, 8)

    if max_lot < VOL_MIN:
        return None, 0.0, False

    new_required_margin = max_lot * contract_size * ep / leverage
    if free_margin < new_required_margin * 1.10:
        return None, 0.0, False

    return max_lot, new_required_margin, True


def _run_simulation_core(combo_name: str, mode_label: str,
                          symbols: list, all_caches: dict,
                          broker_info: dict,
                          resolver_fn,
                          direction_filter: int = None) -> dict:
    """
    Generic simulation loop. resolver_fn(cache, si) → (outcome_r, exit_bar, ep, sl_dist).
    Returns a results dict + populated RTracker.

    MARGIN (mirrors execute_entry's MD-1 patch): `used_margin` tracks margin
    reserved by every currently-open (entered, not yet exited) trade in the
    chronological replay. At each ENTRY, `free_margin = balance - used_margin`
    (balance approximates equity, ignoring floating P&L of open trades — the
    same simplification the rest of this backtest already makes). If margin
    is insufficient, the trade is downsized exactly like live, or rejected if
    even VOL_MIN doesn't fit. Margin is released back at EXIT.
    """
    param_set = {sym: PARAMS_GRID_BEST[sym] for sym in symbols}
    max_trades_per_day_combo = sum(
        param_set[sym]["max_trades_day"] for sym in symbols
    )

    caches = {sym: all_caches[sym] for sym in symbols}

    trades = _resolve_all_trades(symbols, caches, resolver_fn, direction_filter)
    chrono = _build_chrono_events(trades)

    balance  = STARTING_BALANCE
    peak_bal = STARTING_BALANCE
    max_dd   = 0.0
    day_pnl: dict = {}
    max_day_loss  = 0.0
    used_margin   = 0.0

    per_sym = {sym: {"r": [], "pnl": [], "rejected": 0, "margin_downsized": 0,
                      "margin_rejected": 0} for sym in symbols}
    r_tracker = RTracker(combo_name, mode_label)
    open_state: dict = {}   # idx -> {"lot", "rejected", "margin"}

    for t_time, etype, idx in chrono:
        t     = trades[idx]
        sym   = t["sym"]
        cache = t["cache"]
        tvpl  = broker_info[sym]["tvpl"]
        contract_size = broker_info[sym]["contract_size"]
        sl_dist = t["sl_dist"]
        ep      = t["ep"]

        if etype == 1:  # ENTRY
            vol_max_cap = compute_vol_max_cap(sl_dist, tvpl, max_trades_per_day_combo)
            lot, intended, actual_loss, risk_mult, rejected = \
                compute_lot_aware(balance, sl_dist, tvpl, vol_max_cap)

            margin_used_now = 0.0
            downsized = False
            if not rejected:
                free_margin = balance - used_margin
                sym_leverage = broker_info[sym]["leverage"]
                new_lot, margin_used_now, downsized = _check_margin_and_downsize(
                    lot, sl_dist, ep, contract_size, sym_leverage,
                    free_margin, vol_max_cap
                )
                if new_lot is None:
                    rejected = True
                    per_sym[sym]["margin_rejected"] += 1
                else:
                    if downsized:
                        per_sym[sym]["margin_downsized"] += 1
                    lot = new_lot

            open_state[idx] = {"lot": lot, "rejected": rejected, "margin": margin_used_now}
            if rejected:
                per_sym[sym]["rejected"] += 1
            else:
                used_margin += margin_used_now
            continue

        # etype == 0: EXIT
        st = open_state.get(idx)
        if st is None or st["rejected"]:
            continue
        lot = st["lot"]
        used_margin -= st["margin"]
        outcome_r = t["outcome_r"]

        pnl  = outcome_r * lot * sl_dist * tvpl
        pnl -= lot * COMMISSION_PER_LOT.get(sym, 0.0)
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
    total_margin_rej = sum(per_sym[s]["margin_rejected"] for s in symbols)
    total_margin_ds  = sum(per_sym[s]["margin_downsized"] for s in symbols)
    nt      = len(all_r)
    wr      = float((all_r > 0).sum() / nt)   if nt > 0 else 0.0
    ex      = float(all_r.mean())              if nt > 0 else 0.0
    pos     = all_r[all_r > 0]
    neg     = all_r[all_r < 0]
    pf      = float(pos.sum() / -neg.sum()) \
              if len(neg) and neg.sum() != 0 else 0.0
    ret     = (balance - STARTING_BALANCE) / STARTING_BALANCE

    result = {
        "combo":              combo_name,
        "mode":               mode_label,
        "symbols":            "+".join(symbols),
        "trades":             nt,
        "rejected":           total_rej,
        "margin_rejected":    total_margin_rej,
        "margin_downsized":   total_margin_ds,
        "win_rate":           round(wr, 4),
        "expectancy_r":       round(ex, 4),
        "profit_factor":      round(pf, 4),
        "mdd_pct":            round(max_dd * 100, 2),
        "max_day_loss_pct":   round(max_day_loss / STARTING_BALANCE * 100, 2),
        "return_pct":         round(ret * 100, 2),
        "final_balance":      round(balance, 2),
    }
    return result, r_tracker


# ==============================================================================
#  SECTION 10 — COMBO SWEEP WITH MODE COMPARISON
# ==============================================================================

def run_combo_all_modes(combo_name: str, symbols: list,
                         all_caches: dict, broker_info: dict) -> dict:
    results_by_mode = {}
    trackers_by_mode = {}

    res_a, tracker_a = _run_simulation_core(
        combo_name, "A:BE+TRAIL", symbols, all_caches, broker_info,
        lambda cache, si: resolve_mode_a(cache, si)
    )
    results_by_mode["A:BE+TRAIL"] = res_a
    trackers_by_mode["A:BE+TRAIL"] = tracker_a

    res_a_long, tracker_a_long = _run_simulation_core(
        combo_name, "A:LONG_ONLY", symbols, all_caches, broker_info,
        lambda cache, si: resolve_mode_a(cache, si),
        direction_filter=1,
    )
    results_by_mode["A:LONG_ONLY"] = res_a_long
    trackers_by_mode["A:LONG_ONLY"] = tracker_a_long

    best_b = None
    for rr in RR_TARGETS:
        label = f"B:RR{rr}"
        res, tracker = _run_simulation_core(
            combo_name, label, symbols, all_caches, broker_info,
            lambda cache, si, _rr=rr: resolve_mode_b(cache, si, _rr,
                                                      sl_multiplier=1.0)
        )
        results_by_mode[label] = res
        trackers_by_mode[label] = tracker
        if best_b is None or res["expectancy_r"] > best_b["expectancy_r"]:
            best_b = res

    best_c = None
    for rr in RR_TARGETS:
        label = f"C:WIDE_RR{rr}"
        res, tracker = _run_simulation_core(
            combo_name, label, symbols, all_caches, broker_info,
            lambda cache, si, _rr=rr: resolve_mode_b(cache, si, _rr,
                                                      sl_multiplier=WIDE_SL_MULT)
        )
        results_by_mode[label] = res
        trackers_by_mode[label] = tracker
        if best_c is None or res["expectancy_r"] > best_c["expectancy_r"]:
            best_c = res

    return {
        "combo":          combo_name,
        "mode_A":         res_a,
        "mode_A_long":    res_a_long,
        "best_B":         best_b,
        "best_C":         best_c,
        "all_modes":      results_by_mode,
        "trackers":       trackers_by_mode,
    }


# ==============================================================================
#  SECTION 11 — CONSOLIDATED SUMMARY TABLES
# ==============================================================================

def _print_df(title: str, df: pd.DataFrame):
    sep = "=" * max(60, len(title) + 4)
    logger.info(f"\n{sep}")
    logger.info(f"  {title}")
    logger.info(sep)
    logger.info(df.to_string(index=False))
    logger.info(sep)


def build_table_exit_mode_comparison(all_combo_results: list) -> pd.DataFrame:
    rows = []
    for cr in all_combo_results:
        for label, res in [("A:BE+TRAIL", cr["mode_A"]),
                            ("A:LONG_ONLY", cr["mode_A_long"]),
                            (cr["best_B"]["mode"] + " (best)", cr["best_B"]),
                            (cr["best_C"]["mode"] + " (best)", cr["best_C"])]:
            rows.append({
                "Combo":    cr["combo"],
                "Mode":     label,
                "Trades":   res["trades"],
                "MarginDS": res["margin_downsized"],
                "MarginRej": res["margin_rejected"],
                "WinRate%": round(res["win_rate"] * 100, 1),
                "AvgR":     res["expectancy_r"],
                "PF":       res["profit_factor"],
                "MDD%":     res["mdd_pct"],
                "Return%":  res["return_pct"],
                "FinalBal": res["final_balance"],
            })
    return pd.DataFrame(rows)


def build_table_winner_per_combo(all_combo_results: list) -> pd.DataFrame:
    rows = []
    for cr in all_combo_results:
        candidates = [cr["mode_A"], cr["mode_A_long"], cr["best_B"], cr["best_C"]]
        winner = max(candidates, key=lambda x: (x["expectancy_r"], -x["mdd_pct"]))
        rows.append({
            "Combo":    cr["combo"],
            "Winner":   winner["mode"],
            "AvgR":     winner["expectancy_r"],
            "WinRate%": round(winner["win_rate"] * 100, 1),
            "MDD%":     winner["mdd_pct"],
            "Return%":  winner["return_pct"],
            "FinalBal": winner["final_balance"],
        })
    return pd.DataFrame(rows)


def build_table_rr_ladder(all_combo_results: list) -> pd.DataFrame:
    rows = []
    for cr in all_combo_results:
        am  = cr["all_modes"]
        row = {"Combo": cr["combo"], "ModeA": am["A:BE+TRAIL"]["expectancy_r"],
               "ModeA_Long": am["A:LONG_ONLY"]["expectancy_r"]}
        for rr in RR_TARGETS:
            row[f"B_RR{rr}"] = am[f"B:RR{rr}"]["expectancy_r"]
        for rr in RR_TARGETS:
            row[f"C_RR{rr}"] = am[f"C:WIDE_RR{rr}"]["expectancy_r"]
        rows.append(row)
    return pd.DataFrame(rows)


def build_table_monthly(all_combo_results: list, mode_label: str) -> pd.DataFrame:
    frames = []
    for cr in all_combo_results:
        tracker = cr["trackers"].get(mode_label)
        if tracker is None or not tracker.trades:
            continue
        df = pd.DataFrame(tracker.trades)
        df["combo"] = cr["combo"]
        frames.append(df)
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["ym"] = df["trade_date"].dt.to_period("M")

    out = (df.groupby(["combo", "ym"])
             .agg(Trades=("outcome_r", "size"),
                  WinRate=("result", lambda s: (s == "W").mean()),
                  AvgR=("outcome_r", "mean"),
                  SumR=("outcome_r", "sum"),
                  PnL=("pnl", "sum"))
             .reset_index())
    out["WinRate%"]  = (out["WinRate"] * 100).round(1)
    out["AvgR"]      = out["AvgR"].round(3)
    out["SumR"]      = out["SumR"].round(2)
    out["SumR_pct~"] = (out["SumR"] * RISK_PER_TRADE * 100).round(2)
    out["PnL"]       = out["PnL"].round(2)
    out["Month"]     = out["ym"].astype(str)
    out = out[["combo", "Month", "Trades", "WinRate%", "AvgR", "SumR",
               "SumR_pct~", "PnL"]]
    out.columns = ["Combo", "Month", "Trades", "WinRate%", "AvgR", "SumR",
                   "SumR_pct~", "PnL($)"]
    return out.sort_values(["Combo", "Month"]).reset_index(drop=True)


def build_table_ftmo_pass_rate(eval_summary: dict) -> pd.DataFrame:
    rows = []
    for combo_name, mode_rates in eval_summary.items():
        row = {"Combo": combo_name}
        row.update({m: r for m, r in mode_rates.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def print_all_summary_tables(all_combo_results: list, eval_summary: dict):
    _print_df("TABLE 1 — EXIT MODE COMPARISON (Combo x Mode)",
              build_table_exit_mode_comparison(all_combo_results))

    _print_df("TABLE 2 — WINNER PER COMBO (by expectancy R, MDD tiebreak)",
              build_table_winner_per_combo(all_combo_results))

    _print_df("TABLE 3 — RR LADDER DETAIL (expectancy R, Modes B/C)",
              build_table_rr_ladder(all_combo_results))

    monthly_a = build_table_monthly(all_combo_results, "A:BE+TRAIL")
    if not monthly_a.empty:
        _print_df("TABLE 4 — MONTHLY BREAKDOWN — Mode A (BE+TRAIL, both directions)",
                  monthly_a)

    monthly_a_long = build_table_monthly(all_combo_results, "A:LONG_ONLY")
    if not monthly_a_long.empty:
        _print_df("TABLE 5 — MONTHLY BREAKDOWN — Mode A LONG-ONLY",
                  monthly_a_long)

    if eval_summary:
        _print_df("TABLE 6 — FTMO EVAL PASS RATE % (Combo x Mode)",
                  build_table_ftmo_pass_rate(eval_summary))


# ==============================================================================
#  SECTION 12 — FTMO EVAL SIMULATOR
# ==============================================================================

EVAL_P1_TARGET    = 0.10
EVAL_P2_TARGET    = 0.05
EVAL_DAILY_LIMIT  = 0.0475
EVAL_TOTAL_DD_LIM = 0.10
EVAL_MIN_DAYS     = 4


def run_eval_simulation(symbols: list, caches: dict, broker_info: dict,
                         resolver_fn=None, direction_filter: int = None) -> list:
    if resolver_fn is None:
        resolver_fn = resolve_mode_a

    max_trades_per_day = sum(
        PARAMS_GRID_BEST[s]["max_trades_day"] for s in symbols
    )

    trades = _resolve_all_trades(symbols, caches, resolver_fn, direction_filter)
    chrono = _build_chrono_events(trades)
    open_state: dict = {}
    used_margin_holder = [0.0]   # persists across phases, mutable via closure

    def run_phase(target_pct, event_idx):
        nonlocal balance
        phase_start_bal = balance
        phase_target    = phase_start_bal * (1 + target_pct)
        dd_floor        = phase_start_bal * (1 - EVAL_TOTAL_DD_LIM)
        day_pnl: dict   = {}
        trading_days    = set()
        trade_count     = 0

        while event_idx[0] < len(chrono):
            t_time, etype, idx = chrono[event_idx[0]]
            event_idx[0] += 1
            t     = trades[idx]
            sym   = t["sym"]
            cache = t["cache"]
            tvpl  = broker_info[sym]["tvpl"]
            contract_size = broker_info[sym]["contract_size"]
            sl_dist = t["sl_dist"]
            ep      = t["ep"]

            if etype == 1:  # ENTRY
                vol_max_cap = compute_vol_max_cap(sl_dist, tvpl, max_trades_per_day)
                lot, _, _, _, rejected = compute_lot_aware(
                    balance, sl_dist, tvpl, vol_max_cap)

                margin_used_now = 0.0
                if not rejected:
                    free_margin = balance - used_margin_holder[0]
                    sym_leverage = broker_info[sym]["leverage"]
                    new_lot, margin_used_now, _ = _check_margin_and_downsize(
                        lot, sl_dist, ep, contract_size, sym_leverage,
                        free_margin, vol_max_cap
                    )
                    if new_lot is None:
                        rejected = True
                    else:
                        lot = new_lot

                open_state[idx] = {"lot": lot, "rejected": rejected, "margin": margin_used_now}
                if not rejected:
                    used_margin_holder[0] += margin_used_now
                continue

            # etype == 0: EXIT
            st = open_state.get(idx)
            if st is None or st["rejected"]:
                continue
            lot = st["lot"]
            used_margin_holder[0] -= st["margin"]
            outcome_r = t["outcome_r"]

            pnl  = outcome_r * lot * sl_dist * tvpl
            pnl -= lot * COMMISSION_PER_LOT.get(sym, 0.0)
            trade_date = cache["dates"][t["si"] + 1]

            trading_days.add(trade_date)
            day_pnl[trade_date] = day_pnl.get(trade_date, 0.0) + pnl
            balance    += pnl
            trade_count += 1

            if day_pnl[trade_date] < -(phase_start_bal * EVAL_DAILY_LIMIT):
                return False, len(trading_days), trade_count, "DAILY_BREACH"

            if balance < dd_floor:
                return False, len(trading_days), trade_count, "DD_BREACH"

            if balance >= phase_target and len(trading_days) >= EVAL_MIN_DAYS:
                return True, len(trading_days), trade_count, "PASSED"

        return False, len(trading_days), trade_count, "DATA_END"

    balance   = STARTING_BALANCE
    event_idx = [0]
    cycles    = []

    while event_idx[0] < len(chrono):
        balance = STARTING_BALANCE
        used_margin_holder[0] = 0.0

        p1_pass, p1_days, p1_trades, p1_reason = run_phase(EVAL_P1_TARGET, event_idx)

        if not p1_pass:
            cycles.append({"result": "FAIL", "failed_at": "P1", "reason": p1_reason})
            continue

        p2_pass, p2_days, p2_trades, p2_reason = run_phase(EVAL_P2_TARGET, event_idx)

        cycles.append({
            "result":    "PASS" if p2_pass else "FAIL",
            "failed_at": "—" if p2_pass else "P2",
            "reason":    p2_reason,
        })

    return cycles


# ==============================================================================
#  SECTION 12B — LIVE vs BACKTEST TRADE DIFF (merged into the same file so
#  there's no second script, no import-path fragility, no separate process —
#  this runs as part of the same backtest run, using the exact same caches,
#  resolvers, and broker_info already built above).
# ==============================================================================
#
# Pulls your REAL closed-trade history from MT5 (history_deals_get, filtered
# by LIVE_MAGIC) for the same account the live engine trades on, matches each
# real trade to the exact signal bar the backtest's own signal logic fired
# on, runs resolve_mode_a on that same signal, and prints a side-by-side
# comparison: live entry/exit vs backtest-predicted entry/exit, live R/$ vs
# backtest R/$, and the implied slippage per trade and per symbol.
#
# It also separately flags:
#   - live trades with NO matching backtest signal (a live-vs-backtest LOGIC
#     divergence, not slippage — worth investigating on its own)
#   - unmatched backtest signals are visible by comparing this section's
#     per-symbol trade count against Table 1's "Trades" column for Mode A
#     over the same window — a gap there means the backtest thinks a trade
#     should have happened that live never took (margin/risk rejection, or a
#     dropped signal).

def fetch_live_trades(date_from: pd.Timestamp, date_to: pd.Timestamp) -> pd.DataFrame:
    """Pull closed round-trip trades for LIVE_MAGIC from MT5 deal history."""
    deals = mt5.history_deals_get(date_from.to_pydatetime(), date_to.to_pydatetime())
    if deals is None or len(deals) == 0:
        logger.warning("  [live-diff] No deals returned by MT5 for this window — "
                       "check LIVE_DIFF_DATE_FROM and that MT5 is connected to "
                       "the correct account.")
        return pd.DataFrame()

    rows = []
    for d in deals:
        if d.magic != LIVE_MAGIC:
            continue
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
        logger.warning(f"  [live-diff] No deals matched LIVE_MAGIC={LIVE_MAGIC} "
                       f"in this window.")
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


def find_matching_signal(cache: dict, entry_time: pd.Timestamp, direction: int):
    """Find the signal bar (si) whose entry bar (si+1) opens at entry_time,
    rounded down to the 5-minute bar boundary, with matching direction.
    Falls back to +-1 bar tolerance for clock/latency drift."""
    entry_bar_open = entry_time.floor("5min")
    target = np.datetime64(entry_bar_open)

    for si in cache["signal_bars"]:
        if int(cache["signal"][si]) != direction:
            continue
        ei = si + 1
        if ei >= cache["n"]:
            continue
        if cache["times"][ei] == target:
            return si

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


def fetch_live_entry_sl(date_from: pd.Timestamp, date_to: pd.Timestamp) -> dict:
    """Map position_id -> the SL price actually requested on the entry order,
    so we can compute R normalized to LIVE's own real risk instead of the
    backtest's assumed sl_dist. If live's real R on a loss ever exceeds
    ~1.0 in magnitude, that means the stop didn't actually cap the loss at
    the intended risk — a genuine execution/stop-management issue, not
    slippage in the ordinary sense."""
    orders = mt5.history_orders_get(date_from.to_pydatetime(), date_to.to_pydatetime())
    sl_by_position: dict = {}
    if not orders:
        return sl_by_position
    for o in orders:
        if o.magic != LIVE_MAGIC:
            continue
        if o.position_id and o.sl and o.sl > 0 and o.position_id not in sl_by_position:
            sl_by_position[o.position_id] = o.sl   # first SL seen = the original entry SL
    return sl_by_position


def run_live_vs_backtest_diff(all_caches: dict, broker_info: dict,
                               date_from: pd.Timestamp, date_to: pd.Timestamp) -> None:
    """Runs entirely inside the same process/file as the backtest — uses the
    SAME caches and broker_info already built by main(), so there's no
    second script, no import path to get wrong, and no separate MT5 session."""
    logger.info(f"\n{'='*90}")
    logger.info(f"  LIVE vs BACKTEST TRADE DIFF")
    logger.info(f"  Window: {date_from} -> {date_to}   LIVE_MAGIC={LIVE_MAGIC}")
    logger.info(f"{'='*90}")

    live_trades = fetch_live_trades(date_from, date_to)
    if live_trades.empty:
        logger.warning("  [live-diff] Nothing to compare — no live trades found.")
        return
    logger.info(f"  [live-diff] Found {len(live_trades)} closed round-trip live trades.")

    # FIX: map EVERY alias in SYMBOL_ALIASES to its canonical symbol, not just
    # whichever single alias resolve_symbol() currently picks. A deal's
    # `symbol` field records whatever the broker used AT THE TIME OF THE
    # TRADE, which can be a different (also-valid) alias than the one
    # resolve_symbol() happens to return today — that mismatch was silently
    # dropping real trades into "unmatched" for no logic reason at all.
    canon_map = {}
    for canon in SYMBOLS:
        for alias in SYMBOL_ALIASES.get(canon, []):
            canon_map[alias] = canon
        broker = resolve_symbol(canon)   # keep too, in case broker uses an
        if broker:                       # unlisted variant not in the alias list
            canon_map[broker] = canon

    sl_by_position = fetch_live_entry_sl(date_from, date_to)
    logger.info(f"  [live-diff] Found real entry-SL on file for "
                f"{len(sl_by_position)} positions.")

    rows = []
    unmatched_live = []

    for _, lt in live_trades.iterrows():
        canon = canon_map.get(lt["symbol"])
        if canon is None or canon not in all_caches:
            unmatched_live.append(lt)
            continue
        cache = all_caches[canon]
        si = find_matching_signal(cache, lt["entry_time"], lt["direction"])
        if si is None:
            unmatched_live.append(lt)
            continue

        bt_outcome_r, bt_exit_bar, bt_ep, bt_sl_dist = resolve_mode_a(cache, si)
        if bt_outcome_r is None:
            unmatched_live.append(lt)
            continue

        tvpl = broker_info[canon]["tvpl"]
        bt_pnl   = bt_outcome_r * lt["volume"] * bt_sl_dist * tvpl
        live_pnl = lt["profit"] + lt["commission"] + lt["swap"]

        direction = lt["direction"]
        entry_slip = ((lt["entry_price"] - bt_ep) if direction == 1
                       else (bt_ep - lt["entry_price"]))

        bt_exit_price = (cache["c"][bt_exit_bar] if direction == 1
                          else cache["c"][bt_exit_bar] + cache["spread"][bt_exit_bar])
        exit_slip = ((bt_exit_price - lt["exit_price"]) if direction == 1
                      else (lt["exit_price"] - bt_exit_price))

        # Exit TIMING divergence — this is what actually distinguishes
        # "slightly worse fill" from "the live engine exited on a totally
        # different bar than the backtest's BE/trail logic would have."
        bt_exit_time = pd.Timestamp(cache["times"][bt_exit_bar]) + pd.Timedelta(minutes=5)
        exit_bars_diff = round((lt["exit_time"] - bt_exit_time).total_seconds() / 300.0)

        # Live's R normalized to LIVE's OWN actual risk (from the real SL on
        # the entry order), not the backtest's assumed sl_dist. If this ever
        # falls below -1.05 or so, the stop-loss did not actually cap the
        # loss at the intended 1R — a real stop-management problem.
        pos_id = lt.get("position_id")
        live_sl_price = sl_by_position.get(pos_id)
        live_sl_dist = (abs(lt["entry_price"] - live_sl_price)
                         if live_sl_price is not None else None)
        live_r_own = (round(live_pnl / (lt["volume"] * live_sl_dist * tvpl), 4)
                       if live_sl_dist and live_sl_dist > 0 else float("nan"))
        sl_breached = bool(live_r_own == live_r_own and live_r_own < -1.05)  # nan-safe

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
            "ExitBarsΔ":   exit_bars_diff,
            "LiveSLDist":  round(live_sl_dist, 5) if live_sl_dist is not None else None,
            "BTSLDist":    round(bt_sl_dist, 5),
            "LiveR(ownSL)": live_r_own,
            "LiveR(btSL)": round(live_pnl / (lt["volume"] * bt_sl_dist * tvpl), 4)
                            if bt_sl_dist * tvpl != 0 else float("nan"),
            "BTR":         round(bt_outcome_r, 4),
            "LivePnL":     round(live_pnl, 2),
            "BTPnL":       round(bt_pnl, 2),
            "SL_BREACHED": sl_breached,
        })

    if rows:
        df = pd.DataFrame(rows)
        _print_df("LIVE vs BACKTEST — PER-TRADE COMPARISON", df)

        agg = df.groupby("Symbol").agg(
            N=("EntrySlip", "size"),
            MeanEntrySlip=("EntrySlip", "mean"),
            MedianEntrySlip=("EntrySlip", "median"),
            MeanExitSlip=("ExitSlip~", "mean"),
            MedianExitSlip=("ExitSlip~", "median"),
            LivePnL=("LivePnL", "sum"),
            BTPnL=("BTPnL", "sum"),
        ).round(4).reset_index()
        _print_df("LIVE vs BACKTEST — AGGREGATE SLIPPAGE BY SYMBOL "
                  "(paste MeanEntrySlip/MeanExitSlip into SLIPPAGE_PRICE_PER_SYMBOL)",
                  agg)

        total_live = df["LivePnL"].sum()
        total_bt   = df["BTPnL"].sum()
        logger.info(f"\n  TOTAL over matched trades: Live=${total_live:,.2f}  "
                    f"Backtest(no-slippage)=${total_bt:,.2f}  "
                    f"Gap=${total_bt-total_live:,.2f}")

        # ── The signals that actually distinguish "slippage" from "logic
        # divergence": a stop-loss losing MORE than its own configured risk,
        # and exits landing on a materially different bar than the backtest's
        # BE/trail simulation of the SAME entry would have produced.
        breached = df[df["SL_BREACHED"] == True]
        if len(breached):
            logger.warning(f"\n  *** {len(breached)} TRADE(S) LOST MORE THAN 1R AGAINST "
                           f"THEIR OWN REAL STOP — not slippage, a real stop-management "
                           f"issue: ***")
            _print_df("SL-BREACHED TRADES (LiveR(ownSL) < -1.05)",
                      breached[["Symbol","Dir","EntryTime","LiveEntry","LiveExit",
                                "LiveSLDist","LiveR(ownSL)","LivePnL"]])

        big_timing_divergence = df[df["ExitBarsΔ"].abs() >= 3]
        if len(big_timing_divergence):
            logger.warning(f"\n  *** {len(big_timing_divergence)} TRADE(S) EXITED 3+ BARS "
                           f"(15+ min) AWAY FROM where the backtest's BE/trail logic "
                           f"would have exited the SAME entry — this is a live-vs-"
                           f"backtest LOGIC divergence, not price slippage: ***")
            _print_df("LARGE EXIT-TIMING DIVERGENCE (|ExitBarsΔ| >= 3)",
                      big_timing_divergence[["Symbol","Dir","EntryTime","ExitBarsΔ",
                                              "LiveR(btSL)","BTR","LivePnL","BTPnL"]])

        df.to_csv("orb_live_vs_backtest_diff.csv", index=False)
        logger.info("  Saved: orb_live_vs_backtest_diff.csv")
    else:
        logger.warning("  [live-diff] No matched trades to compare.")

    if unmatched_live:
        logger.warning(f"\n  {len(unmatched_live)} LIVE TRADE(S) WITH NO MATCHING "
                       f"BACKTEST SIGNAL (logic divergence, not slippage — check "
                       f"these individually):")
        for lt in unmatched_live:
            logger.warning(f"    {lt['symbol']}  {lt['entry_time']}  "
                          f"dir={'LONG' if lt['direction']==1 else 'SHORT'}  "
                          f"entry={lt['entry_price']}  pnl={lt['profit']:.2f}")


# ==============================================================================
#  SECTION 13 — MAIN
# ==============================================================================

def main():
    logger.info(f"\n{'='*80}")
    logger.info(f"  RUN CONFIG")
    logger.info(f"  Starting balance : ${STARTING_BALANCE:,.2f}")
    logger.info(f"  Trade start date : {TRADE_START_DATE!r}")
    logger.info(f"  Commission/lot   : {COMMISSION_PER_LOT}")
    logger.info(f"  Slippage/side    : {SLIPPAGE_PRICE_PER_SYMBOL}  "
                f"(price units, applied adversely on EVERY entry and exit fill)")
    logger.info(f"{'='*80}\n")

    if MT5_AVAILABLE:
        if not mt5.initialize(path=TERMINAL_PATH, login=LOGIN,
                              password=PASSWORD, server=SERVER):
            raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")
        logger.info("MT5 connected")

        # ── ACCOUNT IDENTITY CHECK ──────────────────────────────────────
        # LOGIN/PASSWORD/SERVER above come from MT5_LOGIN/MT5_PASSWORD/
        # MT5_SERVER env vars, defaulting to 0/""/"" if unset — unlike the
        # live engine (orb_live_v6.py), which hardcodes the real account.
        # If those env vars weren't set, this may have silently attached
        # to whatever account the terminal already had open, NOT your real
        # FundingPips account — which would make every account-derived
        # number below (leverage, margin, stops level, even price/spread
        # feed) come from the wrong source. Verify this line matches your
        # live engine's LOGIN=20051742 / SERVER="FundingPips-SIM1" exactly.
        _acct = mt5.account_info()
        if _acct is not None:
            logger.info(
                f"  *** CONNECTED ACCOUNT: login={_acct.login}  "
                f"server={_acct.server!r}  leverage=1:{_acct.leverage}  "
                f"balance={_acct.balance:.2f}  currency={_acct.currency} ***"
            )
            if LOGIN and _acct.login != LOGIN:
                logger.error(
                    f"  *** MISMATCH: requested LOGIN={LOGIN} but connected "
                    f"account is {_acct.login} — env var MT5_LOGIN likely "
                    f"unset/wrong. Backtest is NOT using your real account. ***"
                )
            if not LOGIN:
                logger.warning(
                    "  *** MT5_LOGIN env var was unset (LOGIN=0) — backtest "
                    "attached to whatever account the terminal already had "
                    "open, which may not be your FundingPips-SIM1 account. "
                    "Set MT5_LOGIN / MT5_PASSWORD / MT5_SERVER to match the "
                    "live engine, or hardcode them the same way orb_live_v6.py "
                    "does. ***"
                )
        else:
            logger.error("  *** account_info() returned None — could not "
                         "verify which account this backtest is connected to ***")
        # ─────────────────────────────────────────────────────────────────
    else:
        logger.warning("MT5 not available — using fallback broker info")

    logger.info("\n=== BROKER INFO (tick value, min SL distance, contract size, leverage) ===")
    broker_info, account_leverage = fetch_broker_info()
    logger.info("=== END BROKER INFO ===\n")

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
        cache = build_cache_and_signals(
            sym, df_all, mt5_start, PARAMS_GRID_BEST[sym],
            min_sl_distance=broker_info[sym]["min_sl_distance"],
            slippage=SLIPPAGE_PRICE_PER_SYMBOL.get(sym, 0.0),
        )
        all_caches[sym] = cache

    if not all_caches:
        mt5.shutdown()
        logger.error("No symbol data. Exiting.")
        return

    if RUN_LIVE_VS_BACKTEST_DIFF:
        try:
            _diff_from = pd.Timestamp(LIVE_DIFF_DATE_FROM or TRADE_START_DATE)
            _diff_to   = pd.Timestamp(datetime.datetime.utcnow())
            run_live_vs_backtest_diff(all_caches, broker_info, _diff_from, _diff_to)
        except Exception:
            logger.error("  [live-diff] Failed with an exception (see traceback below). "
                        "Continuing with the rest of the backtest regardless.")
            logger.error(traceback.format_exc())

    mt5.shutdown()

    logger.info(f"\n{'='*80}")
    logger.info(f"  EXIT MODE SWEEP — {len(SYMBOL_COMBOS)} combos x "
                f"{2 + 2*len(RR_TARGETS)} modes each (incl. Mode A long-only)")
    logger.info(f"{'='*80}\n")

    all_combo_results = []
    flat_rows         = []

    for combo in SYMBOL_COMBOS:
        if not all(s in all_caches for s in combo):
            continue
        combo_name = "+".join(combo)
        logger.info(f"  running combo: {combo_name} ...")

        cr = run_combo_all_modes(combo_name, list(combo), all_caches, broker_info)
        all_combo_results.append(cr)

        for mode_label, res in cr["all_modes"].items():
            flat_rows.append(res)

    pd.DataFrame(flat_rows).to_csv("orb_combo_sweep.csv", index=False)

    eval_combos = [cr["combo"].split("+") for cr in all_combo_results]

    eval_modes: dict = {
        "A:BE+TRAIL":  (resolve_mode_a, None),
        "A:LONG_ONLY": (resolve_mode_a, 1),
    }
    for rr in RR_TARGETS:
        eval_modes[f"B:RR{rr}"] = (
            (lambda _rr=rr: lambda cache, si: resolve_mode_b(cache, si, _rr, sl_multiplier=1.0))(),
            None,
        )
        eval_modes[f"C:WIDE_RR{rr}"] = (
            (lambda _rr=rr: lambda cache, si: resolve_mode_b(cache, si, _rr, sl_multiplier=WIDE_SL_MULT))(),
            None,
        )

    logger.info(f"\n  running FTMO eval sims — {len(eval_combos)} combo(s) x {len(eval_modes)} modes ...")

    all_eval_cycles = []
    eval_summary: dict = {}

    for symbols in eval_combos:
        combo_name = "+".join(symbols)
        missing    = [s for s in symbols if s not in all_caches]
        if missing:
            logger.warning(f"  [{combo_name}] missing {missing} — skipping")
            continue

        caches = {sym: all_caches[sym] for sym in symbols}
        eval_summary[combo_name] = {}

        for mode_label, (resolver_fn, direction_filter) in eval_modes.items():
            cycles = run_eval_simulation(
                symbols, caches, broker_info,
                resolver_fn=resolver_fn, direction_filter=direction_filter,
            )
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

    print_all_summary_tables(all_combo_results, eval_summary)

    logger.info(
        "\nOutputs saved:\n"
        "  orb_combo_sweep.csv   (full per-mode combo results)\n"
        "  orb_eval_cycles.csv   (every FTMO eval attempt)\n"
        "  orb_chrono_compare.log\n"
        "Done."
    )


if __name__ == "__main__":
    print("=" * 70, flush=True)
    print("  ORB CHRONO BACKTEST — starting up", flush=True)
    print(f"  Python: {sys.version.split()[0]}", flush=True)
    print(f"  Working directory: {os.getcwd()}", flush=True)
    print(f"  MetaTrader5 module available: {MT5_AVAILABLE}", flush=True)
    print("=" * 70, flush=True)
    try:
        main()
        print("\nDone. Check orb_chrono_compare.log for the full run.", flush=True)
    except Exception:
        print("\n*** CRASHED — full traceback below ***", flush=True)
        traceback.print_exc()
    finally:
        # Keeps the console window open if this was double-clicked rather
        # than run from an already-open terminal, so a crash is actually
        # visible instead of the window closing instantly.
        try:
            input("\nPress Enter to exit...")
        except Exception:
            pass
