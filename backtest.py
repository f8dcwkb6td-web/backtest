"""
ORB LIVE LOG SUMMARIZER
========================
Reads orb_live_v6.log (or any log from the live engine) and prints ONE
short table of counts — no need to scroll through thousands of lines.

USAGE:
    python log_summary.py orb_live_v6.log

Optionally narrow to a date range:
    python log_summary.py orb_live_v6.log --from 2026-06-04

What it counts, and why each one matters:

  Signals fired            — total SIGNAL LONG/SHORT lines
  Entries confirmed         — total "ENTERED" lines (signal -> actual position)
  Entry failures (10016)    — retcode 10016 occurrences (invalid stops)
  Entries dropped entirely  — "Entry FAILED after N retries" (10016 persisted,
                               trade never happened at all)
  Margin downsizes          — lot was shrunk below the risk-model's intended size
  Margin rejects            — trade skipped entirely, even downsized, no margin
  Avg lot on downsized trades vs avg lot overall — tells you HOW MUCH smaller
      the downsized trades were, not just how often it happened
  BT accuracy per symbol    — sanity check that signal generation still
                               matches the backtest at startup

If "Signals fired" is meaningfully larger than "Entries confirmed," that gap
IS your discrepancy — every signal that didn't become a live position is a
trade the backtest counted that never actually happened (or happened smaller).
"""

import re
import sys
import argparse
from collections import defaultdict


def parse_log(path: str, from_date: str = None):
    sig_re      = re.compile(r"SIGNAL (LONG|SHORT)")
    entered_re  = re.compile(r"\[(\w+)\] ENTERED (LONG|SHORT) ticket=(\d+)")
    retcode_re  = re.compile(r"retcode 10016")
    failed_re   = re.compile(r"Entry FAILED after \d+ retries")
    downsize_re = re.compile(r"\[(\w+)\] margin downsize: ([\d.]+)")
    reject_re   = re.compile(r"entry rejected")
    lot_re      = re.compile(r"lot=([\d.]+)")
    bt_acc_re   = re.compile(r"\[(\w+)\] Signal match: ([\d.]+)%")
    date_re     = re.compile(r"^(\d{4}-\d{2}-\d{2})")

    counts = defaultdict(int)
    downsize_lots = []
    all_entry_lots = []
    bt_accuracy = {}
    per_symbol_signals = defaultdict(int)
    per_symbol_entries = defaultdict(int)

    from_ts = from_date  # simple string compare works for ISO dates

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if from_ts:
                m = date_re.match(line)
                if m and m.group(1) < from_ts:
                    continue

            if sig_re.search(line):
                counts["signals_fired"] += 1
                sym_m = re.search(r"\[(\w+)\] SIGNAL", line)
                if sym_m:
                    per_symbol_signals[sym_m.group(1)] += 1

            m = entered_re.search(line)
            if m:
                counts["entries_confirmed"] += 1
                per_symbol_entries[m.group(1)] += 1
                lot_m = lot_re.search(line)
                if lot_m:
                    all_entry_lots.append(float(lot_m.group(1)))

            if retcode_re.search(line):
                counts["retcode_10016"] += 1

            if failed_re.search(line):
                counts["entries_dropped_entirely"] += 1

            m = downsize_re.search(line)
            if m:
                counts["margin_downsizes"] += 1
                downsize_lots.append(float(m.group(2)))

            if "insufficient margin" in line and reject_re.search(line):
                counts["margin_rejects"] += 1

            m = bt_acc_re.search(line)
            if m:
                bt_accuracy[m.group(1)] = float(m.group(2))

    return {
        "counts": dict(counts),
        "downsize_lots": downsize_lots,
        "all_entry_lots": all_entry_lots,
        "bt_accuracy": bt_accuracy,
        "per_symbol_signals": dict(per_symbol_signals),
        "per_symbol_entries": dict(per_symbol_entries),
    }


def build_summary_text(result: dict) -> str:
    c = result["counts"]
    sig = c.get("signals_fired", 0)
    ent = c.get("entries_confirmed", 0)
    gap = sig - ent

    lines = []
    lines.append("=" * 60)
    lines.append("  ORB LIVE LOG SUMMARY")
    lines.append("=" * 60)
    lines.append(f"  Signals fired            : {sig}")
    lines.append(f"  Entries confirmed         : {ent}")
    if sig:
        lines.append(f"  Signals -> no position    : {gap}"
                      f"  ({gap/sig*100:.1f}% of signals)")
    lines.append("-" * 60)
    lines.append(f"  retcode 10016 occurrences : {c.get('retcode_10016', 0)}")
    lines.append(f"  Entries dropped entirely  : {c.get('entries_dropped_entirely', 0)}"
                 f"   <- trade never happened, 3 retries exhausted")
    lines.append(f"  Margin downsizes          : {c.get('margin_downsizes', 0)}"
                 f"   <- trade happened, but smaller lot than risk model wanted")
    lines.append(f"  Margin rejects (hard)     : {c.get('margin_rejects', 0)}")
    lines.append("-" * 60)

    if result["downsize_lots"]:
        avg_down = sum(result["downsize_lots"]) / len(result["downsize_lots"])
        lines.append(f"  Avg lot on downsized trades : {avg_down:.3f}")
    if result["all_entry_lots"]:
        avg_all = sum(result["all_entry_lots"]) / len(result["all_entry_lots"])
        lines.append(f"  Avg lot across ALL entries  : {avg_all:.3f}")
    if result["downsize_lots"] and result["all_entry_lots"]:
        avg_down = sum(result["downsize_lots"]) / len(result["downsize_lots"])
        avg_all  = sum(result["all_entry_lots"]) / len(result["all_entry_lots"])
        if avg_all > 0:
            shortfall = (1 - avg_down / avg_all) * 100
            lines.append(f"  -> downsized trades average ~{shortfall:.0f}% smaller lot"
                         f" than a typical entry")
    lines.append("-" * 60)

    if result["bt_accuracy"]:
        lines.append("  BT accuracy at startup (per symbol):")
        for sym, acc in result["bt_accuracy"].items():
            flag = "" if acc >= 95.0 else "  <- BELOW 95% THRESHOLD"
            lines.append(f"    {sym:<8} {acc:.2f}%{flag}")
    lines.append("-" * 60)

    lines.append("  Per-symbol signal -> entry conversion:")
    all_syms = set(result["per_symbol_signals"]) | set(result["per_symbol_entries"])
    for sym in sorted(all_syms):
        s = result["per_symbol_signals"].get(sym, 0)
        e = result["per_symbol_entries"].get(sym, 0)
        pct = (e / s * 100) if s else 0.0
        lines.append(f"    {sym:<8} signals={s:<5} entries={e:<5} ({pct:.1f}% converted)")
    lines.append("=" * 60)

    return "\n".join(lines)


def print_summary(result: dict):
    print(build_summary_text(result))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile")
    ap.add_argument("--from", dest="from_date", default=None,
                     help="Only count lines from this ISO date onward, e.g. 2026-06-04")
    ap.add_argument("--out", dest="out_path", default=None,
                     help="Where to write the summary file "
                          "(default: <logfile>_summary.txt next to the log)")
    args = ap.parse_args()

    result = parse_log(args.logfile, args.from_date)
    summary_text = build_summary_text(result)

    print(summary_text)

    out_path = args.out_path or (args.logfile.rsplit(".", 1)[0] + "_summary.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")

    print(f"\nSummary written to: {out_path}")
