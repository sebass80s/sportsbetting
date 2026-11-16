#!/usr/bin/env python3
"""Out-of-sample BTTS calibration bins. Standard library only.

Usage: python scripts/analyze_btts_residuals.py PATH_TO_OOS.csv
Input columns: fixture_id, btts_probability, btts_actual, prediction_timestamp, start_time.
Aliases: predicted_btts_probability/probability/p_btts; btts/won_btts/btts_result.
Only genuine out-of-sample predictions should be supplied.
"""
import argparse
import csv
import math
from datetime import datetime
from pathlib import Path

PROB = ("btts_probability", "predicted_btts_probability", "probability", "p_btts")
ACTUAL = ("btts_actual", "btts", "won_btts", "btts_result")
EDGES = ((0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.0))


def column(names, candidates):
    for candidate in candidates:
        if candidate in names:
            return candidate
    raise ValueError("Missing required column; expected one of: " + ", ".join(candidates))


def parse_date(value):
    return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))


def analyze(source, destination):
    with source.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        names = reader.fieldnames or []
        probability = column(names, PROB)
        actual = column(names, ACTUAL)
        if "fixture_id" not in names:
            raise ValueError("fixture_id is required for duplicate detection")
        if not {"prediction_timestamp", "start_time"} <= set(names):
            raise ValueError("prediction_timestamp and start_time required to check pre-kickoff timing")
        bins = [[] for _ in EDGES]
        seen = set()
        total = 0
        for line, row in enumerate(reader, 2):
            fixture_id = row["fixture_id"].strip()
            if not fixture_id or fixture_id in seen:
                raise ValueError(f"Missing/duplicate fixture_id at line {line}: {fixture_id}")
            seen.add(fixture_id)
            try:
                p = float(row[probability])
                y = float(row[actual])
                prediction_time = parse_date(row["prediction_timestamp"])
                kickoff = parse_date(row["start_time"])
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"Invalid value at line {line}: {exc}") from exc
            if not math.isfinite(p) or not 0 <= p <= 1 or y not in (0, 1):
                raise ValueError(f"Invalid probability or outcome at line {line}")
            if prediction_time.tzinfo is None or kickoff.tzinfo is None:
                raise ValueError(f"Timezone missing at line {line}")
            if prediction_time >= kickoff:
                raise ValueError(f"Prediction not before kickoff at line {line}")
            index = min(int(p * 5), 4)
            bins[index].append((p, y))
            total += 1
    if total == 0:
        raise ValueError("Empty input")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("probability_bin", "matches", "mean_prediction", "actual_btts_rate", "calibration_gap", "brier_score", "small_sample"))
        for (low, high), values in zip(EDGES, bins):
            n = len(values)
            mean_p = sum(p for p, _ in values) / n if n else None
            actual_rate = sum(y for _, y in values) / n if n else None
            brier = sum((p - y) ** 2 for p, y in values) / n if n else None
            writer.writerow((f"{low:.0%}-{high:.0%}", n,
                             f"{mean_p:.6f}" if n else "",
                             f"{actual_rate:.6f}" if n else "",
                             f"{actual_rate - mean_p:+.6f}" if n else "",
                             f"{brier:.6f}" if n else "",
                             "YES" if n < 30 else "NO"))
    print(f"Validated {total} unique pre-kickoff predictions; wrote {destination}")
    print("NOTE: Timestamp checks cannot prove historical feature construction is leak-free.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Out-of-sample prediction CSV")
    parser.add_argument("--output", type=Path, default=Path("reports/btts_calibration_bins.csv"))
    args = parser.parse_args()
    analyze(args.input, args.output)
