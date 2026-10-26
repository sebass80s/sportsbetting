"""Append first-seen V1 signals to the published, persistent forward-test ledger.

The published CSV is committed to Git and is the source of truth across
independent GitHub Actions runs. Never revise a fixture's first signal.
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/forward/v1_predictions.csv"
DESTINATION = ROOT / "docs/v1_live_signals.csv"
FIELDS = [
    "fixture_id", "start_time", "home_team", "away_team", "bet_side",
    "best_bet_odds", "best_bookmaker", "predicted_close_probability",
    "prediction_timestamp",
]


def load_rows(path):
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


# Existing published records always win. This is the permanent ledger,
# including signals generated in earlier CI runs.
existing = load_rows(DESTINATION)
seen = {}
for row in existing:
    fixture_id = str(row.get("fixture_id", "")).strip()
    if fixture_id and row.get("bet_side") in ("YES", "NO"):
        seen.setdefault(fixture_id, {field: row.get(field, "") for field in FIELDS})

before = len(seen)
for row in load_rows(SOURCE):
    fixture_id = str(row.get("fixture_id", "")).strip()
    if fixture_id and row.get("bet_side") in ("YES", "NO"):
        seen.setdefault(fixture_id, {field: row.get(field, "") for field in FIELDS})

DESTINATION.parent.mkdir(parents=True, exist_ok=True)
with DESTINATION.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(seen.values())

print(f"Published {len(seen)} V1 signals ({len(seen) - before} newly frozen)")
