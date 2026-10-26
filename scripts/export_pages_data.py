"""Publish a sanitized, joined V1 forward-test snapshot for GitHub Pages.

Run locally: python3 scripts/export_pages_data.py
Then commit docs/v1_forward_status.csv and push to main.
No API calls, no secrets, no model retraining.
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORWARD = ROOT / "data" / "forward"
OUTPUT = ROOT / "docs" / "v1_forward_status.csv"


def read_csv(name):
    path = FORWARD / name
    if not path.exists():
        raise SystemExit(f"Missing local file: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def first_by_fixture(rows):
    found = {}
    for row in rows:
        fixture_id = str(row.get("fixture_id", "")).strip()
        if fixture_id and fixture_id not in found:
            found[fixture_id] = row
    return found


def main():
    status = read_csv("v1_forward_status.csv")
    results = first_by_fixture(read_csv("v1_forward_results.csv"))
    predictions_path = FORWARD / "v1_predictions.csv"
    frozen = first_by_fixture(read_csv("v1_predictions.csv")) if predictions_path.exists() else {}

    fields = [
        "fixture_id", "start_time", "home_team", "away_team", "bet_side",
        "bookmaker", "original_odds", "latest_odds", "status",
        "result_status", "home_goals", "away_goals", "won", "profit",
        "predicted_close_probability", "current_economic_clv",
    ]
    records = []
    seen = set()
    for row in status:
        fixture_id = str(row.get("fixture_id", "")).strip()
        if not fixture_id or fixture_id in seen:
            raise SystemExit(f"Invalid or duplicate fixture_id: {fixture_id!r}")
        seen.add(fixture_id)
        result = results.get(fixture_id, {})
        prediction = frozen.get(fixture_id, {})
        merged = {key: row.get(key, "") for key in fields}
        for key in ("result_status", "home_goals", "away_goals", "won", "profit"):
            merged[key] = result.get(key, "")
        merged["predicted_close_probability"] = prediction.get("predicted_close_probability", "")
        records.append(merged)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    print(f"Exported {len(records)} signals to {OUTPUT.relative_to(ROOT)}")
    print("WARNING: This CSV is public on GitHub Pages. Review before committing.")


if __name__ == "__main__":
    main()
