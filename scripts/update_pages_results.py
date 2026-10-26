"""Update published frozen BTTS results without regenerating predictions."""
import csv
import os
import time
from pathlib import Path
from datetime import datetime, timezone
import requests

CSV = Path("docs/v1_forward_status.csv")
SIGNALS = Path("docs/v1_live_signals.csv")
API = "https://api.oddspapi.io/v4"
key = os.environ.get("ODDSPAPI_API_KEY")
if not key:
    raise SystemExit("ODDSPAPI_API_KEY secret is missing")

with CSV.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames
    rows = list(reader)
if not rows:
    raise SystemExit("No published signals found; refusing to overwrite")

# Enroll only signals published before kickoff. Existing settled records are
# immutable; the live ledger is the source of truth for new forward entries.
existing_ids = {str(row["fixture_id"]).strip() for row in rows}
enrolled = 0
if SIGNALS.exists():
    with SIGNALS.open(newline="", encoding="utf-8-sig") as f:
        for signal in csv.DictReader(f):
            fixture_id = str(signal.get("fixture_id", "")).strip()
            if not fixture_id or fixture_id in existing_ids:
                continue
            if signal.get("bet_side") not in ("YES", "NO"):
                continue
            try:
                kickoff = datetime.fromisoformat(signal["start_time"].strip().replace("Z", "+00:00"))
                created = datetime.fromisoformat(signal["prediction_timestamp"].strip().replace("Z", "+00:00"))
                odds = float(signal["best_bet_odds"])
                if kickoff.tzinfo is None or created.tzinfo is None or created >= kickoff or not (1 < odds < 1000):
                    print("Skipping invalid or post-kickoff signal:", fixture_id)
                    continue
            except (ValueError, TypeError, KeyError):
                print("Skipping malformed signal:", fixture_id)
                continue
            new_row = {field: "" for field in fields}
            new_row.update({
                "fixture_id": fixture_id,
                "start_time": signal["start_time"],
                "home_team": signal["home_team"],
                "away_team": signal["away_team"],
                "bet_side": signal["bet_side"],
                "bookmaker": signal.get("best_bookmaker", ""),
                "original_odds": signal["best_bet_odds"],
                "latest_odds": "",
                "status": "WAITING",
                "result_status": "PENDING",
                "predicted_close_probability": signal.get("predicted_close_probability", ""),
            })
            rows.append(new_row)
            existing_ids.add(fixture_id)
            enrolled += 1
print("New pre-kickoff signals enrolled:", enrolled)

response = requests.get(API + "/fixtures", params={"apiKey": key, "tournamentId": 17, "language": "en"}, timeout=30)
response.raise_for_status()
fixtures = {str(f.get("fixtureId")): f for f in response.json()}
print("Fixtures received:", len(fixtures))

def fulltime(payload):
    scores = payload.get("scores", {})
    if not isinstance(scores, dict):
        return None
    periods = scores.get("periods", {})
    containers = [periods, scores]
    for container in containers:
        if not isinstance(container, dict):
            continue
        for name in ("fulltime", "result", "full_time", "ft", "0"):
            period = container.get(name)
            if isinstance(period, dict):
                h, a = period.get("participant1Score"), period.get("participant2Score")
                if h is not None and a is not None:
                    return int(h), int(a)
    return None

updated = 0
calls = 0
for row in rows:
    if row.get("result_status") == "FINISHED":
        continue
    fixture = fixtures.get(str(row["fixture_id"]))
    if not fixture or fixture.get("statusId") != 2:
        continue
    if calls:
        time.sleep(1.1)
    resp = requests.get(API + "/scores", params={"apiKey": key, "fixtureId": row["fixture_id"]}, timeout=30)
    calls += 1
    resp.raise_for_status()
    score = fulltime(resp.json())
    if score is None:
        print("No fulltime score for", row["fixture_id"])
        continue
    home, away = score
    side = row.get("bet_side")
    if side not in ("YES", "NO"):
        raise SystemExit("Unknown side: " + str(side))
    btts = home > 0 and away > 0
    won = btts if side == "YES" else not btts
    odds = float(row["original_odds"])
    row.update(home_goals=str(home), away_goals=str(away), won=str(int(won)), profit=str(odds - 1 if won else -1), result_status="FINISHED")
    updated += 1

if updated or enrolled:
    with CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
print(f"Updated {updated} finished results; {calls} score requests")
