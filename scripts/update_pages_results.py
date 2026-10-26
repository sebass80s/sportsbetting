"""Update published frozen BTTS results without regenerating predictions."""
import csv
import os
import time
from pathlib import Path
import requests

CSV = Path("docs/v1_forward_status.csv")
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

if updated:
    with CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
print(f"Updated {updated} finished results; {calls} score requests")
