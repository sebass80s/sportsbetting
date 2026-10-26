"""Publish upcoming Premier League fixtures from OddsPapi, without inventing V1 signals."""
import csv
import os
from datetime import datetime, timezone
from pathlib import Path
import requests

key = os.environ.get("ODDSPAPI_API_KEY")
if not key:
    raise SystemExit("ODDSPAPI_API_KEY secret missing")
response = requests.get("https://api.oddspapi.io/v4/fixtures", params={"apiKey":key,"tournamentId":17,"language":"en","statusId":0}, timeout=30)
response.raise_for_status()
payload = response.json()
if not isinstance(payload, list):
    raise SystemExit("Unexpected fixtures API response")
now = datetime.now(timezone.utc)
rows = []
for fixture in payload:
    if not isinstance(fixture, dict):
        continue
    kickoff = fixture.get("startTime") or fixture.get("start_time")
    if not kickoff:
        continue
    try:
        dt = datetime.fromisoformat(str(kickoff).replace("Z","+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    except ValueError:
        continue
    if dt <= now:
        continue
    participants = fixture.get("participants") or []
    home = fixture.get("participant1Name") or fixture.get("homeTeam") or fixture.get("home_team")
    away = fixture.get("participant2Name") or fixture.get("awayTeam") or fixture.get("away_team")
    if isinstance(home, dict):
        home = home.get("name")
    if isinstance(away, dict):
        away = away.get("name")
    if not home and isinstance(participants,list) and len(participants)>0:
        home = participants[0].get("name") if isinstance(participants[0],dict) else None
    if not away and isinstance(participants,list) and len(participants)>1:
        away = participants[1].get("name") if isinstance(participants[1],dict) else None
    if not home or not away:
        continue
    rows.append({"fixture_id":str(fixture.get("fixtureId","")),"start_time":dt.isoformat(),"home_team":home,"away_team":away})
rows.sort(key=lambda r:r["start_time"])
if not rows:
    raise SystemExit("No future fixtures parsed; preserving previously published data. Check API schema.")
path = Path("docs/upcoming_fixtures.csv")
with path.open("w",newline="",encoding="utf-8") as out:
    writer=csv.DictWriter(out,fieldnames=["fixture_id","start_time","home_team","away_team"])
    writer.writeheader()
    writer.writerows(rows)
print("Published",len(rows),"future fixtures")
