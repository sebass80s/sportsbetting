import requests
from pathlib import Path
import pandas as pd


# ==================================================
# API-NYCKEL
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

import os
API_KEY = os.environ.get("ODDSPAPI_API_KEY")
if not API_KEY:
    raise RuntimeError("ODDSPAPI_API_KEY saknas i miljön")

# ==================================================
# HÄMTA FIXTURES
# ==================================================

BASE_URL = "https://api.oddspapi.io/v4"

url = f"{BASE_URL}/fixtures"

params = {
    "apiKey": API_KEY,
    "tournamentId": 17,
    "statusId": 0,
    "language": "en"
}

response = requests.get(
    url,
    params=params,
    timeout=30
)

print()
print("===================================")
print("UPCOMING PREMIER LEAGUE")
print("===================================")
print("HTTP status:", response.status_code)

if not response.ok:
    print(response.text[:3000])
    raise SystemExit

fixtures = response.json()

print("Antal matcher:", len(fixtures))
print()


# ==================================================
# VISA MATCHER
# ==================================================

rows = []

for fixture in fixtures:

    row = {
        "fixture_id": fixture.get("fixtureId"),
        "start_time": fixture.get("startTime"),
        "home_team": fixture.get("participant1Name"),
        "away_team": fixture.get("participant2Name"),
        "has_odds": fixture.get("hasOdds"),
        "season_id": fixture.get("seasonId")
    }

    rows.append(row)


df = pd.DataFrame(rows)

if len(df) > 0:

    df["start_time"] = pd.to_datetime(
        df["start_time"],
        utc=True
    )

    df = df.sort_values("start_time")

    print(
        df[
            [
                "start_time",
                "home_team",
                "away_team",
                "has_odds",
                "fixture_id"
            ]
        ].to_string(index=False)
    )

else:
    print("Inga kommande matcher hittades.")


# ==================================================
# SPARA
# ==================================================

OUTPUT_DIR = PROJECT_ROOT / "data" / "forward"
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = OUTPUT_DIR / "upcoming_pl.csv"

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("Sparad till:", OUTPUT_FILE)