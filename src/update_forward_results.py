import time
from pathlib import Path

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_FILE = PROJECT_ROOT / "data" / "forward" / "v1_predictions.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "forward" / "v1_forward_results.csv"
ENV_FILE = PROJECT_ROOT / ".env"


# ==================================================
# API-NYCKEL
# ==================================================

API_KEY = None
with open(ENV_FILE, "r", encoding="utf-8-sig") as f:
    for line in f:
        line = line.strip()
        if line.startswith("ODDSPAPI_API_KEY="):
            API_KEY = line.split("=", 1)[1].strip().strip("'\"")
            break

if not API_KEY:
    raise RuntimeError("API-nyckel saknas")


# ==================================================
# LADDA SIGNALER + TIDIGARE RESULTAT
# ==================================================

bets = pd.read_csv(PREDICTIONS_FILE)
bets["start_time"] = pd.to_datetime(bets["start_time"], utc=True)

finished_cache = {}
if OUTPUT_FILE.exists():
    previous = pd.read_csv(OUTPUT_FILE)
    if "fixture_id" in previous.columns and "result_status" in previous.columns:
        previous_finished = previous[previous["result_status"] == "FINISHED"].copy()
        for _, old in previous_finished.iterrows():
            finished_cache[str(old["fixture_id"])] = old.to_dict()


# ==================================================
# HÄMTA FIXTURES
# ==================================================

url = "https://api.oddspapi.io/v4/fixtures"
params = {
    "apiKey": API_KEY,
    "tournamentId": 17,
    "language": "en",
}

response = requests.get(url, params=params, timeout=30)

print()
print("===================================")
print("FORWARD RESULTS UPDATE")
print("===================================")
print("HTTP status:", response.status_code)

if not response.ok:
    print(response.text[:3000])
    raise SystemExit(1)

fixtures = response.json()
fixture_lookup = {
    str(fixture.get("fixtureId")): fixture
    for fixture in fixtures
}


# ==================================================
# SCORE-HJÄLPARE
# ==================================================

SCORES_URL = "https://api.oddspapi.io/v4/scores"


def extract_fulltime_score(score_data):
    """Returnera fulltidsscore från kända OddsPapi-format."""
    scores = score_data.get("scores", {})
    if not isinstance(scores, dict):
        return None, None

    periods = scores.get("periods")
    if isinstance(periods, dict):
        for key in ("fulltime", "result", "full_time", "ft"):
            period = periods.get(key)
            if isinstance(period, dict):
                home = period.get("participant1Score")
                away = period.get("participant2Score")
                if home is not None and away is not None:
                    return home, away

    for key in ("fulltime", "result", "full_time", "ft", "0"):
        period = scores.get(key)
        if isinstance(period, dict):
            home = period.get("participant1Score")
            away = period.get("participant2Score")
            if home is not None and away is not None:
                return home, away

    return None, None


def get_finished_score(fixture_id):
    score_response = requests.get(
        SCORES_URL,
        params={"apiKey": API_KEY, "fixtureId": fixture_id},
        timeout=30,
    )

    if not score_response.ok:
        print(
            f"Score-anrop misslyckades för {fixture_id}: "
            f"HTTP {score_response.status_code}"
        )
        return None, None

    try:
        score_data = score_response.json()
    except ValueError:
        return None, None

    return extract_fulltime_score(score_data)


# ==================================================
# RESULTAT
# ==================================================

rows = []
score_requests = 0
reused_finished = 0

for _, bet in bets.iterrows():
    fixture_id = str(bet["fixture_id"])
    row = bet.to_dict()

    # Redan verifierade slutresultat är frysta. Återanvänd dem och undvik
    # ett nytt score-anrop varje gång dashboarden uppdateras.
    cached = finished_cache.get(fixture_id)
    if cached is not None:
        for col in (
            "home_goals",
            "away_goals",
            "actual_btts",
            "won",
            "profit",
            "result_status",
        ):
            row[col] = cached.get(col)
        rows.append(row)
        reused_finished += 1
        continue

    row["home_goals"] = None
    row["away_goals"] = None
    row["actual_btts"] = None
    row["won"] = None
    row["profit"] = None
    row["result_status"] = "NOT_FINISHED"

    fixture = fixture_lookup.get(fixture_id)
    if fixture is None:
        row["result_status"] = "FIXTURE_NOT_FOUND"
        rows.append(row)
        continue

    # OddsPapi: statusId 2 = färdigspelad. Score hämtas endast första
    # gången en fryst match upptäcks som färdig.
    if fixture.get("statusId") != 2:
        rows.append(row)
        continue

    # Scores-endpointen har 1000 ms cooldown.
    if score_requests > 0:
        time.sleep(1.05)

    home_score, away_score = get_finished_score(fixture_id)
    score_requests += 1

    if home_score is None or away_score is None:
        row["result_status"] = "SCORE_NOT_FOUND"
        rows.append(row)
        continue

    try:
        home_goals = int(home_score)
        away_goals = int(away_score)
    except (ValueError, TypeError):
        row["result_status"] = "INVALID_SCORE"
        rows.append(row)
        continue

    actual_btts = int(home_goals > 0 and away_goals > 0)

    if bet["bet_side"] == "YES":
        won = actual_btts == 1
    elif bet["bet_side"] == "NO":
        won = actual_btts == 0
    else:
        won = False

    odds = bet["best_bet_odds"]
    profit = odds - 1 if won else -1

    row["home_goals"] = home_goals
    row["away_goals"] = away_goals
    row["actual_btts"] = actual_btts
    row["won"] = int(won)
    row["profit"] = profit
    row["result_status"] = "FINISHED"
    rows.append(row)


results = pd.DataFrame(rows)
results.to_csv(OUTPUT_FILE, index=False)


# ==================================================
# VISA
# ==================================================

print()
print("Frysta bets:", len(results))
print("Återanvända färdiga resultat:", reused_finished)
print("Nya score-anrop:", score_requests)
print()

print(
    results[
        [
            "home_team",
            "away_team",
            "bet_side",
            "best_bet_odds",
            "home_goals",
            "away_goals",
            "actual_btts",
            "won",
            "profit",
            "result_status",
        ]
    ].to_string(index=False)
)

finished = results[results["result_status"] == "FINISHED"].copy()

print()
print("===================================")
print("RESULTAT")
print("===================================")
print("Färdigspelade bets:", len(finished))

if len(finished) > 0:
    print("Wins:", finished["won"].sum())
    print("Profit:", round(finished["profit"].sum(), 2), "units")
    print("ROI:", round(finished["profit"].mean() * 100, 2), "%")

print()
print("Sparad till:", OUTPUT_FILE)
