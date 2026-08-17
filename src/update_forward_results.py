import pandas as pd
from pathlib import Path
import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "forward"
    / "v1_predictions.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "forward"
    / "v1_forward_results.csv"
)

ENV_FILE = (
    PROJECT_ROOT
    / ".env"
)


# ==================================================
# API-NYCKEL
# ==================================================

API_KEY = None

with open(
    ENV_FILE,
    "r",
    encoding="utf-8-sig"
) as f:

    for line in f:

        line = line.strip()

        if line.startswith(
            "ODDSPAPI_API_KEY="
        ):

            API_KEY = (
                line
                .split("=", 1)[1]
                .strip()
                .strip("'\"")
            )

            break


if not API_KEY:
    raise RuntimeError(
        "API-nyckel saknas"
    )


# ==================================================
# LADDA SIGNALER
# ==================================================

bets = pd.read_csv(
    PREDICTIONS_FILE
)

bets["start_time"] = pd.to_datetime(
    bets["start_time"],
    utc=True
)


# ==================================================
# HÄMTA FIXTURES
# ==================================================

url = (
    "https://api.oddspapi.io/v4/fixtures"
)

params = {
    "apiKey": API_KEY,
    "tournamentId": 17,
    "language": "en"
}


response = requests.get(
    url,
    params=params,
    timeout=30
)

print()
print("===================================")
print("FORWARD RESULTS UPDATE")
print("===================================")

print(
    "HTTP status:",
    response.status_code
)

if not response.ok:

    print(
        response.text[:3000]
    )

    raise SystemExit


fixtures = response.json()


# ==================================================
# INDEXERA FIXTURES
# ==================================================

fixture_lookup = {
    str(
        fixture.get(
            "fixtureId"
        )
    ):
    fixture

    for fixture in fixtures
}


# ==================================================
# RESULTAT
# ==================================================

rows = []


for _, bet in bets.iterrows():

    fixture_id = str(
        bet["fixture_id"]
    )

    fixture = fixture_lookup.get(
        fixture_id
    )

    row = bet.to_dict()

    row["home_goals"] = None
    row["away_goals"] = None
    row["actual_btts"] = None
    row["won"] = None
    row["profit"] = None
    row["result_status"] = "NOT_FINISHED"


    if fixture is None:

        row["result_status"] = (
            "FIXTURE_NOT_FOUND"
        )

        rows.append(row)

        continue


    # ==================================================
    # FÖRSÖK LÄSA RESULTAT
    # ==================================================

    home_score = fixture.get(
        "participant1Score"
    )

    away_score = fixture.get(
        "participant2Score"
    )

    status_id = fixture.get(
        "statusId"
    )


    # Vi accepterar resultat först när båda
    # målsiffrorna faktiskt finns.
    if (
        home_score is not None
        and
        away_score is not None
    ):

        try:

            home_goals = int(
                home_score
            )

            away_goals = int(
                away_score
            )

        except (
            ValueError,
            TypeError
        ):

            rows.append(row)

            continue


        actual_btts = int(
            home_goals > 0
            and
            away_goals > 0
        )


        if bet["bet_side"] == "YES":

            won = (
                actual_btts == 1
            )

        elif bet["bet_side"] == "NO":

            won = (
                actual_btts == 0
            )

        else:

            won = False


        odds = bet[
            "best_bet_odds"
        ]


        if won:

            profit = (
                odds - 1
            )

        else:

            profit = -1


        row["home_goals"] = (
            home_goals
        )

        row["away_goals"] = (
            away_goals
        )

        row["actual_btts"] = (
            actual_btts
        )

        row["won"] = int(
            won
        )

        row["profit"] = (
            profit
        )

        row["result_status"] = (
            "FINISHED"
        )


    rows.append(row)


results = pd.DataFrame(
    rows
)


# ==================================================
# SPARA
# ==================================================

results.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==================================================
# VISA
# ==================================================

print()
print(
    "Frysta bets:",
    len(results)
)

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
            "result_status"
        ]
    ].to_string(
        index=False
    )
)


# ==================================================
# SUMMERING
# ==================================================

finished = results[
    results["result_status"]
    ==
    "FINISHED"
].copy()


print()
print("===================================")
print("RESULTAT")
print("===================================")

print(
    "Färdigspelade bets:",
    len(finished)
)


if len(finished) > 0:

    print(
        "Wins:",
        finished[
            "won"
        ].sum()
    )

    print(
        "Profit:",
        round(
            finished[
                "profit"
            ].sum(),
            2
        ),
        "units"
    )

    print(
        "ROI:",
        round(
            finished[
                "profit"
            ].mean()
            * 100,
            2
        ),
        "%"
    )


print()
print(
    "Sparad till:",
    OUTPUT_FILE
)