import pandas as pd
import numpy as np
from pathlib import Path


# ==================================================
# INSTÄLLNINGAR
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

HISTORY_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "premier_league_2015_2026.csv"
)

FIXTURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "forward"
    / "upcoming_pl.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "forward"
    / "forward_features.csv"
)

WINDOW = 20


# ==================================================
# LADDA DATA
# ==================================================

history = pd.read_csv(HISTORY_FILE)
fixtures = pd.read_csv(FIXTURE_FILE)

history["date"] = pd.to_datetime(history["date"])
fixtures["start_time"] = pd.to_datetime(
    fixtures["start_time"],
    utc=True
)


# ==================================================
# STANDARDISERA API-LAGNAMN
# ==================================================

NAME_MAP = {
    "Arsenal FC": "Arsenal",
    "Aston Villa": "Aston Villa",
    "AFC Bournemouth": "Bournemouth",
    "Brentford FC": "Brentford",
    "Brighton & Hove Albion": "Brighton",
    "Chelsea FC": "Chelsea",
    "Crystal Palace": "Crystal Palace",
    "Everton FC": "Everton",
    "Fulham FC": "Fulham",
    "Hull City": "Hull",
    "Ipswich Town": "Ipswich",
    "Leeds United": "Leeds",
    "Liverpool FC": "Liverpool",
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Sunderland AFC": "Sunderland",
    "Tottenham Hotspur": "Tottenham",
    "Coventry City": "Coventry"
}

fixtures["home_team_model"] = (
    fixtures["home_team"]
    .replace(NAME_MAP)
)

fixtures["away_team_model"] = (
    fixtures["away_team"]
    .replace(NAME_MAP)
)


# ==================================================
# LIGANS HISTORISKA GENOMSNITT
# ==================================================

avg_home_goals = history["home_goals"].mean()
avg_away_goals = history["away_goals"].mean()


print()
print("===================================")
print("FORWARD FEATURES")
print("===================================")

print(
    "Historiskt hemmamålssnitt:",
    round(avg_home_goals, 4)
)

print(
    "Historiskt bortamålssnitt:",
    round(avg_away_goals, 4)
)


# ==================================================
# BYGG FEATURES
# ==================================================

rows = []


for _, fixture in fixtures.iterrows():

    home_team = fixture["home_team_model"]
    away_team = fixture["away_team_model"]

    # ----------------------------------------------
    # SENASTE 20 HEMMAMATCHER
    # ----------------------------------------------

    home_history = (
        history[
            history["home_team"] == home_team
        ]
        .sort_values("date")
        .tail(WINDOW)
    )

    # ----------------------------------------------
    # SENASTE 20 BORTAMATCHER
    # ----------------------------------------------

    away_history = (
        history[
            history["away_team"] == away_team
        ]
        .sort_values("date")
        .tail(WINDOW)
    )

    home_n = len(home_history)
    away_n = len(away_history)

    home_attack = np.nan
    home_defense = np.nan
    away_attack = np.nan
    away_defense = np.nan

    if home_n >= WINDOW:

        home_attack = (
            home_history["home_goals"].mean()
            /
            avg_home_goals
        )

        home_defense = (
            home_history["away_goals"].mean()
            /
            avg_away_goals
        )

    if away_n >= WINDOW:

        away_attack = (
            away_history["away_goals"].mean()
            /
            avg_away_goals
        )

        away_defense = (
            away_history["home_goals"].mean()
            /
            avg_home_goals
        )

    # ----------------------------------------------
    # STATUS
    # ----------------------------------------------

    if (
        home_n >= WINDOW
        and away_n >= WINDOW
    ):
        status = "READY"
    else:
        status = "NO_PREDICTION"

    rows.append({

        "fixture_id":
            fixture["fixture_id"],

        "start_time":
            fixture["start_time"],

        "home_team":
            fixture["home_team"],

        "away_team":
            fixture["away_team"],

        "home_team_model":
            home_team,

        "away_team_model":
            away_team,

        "home_attack":
            home_attack,

        "home_defense":
            home_defense,

        "away_attack":
            away_attack,

        "away_defense":
            away_defense,

        "home_matches_used":
            home_n,

        "away_matches_used":
            away_n,

        "status":
            status
    })


# ==================================================
# DATAFRAME
# ==================================================

features = pd.DataFrame(rows)


# ==================================================
# SPARA
# ==================================================

features.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==================================================
# VISA
# ==================================================

print()
print(
    features[
        [
            "start_time",
            "home_team",
            "away_team",
            "home_matches_used",
            "away_matches_used",
            "home_attack",
            "home_defense",
            "away_attack",
            "away_defense",
            "status"
        ]
    ].to_string(index=False)
)

print()
print("===================================")
print("SAMMANFATTNING")
print("===================================")

print(
    features["status"].value_counts()
)

print()
print(
    "Sparad till:",
    OUTPUT_FILE
)