import pandas as pd
import numpy as np


# ==================================================
# INSTÄLLNINGAR
# ==================================================

FILE = "data/raw/premier_league_2015_2025.csv"

WINDOW = 20


# ==================================================
# LADDA DATA
# ==================================================

df = pd.read_csv(FILE)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    "date"
).reset_index(drop=True)


# ==================================================
# LAGRING
# ==================================================

rows = []


# ==================================================
# LOOP GENOM MATCHER
# ==================================================

for index, match in df.iterrows():

    current_date = match["date"]

    home_team = match["home_team"]
    away_team = match["away_team"]


    # ----------------------------------------------
    # ENDAST HISTORISKA MATCHER
    # ----------------------------------------------

    history = df[
        df["date"] < current_date
    ]


    # ----------------------------------------------
    # LIGAMEDEL FRAM TILL MATCHEN
    # ----------------------------------------------

    if len(history) > 0:

        avg_home_goals = (
            history["home_goals"].mean()
        )

        avg_away_goals = (
            history["away_goals"].mean()
        )

    else:

        avg_home_goals = np.nan
        avg_away_goals = np.nan


    # ----------------------------------------------
    # HEMMALAGETS SENASTE HEMMAMATCHER
    # ----------------------------------------------

    home_history = (
        history[
            history["home_team"]
            == home_team
        ]
        .tail(WINDOW)
    )


    # ----------------------------------------------
    # BORTALAGETS SENASTE BORTAMATCHER
    # ----------------------------------------------

    away_history = (
        history[
            history["away_team"]
            == away_team
        ]
        .tail(WINDOW)
    )


    # ----------------------------------------------
    # DEFAULT
    # ----------------------------------------------

    home_attack = np.nan
    home_defense = np.nan

    away_attack = np.nan
    away_defense = np.nan


    # ----------------------------------------------
    # HOME STRENGTH
    # ----------------------------------------------

    if (
        len(home_history) > 0
        and
        avg_home_goals > 0
        and
        avg_away_goals > 0
    ):

        home_scored = (
            home_history[
                "home_goals"
            ].mean()
        )

        home_conceded = (
            home_history[
                "away_goals"
            ].mean()
        )

        home_attack = (
            home_scored
            /
            avg_home_goals
        )

        home_defense = (
            home_conceded
            /
            avg_away_goals
        )


    # ----------------------------------------------
    # AWAY STRENGTH
    # ----------------------------------------------

    if (
        len(away_history) > 0
        and
        avg_home_goals > 0
        and
        avg_away_goals > 0
    ):

        away_scored = (
            away_history[
                "away_goals"
            ].mean()
        )

        away_conceded = (
            away_history[
                "home_goals"
            ].mean()
        )

        away_attack = (
            away_scored
            /
            avg_away_goals
        )

        away_defense = (
            away_conceded
            /
            avg_home_goals
        )


    # ----------------------------------------------
    # SPARA
    # ----------------------------------------------

    rows.append({

        "date": current_date,

        "season": match["season"],

        "home_team": home_team,
        "away_team": away_team,

        "home_attack": home_attack,
        "home_defense": home_defense,

        "away_attack": away_attack,
        "away_defense": away_defense,

        "home_matches_used":
            len(home_history),

        "away_matches_used":
            len(away_history)
    })


# ==================================================
# DATAFRAME
# ==================================================

features = pd.DataFrame(rows)


# ==================================================
# SPARA
# ==================================================

OUTPUT = (
    "data/processed/"
    "attack_defense_features.csv"
)

features.to_csv(
    OUTPUT,
    index=False
)


# ==================================================
# RESULTAT
# ==================================================

print()
print("===================================")
print("ATTACK / DEFENSE FEATURES")
print("===================================")

print(
    "Matcher:",
    len(features)
)

print()

print(
    features.tail(20).to_string(
        index=False
    )
)


print()
print("MATCHER MED FULL 20-MATCHERS HISTORIK")
print("-----------------------------------")

full_history = (
    (features["home_matches_used"] >= WINDOW)
    &
    (features["away_matches_used"] >= WINDOW)
)

print(
    full_history.sum()
)

print(
    "Andel:",
    round(
        full_history.mean()
        * 100,
        2
    ),
    "%"
)

print()
print(
    "Sparad till:",
    OUTPUT
)