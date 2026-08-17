import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ODDS_FILE = (
    "external_football_data/data/england/"
    "premier-league.csv"
)

FEATURE_FILE = (
    "data/processed/"
    "attack_defense_features.csv"
)

TEST_SEASONS = [
    "2022-2023",
    "2023-2024",
    "2024-2025"
]

MARKET_FEATURES = [
    "btts_open",
    "over_open",
    "home_open_prob",
    "draw_open_prob",
    "balance"
]

FOOTBALL_FEATURES = [
    "home_attack",
    "home_defense",
    "away_attack",
    "away_defense"
]

ALL_FEATURES = (
    MARKET_FEATURES
    + FOOTBALL_FEATURES
)

SIGNAL_THRESHOLD = 0.005


# ==================================================
# LADDA DATA
# ==================================================

odds = pd.read_csv(ODDS_FILE)
football = pd.read_csv(FEATURE_FILE)

odds["Date"] = pd.to_datetime(
    odds["Date"]
).dt.normalize()

football["date"] = pd.to_datetime(
    football["date"]
).dt.normalize()


# ==================================================
# NO-VIG BTTS OPEN / CLOSE
# ==================================================

open_yes = 1 / odds["bts_yes_open"]
open_no = 1 / odds["bts_no_open"]

odds["btts_open"] = (
    open_yes
    /
    (open_yes + open_no)
)

close_yes = 1 / odds["bts_yes_close"]
close_no = 1 / odds["bts_no_close"]

odds["btts_close"] = (
    close_yes
    /
    (close_yes + close_no)
)


# ==================================================
# O/U OPEN
# ==================================================

over = 1 / odds["over_2.5_open"]
under = 1 / odds["under_2.5_open"]

odds["over_open"] = (
    over
    /
    (over + under)
)


# ==================================================
# 1X2 OPEN
# ==================================================

home = 1 / odds["home_open"]
draw = 1 / odds["draw_open"]
away = 1 / odds["away_open"]

total = (
    home
    + draw
    + away
)

odds["home_open_prob"] = (
    home / total
)

odds["draw_open_prob"] = (
    draw / total
)

odds["away_open_prob"] = (
    away / total
)

odds["balance"] = (
    1
    -
    abs(
        odds["home_open_prob"]
        -
        odds["away_open_prob"]
    )
)


# ==================================================
# FAKTISKT BTTS
# ==================================================

odds["actual_btts"] = (
    (odds["FTHG"] > 0)
    &
    (odds["FTAG"] > 0)
).astype(int)


# ==================================================
# TARGET: OPEN -> CLOSE MOVEMENT
# ==================================================

odds["movement"] = (
    odds["btts_close"]
    -
    odds["btts_open"]
)


# ==================================================
# STANDARDISERA LAGNAMN
# ==================================================

NAME_MAP = {
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Newcastle Utd": "Newcastle",
    "Nottingham": "Nott'm Forest",
    "Sheffield Utd": "Sheffield United",
    "Tottenham Hotspur": "Tottenham",
    "West Bromwich Albion": "West Brom",
    "Wolverhampton": "Wolves"
}

odds["home_team_merge"] = (
    odds["HomeTeam"]
    .replace(NAME_MAP)
)

odds["away_team_merge"] = (
    odds["AwayTeam"]
    .replace(NAME_MAP)
)


# ==================================================
# MERGE
# ==================================================

data = odds.merge(
    football,
    left_on=[
        "Date",
        "home_team_merge",
        "away_team_merge"
    ],
    right_on=[
        "date",
        "home_team",
        "away_team"
    ],
    how="left"
)


# ==================================================
# KOMPLETTA MATCHER
# ==================================================

required = (
    ALL_FEATURES
    + [
        "movement",
        "btts_open",
        "btts_close",
        "bts_yes_open",
        "bts_no_open",
        "bts_yes_close",
        "bts_no_close",
        "actual_btts"
    ]
)

data = data.dropna(
    subset=required
).copy()

data = data[
    (data["home_matches_used"] >= 20)
    &
    (data["away_matches_used"] >= 20)
].copy()


# ==================================================
# MODELL
# ==================================================

def build_model():

    return Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "ridge",
            Ridge(
                alpha=1.0
            )
        )
    ])


# ==================================================
# WALK FORWARD
# ==================================================

all_bets = []

for test_season in TEST_SEASONS:

    train = data[
        data["Season"] < test_season
    ].copy()

    test = data[
        data["Season"] == test_season
    ].copy()

    if len(test) == 0:
        continue

    model = build_model()

    model.fit(
        train[ALL_FEATURES],
        train["movement"]
    )

    predicted_movement = model.predict(
        test[ALL_FEATURES]
    )

    test = test.copy()

    test["predicted_movement"] = (
        predicted_movement
    )

    test["predicted_close"] = (
        test["btts_open"]
        +
        test["predicted_movement"]
    )

    test["predicted_close"] = (
        test["predicted_close"]
        .clip(0.01, 0.99)
    )


    # ==================================================
    # BETTING SIGNAL
    # ==================================================

    test["bet_side"] = "NONE"

    test.loc[
        test["predicted_movement"]
        >= SIGNAL_THRESHOLD,
        "bet_side"
    ] = "YES"

    test.loc[
        test["predicted_movement"]
        <= -SIGNAL_THRESHOLD,
        "bet_side"
    ] = "NO"


    bets = test[
        test["bet_side"] != "NONE"
    ].copy()


    # ==================================================
    # OPENING ODDS SOM VI SPELAR TILL
    # ==================================================

    bets["bet_odds"] = np.where(
        bets["bet_side"] == "YES",
        bets["bts_yes_open"],
        bets["bts_no_open"]
    )


    # ==================================================
    # CLOSING ODDS PÅ SAMMA SIDA
    # ==================================================

    bets["close_odds_same_side"] = (
        np.where(
            bets["bet_side"] == "YES",
            bets["bts_yes_close"],
            bets["bts_no_close"]
        )
    )


    # ==================================================
    # VINST / FÖRLUST
    # ==================================================

    bets["won"] = np.where(
        bets["bet_side"] == "YES",
        bets["actual_btts"] == 1,
        bets["actual_btts"] == 0
    )


    bets["profit"] = np.where(
        bets["won"],
        bets["bet_odds"] - 1,
        -1
    )


    # ==================================================
    # CLV
    #
    # Positivt = vårt opening-odds var bättre
    # än closing-oddset.
    # ==================================================

    bets["clv"] = (
        bets["bet_odds"]
        /
        bets["close_odds_same_side"]
        - 1
    )


    # ==================================================
    # SLOG VI CLOSING LINE?
    # ==================================================

    bets["beat_close"] = (
        bets["bet_odds"]
        >
        bets["close_odds_same_side"]
    )


    bets["test_season"] = (
        test_season
    )

    all_bets.append(bets)


# ==================================================
# SLÅ IHOP ALLA BETS
# ==================================================

bets = pd.concat(
    all_bets,
    ignore_index=True
)


# ==================================================
# TOTALT RESULTAT
# ==================================================

print()
print("===================================")
print("WALK-FORWARD BETTING")
print("===================================")

print(
    "Signal threshold:",
    SIGNAL_THRESHOLD
)

print(
    "Antal bets:",
    len(bets)
)

print(
    "Wins:",
    bets["won"].sum()
)

print(
    "Hit rate:",
    round(
        bets["won"].mean(),
        4
    )
)

print(
    "Genomsnittligt odds:",
    round(
        bets["bet_odds"].mean(),
        3
    )
)

print(
    "Profit:",
    round(
        bets["profit"].sum(),
        2
    ),
    "units"
)

print(
    "ROI:",
    round(
        (
            bets["profit"].sum()
            /
            len(bets)
        )
        * 100,
        2
    ),
    "%"
)


# ==================================================
# CLV
# ==================================================

print()
print("===================================")
print("CLOSING LINE VALUE")
print("===================================")

print(
    "Genomsnittlig CLV:",
    round(
        bets["clv"].mean()
        * 100,
        2
    ),
    "%"
)

print(
    "Median CLV:",
    round(
        bets["clv"].median()
        * 100,
        2
    ),
    "%"
)

print(
    "Slog closing line:",
    round(
        bets["beat_close"].mean()
        * 100,
        2
    ),
    "%"
)


# ==================================================
# PER SÄSONG
# ==================================================

print()
print("===================================")
print("RESULTAT PER SÄSONG")
print("===================================")

for season, group in bets.groupby(
    "test_season"
):

    print()
    print(season)

    print(
        "Bets:",
        len(group)
    )

    print(
        "ROI:",
        round(
            (
                group["profit"].sum()
                /
                len(group)
            )
            * 100,
            2
        ),
        "%"
    )

    print(
        "CLV:",
        round(
            group["clv"].mean()
            * 100,
            2
        ),
        "%"
    )

    print(
        "Beat close:",
        round(
            group["beat_close"].mean()
            * 100,
            2
        ),
        "%"
    )


# ==================================================
# YES / NO SEPARAT
# ==================================================

print()
print("===================================")
print("RESULTAT PER BET-SIDA")
print("===================================")

for side, group in bets.groupby(
    "bet_side"
):

    print()
    print(side)

    print(
        "Bets:",
        len(group)
    )

    print(
        "ROI:",
        round(
            (
                group["profit"].sum()
                /
                len(group)
            )
            * 100,
            2
        ),
        "%"
    )

    print(
        "CLV:",
        round(
            group["clv"].mean()
            * 100,
            2
        ),
        "%"
    )


# ==================================================
# SPARA
# ==================================================

OUTPUT = (
    "data/processed/"
    "walk_forward_bets.csv"
)

bets.to_csv(
    OUTPUT,
    index=False
)

print()
print(
    "Sparad till:",
    OUTPUT
)