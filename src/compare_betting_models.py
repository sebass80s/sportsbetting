import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ==================================================
# INSTÄLLNINGAR
# ==================================================

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

SIGNAL_THRESHOLD = 0.005


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
    +
    FOOTBALL_FEATURES
)


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
    home +
    draw +
    away
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
# RESULTAT + TARGET
# ==================================================

odds["actual_btts"] = (
    (odds["FTHG"] > 0)
    &
    (odds["FTAG"] > 0)
).astype(int)

odds["movement"] = (
    odds["btts_close"]
    -
    odds["btts_open"]
)


# ==================================================
# LAGNAMN
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
    odds["HomeTeam"].replace(NAME_MAP)
)

odds["away_team_merge"] = (
    odds["AwayTeam"].replace(NAME_MAP)
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
# FUNKTION FÖR BETTING
# ==================================================

def run_strategy(
    feature_list,
    model_name
):

    all_bets = []

    for test_season in TEST_SEASONS:

        train = data[
            data["Season"]
            <
            test_season
        ].copy()

        test = data[
            data["Season"]
            ==
            test_season
        ].copy()

        if len(test) == 0:
            continue

        model = build_model()

        model.fit(
            train[feature_list],
            train["movement"]
        )

        prediction = model.predict(
            test[feature_list]
        )

        test = test.copy()

        test[
            "predicted_movement"
        ] = prediction

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
            test["bet_side"]
            != "NONE"
        ].copy()

        if len(bets) == 0:
            continue

        bets["bet_odds"] = np.where(
            bets["bet_side"] == "YES",
            bets["bts_yes_open"],
            bets["bts_no_open"]
        )

        bets["close_odds"] = np.where(
            bets["bet_side"] == "YES",
            bets["bts_yes_close"],
            bets["bts_no_close"]
        )

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

        # ------------------------------------------
        # NO-VIG CLV
        # ------------------------------------------

        open_yes_raw = (
            1 / bets["bts_yes_open"]
        )

        open_no_raw = (
            1 / bets["bts_no_open"]
        )

        open_total = (
            open_yes_raw
            +
            open_no_raw
        )

        close_yes_raw = (
            1 / bets["bts_yes_close"]
        )

        close_no_raw = (
            1 / bets["bts_no_close"]
        )

        close_total = (
            close_yes_raw
            +
            close_no_raw
        )

        open_yes_fair = (
            open_yes_raw
            /
            open_total
        )

        open_no_fair = (
            open_no_raw
            /
            open_total
        )

        close_yes_fair = (
            close_yes_raw
            /
            close_total
        )

        close_no_fair = (
            close_no_raw
            /
            close_total
        )

        bets["novig_clv"] = np.where(
            bets["bet_side"] == "YES",

            close_yes_fair
            -
            open_yes_fair,

            close_no_fair
            -
            open_no_fair
        )

        bets["model"] = model_name
        bets["test_season"] = test_season

        all_bets.append(bets)

    return pd.concat(
        all_bets,
        ignore_index=True
    )


# ==================================================
# KÖR BÅDA MODELLERNA
# ==================================================

market_bets = run_strategy(
    MARKET_FEATURES,
    "MARKET ONLY"
)

football_bets = run_strategy(
    ALL_FEATURES,
    "MARKET + ATTACK/DEFENSE"
)


# ==================================================
# SAMMANFATTNING
# ==================================================

def summarize(name, bets):

    print()
    print(name)
    print("-----------------------------------")

    print(
        "Bets:",
        len(bets)
    )

    print(
        "ROI:",
        round(
            bets["profit"].mean()
            * 100,
            2
        ),
        "%"
    )

    print(
        "No-vig CLV:",
        round(
            bets["novig_clv"].mean()
            * 100,
            3
        ),
        "pp"
    )

    print(
        "Beat close:",
        round(
            (
                bets["novig_clv"] > 0
            ).mean()
            * 100,
            2
        ),
        "%"
    )

    print(
        "YES bets:",
        (
            bets["bet_side"]
            == "YES"
        ).sum()
    )

    print(
        "NO bets:",
        (
            bets["bet_side"]
            == "NO"
        ).sum()
    )


print()
print("===================================")
print("BETTING MODEL COMPARISON")
print("===================================")

summarize(
    "MODEL A - MARKET ONLY",
    market_bets
)

summarize(
    "MODEL B - MARKET + ATTACK/DEFENSE",
    football_bets
)


# ==================================================
# SÄSONGSVIS
# ==================================================

print()
print("===================================")
print("CLV PER SÄSONG")
print("===================================")

combined = pd.concat(
    [
        market_bets,
        football_bets
    ],
    ignore_index=True
)

season_summary = (
    combined.groupby(
        [
            "model",
            "test_season"
        ]
    )
    .agg(
        bets=(
            "novig_clv",
            "size"
        ),
        roi=(
            "profit",
            "mean"
        ),
        clv=(
            "novig_clv",
            "mean"
        )
    )
)

season_summary["roi"] *= 100
season_summary["clv"] *= 100

print(
    season_summary.to_string()
)


# ==================================================
# SPARA
# ==================================================

market_bets.to_csv(
    "data/processed/"
    "market_only_bets.csv",
    index=False
)

football_bets.to_csv(
    "data/processed/"
    "football_model_bets.csv",
    index=False
)