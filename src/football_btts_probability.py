import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss


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
# FAKTISKT BTTS
# ==================================================

odds["actual_btts"] = (
    (odds["FTHG"] > 0)
    &
    (odds["FTAG"] > 0)
).astype(int)


# ==================================================
# BTTS MARKNADENS NO-VIG PROBABILITY
# ==================================================

yes_raw = (
    1 / odds["bts_yes_open"]
)

no_raw = (
    1 / odds["bts_no_open"]
)

odds["market_btts_probability"] = (
    yes_raw
    /
    (yes_raw + no_raw)
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
# FOOTBALL FEATURES
# ==================================================

data["home_scoring_strength"] = (
    data["home_attack"]
    *
    data["away_defense"]
)

data["away_scoring_strength"] = (
    data["away_attack"]
    *
    data["home_defense"]
)

data["combined_scoring_strength"] = (
    data["home_scoring_strength"]
    +
    data["away_scoring_strength"]
)

data["scoring_balance"] = (
    1
    -
    abs(
        data["home_scoring_strength"]
        -
        data["away_scoring_strength"]
    )
)


FEATURES = [
    "home_attack",
    "home_defense",
    "away_attack",
    "away_defense",

    "home_scoring_strength",
    "away_scoring_strength",
    "combined_scoring_strength",
    "scoring_balance"
]


# ==================================================
# KOMPLETTA MATCHER
# ==================================================

required = (
    FEATURES
    +
    [
        "actual_btts",
        "market_btts_probability",
        "bts_yes_open",
        "bts_no_open",
        "bts_yes_close",
        "bts_no_close"
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
            "logistic",
            LogisticRegression(
                max_iter=2000,
                C=1.0
            )
        )
    ])


# ==================================================
# WALK FORWARD
# ==================================================

all_predictions = []


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
        train[FEATURES],
        train["actual_btts"]
    )

    probability = (
        model.predict_proba(
            test[FEATURES]
        )[:, 1]
    )

    test = test.copy()

    test[
        "football_probability"
    ] = probability

    test[
        "probability_edge"
    ] = (
        test["football_probability"]
        -
        test["market_btts_probability"]
    )

    test[
        "test_season"
    ] = test_season

    all_predictions.append(
        test
    )


predictions = pd.concat(
    all_predictions,
    ignore_index=True
)


# ==================================================
# BRIER
# ==================================================

football_brier = brier_score_loss(
    predictions["actual_btts"],
    predictions["football_probability"]
)

market_brier = brier_score_loss(
    predictions["actual_btts"],
    predictions["market_btts_probability"]
)


print()
print("===================================")
print("FOOTBALL BTTS PROBABILITY MODEL")
print("===================================")

print(
    "Matcher:",
    len(predictions)
)

print(
    "Faktisk BTTS:",
    round(
        predictions["actual_btts"].mean(),
        4
    )
)

print(
    "Football model avg:",
    round(
        predictions["football_probability"].mean(),
        4
    )
)

print(
    "Market avg:",
    round(
        predictions["market_btts_probability"].mean(),
        4
    )
)

print()

print(
    "Football Brier:",
    round(
        football_brier,
        6
    )
)

print(
    "Market Brier:",
    round(
        market_brier,
        6
    )
)

print(
    "Football minus Market:",
    round(
        football_brier
        -
        market_brier,
        6
    )
)


# ==================================================
# PER SÄSONG
# ==================================================

print()
print("===================================")
print("RESULTAT PER SÄSONG")
print("===================================")

for season, group in predictions.groupby(
    "test_season"
):

    fb = brier_score_loss(
        group["actual_btts"],
        group["football_probability"]
    )

    mk = brier_score_loss(
        group["actual_btts"],
        group["market_btts_probability"]
    )

    print()
    print(season)

    print(
        "Matcher:",
        len(group)
    )

    print(
        "Football avg:",
        round(
            group["football_probability"].mean(),
            4
        )
    )

    print(
        "Market avg:",
        round(
            group["market_btts_probability"].mean(),
            4
        )
    )

    print(
        "Actual:",
        round(
            group["actual_btts"].mean(),
            4
        )
    )

    print(
        "Football Brier:",
        round(
            fb,
            6
        )
    )

    print(
        "Market Brier:",
        round(
            mk,
            6
        )
    )


# ==================================================
# EDGE ANALYS
# ==================================================

print()
print("===================================")
print("EDGE ANALYS")
print("===================================")

bins = [
    -1,
    -0.10,
    -0.05,
    -0.025,
    0,
    0.025,
    0.05,
    0.10,
    1
]

labels = [
    "< -10 pp",
    "-10 to -5 pp",
    "-5 to -2.5 pp",
    "-2.5 to 0 pp",
    "0 to 2.5 pp",
    "2.5 to 5 pp",
    "5 to 10 pp",
    "> 10 pp"
]

predictions["edge_bin"] = pd.cut(
    predictions["probability_edge"],
    bins=bins,
    labels=labels,
    include_lowest=True
)


edge_summary = (
    predictions.groupby(
        "edge_bin",
        observed=True
    )
    .agg(
        matches=(
            "actual_btts",
            "size"
        ),
        football_probability=(
            "football_probability",
            "mean"
        ),
        market_probability=(
            "market_btts_probability",
            "mean"
        ),
        actual_btts=(
            "actual_btts",
            "mean"
        )
    )
)

print(
    edge_summary.to_string()
)


# ==================================================
# SPARA
# ==================================================

OUTPUT = (
    "data/processed/"
    "football_btts_predictions.csv"
)

predictions.to_csv(
    OUTPUT,
    index=False
)

print()
print(
    "Sparad till:",
    OUTPUT
)