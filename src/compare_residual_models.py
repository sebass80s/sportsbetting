import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error


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

TRAIN_SEASONS = [
    "2019-2020",
    "2020-2021",
    "2021-2022"
]

VALIDATION_SEASON = "2022-2023"


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
    open_yes /
    (open_yes + open_no)
)


close_yes = 1 / odds["bts_yes_close"]
close_no = 1 / odds["bts_no_close"]

odds["btts_close"] = (
    close_yes /
    (close_yes + close_no)
)


# ==================================================
# OVER / UNDER OPEN
# ==================================================

over = 1 / odds["over_2.5_open"]
under = 1 / odds["under_2.5_open"]

odds["over_open"] = (
    over /
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
    1 -
    abs(
        odds["home_open_prob"]
        -
        odds["away_open_prob"]
    )
)


# ==================================================
# TARGET = CLOSING MOVEMENT
# ==================================================

odds["movement"] = (
    odds["btts_close"]
    -
    odds["btts_open"]
)


# ==================================================
# STANDARDISERA LAGNAMN FÖR MERGE
# ==================================================

NAME_MAP = {

    "Manchester United":
        "Man United",

    "Manchester City":
        "Man City",

    "Newcastle Utd":
        "Newcastle",

    "Nottingham":
        "Nott'm Forest",

    "Sheffield Utd":
        "Sheffield United",

    "Tottenham Hotspur":
        "Tottenham",

    "West Bromwich Albion":
        "West Brom",

    "Wolverhampton":
        "Wolves"
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
# FEATURES
# ==================================================

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
# BEHÅLL KOMPLETTA MATCHER
# ==================================================

required = (
    ALL_FEATURES
    +
    [
        "movement",
        "btts_close"
    ]
)

data = data.dropna(
    subset=required
).copy()


# ==================================================
# KRÄV FULL 20-MATCHERS HISTORIK
# ==================================================

data = data[
    (data["home_matches_used"] >= 20)
    &
    (data["away_matches_used"] >= 20)
].copy()


# ==================================================
# TRAIN / VALIDATION
# ==================================================

train = data[
    data["Season"].isin(
        TRAIN_SEASONS
    )
].copy()

validation = data[
    data["Season"]
    ==
    VALIDATION_SEASON
].copy()


print()
print("===================================")
print("DATA")
print("===================================")

print(
    "Totalt användbara matcher:",
    len(data)
)

print(
    "Train:",
    len(train)
)

print(
    "Validation:",
    len(validation)
)


# ==================================================
# BASELINE
#
# Predikterad movement = 0
# dvs closing = opening
# ==================================================

actual = validation[
    "movement"
].values

baseline = np.zeros(
    len(validation)
)

baseline_mse = mean_squared_error(
    actual,
    baseline
)

baseline_mae = mean_absolute_error(
    actual,
    baseline
)


# ==================================================
# FUNKTION FÖR MODELL
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
# MODELL A
# MARKET ONLY
# ==================================================

market_model = build_model()

market_model.fit(

    train[
        MARKET_FEATURES
    ],

    train[
        "movement"
    ]
)


market_prediction = (
    market_model.predict(
        validation[
            MARKET_FEATURES
        ]
    )
)


# ==================================================
# MODELL B
# MARKET + FOOTBALL
# ==================================================

football_model = build_model()

football_model.fit(

    train[
        ALL_FEATURES
    ],

    train[
        "movement"
    ]
)


football_prediction = (
    football_model.predict(
        validation[
            ALL_FEATURES
        ]
    )
)


# ==================================================
# UTVÄRDERING
# ==================================================

def evaluate(
    name,
    prediction
):

    mse = mean_squared_error(
        actual,
        prediction
    )

    mae = mean_absolute_error(
        actual,
        prediction
    )

    direction = np.mean(

        np.sign(actual)
        ==
        np.sign(prediction)
    )

    correlation = np.corrcoef(
        actual,
        prediction
    )[0, 1]

    print()
    print(name)
    print("-----------------------------------")

    print(
        "MSE:",
        round(
            mse,
            8
        )
    )

    print(
        "MAE:",
        round(
            mae,
            6
        )
    )

    print(
        "Direction accuracy:",
        round(
            direction,
            4
        )
    )

    print(
        "Correlation:",
        round(
            correlation,
            4
        )
    )

    return mse, mae


print()
print("===================================")
print("RESIDUAL MODEL COMPARISON")
print("===================================")

print()
print("BASELINE")
print("-----------------------------------")

print(
    "MSE:",
    round(
        baseline_mse,
        8
    )
)

print(
    "MAE:",
    round(
        baseline_mae,
        6
    )
)


market_mse, market_mae = evaluate(
    "MODEL A - MARKET ONLY",
    market_prediction
)


football_mse, football_mae = evaluate(
    "MODEL B - MARKET + ATTACK/DEFENSE",
    football_prediction
)


# ==================================================
# DIREKT JÄMFÖRELSE
# ==================================================

print()
print("===================================")
print("ATTACK/DEFENSE MARGINAL VALUE")
print("===================================")

print(
    "MSE improvement vs Market:",
    round(
        market_mse
        -
        football_mse,
        8
    )
)

print(
    "MAE improvement vs Market:",
    round(
        market_mae
        -
        football_mae,
        6
    )
)


# ==================================================
# SIGNALSTYRKA FÖR FOOTBALL MODELL
# ==================================================

print()
print("===================================")
print("MODEL B SIGNALSTYRKA")
print("===================================")


thresholds = [
    0.005,
    0.01,
    0.02,
    0.03
]


for threshold in thresholds:

    mask = (
        abs(
            football_prediction
        )
        >= threshold
    )

    count = mask.sum()

    if count == 0:
        continue

    actual_subset = (
        actual[mask]
    )

    predicted_subset = (
        football_prediction[
            mask
        ]
    )

    direction = np.mean(

        np.sign(
            actual_subset
        )
        ==
        np.sign(
            predicted_subset
        )
    )

    signed_movement = np.mean(

        actual_subset
        *
        np.sign(
            predicted_subset
        )
    )

    print(
        f"Signal >= {threshold*100:.1f} pp",
        "| matcher:",
        count,
        "| direction:",
        round(
            direction,
            4
        ),
        "| actual movement:",
        round(
            signed_movement,
            4
        )
    )


# ==================================================
# KOEFFICIENTER MODEL B
# ==================================================

ridge = (
    football_model
    .named_steps["ridge"]
)

print()
print("===================================")
print("STANDARDISERADE KOEFFICIENTER")
print("===================================")

for feature, coef in zip(
    ALL_FEATURES,
    ridge.coef_
):

    print(
        feature,
        ":",
        round(
            coef,
            6
        )
    )