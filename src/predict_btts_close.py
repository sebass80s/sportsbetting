import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# ==================================================
# INSTÄLLNINGAR
# ==================================================

FILE = (
    "external_football_data/data/england/"
    "premier-league.csv"
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

df = pd.read_csv(FILE)

df["Date"] = pd.to_datetime(df["Date"])


# ==================================================
# KRÄV ALLA OPENING + BTTS CLOSE
# ==================================================

required = [
    "bts_yes_open",
    "bts_no_open",
    "bts_yes_close",
    "bts_no_close",

    "over_2.5_open",
    "under_2.5_open",

    "home_open",
    "draw_open",
    "away_open"
]

df = df.dropna(
    subset=required
).copy()


# ==================================================
# FUNKTION: NO-VIG TVÅVÄGSMARKNAD
# ==================================================

def no_vig_two_way(odds_a, odds_b):

    raw_a = 1 / odds_a
    raw_b = 1 / odds_b

    total = raw_a + raw_b

    return (
        raw_a / total,
        raw_b / total
    )


# ==================================================
# BTTS OPEN
# ==================================================

(
    df["btts_open_probability"],
    df["btts_no_open_probability"]

) = no_vig_two_way(
    df["bts_yes_open"],
    df["bts_no_open"]
)


# ==================================================
# BTTS CLOSE
# ==================================================

(
    df["btts_close_probability"],
    df["btts_no_close_probability"]

) = no_vig_two_way(
    df["bts_yes_close"],
    df["bts_no_close"]
)


# ==================================================
# OVER / UNDER OPEN
# ==================================================

(
    df["over_open_probability"],
    df["under_open_probability"]

) = no_vig_two_way(
    df["over_2.5_open"],
    df["under_2.5_open"]
)


# ==================================================
# 1X2 OPEN
# ==================================================

df["raw_home_open"] = (
    1 / df["home_open"]
)

df["raw_draw_open"] = (
    1 / df["draw_open"]
)

df["raw_away_open"] = (
    1 / df["away_open"]
)

df["open_1x2_total"] = (
    df["raw_home_open"]
    +
    df["raw_draw_open"]
    +
    df["raw_away_open"]
)

df["home_open_probability"] = (
    df["raw_home_open"]
    /
    df["open_1x2_total"]
)

df["draw_open_probability"] = (
    df["raw_draw_open"]
    /
    df["open_1x2_total"]
)

df["away_open_probability"] = (
    df["raw_away_open"]
    /
    df["open_1x2_total"]
)


# ==================================================
# BALANS
# ==================================================

df["open_balance"] = (
    1
    -
    abs(
        df["home_open_probability"]
        -
        df["away_open_probability"]
    )
)


# ==================================================
# TRAIN / VALIDATION
# ==================================================

train = df[
    df["Season"].isin(
        TRAIN_SEASONS
    )
].copy()

validation = df[
    df["Season"]
    == VALIDATION_SEASON
].copy()


FEATURES = [
    "btts_open_probability",
    "over_open_probability",
    "home_open_probability",
    "draw_open_probability",
    "open_balance"
]


X_train = train[FEATURES]

y_train = train[
    "btts_close_probability"
]


X_validation = validation[FEATURES]

y_validation = validation[
    "btts_close_probability"
]


print()
print("===================================")
print("DATA")
print("===================================")

print(
    "Train matcher:",
    len(train)
)

print(
    "Validation matcher:",
    len(validation)
)


# ==================================================
# BASELINE
#
# Baseline = opening probability
# ==================================================

baseline_prediction = validation[
    "btts_open_probability"
].values


baseline_mse = mean_squared_error(
    y_validation,
    baseline_prediction
)

baseline_mae = np.mean(
    abs(
        y_validation
        -
        baseline_prediction
    )
)


# ==================================================
# LINEAR REGRESSION
# ==================================================

model = LinearRegression()

model.fit(
    X_train,
    y_train
)

prediction = model.predict(
    X_validation
)

# Sannolikheter får inte gå utanför 0-1
prediction = np.clip(
    prediction,
    0,
    1
)


# ==================================================
# MODELLENS FEL
# ==================================================

model_mse = mean_squared_error(
    y_validation,
    prediction
)

model_mae = np.mean(
    abs(
        y_validation
        -
        prediction
    )
)


# ==================================================
# RESULTAT
# ==================================================

print()
print("===================================")
print("OPEN → CLOSE MODEL")
print("===================================")

print(
    "Baseline MSE:",
    round(
        baseline_mse,
        6
    )
)

print(
    "Model MSE:",
    round(
        model_mse,
        6
    )
)

print(
    "MSE improvement:",
    round(
        baseline_mse
        -
        model_mse,
        6
    )
)

print()

print(
    "Baseline MAE:",
    round(
        baseline_mae,
        6
    )
)

print(
    "Model MAE:",
    round(
        model_mae,
        6
    )
)

print(
    "MAE improvement:",
    round(
        baseline_mae
        -
        model_mae,
        6
    )
)


# ==================================================
# KOEFFICIENTER
# ==================================================

print()
print("KOEFFICIENTER")
print("-----------------------------------")

for feature, coefficient in zip(
    FEATURES,
    model.coef_
):

    print(
        feature,
        ":",
        round(
            coefficient,
            4
        )
    )

print(
    "intercept:",
    round(
        model.intercept_,
        4
    )
)


# ==================================================
# KAN MODELLEN FÖRUTSÄGA RIKTNING?
# ==================================================

validation = validation.copy()

validation[
    "model_close_probability"
] = prediction

validation[
    "actual_movement"
] = (
    validation[
        "btts_close_probability"
    ]
    -
    validation[
        "btts_open_probability"
    ]
)

validation[
    "predicted_movement"
] = (
    validation[
        "model_close_probability"
    ]
    -
    validation[
        "btts_open_probability"
    ]
)


actual_direction = np.sign(
    validation["actual_movement"]
)

predicted_direction = np.sign(
    validation["predicted_movement"]
)


# Ignorera helt oförändrade closing lines
mask = (
    actual_direction != 0
)

direction_accuracy = np.mean(
    actual_direction[mask]
    ==
    predicted_direction[mask]
)


print()
print("RIKTNING")
print("-----------------------------------")

print(
    "Direction accuracy:",
    round(
        direction_accuracy,
        4
    )
)


# ==================================================
# STORA MODELLSIGNALER
# ==================================================

print()
print("SIGNALSTYRKA")
print("-----------------------------------")

thresholds = [
    0.01,
    0.02,
    0.03,
    0.05
]


for threshold in thresholds:

    subset = validation[
        abs(
            validation[
                "predicted_movement"
            ]
        )
        >= threshold
    ]

    if len(subset) == 0:
        continue

    correct = np.mean(
        np.sign(
            subset[
                "actual_movement"
            ]
        )
        ==
        np.sign(
            subset[
                "predicted_movement"
            ]
        )
    )

    avg_actual = np.mean(
        subset[
            "actual_movement"
        ]
        *
        np.sign(
            subset[
                "predicted_movement"
            ]
        )
    )

    print(
        f"Signal >= {threshold*100:.0f} pp",
        "| matcher:",
        len(subset),
        "| direction:",
        round(
            correct,
            4
        ),
        "| avg movement rätt riktning:",
        round(
            avg_actual,
            4
        )
    )