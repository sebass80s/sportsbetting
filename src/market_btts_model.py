import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


# ==================================================
# INSTÄLLNINGAR
# ==================================================

FILE = "data/processed/market_features_v2.csv"

TRAIN_SEASONS = [
    1920,
    2021,
    2122
]

VALIDATION_SEASON = 2223

FEATURES = [
    "market_over_probability",
    "market_home_probability",
    "market_draw_probability",
    "market_balance"
]


# ==================================================
# LADDA DATA
# ==================================================

df = pd.read_csv(FILE)

df["season"] = df["season"].astype(int)

# Konvertera BTTS till 0/1
if df["btts"].dtype == object:

    df["btts"] = (
        df["btts"]
        .astype(str)
        .str.lower()
        .map({
            "true": 1,
            "false": 0
        })
    )

else:

    df["btts"] = (
        df["btts"].astype(int)
    )


# ==================================================
# TRAIN / VALIDATION
# ==================================================

train = df[
    df["season"].isin(
        TRAIN_SEASONS
    )
].copy()

validation = df[
    df["season"]
    == VALIDATION_SEASON
].copy()


X_train = train[FEATURES]
y_train = train["btts"]

X_validation = validation[FEATURES]
y_validation = validation["btts"]


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

print(
    "Train BTTS:",
    round(
        y_train.mean(),
        4
    )
)

print(
    "Validation BTTS:",
    round(
        y_validation.mean(),
        4
    )
)


# ==================================================
# LOGISTISK REGRESSION
# ==================================================

model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train,
    y_train
)


# ==================================================
# PREDIKTION
# ==================================================

probability = model.predict_proba(
    X_validation
)[:, 1]


# ==================================================
# BRIER
# ==================================================

model_brier = brier_score_loss(
    y_validation,
    probability
)


# Baseline använder endast TRAIN BTTS-rate
baseline_probability = (
    y_train.mean()
)

baseline_predictions = np.full(
    len(y_validation),
    baseline_probability
)

baseline_brier = brier_score_loss(
    y_validation,
    baseline_predictions
)


# ==================================================
# RESULTAT
# ==================================================

print()
print("===================================")
print("MARKET BTTS MODEL")
print("===================================")

print(
    "Faktisk BTTS:",
    round(
        y_validation.mean(),
        4
    )
)

print(
    "Modellens genomsnitt:",
    round(
        probability.mean(),
        4
    )
)

print(
    "Model Brier:",
    round(
        model_brier,
        6
    )
)

print(
    "Baseline Brier:",
    round(
        baseline_brier,
        6
    )
)

print(
    "Skill vs baseline:",
    round(
        baseline_brier
        - model_brier,
        6
    )
)


# ==================================================
# KOEFFICIENTER
# ==================================================

print()
print("MODELLENS KOEFFICIENTER")
print("-----------------------------------")

for feature, coefficient in zip(
    FEATURES,
    model.coef_[0]
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
        model.intercept_[0],
        4
    )
)


# ==================================================
# KALIBRERING
# ==================================================

validation = validation.copy()

validation[
    "predicted_btts"
] = probability


bins = [
    0,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    1.0
]

validation[
    "probability_bin"
] = pd.cut(
    validation[
        "predicted_btts"
    ],
    bins=bins,
    include_lowest=True
)


calibration = (
    validation.groupby(
        "probability_bin",
        observed=True
    )
    .agg(
        matches=(
            "btts",
            "size"
        ),
        predicted=(
            "predicted_btts",
            "mean"
        ),
        actual=(
            "btts",
            "mean"
        )
    )
)


print()
print("KALIBRERING")
print("-----------------------------------")

print(
    calibration.to_string()
)