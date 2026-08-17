import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss


# ==================================================
# INSTÄLLNINGAR
# ==================================================

FILE = (
    "data/processed/"
    "football_btts_predictions.csv"
)

TEST_SEASONS = [
    "2022-2023",
    "2023-2024",
    "2024-2025"
]


# ==================================================
# LADDA DATA
# ==================================================

df = pd.read_csv(FILE)


# ==================================================
# FEATURES
# ==================================================

FEATURES = [
    "market_btts_probability",

    "home_attack",
    "home_defense",

    "away_attack",
    "away_defense",

    "home_scoring_strength",
    "away_scoring_strength"
]


# ==================================================
# DATA
# ==================================================

df = df.dropna(
    subset=FEATURES + ["actual_btts"]
).copy()


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
                solver="liblinear",
                C=1.0,
                max_iter=2000
            )
        )
    ])


# ==================================================
# WALK FORWARD
# ==================================================

all_predictions = []


for test_season in TEST_SEASONS:

    # Här använder vi endast tidigare testperioder
    # plus all historik som redan finns i filen.

    test = df[
        df["test_season"]
        == test_season
    ].copy()

    train = df[
        df["test_season"]
        < test_season
    ].copy()

    # För första testsäsongen finns ingen
    # tidigare walk-forward-data i denna fil.
    # Vi hoppar därför över den.
    if len(train) == 0:
        print(
            "Hoppar över",
            test_season,
            "- ingen tidigare träningsdata."
        )
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
        "hybrid_probability"
    ] = probability

    all_predictions.append(
        test
    )


predictions = pd.concat(
    all_predictions,
    ignore_index=True
)


# ==================================================
# TOTALT
# ==================================================

actual = predictions[
    "actual_btts"
]

hybrid = predictions[
    "hybrid_probability"
]

market = predictions[
    "market_btts_probability"
]


hybrid_brier = brier_score_loss(
    actual,
    hybrid
)

market_brier = brier_score_loss(
    actual,
    market
)


print()
print("===================================")
print("HYBRID BTTS MODEL")
print("===================================")

print(
    "Matcher:",
    len(predictions)
)

print(
    "Actual BTTS:",
    round(
        actual.mean(),
        4
    )
)

print(
    "Market average:",
    round(
        market.mean(),
        4
    )
)

print(
    "Hybrid average:",
    round(
        hybrid.mean(),
        4
    )
)

print()

print(
    "Market Brier:",
    round(
        market_brier,
        6
    )
)

print(
    "Hybrid Brier:",
    round(
        hybrid_brier,
        6
    )
)

print(
    "Improvement:",
    round(
        market_brier
        -
        hybrid_brier,
        6
    )
)


# ==================================================
# PER SÄSONG
# ==================================================

print()
print("===================================")
print("PER SÄSONG")
print("===================================")

for season, group in predictions.groupby(
    "test_season"
):

    market_score = brier_score_loss(
        group["actual_btts"],
        group[
            "market_btts_probability"
        ]
    )

    hybrid_score = brier_score_loss(
        group["actual_btts"],
        group[
            "hybrid_probability"
        ]
    )

    print()
    print(season)

    print(
        "Matcher:",
        len(group)
    )

    print(
        "Actual:",
        round(
            group[
                "actual_btts"
            ].mean(),
            4
        )
    )

    print(
        "Market:",
        round(
            group[
                "market_btts_probability"
            ].mean(),
            4
        )
    )

    print(
        "Hybrid:",
        round(
            group[
                "hybrid_probability"
            ].mean(),
            4
        )
    )

    print(
        "Market Brier:",
        round(
            market_score,
            6
        )
    )

    print(
        "Hybrid Brier:",
        round(
            hybrid_score,
            6
        )
    )

    print(
        "Improvement:",
        round(
            market_score
            -
            hybrid_score,
            6
        )
    )


# ==================================================
# HYBRID EDGE MOT MARKNAD
# ==================================================

predictions[
    "hybrid_edge"
] = (
    predictions[
        "hybrid_probability"
    ]
    -
    predictions[
        "market_btts_probability"
    ]
)


bins = [
    -1,
    -0.05,
    -0.025,
    0,
    0.025,
    0.05,
    1
]

labels = [
    "< -5 pp",
    "-5 to -2.5 pp",
    "-2.5 to 0 pp",
    "0 to 2.5 pp",
    "2.5 to 5 pp",
    "> 5 pp"
]


predictions[
    "edge_bin"
] = pd.cut(
    predictions[
        "hybrid_edge"
    ],
    bins=bins,
    labels=labels,
    include_lowest=True
)


summary = (
    predictions.groupby(
        "edge_bin",
        observed=True
    )
    .agg(
        matches=(
            "actual_btts",
            "size"
        ),

        market=(
            "market_btts_probability",
            "mean"
        ),

        hybrid=(
            "hybrid_probability",
            "mean"
        ),

        actual=(
            "actual_btts",
            "mean"
        )
    )
)


print()
print("===================================")
print("HYBRID EDGE")
print("===================================")

print(
    summary.to_string()
)


# ==================================================
# SPARA
# ==================================================

OUTPUT = (
    "data/processed/"
    "hybrid_btts_predictions.csv"
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