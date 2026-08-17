import pandas as pd
import numpy as np
import math
import statsmodels.api as sm
import statsmodels.formula.api as smf


# ==================================================
# INSTÄLLNINGAR
# ==================================================

MATCH_FILE = "data/raw/premier_league_2015_2025.csv"
POISSON_FILE = "data/processed/poisson_data.csv"

HALF_LIFE = 180

TEST_SEASONS = [
    2223,
    2324,
    2425
]

DEVELOPMENT_SEASONS = [
    1516,
    1617,
    1718,
    1819,
    1920,
    2021,
    2122
]


# ==================================================
# LADDA DATA
# ==================================================

matches = pd.read_csv(MATCH_FILE)
poisson_data = pd.read_csv(POISSON_FILE)

matches["date"] = pd.to_datetime(matches["date"])
poisson_data["date"] = pd.to_datetime(poisson_data["date"])

matches["season"] = matches["season"].astype(int)
poisson_data["season"] = poisson_data["season"].astype(int)


# ==================================================
# BTTS
# ==================================================

def calculate_btts(lambda_home, lambda_away):

    return (
        (1 - math.exp(-lambda_home))
        *
        (1 - math.exp(-lambda_away))
    )


# ==================================================
# WALK FORWARD
# ==================================================

all_predictions = []


for test_season in TEST_SEASONS:

    print()
    print("===================================")
    print("TESTSÄSONG:", test_season)
    print("===================================")

    test_matches = matches[
        matches["season"] == test_season
    ].copy()

    training = poisson_data[
        poisson_data["season"] < test_season
    ].copy()

    # Referensdatum:
    # dagen före testsäsongens första match
    reference_date = (
        test_matches["date"].min()
        - pd.Timedelta(days=1)
    )

    # ----------------------------------------------
    # TIME DECAY
    # ----------------------------------------------

    training["age_days"] = (
        reference_date
        - training["date"]
    ).dt.days

    training["age_days"] = (
        training["age_days"].clip(lower=0)
    )

    training["weight"] = np.exp(
        -np.log(2)
        * training["age_days"]
        / HALF_LIFE
    )

    print(
        "Träningsobservationer:",
        len(training)
    )

    print(
        "Testmatcher:",
        len(test_matches)
    )

    # ----------------------------------------------
    # TRÄNA MODELL
    # ----------------------------------------------

    model = smf.glm(
        formula=(
            "goals ~ home "
            "+ C(team) "
            "+ C(opponent)"
        ),
        data=training,
        family=sm.families.Poisson(),
        freq_weights=training["weight"]
    ).fit()

    known_teams = set(
        training["team"].unique()
    )

    # ----------------------------------------------
    # VIKTADE LIGAMEDEL
    # ----------------------------------------------

    home_training = training[
        training["home"] == 1
    ]

    away_training = training[
        training["home"] == 0
    ]

    avg_home_goals = np.average(
        home_training["goals"],
        weights=home_training["weight"]
    )

    avg_away_goals = np.average(
        away_training["goals"],
        weights=away_training["weight"]
    )

    fallback_count = 0

    # ----------------------------------------------
    # PREDIKTERA
    # ----------------------------------------------

    for _, match in test_matches.iterrows():

        home_team = match["home_team"]
        away_team = match["away_team"]

        if (
            home_team in known_teams
            and away_team in known_teams
        ):

            home_input = pd.DataFrame({
                "team": [home_team],
                "opponent": [away_team],
                "home": [1]
            })

            away_input = pd.DataFrame({
                "team": [away_team],
                "opponent": [home_team],
                "home": [0]
            })

            lambda_home = (
                model.predict(
                    home_input
                ).iloc[0]
            )

            lambda_away = (
                model.predict(
                    away_input
                ).iloc[0]
            )

            prediction_type = "model"

        else:

            lambda_home = avg_home_goals
            lambda_away = avg_away_goals

            prediction_type = "fallback"

            fallback_count += 1

        btts_probability = calculate_btts(
            lambda_home,
            lambda_away
        )

        fair_odds = (
            1 / btts_probability
            if btts_probability > 0
            else np.nan
        )

        all_predictions.append({
            "date": match["date"],
            "season": test_season,
            "home_team": home_team,
            "away_team": away_team,

            "lambda_home": lambda_home,
            "lambda_away": lambda_away,

            "btts_probability":
                btts_probability,

            "fair_odds":
                fair_odds,

            "actual_btts":
                int(match["btts"]),

            "prediction_type":
                prediction_type
        })

    print(
        "Fallback-matcher:",
        fallback_count
    )


# ==================================================
# DATAFRAME
# ==================================================

predictions = pd.DataFrame(
    all_predictions
)


# ==================================================
# SPARA
# ==================================================

output_file = (
    "data/processed/"
    "decay_predictions.csv"
)

predictions.to_csv(
    output_file,
    index=False
)


# ==================================================
# BASELINE
# ==================================================

development = matches[
    matches["season"].isin(
        DEVELOPMENT_SEASONS
    )
]

development_btts_rate = (
    development["btts"].mean()
)


# ==================================================
# TOTALT RESULTAT
# ==================================================

actual = predictions["actual_btts"]
probability = predictions["btts_probability"]

model_brier = np.mean(
    (probability - actual) ** 2
)

baseline_brier = np.mean(
    (
        development_btts_rate
        - actual
    ) ** 2
)


print()
print("===================================")
print("TIME-DECAY WALK-FORWARD")
print("===================================")

print(
    "Halveringstid:",
    HALF_LIFE,
    "dagar"
)

print(
    "Antal matcher:",
    len(predictions)
)

print(
    "Faktisk BTTS:",
    round(
        actual.mean(),
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
    "Decay Brier:",
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
# PER SÄSONG
# ==================================================

print()
print("RESULTAT PER SÄSONG")
print("-----------------------------------")

for season, group in predictions.groupby(
    "season"
):

    actual = group["actual_btts"]
    probability = group["btts_probability"]

    season_brier = np.mean(
        (probability - actual) ** 2
    )

    season_baseline = np.mean(
        (
            development_btts_rate
            - actual
        ) ** 2
    )

    print(
        season,
        "| BTTS:",
        round(actual.mean(), 4),
        "| Model:",
        round(probability.mean(), 4),
        "| Brier:",
        round(season_brier, 6),
        "| Baseline:",
        round(season_baseline, 6),
        "| Skill:",
        round(
            season_baseline
            - season_brier,
            6
        )
    )


print()
print(
    "Sparad till:",
    output_file
)