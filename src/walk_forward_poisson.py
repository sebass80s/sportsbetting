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

TEST_SEASONS = [2223, 2324, 2425]


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
# BTTS-FUNKTION
# ==================================================

def calculate_btts(lambda_home, lambda_away):

    probability = (
        (1 - math.exp(-lambda_home))
        *
        (1 - math.exp(-lambda_away))
    )

    return probability


# ==================================================
# WALK-FORWARD
# ==================================================

all_predictions = []


for test_season in TEST_SEASONS:

    print()
    print("===================================")
    print("TESTSÄSONG:", test_season)
    print("===================================")

    # All data FÖRE testsäsongen
    training = poisson_data[
        poisson_data["season"] < test_season
    ].copy()

    test_matches = matches[
        matches["season"] == test_season
    ].copy()

    print(
        "Träningsobservationer:",
        len(training)
    )

    print(
        "Testmatcher:",
        len(test_matches)
    )

    # ----------------------------------------------
    # TRÄNA POISSON
    # ----------------------------------------------

    model = smf.glm(
        formula=(
            "goals ~ home "
            "+ C(team) "
            "+ C(opponent)"
        ),
        data=training,
        family=sm.families.Poisson()
    ).fit()

    known_teams = set(
        training["team"].unique()
    )

    # Ligamedel används som fallback
    avg_home_goals = (
        training[
            training["home"] == 1
        ]["goals"].mean()
    )

    avg_away_goals = (
        training[
            training["home"] == 0
        ]["goals"].mean()
    )

    skipped = 0

    # ----------------------------------------------
    # PREDIKTERA TESTSÄSONGEN
    # ----------------------------------------------

    for _, match in test_matches.iterrows():

        home_team = match["home_team"]
        away_team = match["away_team"]

        home_known = (
            home_team in known_teams
        )

        away_known = (
            away_team in known_teams
        )

        # ------------------------------------------
        # BÅDA LAGEN KÄNDA
        # ------------------------------------------

        if home_known and away_known:

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

        # ------------------------------------------
        # MINST ETT NYTT LAG
        # ------------------------------------------

        else:

            lambda_home = avg_home_goals
            lambda_away = avg_away_goals

            prediction_type = "fallback"
            skipped += 1

        # ------------------------------------------
        # BTTS
        # ------------------------------------------

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
        skipped
    )


# ==================================================
# RESULTAT
# ==================================================

predictions = pd.DataFrame(
    all_predictions
)

output_file = (
    "data/processed/"
    "poisson_predictions.csv"
)

predictions.to_csv(
    output_file,
    index=False
)


# ==================================================
# UTVÄRDERING
# ==================================================

actual = predictions[
    "actual_btts"
]

probability = predictions[
    "btts_probability"
]

model_brier = np.mean(
    (probability - actual) ** 2
)


# Baseline från DEVELOPMENT-perioden
development_matches = matches[
    matches["season"] < 2223
]

development_btts_rate = (
    development_matches["btts"].mean()
)

baseline_brier = np.mean(
    (
        development_btts_rate
        - actual
    ) ** 2
)


print()
print("===================================")
print("POISSON WALK-FORWARD RESULTAT")
print("===================================")

print(
    "Antal matcher:",
    len(predictions)
)

print(
    "Development baseline:",
    round(
        development_btts_rate,
        4
    )
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
    "Poisson Brier:",
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
# RESULTAT PER SÄSONG
# ==================================================

print()
print("RESULTAT PER SÄSONG")
print("-----------------------------------")

for season, group in predictions.groupby(
    "season"
):

    season_actual = group[
        "actual_btts"
    ]

    season_probability = group[
        "btts_probability"
    ]

    season_brier = np.mean(
        (
            season_probability
            - season_actual
        ) ** 2
    )

    season_baseline = np.mean(
        (
            development_btts_rate
            - season_actual
        ) ** 2
    )

    print(
        season,
        "| BTTS:",
        round(
            season_actual.mean(),
            4
        ),
        "| Model:",
        round(
            season_probability.mean(),
            4
        ),
        "| Brier:",
        round(
            season_brier,
            6
        ),
        "| Baseline:",
        round(
            season_baseline,
            6
        ),
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