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

# Dessa testar vi ENDAST på development-perioden
HALF_LIVES = [90, 180, 365, 730, 1460]

# Vi använder 2021/22 som development-validering.
# Allt före 2021/22 används som träning.
VALIDATION_SEASON = 2122


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
# TRÄNA OCH TESTA EN HALVERINGSTID
# ==================================================

def evaluate_half_life(half_life):

    training = poisson_data[
        poisson_data["season"] < VALIDATION_SEASON
    ].copy()

    validation = matches[
        matches["season"] == VALIDATION_SEASON
    ].copy()

    # Referensdatum = dagen före första
    # valideringsmatchen
    reference_date = (
        validation["date"].min()
        - pd.Timedelta(days=1)
    )

    # Hur gammal är varje träningsobservation?
    training["age_days"] = (
        reference_date - training["date"]
    ).dt.days

    # Säkerhetskontroll
    training["age_days"] = (
        training["age_days"].clip(lower=0)
    )

    # Exponentiell decay:
    # efter en half-life återstår vikt 0.5
    training["weight"] = np.exp(
        -np.log(2)
        * training["age_days"]
        / half_life
    )

    # Träna viktad Poisson
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

    # Viktade ligamedel som fallback
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

    predictions = []

    fallback_count = 0

    for _, match in validation.iterrows():

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

            lambda_home = model.predict(
                home_input
            ).iloc[0]

            lambda_away = model.predict(
                away_input
            ).iloc[0]

        else:

            lambda_home = avg_home_goals
            lambda_away = avg_away_goals

            fallback_count += 1

        probability = calculate_btts(
            lambda_home,
            lambda_away
        )

        predictions.append({
            "probability": probability,
            "actual": int(match["btts"])
        })

    predictions = pd.DataFrame(predictions)

    brier = np.mean(
        (
            predictions["probability"]
            - predictions["actual"]
        ) ** 2
    )

    return {
        "half_life": half_life,
        "brier": brier,
        "average_prediction":
            predictions["probability"].mean(),
        "actual_btts":
            predictions["actual"].mean(),
        "fallback_matches":
            fallback_count
    }


# ==================================================
# TESTA HALVERINGSTIDER
# ==================================================

results = []

for half_life in HALF_LIVES:

    print(
        "Testar halveringstid:",
        half_life,
        "dagar..."
    )

    result = evaluate_half_life(
        half_life
    )

    results.append(result)


results = pd.DataFrame(results)

results = results.sort_values(
    "brier"
)


# ==================================================
# RESULTAT
# ==================================================

print()
print("===================================")
print("TIME DECAY - DEVELOPMENT")
print("===================================")

print(
    results.to_string(
        index=False
    )
)

best_half_life = int(
    results.iloc[0]["half_life"]
)

print()
print(
    "Bästa halveringstid:",
    best_half_life,
    "dagar"
)

print()
print(
    "OBS: valet är gjort på 2021/22, "
    "inte på 2022/23–2024/25."
)