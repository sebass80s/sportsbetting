import pandas as pd
import numpy as np
import math
from collections import defaultdict, deque


# ==================================================
# INSTÄLLNINGAR
# ==================================================

DATA_FILE = "data/raw/premier_league_2015_2025.csv"

WINDOW_SIZES = [5, 10, 20, 30, 50]

SMOOTHING_K = 5

INITIAL_HOME_GOALS = 1.50
INITIAL_AWAY_GOALS = 1.20

LEAGUE_PRIOR_MATCHES = 20

DEVELOPMENT_SEASONS = [
    1516,
    1617,
    1718,
    1819,
    1920,
    2021,
    2122
]

TEST_SEASONS = [
    2223,
    2324,
    2425
]


# ==================================================
# LADDA DATA
# ==================================================

matches = pd.read_csv(DATA_FILE)

matches["date"] = pd.to_datetime(matches["date"])

matches = matches.sort_values(
    "date"
).reset_index(drop=True)


# ==================================================
# FUNKTION FÖR EN MODELL
# ==================================================

def run_model(window_size):

    # Separat hemma- och bortahistorik
    home_history = defaultdict(
        lambda: deque(maxlen=window_size)
    )

    away_history = defaultdict(
        lambda: deque(maxlen=window_size)
    )

    league_matches = 0
    league_home_goals = 0
    league_away_goals = 0

    predictions = []

    for _, match in matches.iterrows():

        home_team = match["home_team"]
        away_team = match["away_team"]

        # ------------------------------------------
        # DYNAMISKT LIGAMEDEL
        # ------------------------------------------

        avg_home_goals = (
            league_home_goals
            + LEAGUE_PRIOR_MATCHES
            * INITIAL_HOME_GOALS
        ) / (
            league_matches
            + LEAGUE_PRIOR_MATCHES
        )

        avg_away_goals = (
            league_away_goals
            + LEAGUE_PRIOR_MATCHES
            * INITIAL_AWAY_GOALS
        ) / (
            league_matches
            + LEAGUE_PRIOR_MATCHES
        )

        # ------------------------------------------
        # HEMMALAGETS HISTORIK
        # ------------------------------------------

        home_games = list(
            home_history[home_team]
        )

        home_n = len(home_games)

        home_scored = sum(
            game[0] for game in home_games
        )

        home_conceded = sum(
            game[1] for game in home_games
        )

        # Smoothing
        home_scoring_avg = (
            home_scored
            + SMOOTHING_K * avg_home_goals
        ) / (
            home_n + SMOOTHING_K
        )

        home_conceding_avg = (
            home_conceded
            + SMOOTHING_K * avg_away_goals
        ) / (
            home_n + SMOOTHING_K
        )

        home_attack = (
            home_scoring_avg
            / avg_home_goals
        )

        home_defense = (
            home_conceding_avg
            / avg_away_goals
        )

        # ------------------------------------------
        # BORTALAGETS HISTORIK
        # ------------------------------------------

        away_games = list(
            away_history[away_team]
        )

        away_n = len(away_games)

        away_scored = sum(
            game[0] for game in away_games
        )

        away_conceded = sum(
            game[1] for game in away_games
        )

        away_scoring_avg = (
            away_scored
            + SMOOTHING_K * avg_away_goals
        ) / (
            away_n + SMOOTHING_K
        )

        away_conceding_avg = (
            away_conceded
            + SMOOTHING_K * avg_home_goals
        ) / (
            away_n + SMOOTHING_K
        )

        away_attack = (
            away_scoring_avg
            / avg_away_goals
        )

        away_defense = (
            away_conceding_avg
            / avg_home_goals
        )

        # ------------------------------------------
        # EXPECTED GOALS
        # ------------------------------------------

        lambda_home = (
            avg_home_goals
            * home_attack
            * away_defense
        )

        lambda_away = (
            avg_away_goals
            * away_attack
            * home_defense
        )

        # ------------------------------------------
        # BTTS
        # ------------------------------------------

        btts_probability = (
            1
            - math.exp(-lambda_home)
            - math.exp(-lambda_away)
            + math.exp(
                -(lambda_home + lambda_away)
            )
        )

        predictions.append({
            "season": int(match["season"]),
            "btts_probability":
                btts_probability,
            "actual_btts":
                int(match["btts"])
        })

        # ------------------------------------------
        # RESULTATET BLIR NU HISTORIK
        # ------------------------------------------

        home_history[home_team].append(
            (
                match["home_goals"],
                match["away_goals"]
            )
        )

        away_history[away_team].append(
            (
                match["away_goals"],
                match["home_goals"]
            )
        )

        league_matches += 1

        league_home_goals += (
            match["home_goals"]
        )

        league_away_goals += (
            match["away_goals"]
        )

    return pd.DataFrame(predictions)


# ==================================================
# TESTA OLIKA WINDOW-STORLEKAR
# ==================================================

results = []

for window_size in WINDOW_SIZES:

    print(
        f"Testar rolling window = "
        f"{window_size}..."
    )

    predictions = run_model(
        window_size
    )

    development = predictions[
        predictions["season"].isin(
            DEVELOPMENT_SEASONS
        )
    ]

    actual = development[
        "actual_btts"
    ]

    probability = development[
        "btts_probability"
    ]

    brier = np.mean(
        (probability - actual) ** 2
    )

    results.append({
        "window": window_size,
        "matches": len(development),
        "brier": brier
    })


# ==================================================
# RESULTAT
# ==================================================

results = pd.DataFrame(results)

results = results.sort_values(
    "brier"
)

print()
print(
    "==================================="
)
print("DEVELOPMENT RESULTAT")
print(
    "==================================="
)

print(
    results.to_string(
        index=False
    )
)

best_window = int(
    results.iloc[0]["window"]
)

print()
print(
    "Bästa rolling window:",
    best_window
)

print()
print(
    "OBS: Testperioden 2223–2425 "
    "har inte använts för valet."
)
# ==================================================
# SLUTLIGT TEST PÅ LÅST TESTPERIOD
# ==================================================

final_predictions = run_model(best_window)

test = final_predictions[
    final_predictions["season"].isin(
        TEST_SEASONS
    )
].copy()

test_actual = test["actual_btts"]
test_probability = test["btts_probability"]

# Modellens Brier score
test_brier = np.mean(
    (test_probability - test_actual) ** 2
)

# Naiv baseline:
# Här använder vi development-periodens BTTS-andel,
# INTE testperiodens framtida facit.
development_predictions = final_predictions[
    final_predictions["season"].isin(
        DEVELOPMENT_SEASONS
    )
]

development_btts_rate = (
    development_predictions["actual_btts"].mean()
)

baseline_brier = np.mean(
    (
        development_btts_rate
        - test_actual
    ) ** 2
)

print()
print("===================================")
print("LÅST TESTPERIOD 2022/23–2024/25")
print("===================================")

print("Rolling window:", best_window)
print("Antal matcher:", len(test))

print(
    "Development BTTS baseline:",
    round(development_btts_rate, 4)
)

print(
    "Faktisk BTTS i testperioden:",
    round(test_actual.mean(), 4)
)

print(
    "Modellens genomsnitt:",
    round(test_probability.mean(), 4)
)

print(
    "Modell Brier:",
    round(test_brier, 6)
)

print(
    "Baseline Brier:",
    round(baseline_brier, 6)
)

print(
    "Skill vs baseline:",
    round(
        baseline_brier - test_brier,
        6
    )
)


# ==================================================
# RESULTAT PER TESTSÄSONG
# ==================================================

print()
print("TESTRESULTAT PER SÄSONG")
print("-----------------------------------")

for season, group in test.groupby("season"):

    actual = group["actual_btts"]
    probability = group["btts_probability"]

    model_brier = np.mean(
        (probability - actual) ** 2
    )

    baseline_season_brier = np.mean(
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
        "| Model Brier:",
        round(model_brier, 6),
        "| Baseline:",
        round(baseline_season_brier, 6)
    )