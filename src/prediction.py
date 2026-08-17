import pandas as pd
import math

# --------------------------------------------------
# INSTÄLLNINGAR
# --------------------------------------------------

DATA_FILE = "data/raw/premier_league_2015_2025.csv"

SMOOTHING_K = 5

# Startvärden innan vi har historisk ligadata
INITIAL_HOME_GOALS = 1.50
INITIAL_AWAY_GOALS = 1.20

# Hur starkt vi väger in våra initiala ligagenomsnitt
LEAGUE_PRIOR_MATCHES = 20


# --------------------------------------------------
# LADDA DATA
# --------------------------------------------------

matches = pd.read_csv(DATA_FILE)

matches["date"] = pd.to_datetime(matches["date"])

matches = matches.sort_values("date").reset_index(drop=True)


# --------------------------------------------------
# STATISTIK
# --------------------------------------------------

team_stats = {}

league_matches = 0
league_home_goals = 0
league_away_goals = 0


def get_team_stats(team):
    if team not in team_stats:
        team_stats[team] = {
            "home_matches": 0,
            "home_goals_scored": 0,
            "home_goals_conceded": 0,
            "away_matches": 0,
            "away_goals_scored": 0,
            "away_goals_conceded": 0
        }

    return team_stats[team]


def smoothed_average(
    matches_played,
    goals,
    league_average,
    k=SMOOTHING_K
):
    return (
        goals + k * league_average
    ) / (
        matches_played + k
    )


# --------------------------------------------------
# PREDIKTIONER
# --------------------------------------------------

predictions = []


for index, match in matches.iterrows():

    home_team = match["home_team"]
    away_team = match["away_team"]

    home = get_team_stats(home_team)
    away = get_team_stats(away_team)

    # ----------------------------------------------
    # LIGANS GENOMSNITT FÖRE MATCHEN
    # ----------------------------------------------

    avg_home_goals = (
        league_home_goals
        + LEAGUE_PRIOR_MATCHES * INITIAL_HOME_GOALS
    ) / (
        league_matches + LEAGUE_PRIOR_MATCHES
    )

    avg_away_goals = (
        league_away_goals
        + LEAGUE_PRIOR_MATCHES * INITIAL_AWAY_GOALS
    ) / (
        league_matches + LEAGUE_PRIOR_MATCHES
    )

    # ----------------------------------------------
    # HEMMALAG
    # ----------------------------------------------

    home_scoring = smoothed_average(
        home["home_matches"],
        home["home_goals_scored"],
        avg_home_goals
    )

    home_conceding = smoothed_average(
        home["home_matches"],
        home["home_goals_conceded"],
        avg_away_goals
    )

    home_attack = (
        home_scoring / avg_home_goals
    )

    home_defense = (
        home_conceding / avg_away_goals
    )

    # ----------------------------------------------
    # BORTALAG
    # ----------------------------------------------

    away_scoring = smoothed_average(
        away["away_matches"],
        away["away_goals_scored"],
        avg_away_goals
    )

    away_conceding = smoothed_average(
        away["away_matches"],
        away["away_goals_conceded"],
        avg_home_goals
    )

    away_attack = (
        away_scoring / avg_away_goals
    )

    away_defense = (
        away_conceding / avg_home_goals
    )

    # ----------------------------------------------
    # EXPECTED GOALS
    # ----------------------------------------------

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

    # ----------------------------------------------
    # BTTS-SANNOLIKHET
    # ----------------------------------------------

    prob_home_zero = math.exp(-lambda_home)
    prob_away_zero = math.exp(-lambda_away)

    btts_probability = (
        1
        - prob_home_zero
        - prob_away_zero
        + math.exp(-(lambda_home + lambda_away))
    )

    fair_odds = (
        1 / btts_probability
        if btts_probability > 0
        else None
    )

    # ----------------------------------------------
    # SPARA PREDIKTION
    # ----------------------------------------------

    predictions.append({
        "date": match["date"],
        "season": match["season"],
        "home_team": home_team,
        "away_team": away_team,

        "league_avg_home_goals": avg_home_goals,
        "league_avg_away_goals": avg_away_goals,

        "home_attack": home_attack,
        "home_defense": home_defense,

        "away_attack": away_attack,
        "away_defense": away_defense,

        "expected_home_goals": lambda_home,
        "expected_away_goals": lambda_away,

        "btts_probability": btts_probability,
        "fair_odds": fair_odds,

        "actual_home_goals": match["home_goals"],
        "actual_away_goals": match["away_goals"],
        "actual_btts": match["btts"]
    })

    # ----------------------------------------------
    # UPPDATERA STATISTIK EFTER MATCHEN
    # ----------------------------------------------

    home["home_matches"] += 1
    home["home_goals_scored"] += match["home_goals"]
    home["home_goals_conceded"] += match["away_goals"]

    away["away_matches"] += 1
    away["away_goals_scored"] += match["away_goals"]
    away["away_goals_conceded"] += match["home_goals"]

    league_matches += 1
    league_home_goals += match["home_goals"]
    league_away_goals += match["away_goals"]


# --------------------------------------------------
# SKAPA DATAFRAME
# --------------------------------------------------

predictions = pd.DataFrame(predictions)


# --------------------------------------------------
# SPARA
# --------------------------------------------------

output_file = "data/processed/predictions.csv"

predictions.to_csv(
    output_file,
    index=False
)


# --------------------------------------------------
# RESULTAT
# --------------------------------------------------

print()
print("===================================")
print("PREDIKTIONER KLARA")
print("===================================")

print("Antal matcher:", len(predictions))

print()

columns_to_show = [
    "date",
    "home_team",
    "away_team",
    "expected_home_goals",
    "expected_away_goals",
    "btts_probability",
    "fair_odds",
    "actual_btts"
]

print(
    predictions[
        columns_to_show
    ].tail(10).to_string(index=False)
)

print()
print("Sparad till:", output_file)