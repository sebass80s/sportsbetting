import pandas as pd

# Läs historisk data
file_path = "data/raw/premier_league_2015_2025.csv"

matches = pd.read_csv(file_path)

# Datum
matches["date"] = pd.to_datetime(matches["date"])

# Sortera kronologiskt
matches = matches.sort_values("date").reset_index(drop=True)

# Ligans genomsnitt
average_home_goals = matches["home_goals"].mean()
average_away_goals = matches["away_goals"].mean()

print("Ligans genomsnitt")
print("------------------")
print("Hemmamål:", average_home_goals)
print("Bortamål:", average_away_goals)
print()


# Statistik som vi bygger upp match för match
team_stats = {}


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
def calculate_strength(
    matches_played,
    goals_scored,
    goals_conceded,
    league_scoring_average,
    k=5
):
    """
    Beräknar en smoothed attack- och försvarsstyrka.
    """

    if matches_played == 0:
        adjusted_scored = league_scoring_average
        adjusted_conceded = league_scoring_average

    else:
        team_scored_average = (
            goals_scored / matches_played
        )

        team_conceded_average = (
            goals_conceded / matches_played
        )

        adjusted_scored = (
            matches_played * team_scored_average
            + k * league_scoring_average
        ) / (matches_played + k)

        adjusted_conceded = (
            matches_played * team_conceded_average
            + k * league_scoring_average
        ) / (matches_played + k)

    attack = (
        adjusted_scored / league_scoring_average
    )

    defense = (
        adjusted_conceded / league_scoring_average
    )

    return attack, defense

# Testa modellen på de första 10 matcherna

for index, match in matches.head(10).iterrows():

    home_team = match["home_team"]
    away_team = match["away_team"]

    home = get_team_stats(home_team)
    away = get_team_stats(away_team)

    # Beräkna hemmalagets styrka
    home_attack, home_defense = calculate_strength(
        home["home_matches"],
        home["home_goals_scored"],
        home["home_goals_conceded"],
        average_home_goals
    )

    # Beräkna bortalagets styrka
    away_attack, away_defense = calculate_strength(
        away["away_matches"],
        away["away_goals_scored"],
        away["away_goals_conceded"],
        average_away_goals
    )

    print(
        f"{match['date'].date()} | "
        f"{home_team} - {away_team}"
    )

    print(
        f"  {home_team}: "
        f"Attack={home_attack:.3f}, "
        f"Defense={home_defense:.3f}"
    )

    print(
        f"  {away_team}: "
        f"Attack={away_attack:.3f}, "
        f"Defense={away_defense:.3f}"
    )

    print()

    # Lägg till matchens resultat EFTER prediktionen

    home["home_matches"] += 1
    home["home_goals_scored"] += match["home_goals"]
    home["home_goals_conceded"] += match["away_goals"]

    away["away_matches"] += 1
    away["away_goals_scored"] += match["away_goals"]
    away["away_goals_conceded"] += match["home_goals"]

    # Lägg till matchens resultat EFTER prediktionen
    home["home_matches"] += 1
    home["home_goals_scored"] += match["home_goals"]
    home["home_goals_conceded"] += match["away_goals"]

    away["away_matches"] += 1
    away["away_goals_scored"] += match["away_goals"]
    away["away_goals_conceded"] += match["home_goals"]