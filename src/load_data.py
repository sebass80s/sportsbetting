import pandas as pd

file_path = "data/raw/matches.csv"

matches = pd.read_csv(file_path)

print(matches)
print()
print("Antal matcher:", len(matches))

matches["btts"] = (
    (matches["home_goals"] > 0) &
    (matches["away_goals"] > 0)
)

print()
print(matches)

# Ligans genomsnittliga antal mål
average_home_goals = matches["home_goals"].mean()
average_away_goals = matches["away_goals"].mean()

print()
print("Genomsnittliga hemmamål:", average_home_goals)
print("Genomsnittliga bortamål:", average_away_goals)

# Beräkna genomsnittliga mål hemma och borta för varje lag

home_stats = matches.groupby("home_team").agg(
    home_goals_scored=("home_goals", "mean"),
    home_goals_conceded=("away_goals", "mean")
)

away_stats = matches.groupby("away_team").agg(
    away_goals_scored=("away_goals", "mean"),
    away_goals_conceded=("home_goals", "mean")
)

print()
print("HOME STATS")
print(home_stats)

print()
print("AWAY STATS")
print(away_stats)

# Normalisera attack och defense mot ligans genomsnitt

home_stats["home_attack"] = (
    home_stats["home_goals_scored"] / average_home_goals
)

home_stats["home_defense"] = (
    home_stats["home_goals_conceded"] / average_away_goals
)

away_stats["away_attack"] = (
    away_stats["away_goals_scored"] / average_away_goals
)

away_stats["away_defense"] = (
    away_stats["away_goals_conceded"] / average_home_goals
)

print()
print("HOME STRENGTH")
print(home_stats[["home_attack", "home_defense"]])

print()
print("AWAY STRENGTH")
print(away_stats[["away_attack", "away_defense"]])