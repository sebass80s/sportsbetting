import pandas as pd

# Football-Data: Premier League 2024/25
url = "https://www.football-data.co.uk/mmz4281/2425/E0.csv"

# Läs in data
df = pd.read_csv(url)

# Välj de kolumner vi behöver just nu
matches = df[
    [
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "AvgH",
        "AvgD",
        "AvgA",
        "Avg>2.5",
        "Avg<2.5"
    ]
].copy()

# Byt till våra egna kolumnnamn
matches = matches.rename(columns={
    "Date": "date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "AvgH": "avg_home_odds",
    "AvgD": "avg_draw_odds",
    "AvgA": "avg_away_odds",
    "Avg>2.5": "avg_over_2_5_odds",
    "Avg<2.5": "avg_under_2_5_odds"
})

# Lägg till liga
matches["league"] = "Premier League"

# Konvertera datum
matches["date"] = pd.to_datetime(
    matches["date"],
    dayfirst=True
)

# Lägg till BTTS
matches["btts"] = (
    (matches["home_goals"] > 0) &
    (matches["away_goals"] > 0)
)

# Spara den bearbetade datan
output_file = "data/raw/premier_league_2024_25.csv"

matches.to_csv(
    output_file,
    index=False
)

print(matches.head())
print()
print("Antal matcher:", len(matches))
print()
print("BTTS-andel:", matches["btts"].mean())
print()
print("Sparad till:", output_file)