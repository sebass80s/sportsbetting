import pandas as pd
import os

# Premier League-säsonger
seasons = [
    "1516",
    "1617",
    "1718",
    "1819",
    "1920",
    "2021",
    "2122",
    "2223",
    "2324",
    "2425"
]

output_folder = "data/raw"

os.makedirs(output_folder, exist_ok=True)

all_matches = []

for season in seasons:

    url = f"https://www.football-data.co.uk/mmz4281/{season}/E0.csv"

    print(f"Hämtar säsong {season}...")

    try:
        df = pd.read_csv(url)

        # Kolumner som alltid behövs
        required_columns = [
            "Date",
            "HomeTeam",
            "AwayTeam",
            "FTHG",
            "FTAG"
        ]

        # Kontrollera att grunddata finns
        missing_required = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing_required:
            print(f"  FEL - saknade grundkolumner: {missing_required}")
            continue

        # Börja med grunddata
        matches = df[required_columns].copy()

        # Odds som vi gärna vill ha
        optional_columns = [
            "AvgH",
            "AvgD",
            "AvgA",
            "Avg>2.5",
            "Avg<2.5"
        ]

        # Lägg till odds om de finns
        for column in optional_columns:
            if column in df.columns:
                matches[column] = df[column]
            else:
                matches[column] = pd.NA

        # Byt namn
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

        # Metadata
        matches["league"] = "Premier League"
        matches["season"] = season

        # Datum
        matches["date"] = pd.to_datetime(
            matches["date"],
            dayfirst=True,
            errors="coerce"
        )

        # BTTS
        matches["btts"] = (
            (matches["home_goals"] > 0) &
            (matches["away_goals"] > 0)
        )

        all_matches.append(matches)

        print(f"  {len(matches)} matcher")

    except Exception as e:
        print(f"  FEL: {e}")


# Slå ihop alla säsonger
matches = pd.concat(
    all_matches,
    ignore_index=True
)

# Sortera kronologiskt
matches = matches.sort_values("date")

# Spara
output_file = (
    f"{output_folder}/premier_league_2015_2025.csv"
)

matches.to_csv(
    output_file,
    index=False
)

print()
print("===================================")
print("KLART")
print("===================================")
print("Antal matcher:", len(matches))
print("Antal säsonger:", matches["season"].nunique())
print("BTTS-andel:", matches["btts"].mean())
print()
print("Matcher per säsong:")
print(matches.groupby("season").size())
print()
print("Fil:", output_file)