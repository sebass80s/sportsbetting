import pandas as pd


# ==================================================
# LADDA DATA
# ==================================================

DATA_FILE = "data/raw/premier_league_2015_2025.csv"

matches = pd.read_csv(DATA_FILE)

matches["date"] = pd.to_datetime(matches["date"])

matches = matches.sort_values(
    "date"
).reset_index(drop=True)


# ==================================================
# SKAPA POISSON-DATA
# ==================================================

rows = []


for _, match in matches.iterrows():

    # Hemmalagets observation
    rows.append({
        "date": match["date"],
        "season": match["season"],
        "team": match["home_team"],
        "opponent": match["away_team"],
        "home": 1,
        "goals": match["home_goals"]
    })

    # Bortalagets observation
    rows.append({
        "date": match["date"],
        "season": match["season"],
        "team": match["away_team"],
        "opponent": match["home_team"],
        "home": 0,
        "goals": match["away_goals"]
    })


poisson_data = pd.DataFrame(rows)


# ==================================================
# VISA RESULTAT
# ==================================================

print()
print("===================================")
print("POISSON DATASET")
print("===================================")

print(
    poisson_data.head(10).to_string(
        index=False
    )
)

print()

print(
    "Antal matcher:",
    len(matches)
)

print(
    "Antal Poisson-observationer:",
    len(poisson_data)
)

print(
    "Antal lag:",
    poisson_data["team"].nunique()
)


# ==================================================
# SPARA
# ==================================================

output_file = (
    "data/processed/poisson_data.csv"
)

poisson_data.to_csv(
    output_file,
    index=False
)

print()
print(
    "Sparad till:",
    output_file
)