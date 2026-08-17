import pandas as pd
from pathlib import Path


# ==================================================
# FILER
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OLD_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "premier_league_2015_2025.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "premier_league_2015_2026.csv"
)


# ==================================================
# FOOTBALL-DATA 2025/26
# ==================================================

URL = (
    "https://www.football-data.co.uk/"
    "mmz4281/2526/E0.csv"
)


print()
print("===================================")
print("HÄMTAR PREMIER LEAGUE 2025/26")
print("===================================")


new = pd.read_csv(URL)

print(
    "Rader hämtade:",
    len(new)
)


# ==================================================
# KONTROLL
# ==================================================

required = [
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG"
]

missing = [
    col
    for col in required
    if col not in new.columns
]

if missing:
    raise RuntimeError(
        f"Saknade kolumner: {missing}"
    )


# ==================================================
# RENSA MATCHER
# ==================================================

new = new.dropna(
    subset=[
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG"
    ]
).copy()


# ==================================================
# DATUM
# ==================================================

new["date"] = pd.to_datetime(
    new["Date"],
    dayfirst=True,
    errors="coerce"
)

new = new.dropna(
    subset=["date"]
).copy()


# ==================================================
# VÅRT FORMAT
# ==================================================

season_2526 = pd.DataFrame({

    "date":
        new["date"],

    "season":
        2526,

    "league":
        "Premier League",

    "home_team":
        new["HomeTeam"],

    "away_team":
        new["AwayTeam"],

    "home_goals":
        new["FTHG"].astype(int),

    "away_goals":
        new["FTAG"].astype(int)
})


season_2526["btts"] = (
    (season_2526["home_goals"] > 0)
    &
    (season_2526["away_goals"] > 0)
)


# ==================================================
# LADDA GAMMAL HISTORIK
# ==================================================

old = pd.read_csv(
    OLD_FILE
)

old["date"] = pd.to_datetime(
    old["date"]
)


# ==================================================
# KONTROLLERA DUBLETTER
# ==================================================

existing_seasons = sorted(
    old["season"].unique()
)

print(
    "Gamla säsonger:",
    existing_seasons
)

if 2526 in existing_seasons:
    raise RuntimeError(
        "2526 finns redan i gamla filen."
    )


# ==================================================
# GEMENSAMMA KOLUMNER
# ==================================================

# Vi behöver framför allt dessa för V1:s
# attack/defense-features.

base_columns = [
    "date",
    "season",
    "league",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "btts"
]


old_base = old[
    base_columns
].copy()


# ==================================================
# SLÅ IHOP
# ==================================================

combined = pd.concat(
    [
        old_base,
        season_2526
    ],
    ignore_index=True
)

combined = combined.sort_values(
    "date"
).reset_index(drop=True)


# ==================================================
# SPARA
# ==================================================

combined.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==================================================
# RESULTAT
# ==================================================

print()
print("===================================")
print("2025/26")
print("===================================")

print(
    "Matcher:",
    len(season_2526)
)

print(
    "Från:",
    season_2526["date"].min()
)

print(
    "Till:",
    season_2526["date"].max()
)

print(
    "BTTS:",
    round(
        season_2526["btts"].mean(),
        4
    )
)


print()
print("===================================")
print("KOMBINERAD HISTORIK")
print("===================================")

print(
    "Matcher:",
    len(combined)
)

print()

print(
    combined.groupby(
        "season"
    ).size()
)

print()
print(
    "Sparad till:",
    OUTPUT_FILE
)