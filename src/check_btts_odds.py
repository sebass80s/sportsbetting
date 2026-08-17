import pandas as pd


FILE = (
    "external_football_data/data/england/"
    "premier-league.csv"
)

df = pd.read_csv(FILE)

df["Date"] = pd.to_datetime(df["Date"])


# ==================================================
# VÅRA SÄSONGER
# ==================================================

SEASONS = [
    "2019-2020",
    "2020-2021",
    "2021-2022",
    "2022-2023",
    "2023-2024",
    "2024-2025"
]

df = df[
    df["Season"].isin(SEASONS)
].copy()


# ==================================================
# KONTROLL PER SÄSONG
# ==================================================

print()
print("===================================")
print("BTTS ODDS COVERAGE")
print("===================================")

for season, group in df.groupby("Season"):

    total = len(group)

    yes_close = (
        group["bts_yes_close"]
        .notna()
        .sum()
    )

    no_close = (
        group["bts_no_close"]
        .notna()
        .sum()
    )

    both = (
        group["bts_yes_close"].notna()
        &
        group["bts_no_close"].notna()
    ).sum()

    print(
        season,
        "| matcher:",
        total,
        "| Yes:",
        yes_close,
        "| No:",
        no_close,
        "| båda:",
        both,
        "| från:",
        group["Date"].min().date(),
        "| till:",
        group["Date"].max().date()
    )


# ==================================================
# TOTALT
# ==================================================

complete = (
    df["bts_yes_close"].notna()
    &
    df["bts_no_close"].notna()
)

print()
print("TOTALT")
print("-----------------------------------")

print(
    "Matcher:",
    len(df)
)

print(
    "Kompletta BTTS closing odds:",
    complete.sum()
)

print(
    "Andel:",
    round(
        complete.mean() * 100,
        2
    ),
    "%"
)


# ==================================================
# BOOKMAKER-MARGINAL
# ==================================================

odds = df[complete].copy()

odds["raw_yes_probability"] = (
    1 / odds["bts_yes_close"]
)

odds["raw_no_probability"] = (
    1 / odds["bts_no_close"]
)

odds["btts_overround"] = (
    odds["raw_yes_probability"]
    +
    odds["raw_no_probability"]
)

odds["market_yes_probability"] = (
    odds["raw_yes_probability"]
    /
    odds["btts_overround"]
)

odds["market_no_probability"] = (
    odds["raw_no_probability"]
    /
    odds["btts_overround"]
)


print()
print("MARKNAD")
print("-----------------------------------")

print(
    "Genomsnittlig BTTS-overround:",
    round(
        odds["btts_overround"].mean(),
        4
    )
)

print(
    "Genomsnittlig fair BTTS Yes:",
    round(
        odds["market_yes_probability"].mean(),
        4
    )
)


# ==================================================
# FAKTISKT RESULTAT
# ==================================================

odds["actual_btts"] = (
    (odds["FTHG"] > 0)
    &
    (odds["FTAG"] > 0)
).astype(int)

print(
    "Faktisk BTTS:",
    round(
        odds["actual_btts"].mean(),
        4
    )
)