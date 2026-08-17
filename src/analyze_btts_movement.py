import pandas as pd
import numpy as np


# ==================================================
# LADDA DATA
# ==================================================

FILE = (
    "external_football_data/data/england/"
    "premier-league.csv"
)

df = pd.read_csv(FILE)

df["Date"] = pd.to_datetime(df["Date"])


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
# KRÄV OPEN + CLOSE
# ==================================================

required = [
    "bts_yes_open",
    "bts_no_open",
    "bts_yes_close",
    "bts_no_close"
]

df = df.dropna(
    subset=required
).copy()


# ==================================================
# OPENING NO-VIG
# ==================================================

df["open_raw_yes"] = (
    1 / df["bts_yes_open"]
)

df["open_raw_no"] = (
    1 / df["bts_no_open"]
)

df["open_overround"] = (
    df["open_raw_yes"]
    +
    df["open_raw_no"]
)

df["open_probability"] = (
    df["open_raw_yes"]
    /
    df["open_overround"]
)


# ==================================================
# CLOSING NO-VIG
# ==================================================

df["close_raw_yes"] = (
    1 / df["bts_yes_close"]
)

df["close_raw_no"] = (
    1 / df["bts_no_close"]
)

df["close_overround"] = (
    df["close_raw_yes"]
    +
    df["close_raw_no"]
)

df["close_probability"] = (
    df["close_raw_yes"]
    /
    df["close_overround"]
)


# ==================================================
# RÖRELSE
# ==================================================

df["movement"] = (
    df["close_probability"]
    -
    df["open_probability"]
)

df["absolute_movement"] = abs(
    df["movement"]
)


# ==================================================
# ODDSRÖRELSE
# ==================================================

df["odds_change"] = (
    df["bts_yes_close"]
    -
    df["bts_yes_open"]
)


# ==================================================
# RESULTAT
# ==================================================

print()
print("===================================")
print("BTTS OPEN → CLOSE")
print("===================================")

print(
    "Matcher:",
    len(df)
)

print(
    "Genomsnitt opening probability:",
    round(
        df["open_probability"].mean(),
        4
    )
)

print(
    "Genomsnitt closing probability:",
    round(
        df["close_probability"].mean(),
        4
    )
)

print(
    "Genomsnittlig movement:",
    round(
        df["movement"].mean(),
        6
    )
)

print(
    "Genomsnittlig absolut movement:",
    round(
        df["absolute_movement"].mean(),
        4
    )
)

print(
    "Median absolut movement:",
    round(
        df["absolute_movement"].median(),
        4
    )
)


# ==================================================
# HUR OFTA RÖR SIG MARKNADEN?
# ==================================================

print()
print("MOVEMENT DISTRIBUTION")
print("-----------------------------------")

thresholds = [
    0.01,
    0.02,
    0.03,
    0.05,
    0.075,
    0.10
]

for threshold in thresholds:

    count = (
        df["absolute_movement"]
        >= threshold
    ).sum()

    percentage = (
        count / len(df)
    )

    print(
        f">= {threshold * 100:.1f} pp:",
        count,
        "matcher",
        f"({percentage * 100:.2f}%)"
    )


# ==================================================
# RIKTNING
# ==================================================

up = (
    df["movement"] > 0
).sum()

down = (
    df["movement"] < 0
).sum()

unchanged = (
    df["movement"] == 0
).sum()


print()
print("RIKTNING")
print("-----------------------------------")

print(
    "BTTS probability upp:",
    up
)

print(
    "BTTS probability ner:",
    down
)

print(
    "Oförändrad:",
    unchanged
)


# ==================================================
# KORRELATION OPEN / CLOSE
# ==================================================

correlation = df[
    [
        "open_probability",
        "close_probability"
    ]
].corr().iloc[0, 1]


print()
print(
    "Korrelation open/close:",
    round(
        correlation,
        4
    )
)


# ==================================================
# STÖRSTA RÖRELSERNA
# ==================================================

print()
print("STÖRSTA RÖRELSER")
print("-----------------------------------")

columns = [
    "Date",
    "HomeTeam",
    "AwayTeam",
    "bts_yes_open",
    "bts_yes_close",
    "open_probability",
    "close_probability",
    "movement"
]

largest = (
    df.reindex(
        df["absolute_movement"]
        .sort_values(
            ascending=False
        )
        .index
    )
    .head(15)
)

print(
    largest[
        columns
    ].to_string(
        index=False
    )
)