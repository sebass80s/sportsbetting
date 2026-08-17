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
# KOMPLETTA BTTS ODDS
# ==================================================

df = df[
    df["bts_yes_close"].notna()
    &
    df["bts_no_close"].notna()
].copy()


# ==================================================
# RESULTAT
# ==================================================

df["actual_btts"] = (
    (df["FTHG"] > 0)
    &
    (df["FTAG"] > 0)
).astype(int)


# ==================================================
# NO-VIG MARKET PROBABILITY
# ==================================================

df["raw_yes"] = (
    1 / df["bts_yes_close"]
)

df["raw_no"] = (
    1 / df["bts_no_close"]
)

df["overround"] = (
    df["raw_yes"]
    +
    df["raw_no"]
)

df["market_probability"] = (
    df["raw_yes"]
    /
    df["overround"]
)


# ==================================================
# PROFIT VID 1 UNIT PER MATCH
# ==================================================

df["profit"] = np.where(
    df["actual_btts"] == 1,

    # Vinst:
    # tillbaka odds inklusive insats,
    # alltså nettovinst odds - 1
    df["bts_yes_close"] - 1,

    # Förlust
    -1
)


# ==================================================
# SANNOLIKHETSINTERVALL
# ==================================================

bins = [
    0,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    1.0
]

df["probability_bin"] = pd.cut(
    df["market_probability"],
    bins=bins,
    include_lowest=True
)


results = (
    df.groupby(
        "probability_bin",
        observed=True
    )
    .agg(
        bets=(
            "actual_btts",
            "size"
        ),

        wins=(
            "actual_btts",
            "sum"
        ),

        actual_btts=(
            "actual_btts",
            "mean"
        ),

        market_probability=(
            "market_probability",
            "mean"
        ),

        avg_odds=(
            "bts_yes_close",
            "mean"
        ),

        profit=(
            "profit",
            "sum"
        )
    )
)


results["roi"] = (
    results["profit"]
    /
    results["bets"]
)


print()
print("===================================")
print("BTTS YES - MARKET BACKTEST")
print("===================================")

print(
    results.to_string()
)


# ==================================================
# TOTALT
# ==================================================

total_profit = df["profit"].sum()

total_roi = (
    total_profit / len(df)
)

print()
print("TOTALT")
print("-----------------------------------")

print(
    "Bets:",
    len(df)
)

print(
    "Wins:",
    df["actual_btts"].sum()
)

print(
    "Hit rate:",
    round(
        df["actual_btts"].mean(),
        4
    )
)

print(
    "Profit:",
    round(
        total_profit,
        2
    ),
    "units"
)

print(
    "ROI:",
    round(
        total_roi * 100,
        2
    ),
    "%"
)


# ==================================================
# PER SÄSONG
# ==================================================

print()
print("RESULTAT PER SÄSONG")
print("-----------------------------------")

for season, group in df.groupby(
    "Season"
):

    profit = group["profit"].sum()

    roi = (
        profit / len(group)
    )

    print(
        season,
        "| bets:",
        len(group),
        "| hit:",
        round(
            group["actual_btts"].mean(),
            4
        ),
        "| odds:",
        round(
            group["bts_yes_close"].mean(),
            3
        ),
        "| profit:",
        round(
            profit,
            2
        ),
        "| ROI:",
        round(
            roi * 100,
            2
        ),
        "%"
    )