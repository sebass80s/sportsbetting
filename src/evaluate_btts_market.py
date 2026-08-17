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
# BEHÅLL KOMPLETTA BTTS-ODDS
# ==================================================

df = df[
    df["bts_yes_close"].notna()
    &
    df["bts_no_close"].notna()
].copy()


# ==================================================
# FAKTISKT BTTS
# ==================================================

df["actual_btts"] = (
    (df["FTHG"] > 0)
    &
    (df["FTAG"] > 0)
).astype(int)


# ==================================================
# TA BORT BOOKMAKER-MARGINAL
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

df["market_btts_probability"] = (
    df["raw_yes"]
    /
    df["overround"]
)


# ==================================================
# BRIER SCORE
# ==================================================

market_brier = np.mean(
    (
        df["market_btts_probability"]
        -
        df["actual_btts"]
    ) ** 2
)


# Konstant baseline
baseline_probability = (
    df["actual_btts"].mean()
)

baseline_brier = np.mean(
    (
        baseline_probability
        -
        df["actual_btts"]
    ) ** 2
)


print()
print("===================================")
print("BTTS MARKET EVALUATION")
print("===================================")

print(
    "Matcher:",
    len(df)
)

print(
    "Faktisk BTTS:",
    round(
        df["actual_btts"].mean(),
        4
    )
)

print(
    "Marknadens genomsnitt:",
    round(
        df["market_btts_probability"].mean(),
        4
    )
)

print(
    "Market Brier:",
    round(
        market_brier,
        6
    )
)

print(
    "Constant baseline Brier:",
    round(
        baseline_brier,
        6
    )
)

print(
    "Market skill:",
    round(
        baseline_brier
        -
        market_brier,
        6
    )
)


# ==================================================
# KALIBRERING
# ==================================================

bins = [
    0,
    0.35,
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
    df["market_btts_probability"],
    bins=bins,
    include_lowest=True
)


calibration = (
    df.groupby(
        "probability_bin",
        observed=True
    )
    .agg(
        matches=(
            "actual_btts",
            "size"
        ),
        market_probability=(
            "market_btts_probability",
            "mean"
        ),
        actual_btts=(
            "actual_btts",
            "mean"
        ),
        avg_yes_odds=(
            "bts_yes_close",
            "mean"
        )
    )
)


print()
print("KALIBRERING")
print("-----------------------------------")

print(
    calibration.to_string()
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

    brier = np.mean(
        (
            group["market_btts_probability"]
            -
            group["actual_btts"]
        ) ** 2
    )

    print(
        season,
        "| matcher:",
        len(group),
        "| Market:",
        round(
            group[
                "market_btts_probability"
            ].mean(),
            4
        ),
        "| Actual:",
        round(
            group[
                "actual_btts"
            ].mean(),
            4
        ),
        "| Brier:",
        round(
            brier,
            6
        )
    )