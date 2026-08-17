import pandas as pd
import numpy as np


# ==================================================
# LADDA DATA
# ==================================================

FILE = "data/processed/market_features.csv"

df = pd.read_csv(FILE)

df["season"] = df["season"].astype(int)

# BTTS till 0/1
if df["btts"].dtype == object:
    df["btts"] = (
        df["btts"]
        .astype(str)
        .str.lower()
        .map({
            "true": 1,
            "false": 0
        })
    )
else:
    df["btts"] = df["btts"].astype(int)


# ==================================================
# KORRELATION
# ==================================================

correlation = df[
    [
        "market_over_probability",
        "btts"
    ]
].corr().iloc[0, 1]


print()
print("===================================")
print("MARKNAD VS BTTS")
print("===================================")

print(
    "Antal matcher:",
    len(df)
)

print(
    "Korrelation Over 2.5 / BTTS:",
    round(correlation, 4)
)


# ==================================================
# DELA IN MARKNADEN I INTERVALL
# ==================================================

bins = [
    0.0,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    1.0
]

df["market_bin"] = pd.cut(
    df["market_over_probability"],
    bins=bins,
    include_lowest=True
)


market_groups = (
    df.groupby(
        "market_bin",
        observed=True
    )
    .agg(
        matches=("btts", "size"),
        avg_over_probability=(
            "market_over_probability",
            "mean"
        ),
        actual_btts_rate=(
            "btts",
            "mean"
        )
    )
)


print()
print("BTTS PER OVER 2.5-INTERVALL")
print("-----------------------------------")

print(
    market_groups.to_string()
)


# ==================================================
# RESULTAT PER SÄSONG
# ==================================================

print()
print("RESULTAT PER SÄSONG")
print("-----------------------------------")

season_results = (
    df.groupby("season")
    .agg(
        matches=("btts", "size"),
        avg_market_over=(
            "market_over_probability",
            "mean"
        ),
        actual_btts=(
            "btts",
            "mean"
        )
    )
)

print(
    season_results.to_string()
)