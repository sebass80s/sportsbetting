import pandas as pd
import numpy as np


FILE = (
    "external_football_data/data/england/"
    "premier-league.csv"
)

df = pd.read_csv(FILE)


COLUMNS = [
    "bts_yes_open",
    "bts_no_open",
    "over_2.5_open",
    "under_2.5_open",
    "home_open",
    "draw_open",
    "away_open"
]


print()
print("===================================")
print("OPENING DATA QUALITY")
print("===================================")


# ==================================================
# DATATYPER
# ==================================================

print()
print("DATATYPER")
print("-----------------------------------")

print(
    df[COLUMNS].dtypes
)


# ==================================================
# MISSING / INFINITE / ZERO / NEGATIVE
# ==================================================

print()
print("PROBLEMVALUES")
print("-----------------------------------")

for column in COLUMNS:

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    missing = values.isna().sum()

    infinite = np.isinf(
        values
    ).sum()

    zero = (
        values == 0
    ).sum()

    negative = (
        values < 0
    ).sum()

    print(
        column,
        "| missing:", missing,
        "| inf:", infinite,
        "| zero:", zero,
        "| negative:", negative
    )


# ==================================================
# MIN / MAX
# ==================================================

print()
print("MIN / MAX")
print("-----------------------------------")

for column in COLUMNS:

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    print(
        column,
        "| min:",
        values.min(),
        "| max:",
        values.max()
    )


# ==================================================
# EXTREMA ODDS
# ==================================================

print()
print("EXTREMA ODDS (> 50)")
print("-----------------------------------")

mask = False

for column in COLUMNS:

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    mask = mask | (values > 50)


extreme = df[mask].copy()

print(
    "Antal matcher:",
    len(extreme)
)

if len(extreme) > 0:

    show_columns = [
        "Date",
        "Season",
        "HomeTeam",
        "AwayTeam"
    ] + COLUMNS

    print(
        extreme[
            show_columns
        ].to_string(
            index=False
        )
    )


# ==================================================
# KONTROLLERA PERIODEN VI ANVÄNDER
# ==================================================

seasons = [
    "2019-2020",
    "2020-2021",
    "2021-2022",
    "2022-2023"
]

subset = df[
    df["Season"].isin(seasons)
].copy()


print()
print("MODELLPERIOD")
print("-----------------------------------")

print(
    "Matcher:",
    len(subset)
)

print()

print(
    subset[COLUMNS]
    .describe()
    .T[
        [
            "count",
            "mean",
            "std",
            "min",
            "25%",
            "50%",
            "75%",
            "max"
        ]
    ]
)