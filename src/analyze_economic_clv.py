import pandas as pd
import numpy as np


# ==================================================
# LADDA DATA
# ==================================================

FILE = "data/processed/walk_forward_bets.csv"

df = pd.read_csv(FILE)


print()
print("===================================")
print("ECONOMIC CLV")
print("===================================")

print("Bets:", len(df))


# ==================================================
# KOLLA VILKA KOLUMNER SOM FINNS
# ==================================================

print()
print("Kolumner:")
print(df.columns.tolist())


# ==================================================
# HITTA OPEN/CLOSE ODDS
# ==================================================
#
# Vi stödjer några olika namn eftersom vår tidigare
# fil kan ha sparat dem med olika kolumnnamn.
# ==================================================

def find_column(candidates):

    for col in candidates:
        if col in df.columns:
            return col

    return None


open_col = find_column([
    "bet_odds",
    "open_odds",
    "opening_odds",
    "odds_open"
])

close_col = find_column([
    "close_odds_same_side",
    "close_odds",
    "closing_odds",
    "odds_close"
])


# ==================================================
# OM CLOSE ODDS INTE FINNS
# ==================================================

if open_col is None or close_col is None:

    print()
    print("===================================")
    print("SAKNAD ODDSKOLUMN")
    print("===================================")

    print("Open column:", open_col)
    print("Close column:", close_col)

    print()
    print(
        "Vi behöver komplettera "
        "walk_forward_bets.csv med "
        "opening och closing odds."
    )

    raise SystemExit


# ==================================================
# RENGÖR
# ==================================================

df[open_col] = pd.to_numeric(
    df[open_col],
    errors="coerce"
)

df[close_col] = pd.to_numeric(
    df[close_col],
    errors="coerce"
)

df = df.dropna(
    subset=[
        open_col,
        close_col
    ]
).copy()

df = df[
    (df[open_col] > 1)
    &
    (df[close_col] > 1)
].copy()


# ==================================================
# ECONOMIC CLV
# ==================================================
#
# Exempel:
#
# Bet @ 2.00
# Close @ 1.90
#
# 2.00 / 1.90 - 1
# = +5.26 %
#
# Positivt = vi fick bättre odds än closing.
# ==================================================

df["economic_clv"] = (
    df[open_col]
    /
    df[close_col]
    -
    1
)


# ==================================================
# TOTALT
# ==================================================

print()
print("===================================")
print("TOTALT")
print("===================================")

print(
    "Bets:",
    len(df)
)

print(
    "Mean economic CLV:",
    round(
        df["economic_clv"].mean()
        * 100,
        3
    ),
    "%"
)

print(
    "Median economic CLV:",
    round(
        df["economic_clv"].median()
        * 100,
        3
    ),
    "%"
)

print(
    "Beat closing odds:",
    round(
        (
            df["economic_clv"] > 0
        ).mean()
        * 100,
        2
    ),
    "%"
)


# ==================================================
# YES / NO
# ==================================================

print()
print("===================================")
print("YES / NO")
print("===================================")

for side, group in df.groupby(
    "bet_side"
):

    print()
    print(side)

    print(
        "Bets:",
        len(group)
    )

    print(
        "Economic CLV:",
        round(
            group[
                "economic_clv"
            ].mean()
            * 100,
            3
        ),
        "%"
    )

    print(
        "Median:",
        round(
            group[
                "economic_clv"
            ].median()
            * 100,
            3
        ),
        "%"
    )

    print(
        "Beat close:",
        round(
            (
                group[
                    "economic_clv"
                ] > 0
            ).mean()
            * 100,
            2
        ),
        "%"
    )

    if "profit" in group.columns:

        print(
            "ROI:",
            round(
                group[
                    "profit"
                ].mean()
                * 100,
                2
            ),
            "%"
        )


# ==================================================
# PER SÄSONG
# ==================================================

print()
print("===================================")
print("PER SÄSONG")
print("===================================")

for season, group in df.groupby(
    "test_season"
):

    print()
    print(season)

    print(
        "Bets:",
        len(group)
    )

    print(
        "Economic CLV:",
        round(
            group[
                "economic_clv"
            ].mean()
            * 100,
            3
        ),
        "%"
    )

    print(
        "Beat close:",
        round(
            (
                group[
                    "economic_clv"
                ] > 0
            ).mean()
            * 100,
            2
        ),
        "%"
    )

    if "profit" in group.columns:

        print(
            "ROI:",
            round(
                group[
                    "profit"
                ].mean()
                * 100,
            2
            ),
            "%"
        )


# ==================================================
# SIGNAL BUCKETS
# ==================================================

if "signal" in df.columns:

    signal = df["signal"].abs()

elif "predicted_movement" in df.columns:

    signal = (
        df[
            "predicted_movement"
        ].abs()
    )

else:

    signal = None


if signal is not None:

    df["signal_abs"] = signal

    bins = [
        0.005,
        0.0075,
        0.010,
        0.015,
        0.020,
        np.inf
    ]

    labels = [
        "0.5-0.75 pp",
        "0.75-1.0 pp",
        "1.0-1.5 pp",
        "1.5-2.0 pp",
        "2.0+ pp"
    ]

    df["signal_bin"] = pd.cut(
        df["signal_abs"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    print()
    print("===================================")
    print("SIGNAL BUCKETS")
    print("===================================")

    summary = (
        df.groupby(
            "signal_bin",
            observed=True
        )
        .agg(
            bets=(
                "economic_clv",
                "size"
            ),

            economic_clv=(
                "economic_clv",
                "mean"
            ),

            beat_close=(
                "economic_clv",
                lambda x:
                    (x > 0).mean()
            )
        )
    )

    summary[
        "economic_clv"
    ] *= 100

    summary[
        "beat_close"
    ] *= 100

    print(
        summary.to_string()
    )


# ==================================================
# BLOCK BOOTSTRAP
# ==================================================

#
# Vi använder veckoblock för att inte låtsas
# att alla matcher är helt oberoende.
#

date_col = find_column([
    "Date",
    "date"
])


if date_col is not None:

    df[date_col] = pd.to_datetime(
        df[date_col]
    )

    df["week"] = (
        df[date_col]
        .dt.to_period("W")
        .astype(str)
    )

    blocks = [
        group["economic_clv"].to_numpy()
        for _, group
        in df.groupby("week")
    ]

    rng = np.random.default_rng(42)

    bootstrap = []

    for _ in range(10000):

        selected = rng.choice(
            len(blocks),
            size=len(blocks),
            replace=True
        )

        sample = np.concatenate(
            [
                blocks[i]
                for i in selected
            ]
        )

        bootstrap.append(
            sample.mean()
        )


    lower = np.percentile(
        bootstrap,
        2.5
    )

    upper = np.percentile(
        bootstrap,
        97.5
    )

    probability_negative = (
        np.mean(
            np.array(
                bootstrap
            ) <= 0
        )
    )


    print()
    print("===================================")
    print("BLOCK BOOTSTRAP")
    print("===================================")

    print(
        "Veckoblock:",
        len(blocks)
    )

    print(
        "Observed economic CLV:",
        round(
            df[
                "economic_clv"
            ].mean()
            * 100,
            3
        ),
        "%"
    )

    print(
        "95% interval:",
        round(
            lower * 100,
            3
        ),
        "to",
        round(
            upper * 100,
            3
        ),
        "%"
    )

    print(
        "P(CLV <= 0):",
        round(
            probability_negative,
            4
        )
    )


# ==================================================
# SPARA
# ==================================================

OUTPUT = (
    "data/processed/"
    "economic_clv_bets.csv"
)

df.to_csv(
    OUTPUT,
    index=False
)

print()
print(
    "Sparad till:",
    OUTPUT
)