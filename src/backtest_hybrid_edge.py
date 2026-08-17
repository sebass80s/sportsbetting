import pandas as pd
import numpy as np


# ==================================================
# DATA
# ==================================================

FILE = "data/processed/hybrid_btts_predictions.csv"

df = pd.read_csv(FILE)


# ==================================================
# KONTROLLERA ODDS
# ==================================================

required = [
    "hybrid_probability",
    "market_btts_probability",
    "actual_btts",
    "bts_yes_open",
    "bts_no_open",
    "bts_yes_close",
    "bts_no_close",
    "test_season"
]

df = df.dropna(subset=required).copy()


# ==================================================
# NO-VIG OPENING PROBABILITIES
# ==================================================

yes_open_raw = 1 / df["bts_yes_open"]
no_open_raw = 1 / df["bts_no_open"]

total_open = yes_open_raw + no_open_raw

df["yes_open_novig"] = (
    yes_open_raw / total_open
)

df["no_open_novig"] = (
    no_open_raw / total_open
)


# ==================================================
# NO-VIG CLOSING PROBABILITIES
# ==================================================

yes_close_raw = 1 / df["bts_yes_close"]
no_close_raw = 1 / df["bts_no_close"]

total_close = yes_close_raw + no_close_raw

df["yes_close_novig"] = (
    yes_close_raw / total_close
)

df["no_close_novig"] = (
    no_close_raw / total_close
)


# ==================================================
# HYBRID EDGE
# ==================================================

df["yes_edge"] = (
    df["hybrid_probability"]
    -
    df["yes_open_novig"]
)

df["no_edge"] = (
    (1 - df["hybrid_probability"])
    -
    df["no_open_novig"]
)


# ==================================================
# SKAPA BETS
# ==================================================

bets = []


for _, row in df.iterrows():

    # YES
    if row["yes_edge"] > 0:

        won = (
            row["actual_btts"] == 1
        )

        profit = (
            row["bts_yes_open"] - 1
            if won
            else -1
        )

        clv = (
            row["yes_close_novig"]
            -
            row["yes_open_novig"]
        )

        bets.append({
            "test_season":
                row["test_season"],

            "bet_side":
                "YES",

            "edge":
                row["yes_edge"],

            "open_probability":
                row["yes_open_novig"],

            "close_probability":
                row["yes_close_novig"],

            "open_odds":
                row["bts_yes_open"],

            "won":
                int(won),

            "profit":
                profit,

            "novig_clv":
                clv
        })

    # NO
    if row["no_edge"] > 0:

        won = (
            row["actual_btts"] == 0
        )

        profit = (
            row["bts_no_open"] - 1
            if won
            else -1
        )

        clv = (
            row["no_close_novig"]
            -
            row["no_open_novig"]
        )

        bets.append({
            "test_season":
                row["test_season"],

            "bet_side":
                "NO",

            "edge":
                row["no_edge"],

            "open_probability":
                row["no_open_novig"],

            "close_probability":
                row["no_close_novig"],

            "open_odds":
                row["bts_no_open"],

            "won":
                int(won),

            "profit":
                profit,

            "novig_clv":
                clv
        })


bets = pd.DataFrame(bets)


# ==================================================
# EDGE BUCKETS
# ==================================================

bins = [
    0,
    0.025,
    0.05,
    0.10,
    1
]

labels = [
    "0-2.5 pp",
    "2.5-5 pp",
    "5-10 pp",
    "10+ pp"
]

bets["edge_bin"] = pd.cut(
    bets["edge"],
    bins=bins,
    labels=labels,
    include_lowest=True
)


# ==================================================
# SUMMARY FUNCTION
# ==================================================

def summarize(df, title):

    print()
    print(title)
    print("-----------------------------------")

    if len(df) == 0:
        print("Inga bets")
        return

    print(
        "Bets:",
        len(df)
    )

    print(
        "Hit rate:",
        round(
            df["won"].mean() * 100,
            2
        ),
        "%"
    )

    print(
        "Average odds:",
        round(
            df["open_odds"].mean(),
            3
        )
    )

    print(
        "Profit:",
        round(
            df["profit"].sum(),
            2
        ),
        "units"
    )

    print(
        "ROI:",
        round(
            df["profit"].mean() * 100,
            2
        ),
        "%"
    )

    print(
        "No-vig CLV:",
        round(
            df["novig_clv"].mean() * 100,
            3
        ),
        "pp"
    )

    print(
        "Beat close:",
        round(
            (df["novig_clv"] > 0).mean()
            * 100,
            2
        ),
        "%"
    )


# ==================================================
# TOTALT
# ==================================================

print()
print("===================================")
print("HYBRID EDGE BETTING")
print("===================================")

summarize(
    bets,
    "ALLA POSITIVA EDGES"
)


# ==================================================
# YES / NO
# ==================================================

print()
print("===================================")
print("PER BET-SIDA")
print("===================================")

for side in ["YES", "NO"]:

    subset = bets[
        bets["bet_side"] == side
    ]

    summarize(
        subset,
        side
    )


# ==================================================
# EDGE BUCKETS
# ==================================================

print()
print("===================================")
print("EDGE BUCKETS")
print("===================================")

for edge_bin in labels:

    subset = bets[
        bets["edge_bin"] == edge_bin
    ]

    summarize(
        subset,
        edge_bin
    )


# ==================================================
# YES EDGE BUCKETS
# ==================================================

print()
print("===================================")
print("YES EDGE BUCKETS")
print("===================================")

yes_bets = bets[
    bets["bet_side"] == "YES"
]

for edge_bin in labels:

    subset = yes_bets[
        yes_bets["edge_bin"] == edge_bin
    ]

    summarize(
        subset,
        edge_bin
    )


# ==================================================
# NO EDGE BUCKETS
# ==================================================

print()
print("===================================")
print("NO EDGE BUCKETS")
print("===================================")

no_bets = bets[
    bets["bet_side"] == "NO"
]

for edge_bin in labels:

    subset = no_bets[
        no_bets["edge_bin"] == edge_bin
    ]

    summarize(
        subset,
        edge_bin
    )


# ==================================================
# PER SÄSONG
# ==================================================

print()
print("===================================")
print("PER SÄSONG")
print("===================================")

for season, group in bets.groupby(
    "test_season"
):

    summarize(
        group,
        str(season)
    )


# ==================================================
# BOOTSTRAP CLV
# ==================================================

print()
print("===================================")
print("BOOTSTRAP CLV")
print("===================================")

rng = np.random.default_rng(42)

values = bets["novig_clv"].to_numpy()

bootstrap = []

for _ in range(10000):

    sample = rng.choice(
        values,
        size=len(values),
        replace=True
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

print(
    "Mean:",
    round(
        values.mean() * 100,
        3
    ),
    "pp"
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
    "pp"
)


# ==================================================
# SPARA
# ==================================================

OUTPUT = (
    "data/processed/"
    "hybrid_edge_bets.csv"
)

bets.to_csv(
    OUTPUT,
    index=False
)

print()
print(
    "Sparad till:",
    OUTPUT
)