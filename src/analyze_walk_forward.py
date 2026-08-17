import pandas as pd
import numpy as np


FILE = (
    "data/processed/"
    "walk_forward_bets.csv"
)

df = pd.read_csv(FILE)


# ==================================================
# NO-VIG CLV
# ==================================================

# Opening probabilities
open_yes_raw = 1 / df["bts_yes_open"]
open_no_raw = 1 / df["bts_no_open"]

open_total = (
    open_yes_raw
    +
    open_no_raw
)

df["open_yes_fair"] = (
    open_yes_raw
    /
    open_total
)

df["open_no_fair"] = (
    open_no_raw
    /
    open_total
)


# Closing probabilities
close_yes_raw = 1 / df["bts_yes_close"]
close_no_raw = 1 / df["bts_no_close"]

close_total = (
    close_yes_raw
    +
    close_no_raw
)

df["close_yes_fair"] = (
    close_yes_raw
    /
    close_total
)

df["close_no_fair"] = (
    close_no_raw
    /
    close_total
)


# ==================================================
# FAIR PROBABILITY PÅ VÅR BET-SIDA
# ==================================================

df["open_side_probability"] = np.where(
    df["bet_side"] == "YES",
    df["open_yes_fair"],
    df["open_no_fair"]
)

df["close_side_probability"] = np.where(
    df["bet_side"] == "YES",
    df["close_yes_fair"],
    df["close_no_fair"]
)


# Positivt = marknaden rörde sig i vår riktning
df["novig_clv_pp"] = (
    df["close_side_probability"]
    -
    df["open_side_probability"]
)


# ==================================================
# SIGNALSTYRKA
# ==================================================

df["signal_strength"] = abs(
    df["predicted_movement"]
)


# ==================================================
# TOTALT
# ==================================================

print()
print("===================================")
print("WALK-FORWARD DIAGNOSTIK")
print("===================================")

print(
    "Bets:",
    len(df)
)

print(
    "ROI:",
    round(
        df["profit"].mean()
        * 100,
        2
    ),
    "%"
)

print(
    "Raw odds CLV:",
    round(
        df["clv"].mean()
        * 100,
        2
    ),
    "%"
)

print(
    "No-vig CLV:",
    round(
        df["novig_clv_pp"].mean()
        * 100,
        3
    ),
    "pp"
)

print(
    "Median no-vig CLV:",
    round(
        df["novig_clv_pp"].median()
        * 100,
        3
    ),
    "pp"
)

print(
    "Marknaden rörde sig vår riktning:",
    round(
        (
            df["novig_clv_pp"] > 0
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
        "Hit rate:",
        round(
            group["won"].mean()
            * 100,
            2
        ),
        "%"
    )

    print(
        "Average odds:",
        round(
            group["bet_odds"].mean(),
            3
        )
    )

    print(
        "ROI:",
        round(
            group["profit"].mean()
            * 100,
            2
        ),
        "%"
    )

    print(
        "No-vig CLV:",
        round(
            group["novig_clv_pp"].mean()
            * 100,
            3
        ),
        "pp"
    )


# ==================================================
# SIGNAL BUCKETS
# ==================================================

bins = [
    0.005,
    0.0075,
    0.010,
    0.015,
    0.020,
    1.0
]

labels = [
    "0.5-0.75 pp",
    "0.75-1.0 pp",
    "1.0-1.5 pp",
    "1.5-2.0 pp",
    "2.0+ pp"
]

df["signal_bin"] = pd.cut(
    df["signal_strength"],
    bins=bins,
    labels=labels,
    include_lowest=True,
    right=False
)


summary = (
    df.groupby(
        "signal_bin",
        observed=True
    )
    .agg(
        bets=(
            "profit",
            "size"
        ),

        avg_signal=(
            "signal_strength",
            "mean"
        ),

        roi=(
            "profit",
            "mean"
        ),

        novig_clv=(
            "novig_clv_pp",
            "mean"
        ),

        beat_close=(
            "novig_clv_pp",
            lambda x: (
                x > 0
            ).mean()
        )
    )
)


summary["avg_signal"] *= 100
summary["roi"] *= 100
summary["novig_clv"] *= 100
summary["beat_close"] *= 100


print()
print("===================================")
print("SIGNAL BUCKETS")
print("===================================")

print(
    summary.to_string()
)


# ==================================================
# PER SÄSONG OCH SIDA
# ==================================================

print()
print("===================================")
print("SÄSONG × SIDA")
print("===================================")

season_side = (
    df.groupby(
        [
            "test_season",
            "bet_side"
        ]
    )
    .agg(
        bets=(
            "profit",
            "size"
        ),

        roi=(
            "profit",
            "mean"
        ),

        novig_clv=(
            "novig_clv_pp",
            "mean"
        )
    )
)

season_side["roi"] *= 100
season_side["novig_clv"] *= 100

print(
    season_side.to_string()
)


# ==================================================
# BOOTSTRAP CLV
# ==================================================

rng = np.random.default_rng(
    42
)

values = df[
    "novig_clv_pp"
].to_numpy()

bootstrap_means = []

for _ in range(10000):

    sample = rng.choice(
        values,
        size=len(values),
        replace=True
    )

    bootstrap_means.append(
        sample.mean()
    )


lower = np.percentile(
    bootstrap_means,
    2.5
)

upper = np.percentile(
    bootstrap_means,
    97.5
)


print()
print("===================================")
print("BOOTSTRAP NO-VIG CLV")
print("===================================")

print(
    "Mean:",
    round(
        values.mean()
        * 100,
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
# BOOTSTRAP ROI
# ==================================================

profits = df[
    "profit"
].to_numpy()

bootstrap_roi = []

for _ in range(10000):

    sample = rng.choice(
        profits,
        size=len(profits),
        replace=True
    )

    bootstrap_roi.append(
        sample.mean()
    )


roi_lower = np.percentile(
    bootstrap_roi,
    2.5
)

roi_upper = np.percentile(
    bootstrap_roi,
    97.5
)


print()
print("===================================")
print("BOOTSTRAP ROI")
print("===================================")

print(
    "Mean ROI:",
    round(
        profits.mean()
        * 100,
        2
    ),
    "%"
)

print(
    "95% interval:",
    round(
        roi_lower * 100,
        2
    ),
    "to",
    round(
        roi_upper * 100,
        2
    ),
    "%"
)