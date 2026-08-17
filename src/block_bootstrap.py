import pandas as pd
import numpy as np


# ==================================================
# INSTÄLLNINGAR
# ==================================================

FILE = (
    "data/processed/"
    "walk_forward_bets.csv"
)

N_BOOTSTRAPS = 5000
RANDOM_SEED = 42


# ==================================================
# LADDA DATA
# ==================================================

df = pd.read_csv(FILE)

df["Date"] = pd.to_datetime(
    df["Date"]
)


# ==================================================
# BERÄKNA NO-VIG CLV IGEN
# ==================================================

open_yes_raw = (
    1 / df["bts_yes_open"]
)

open_no_raw = (
    1 / df["bts_no_open"]
)

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


close_yes_raw = (
    1 / df["bts_yes_close"]
)

close_no_raw = (
    1 / df["bts_no_close"]
)

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


df["novig_clv"] = (
    df["close_side_probability"]
    -
    df["open_side_probability"]
)


# ==================================================
# SKAPA VECKOBLOCK
# ==================================================

# Perioden W-SUN innebär att måndag-söndag
# hamnar i samma block.

df["week"] = (
    df["Date"]
    .dt.to_period("W-SUN")
    .astype(str)
)


print()
print("===================================")
print("BLOCK BOOTSTRAP DATA")
print("===================================")

print(
    "Bets:",
    len(df)
)

print(
    "Veckoblock:",
    df["week"].nunique()
)

print(
    "Genomsnitt bets per block:",
    round(
        len(df)
        /
        df["week"].nunique(),
        2
    )
)


# ==================================================
# GRUPPERA
# ==================================================

blocks = [
    group.copy()
    for _, group
    in df.groupby("week")
]

n_blocks = len(blocks)


# ==================================================
# BOOTSTRAP
# ==================================================

rng = np.random.default_rng(
    RANDOM_SEED
)

bootstrap_clv = []
bootstrap_roi = []


for _ in range(N_BOOTSTRAPS):

    sampled_indices = rng.integers(
        0,
        n_blocks,
        size=n_blocks
    )

    sampled_blocks = [
        blocks[i]
        for i in sampled_indices
    ]

    sample = pd.concat(
        sampled_blocks,
        ignore_index=True
    )

    bootstrap_clv.append(
        sample["novig_clv"].mean()
    )

    bootstrap_roi.append(
        sample["profit"].mean()
    )


bootstrap_clv = np.array(
    bootstrap_clv
)

bootstrap_roi = np.array(
    bootstrap_roi
)


# ==================================================
# CLV RESULTAT
# ==================================================

clv_lower = np.percentile(
    bootstrap_clv,
    2.5
)

clv_upper = np.percentile(
    bootstrap_clv,
    97.5
)


print()
print("===================================")
print("BLOCK BOOTSTRAP - NO-VIG CLV")
print("===================================")

print(
    "Observerad CLV:",
    round(
        df["novig_clv"].mean()
        * 100,
        3
    ),
    "pp"
)

print(
    "95% interval:",
    round(
        clv_lower * 100,
        3
    ),
    "to",
    round(
        clv_upper * 100,
        3
    ),
    "pp"
)

print(
    "P(CL V <= 0):",
    round(
        (
            bootstrap_clv <= 0
        ).mean(),
        4
    )
)


# ==================================================
# ROI RESULTAT
# ==================================================

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
print("BLOCK BOOTSTRAP - ROI")
print("===================================")

print(
    "Observerad ROI:",
    round(
        df["profit"].mean()
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

print(
    "P(ROI > 0):",
    round(
        (
            bootstrap_roi > 0
        ).mean(),
        4
    )
)


# ==================================================
# ROBUSTHET PER SÄSONG
# ==================================================

print()
print("===================================")
print("CLV PER SÄSONG")
print("===================================")

season_results = (
    df.groupby("test_season")
    .agg(
        bets=("novig_clv", "size"),
        clv=("novig_clv", "mean"),
        roi=("profit", "mean")
    )
)

season_results["clv"] *= 100
season_results["roi"] *= 100

print(
    season_results.to_string()
)