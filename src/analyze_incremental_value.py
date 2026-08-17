import pandas as pd
import numpy as np


MARKET_FILE = (
    "data/processed/"
    "market_only_bets.csv"
)

FOOTBALL_FILE = (
    "data/processed/"
    "football_model_bets.csv"
)


market = pd.read_csv(MARKET_FILE)
football = pd.read_csv(FOOTBALL_FILE)


# ==================================================
# MATCH-ID
# ==================================================

for df in [market, football]:

    df["Date"] = pd.to_datetime(
        df["Date"]
    ).dt.normalize()

    df["match_id"] = (
        df["Date"].astype(str)
        + "|"
        + df["HomeTeam"].astype(str)
        + "|"
        + df["AwayTeam"].astype(str)
    )


# ==================================================
# INDEXERA
# ==================================================

market = market.set_index(
    "match_id"
)

football = football.set_index(
    "match_id"
)


market_ids = set(
    market.index
)

football_ids = set(
    football.index
)


# ==================================================
# GRUPPER
# ==================================================

common_ids = (
    market_ids
    &
    football_ids
)

football_only_ids = (
    football_ids
    -
    market_ids
)

market_only_ids = (
    market_ids
    -
    football_ids
)


# ==================================================
# COMMON: SAMMA / MOTSATT SIDA
# ==================================================

common_market = market.loc[
    list(common_ids)
].copy()

common_football = football.loc[
    list(common_ids)
].copy()


comparison = pd.DataFrame(
    index=list(common_ids)
)

comparison["market_side"] = (
    common_market["bet_side"]
)

comparison["football_side"] = (
    common_football["bet_side"]
)

same_side_ids = comparison[
    comparison["market_side"]
    ==
    comparison["football_side"]
].index

opposite_side_ids = comparison[
    comparison["market_side"]
    !=
    comparison["football_side"]
].index


# ==================================================
# SAMMANFATTNINGSFUNKTION
# ==================================================

def summarize(
    name,
    df
):

    print()
    print(name)
    print("-----------------------------------")

    print(
        "Bets:",
        len(df)
    )

    if len(df) == 0:
        return

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
        "No-vig CLV:",
        round(
            df["novig_clv"].mean()
            * 100,
            3
        ),
        "pp"
    )

    print(
        "Beat close:",
        round(
            (
                df["novig_clv"] > 0
            ).mean()
            * 100,
            2
        ),
        "%"
    )

    print(
        "YES:",
        (
            df["bet_side"]
            == "YES"
        ).sum()
    )

    print(
        "NO:",
        (
            df["bet_side"]
            == "NO"
        ).sum()
    )


# ==================================================
# SKAPA DATASETS
# ==================================================

same_market = market.loc[
    list(same_side_ids)
].copy()

same_football = football.loc[
    list(same_side_ids)
].copy()

opposite_market = market.loc[
    list(opposite_side_ids)
].copy()

opposite_football = football.loc[
    list(opposite_side_ids)
].copy()

football_only = football.loc[
    list(football_only_ids)
].copy()

market_only = market.loc[
    list(market_only_ids)
].copy()


# ==================================================
# ÖVERSIKT
# ==================================================

print()
print("===================================")
print("INCREMENTAL VALUE")
print("===================================")

print(
    "Market bets:",
    len(market)
)

print(
    "Football bets:",
    len(football)
)

print(
    "Gemensamma matcher:",
    len(common_ids)
)

print(
    "Samma sida:",
    len(same_side_ids)
)

print(
    "Motsatt sida:",
    len(opposite_side_ids)
)

print(
    "Football-only:",
    len(football_only_ids)
)

print(
    "Market-only:",
    len(market_only_ids)
)


# ==================================================
# SAMMA SIGNAL
# ==================================================

summarize(
    "BÅDA MODELLER - SAMMA SIDA",
    same_football
)


# ==================================================
# FOOTBALL SKAPAR SPELET
# ==================================================

summarize(
    "FOOTBALL-ONLY BETS",
    football_only
)


# ==================================================
# MARKET SKAPAR SPELET
# ==================================================

summarize(
    "MARKET-ONLY BETS",
    market_only
)


# ==================================================
# MOTSATTA SIGNALER
# ==================================================

summarize(
    "MOTSATT SIDA - MARKET",
    opposite_market
)

summarize(
    "MOTSATT SIDA - FOOTBALL",
    opposite_football
)


# ==================================================
# FOOTBALL-ONLY PER SÄSONG
# ==================================================

print()
print("===================================")
print("FOOTBALL-ONLY PER SÄSONG")
print("===================================")

if len(football_only) > 0:

    season = (
        football_only
        .groupby("test_season")
        .agg(
            bets=(
                "profit",
                "size"
            ),
            roi=(
                "profit",
                "mean"
            ),
            clv=(
                "novig_clv",
                "mean"
            )
        )
    )

    season["roi"] *= 100
    season["clv"] *= 100

    print(
        season.to_string()
    )


# ==================================================
# BOOTSTRAP FOOTBALL-ONLY CLV
# ==================================================

if len(football_only) > 0:

    rng = np.random.default_rng(
        42
    )

    values = (
        football_only[
            "novig_clv"
        ]
        .to_numpy()
    )

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

    print()
    print("===================================")
    print("FOOTBALL-ONLY CLV BOOTSTRAP")
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