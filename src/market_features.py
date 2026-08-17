import pandas as pd


# ==================================================
# LADDA DATA
# ==================================================

FILE = "data/raw/premier_league_2015_2025.csv"

df = pd.read_csv(FILE)

df["date"] = pd.to_datetime(df["date"])


# ==================================================
# KRÄV KOMPLETTA MARKNADSODDS
# ==================================================

market = df[
    df["avg_over_2_5_odds"].notna()
    &
    df["avg_under_2_5_odds"].notna()
    &
    df["avg_home_odds"].notna()
    &
    df["avg_draw_odds"].notna()
    &
    df["avg_away_odds"].notna()
].copy()


# ==================================================
# OVER / UNDER 2.5
# ==================================================

market["raw_over_prob"] = (
    1 / market["avg_over_2_5_odds"]
)

market["raw_under_prob"] = (
    1 / market["avg_under_2_5_odds"]
)

market["ou_overround"] = (
    market["raw_over_prob"]
    +
    market["raw_under_prob"]
)

market["market_over_probability"] = (
    market["raw_over_prob"]
    /
    market["ou_overround"]
)

market["market_under_probability"] = (
    market["raw_under_prob"]
    /
    market["ou_overround"]
)


# ==================================================
# 1X2
# ==================================================

market["raw_home_prob"] = (
    1 / market["avg_home_odds"]
)

market["raw_draw_prob"] = (
    1 / market["avg_draw_odds"]
)

market["raw_away_prob"] = (
    1 / market["avg_away_odds"]
)

market["match_overround"] = (
    market["raw_home_prob"]
    +
    market["raw_draw_prob"]
    +
    market["raw_away_prob"]
)

market["market_home_probability"] = (
    market["raw_home_prob"]
    /
    market["match_overround"]
)

market["market_draw_probability"] = (
    market["raw_draw_prob"]
    /
    market["match_overround"]
)

market["market_away_probability"] = (
    market["raw_away_prob"]
    /
    market["match_overround"]
)


# ==================================================
# BALANS / DOMINANS
# ==================================================

# Hur stor skillnad ser marknaden mellan lagen?
market["market_strength_difference"] = abs(
    market["market_home_probability"]
    -
    market["market_away_probability"]
)

# Högre värde = jämnare match
market["market_balance"] = (
    1
    -
    market["market_strength_difference"]
)


# ==================================================
# VISA
# ==================================================

columns = [
    "date",
    "home_team",
    "away_team",

    "market_over_probability",

    "market_home_probability",
    "market_draw_probability",
    "market_away_probability",

    "market_balance",

    "btts"
]


print()
print("===================================")
print("MARKET FEATURES V2")
print("===================================")

print(
    market[
        columns
    ].head(10).to_string(index=False)
)


print()
print("Antal matcher:", len(market))

print(
    "Genomsnitt Over probability:",
    round(
        market[
            "market_over_probability"
        ].mean(),
        4
    )
)

print(
    "Genomsnitt Home probability:",
    round(
        market[
            "market_home_probability"
        ].mean(),
        4
    )
)

print(
    "Genomsnitt Draw probability:",
    round(
        market[
            "market_draw_probability"
        ].mean(),
        4
    )
)

print(
    "Genomsnitt Away probability:",
    round(
        market[
            "market_away_probability"
        ].mean(),
        4
    )
)


# ==================================================
# KONTROLL
# ==================================================

probability_sum = (
    market["market_home_probability"]
    +
    market["market_draw_probability"]
    +
    market["market_away_probability"]
)

print(
    "1X2 probability sum:",
    round(
        probability_sum.mean(),
        6
    )
)


# ==================================================
# SPARA
# ==================================================

output_file = (
    "data/processed/market_features_v2.csv"
)

market.to_csv(
    output_file,
    index=False
)

print()
print(
    "Sparad till:",
    output_file
)