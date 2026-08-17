import pandas as pd

FILE = "data/raw/premier_league_2015_2025.csv"

df = pd.read_csv(FILE)

print()
print("===================================")
print("ODDS DATA")
print("===================================")
print()

for season, group in df.groupby("season"):

    total = len(group)

    over_available = (
        group["avg_over_2_5_odds"]
        .notna()
        .sum()
    )

    under_available = (
        group["avg_under_2_5_odds"]
        .notna()
        .sum()
    )

    both_available = (
        group["avg_over_2_5_odds"].notna()
        &
        group["avg_under_2_5_odds"].notna()
    ).sum()

    print(
        season,
        "| matcher:",
        total,
        "| O2.5:",
        over_available,
        "| U2.5:",
        under_available,
        "| båda:",
        both_available
    )


print()
print("TOTALT")
print("-----------------------------------")

both = (
    df["avg_over_2_5_odds"].notna()
    &
    df["avg_under_2_5_odds"].notna()
)

print(
    "Matcher med kompletta O/U-odds:",
    both.sum()
)

print(
    "Andel:",
    round(
        both.mean() * 100,
        2
    ),
    "%"
)