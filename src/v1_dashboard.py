import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

STATUS_FILE = (
    PROJECT_ROOT
    / "data"
    / "forward"
    / "v1_forward_status.csv"
)

RESULT_FILE = (
    PROJECT_ROOT
    / "data"
    / "forward"
    / "v1_forward_results.csv"
)

CLV_FILE = (
    PROJECT_ROOT
    / "data"
    / "forward"
    / "v1_same_bookmaker_clv.csv"
)


# ==================================================
# LADDA DATA
# ==================================================

status = pd.read_csv(STATUS_FILE)

results = pd.read_csv(RESULT_FILE)

if CLV_FILE.exists():
    clv = pd.read_csv(CLV_FILE)
else:
    clv = pd.DataFrame()


# ==================================================
# BAS
# ==================================================

dashboard = status.copy()


# ==================================================
# RESULTAT
# ==================================================

result_cols = [
    "fixture_id",
    "home_goals",
    "away_goals",
    "actual_btts",
    "won",
    "profit",
    "result_status"
]

available = [
    c for c in result_cols
    if c in results.columns
]

dashboard = dashboard.merge(
    results[available],
    on="fixture_id",
    how="left"
)


# ==================================================
# FINAL CLV OM DEN FINNS
# ==================================================

if len(clv) > 0:

    clv_cols = [
        "fixture_id",
        "closing_odds",
        "economic_clv",
        "status"
    ]

    available = [
        c for c in clv_cols
        if c in clv.columns
    ]

    clv_small = clv[
        available
    ].copy()

    clv_small = clv_small.rename(
        columns={
            "economic_clv":
                "final_economic_clv",

            "status":
                "clv_status"
        }
    )

    dashboard = dashboard.merge(
        clv_small,
        on="fixture_id",
        how="left"
    )


# ==================================================
# FORMAT
# ==================================================

dashboard["current_clv_pct"] = (
    dashboard[
        "current_economic_clv"
    ]
    * 100
)

if "final_economic_clv" in dashboard.columns:

    dashboard["final_clv_pct"] = (
        dashboard[
            "final_economic_clv"
        ]
        * 100
    )

else:

    dashboard["final_clv_pct"] = None


dashboard["hours_to_kickoff"] = (
    dashboard[
        "minutes_to_kickoff"
    ]
    / 60
)


# ==================================================
# VISA
# ==================================================

print()
print("===================================")
print("V1 SPORTS BETTING DASHBOARD")
print("===================================")

print()

columns = [
    "start_time",
    "home_team",
    "away_team",

    "bet_side",

    "bookmaker",

    "original_odds",
    "latest_odds",

    "current_clv_pct",
    "final_clv_pct",

    "hours_to_kickoff",

    "status",

    "home_goals",
    "away_goals",

    "won",
    "profit",

    "result_status"
]


columns = [
    c for c in columns
    if c in dashboard.columns
]


print(
    dashboard[
        columns
    ].to_string(
        index=False
    )
)


# ==================================================
# SAMMANFATTNING
# ==================================================

print()
print("===================================")
print("SAMMANFATTNING")
print("===================================")

print(
    "Totala V1 bets:",
    len(dashboard)
)

finished = dashboard[
    dashboard[
        "result_status"
    ]
    ==
    "FINISHED"
]

print(
    "Färdigspelade:",
    len(finished)
)


if len(finished) > 0:

    print(
        "Wins:",
        finished[
            "won"
        ].sum()
    )

    print(
        "Profit:",
        round(
            finished[
                "profit"
            ].sum(),
            2
        ),
        "units"
    )

    print(
        "ROI:",
        round(
            finished[
                "profit"
            ].mean()
            * 100,
            2
        ),
        "%"
    )


valid_current_clv = dashboard[
    "current_economic_clv"
].dropna()

if len(valid_current_clv) > 0:

    print(
        "Current mean CLV:",
        round(
            valid_current_clv.mean()
            * 100,
            3
        ),
        "%"
    )


if "final_economic_clv" in dashboard.columns:

    valid_final_clv = dashboard[
        "final_economic_clv"
    ].dropna()

    if len(valid_final_clv) > 0:

        print(
            "Final mean CLV:",
            round(
                valid_final_clv.mean()
                * 100,
                3
            ),
            "%"
        )


# ==================================================
# SPARA
# ==================================================

OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "forward"
    / "v1_dashboard.csv"
)

dashboard.to_csv(
    OUTPUT,
    index=False
)

print()
print(
    "Sparad till:",
    OUTPUT
)