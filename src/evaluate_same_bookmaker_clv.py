import pandas as pd
import numpy as np
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "forward"
    / "v1_predictions.csv"
)

RAW_HISTORY_FILE = (
    PROJECT_ROOT
    / "data"
    / "forward"
    / "market_raw_history.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "forward"
    / "v1_same_bookmaker_clv.csv"
)


# ==================================================
# LADDA
# ==================================================

predictions = pd.read_csv(
    PREDICTIONS_FILE
)

history = pd.read_csv(
    RAW_HISTORY_FILE
)


predictions["start_time"] = pd.to_datetime(
    predictions["start_time"],
    utc=True
)

predictions["prediction_timestamp"] = pd.to_datetime(
    predictions["prediction_timestamp"],
    utc=True
)

history["snapshot_timestamp"] = pd.to_datetime(
    history["snapshot_timestamp"],
    utc=True
)

history["start_time"] = pd.to_datetime(
    history["start_time"],
    utc=True
)


# ==================================================
# RESULTAT
# ==================================================

rows = []


for _, bet in predictions.iterrows():

    fixture_id = bet["fixture_id"]
    bookmaker = bet["best_bookmaker"]
    side = bet["bet_side"]
    kickoff = bet["start_time"]

    # ----------------------------------------------
    # SAMMA FIXTURE + SAMMA BOOKMAKER
    # ----------------------------------------------

    snapshots = history[
        (history["fixture_id"] == fixture_id)
        &
        (history["bookmaker"] == bookmaker)
        &
        (history["snapshot_timestamp"] < kickoff)
    ].copy()

    snapshots = snapshots.sort_values(
        "snapshot_timestamp"
    )


    if len(snapshots) == 0:

        rows.append({
            **bet.to_dict(),

            "closing_snapshot":
                pd.NaT,

            "closing_odds":
                np.nan,

            "economic_clv":
                np.nan,

            "minutes_before_kickoff":
                np.nan,

            "status":
                "NO_SNAPSHOT"
        })

        continue


    # ----------------------------------------------
    # SISTA OBSERVATIONEN FÖRE AVSPARK
    # ----------------------------------------------

    close = snapshots.iloc[-1]


    if side == "YES":

        closing_odds = (
            close["btts_yes"]
        )

    elif side == "NO":

        closing_odds = (
            close["btts_no"]
        )

    else:

        continue


    minutes_before = (
        kickoff
        -
        close["snapshot_timestamp"]
    ).total_seconds() / 60


    # ----------------------------------------------
    # ECONOMIC CLV
    # ----------------------------------------------

    opening_odds = bet[
        "best_bet_odds"
    ]

    if (
        pd.notna(opening_odds)
        and
        pd.notna(closing_odds)
        and
        closing_odds > 1
    ):

        economic_clv = (
            opening_odds
            /
            closing_odds
            -
            1
        )

    else:

        economic_clv = np.nan


    # ----------------------------------------------
    # STATUS
    # ----------------------------------------------

    if minutes_before <= 180:

        status = "CLOSING_PROXY"

    else:

        status = "TOO_EARLY"


    rows.append({
        **bet.to_dict(),

        "closing_snapshot":
            close[
                "snapshot_timestamp"
            ],

        "closing_odds":
            closing_odds,

        "economic_clv":
            economic_clv,

        "minutes_before_kickoff":
            minutes_before,

        "status":
            status
    })


results = pd.DataFrame(rows)


# ==================================================
# SPARA
# ==================================================

results.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==================================================
# VISA
# ==================================================

print()
print("===================================")
print("SAME-BOOKMAKER CLV")
print("===================================")

columns = [
    "home_team",
    "away_team",
    "bet_side",
    "best_bookmaker",
    "best_bet_odds",
    "closing_odds",
    "economic_clv",
    "minutes_before_kickoff",
    "status"
]

print(
    results[
        columns
    ].to_string(index=False)
)


# ==================================================
# SUMMERING
# ==================================================

valid = results[
    results["status"]
    ==
    "CLOSING_PROXY"
].copy()


print()
print("===================================")
print("SAMMANFATTNING")
print("===================================")

print(
    "Forward bets:",
    len(results)
)

print(
    "Closing proxies:",
    len(valid)
)


if len(valid) > 0:

    print(
        "Mean economic CLV:",
        round(
            valid[
                "economic_clv"
            ].mean()
            * 100,
            3
        ),
        "%"
    )

    print(
        "Median economic CLV:",
        round(
            valid[
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
                valid[
                    "economic_clv"
                ] > 0
            ).mean()
            * 100,
            2
        ),
        "%"
    )


print()
print(
    "Sparad till:",
    OUTPUT_FILE
)