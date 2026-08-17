import pandas as pd
import numpy as np

from pathlib import Path
from datetime import datetime, timezone


# ==================================================
# FILER
# ==================================================

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
    / "v1_forward_status.csv"
)


# ==================================================
# INSTÄLLNINGAR
# ==================================================

CLOSING_WINDOW_MINUTES = 180

# När detta program körs nära match:
# vi flaggar matcher inom 6 timmar.
ATTENTION_WINDOW_HOURS = 6


# ==================================================
# LADDA DATA
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


now = pd.Timestamp(
    datetime.now(timezone.utc)
)


# ==================================================
# ANALYSERA VARJE FRYST V1-SIGNAL
# ==================================================

rows = []


for _, bet in predictions.iterrows():

    fixture_id = bet["fixture_id"]
    kickoff = bet["start_time"]
    bookmaker = bet["best_bookmaker"]
    side = bet["bet_side"]
    original_odds = bet["best_bet_odds"]

    minutes_to_kickoff = (
        kickoff - now
    ).total_seconds() / 60


    # ==================================================
    # BOOKMAKER-SNAPSHOTS FÖR DENNA MATCH
    # ==================================================

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


    latest_snapshot = pd.NaT
    latest_odds = np.nan
    economic_clv = np.nan
    minutes_snapshot_before_kickoff = np.nan


    if len(snapshots) > 0:

        latest = snapshots.iloc[-1]

        latest_snapshot = latest[
            "snapshot_timestamp"
        ]

        if side == "YES":

            latest_odds = latest[
                "btts_yes"
            ]

        elif side == "NO":

            latest_odds = latest[
                "btts_no"
            ]


        minutes_snapshot_before_kickoff = (
            kickoff
            -
            latest_snapshot
        ).total_seconds() / 60


        if (
            pd.notna(original_odds)
            and
            pd.notna(latest_odds)
            and
            latest_odds > 1
        ):

            economic_clv = (
                original_odds
                /
                latest_odds
                -
                1
            )


    # ==================================================
    # STATUS
    # ==================================================

    if minutes_to_kickoff > ATTENTION_WINDOW_HOURS * 60:

        status = "WAITING"

    elif (
        minutes_to_kickoff > 0
        and
        minutes_to_kickoff
        <= ATTENTION_WINDOW_HOURS * 60
    ):

        if (
            pd.notna(
                minutes_snapshot_before_kickoff
            )
            and
            minutes_snapshot_before_kickoff
            <= CLOSING_WINDOW_MINUTES
        ):

            status = "CLOSING_SNAPSHOT_READY"

        else:

            status = "TAKE_SNAPSHOT"

    else:

        # Matchen har startat
        if (
            pd.notna(
                minutes_snapshot_before_kickoff
            )
            and
            minutes_snapshot_before_kickoff
            <= CLOSING_WINDOW_MINUTES
        ):

            status = "AWAITING_RESULT"

        else:

            status = "NO_CLOSE_SNAPSHOT"


    rows.append({

        "fixture_id":
            fixture_id,

        "start_time":
            kickoff,

        "home_team":
            bet["home_team"],

        "away_team":
            bet["away_team"],

        "bet_side":
            side,

        "original_odds":
            original_odds,

        "bookmaker":
            bookmaker,

        "predicted_movement":
            bet["predicted_movement"],

        "minutes_to_kickoff":
            minutes_to_kickoff,

        "latest_snapshot":
            latest_snapshot,

        "latest_odds":
            latest_odds,

        "snapshot_minutes_before_kickoff":
            minutes_snapshot_before_kickoff,

        "current_economic_clv":
            economic_clv,

        "status":
            status
    })


status_df = pd.DataFrame(
    rows
)


# ==================================================
# SPARA
# ==================================================

status_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==================================================
# VISA
# ==================================================

print()
print("===================================")
print("V1 FORWARD MONITOR")
print("===================================")

print(
    "Nu:",
    now
)

print(
    "Frysta V1-signaler:",
    len(status_df)
)

print()


display = status_df.copy()

display[
    "hours_to_kickoff"
] = (
    display[
        "minutes_to_kickoff"
    ]
    /
    60
)

display[
    "clv_percent"
] = (
    display[
        "current_economic_clv"
    ]
    *
    100
)


columns = [
    "start_time",
    "home_team",
    "away_team",
    "bet_side",
    "bookmaker",
    "original_odds",
    "latest_odds",
    "clv_percent",
    "hours_to_kickoff",
    "status"
]


print(
    display[
        columns
    ].to_string(
        index=False,
        formatters={
            "clv_percent":
                lambda x:
                    ""
                    if pd.isna(x)
                    else f"{x:.2f}%",

            "hours_to_kickoff":
                lambda x:
                    f"{x:.1f}"
        }
    )
)


# ==================================================
# MATCHER SOM KRÄVER UPPMÄRKSAMHET
# ==================================================

attention = status_df[
    status_df["status"].isin(
        [
            "TAKE_SNAPSHOT",
            "NO_CLOSE_SNAPSHOT"
        ]
    )
]


print()
print("===================================")
print("KRÄVER UPPMÄRKSAMHET")
print("===================================")

if len(attention) == 0:

    print(
        "Inga matcher kräver åtgärd just nu."
    )

else:

    print(
        attention[
            [
                "start_time",
                "home_team",
                "away_team",
                "status"
            ]
        ].to_string(
            index=False
        )
    )


# ==================================================
# CLOSING READY
# ==================================================

ready = status_df[
    status_df["status"].isin(
        [
            "CLOSING_SNAPSHOT_READY",
            "AWAITING_RESULT"
        ]
    )
]


print()
print("===================================")
print("CLOSING DATA READY")
print("===================================")

if len(ready) == 0:

    print(
        "Inga closing-proxies klara ännu."
    )

else:

    print(
        ready[
            [
                "home_team",
                "away_team",
                "bet_side",
                "original_odds",
                "latest_odds",
                "current_economic_clv"
            ]
        ].to_string(
            index=False
        )
    )


print()
print(
    "Sparad till:",
    OUTPUT_FILE
)