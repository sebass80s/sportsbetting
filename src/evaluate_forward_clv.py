import pandas as pd
import numpy as np
from pathlib import Path


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

SNAPSHOT_FILE = (
    PROJECT_ROOT
    / "data"
    / "forward"
    / "market_snapshot_history.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "forward"
    / "v1_forward_clv.csv"
)


# ==================================================
# LADDA
# ==================================================

predictions = pd.read_csv(
    PREDICTIONS_FILE
)

snapshots = pd.read_csv(
    SNAPSHOT_FILE
)


predictions["start_time"] = pd.to_datetime(
    predictions["start_time"],
    utc=True
)

predictions["prediction_timestamp"] = pd.to_datetime(
    predictions["prediction_timestamp"],
    utc=True
)

snapshots["start_time"] = pd.to_datetime(
    snapshots["start_time"],
    utc=True
)

snapshots["snapshot_timestamp"] = pd.to_datetime(
    snapshots["snapshot_timestamp"],
    utc=True
)


# ==================================================
# RESULTATRADER
# ==================================================

rows = []


for _, prediction in predictions.iterrows():

    fixture_id = prediction["fixture_id"]
    kickoff = prediction["start_time"]
    side = prediction["bet_side"]

    fixture_snapshots = snapshots[
        snapshots["fixture_id"]
        ==
        fixture_id
    ].copy()

    # Endast snapshots före kickoff
    fixture_snapshots = fixture_snapshots[
        fixture_snapshots[
            "snapshot_timestamp"
        ]
        <
        kickoff
    ].copy()

    fixture_snapshots = fixture_snapshots.sort_values(
        "snapshot_timestamp"
    )


    # ----------------------------------------------
    # INGEN SNAPSHOT
    # ----------------------------------------------

    if len(fixture_snapshots) == 0:

        rows.append({
            **prediction.to_dict(),

            "closing_snapshot_timestamp":
                pd.NaT,

            "closing_side_probability":
                np.nan,

            "snapshot_open_side_probability":
                np.nan,

            "probability_clv":
                np.nan,

            "estimated_closing_odds":
                np.nan,

            "economic_clv":
                np.nan,

            "closing_status":
                "NO_SNAPSHOT"
        })

        continue


    # ----------------------------------------------
    # SISTA SNAPSHOT FÖRE KICKOFF
    # ----------------------------------------------

    close = fixture_snapshots.iloc[-1]


    # ----------------------------------------------
    # SNAPSHOT VID/FÖRST EFTER PREDICTION
    #
    # Detta blir vår observerade startpunkt
    # för probability-CLV.
    # ----------------------------------------------

    after_prediction = fixture_snapshots[
        fixture_snapshots[
            "snapshot_timestamp"
        ]
        >=
        prediction[
            "prediction_timestamp"
        ]
    ]

    if len(after_prediction) > 0:
        first = after_prediction.iloc[0]
    else:
        first = fixture_snapshots.iloc[0]


    # ==================================================
    # YES / NO
    # ==================================================

    if side == "YES":

        first_prob = (
            first["btts_yes_prob"]
        )

        close_prob = (
            close["btts_yes_prob"]
        )

    elif side == "NO":

        first_prob = (
            first["btts_no_prob"]
        )

        close_prob = (
            close["btts_no_prob"]
        )

    else:

        continue


    # ==================================================
    # PROBABILITY CLV
    # ==================================================

    probability_clv = (
        close_prob
        -
        first_prob
    )


    # ==================================================
    # APPROX FAIR CLOSING ODDS
    # ==================================================

    if close_prob > 0:

        estimated_closing_odds = (
            1 / close_prob
        )

    else:

        estimated_closing_odds = np.nan


    # ==================================================
    # ECONOMIC CLV
    #
    # Vi använder det faktiska bästa odds som
    # loggades när signalen skapades.
    #
    # Positivt = vårt pris var bättre än
    # uppskattat fair closingpris.
    # ==================================================

    bet_odds = prediction[
        "best_bet_odds"
    ]

    if (
        pd.notna(bet_odds)
        and
        pd.notna(
            estimated_closing_odds
        )
    ):

        economic_clv = (
            bet_odds
            /
            estimated_closing_odds
            -
            1
        )

    else:

        economic_clv = np.nan


    # ==================================================
    # HUR NÄRA KICKOFF ÄR SNAPSHOTEN?
    # ==================================================

    minutes_before_kickoff = (
        kickoff
        -
        close[
            "snapshot_timestamp"
        ]
    ).total_seconds() / 60


    # Vi kallar inte snapshoten riktig closing
    # om den ligger alltför långt från kickoff.

    if minutes_before_kickoff <= 180:

        closing_status = (
            "CLOSING_PROXY"
        )

    else:

        closing_status = (
            "TOO_EARLY"
        )


    rows.append({

        **prediction.to_dict(),

        "closing_snapshot_timestamp":
            close[
                "snapshot_timestamp"
            ],

        "minutes_before_kickoff":
            minutes_before_kickoff,

        "snapshot_open_side_probability":
            first_prob,

        "closing_side_probability":
            close_prob,

        "probability_clv":
            probability_clv,

        "estimated_closing_odds":
            estimated_closing_odds,

        "economic_clv":
            economic_clv,

        "closing_status":
            closing_status
    })


# ==================================================
# DATAFRAME
# ==================================================

results = pd.DataFrame(
    rows
)


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
print("V1 FORWARD CLV")
print("===================================")

display = [
    "home_team",
    "away_team",
    "bet_side",
    "best_bet_odds",
    "closing_side_probability",
    "estimated_closing_odds",
    "probability_clv",
    "economic_clv",
    "minutes_before_kickoff",
    "closing_status"
]


print(
    results[
        display
    ].to_string(
        index=False
    )
)


# ==================================================
# SUMMERING ENDAST CLOSING PROXY
# ==================================================

valid = results[
    results["closing_status"]
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
    "Med closing proxy:",
    len(valid)
)


if len(valid) > 0:

    print(
        "Mean probability CLV:",
        round(
            valid[
                "probability_clv"
            ].mean()
            * 100,
            3
        ),
        "pp"
    )

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
        "Beat close:",
        round(
            (
                valid[
                    "economic_clv"
                ]
                >
                0
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