import pandas as pd
from pathlib import Path
from datetime import datetime, timezone


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONSENSUS_FILE = (
    PROJECT_ROOT
    / "data"
    / "forward"
    / "market_consensus.csv"
)

HISTORY_FILE = (
    PROJECT_ROOT
    / "data"
    / "forward"
    / "market_snapshot_history.csv"
)


# ==================================================
# LADDA SENASTE MARKNAD
# ==================================================

current = pd.read_csv(
    CONSENSUS_FILE
)

snapshot_time = datetime.now(
    timezone.utc
).isoformat()

current[
    "snapshot_timestamp"
] = snapshot_time


# ==================================================
# KOLUMNER VI VILL BEVARA
# ==================================================

columns = [
    "snapshot_timestamp",
    "fixture_id",
    "start_time",
    "home_team",
    "away_team",
    "bookmakers",

    "btts_yes_prob",
    "btts_no_prob",

    "avg_btts_yes_odds",
    "avg_btts_no_odds",

    "over_25_prob",

    "home_prob",
    "draw_prob",
    "away_prob"
]

current = current[
    columns
].copy()


# ==================================================
# LADDA TIDIGARE SNAPSHOTS
# ==================================================

if HISTORY_FILE.exists():

    try:
        history = pd.read_csv(
            HISTORY_FILE
        )

    except pd.errors.EmptyDataError:
        history = pd.DataFrame()

else:
    history = pd.DataFrame()


# ==================================================
# LÄGG TILL
# ==================================================

history = pd.concat(
    [
        history,
        current
    ],
    ignore_index=True
)


history.to_csv(
    HISTORY_FILE,
    index=False
)


# ==================================================
# RESULTAT
# ==================================================

print()
print("===================================")
print("MARKET SNAPSHOT ARCHIVED")
print("===================================")

print(
    "Timestamp:",
    snapshot_time
)

print(
    "Matcher i snapshot:",
    len(current)
)

print(
    "Totalt snapshots-rader:",
    len(history)
)

print(
    "Unika snapshot-tider:",
    history[
        "snapshot_timestamp"
    ].nunique()
)

print()
print(
    "Sparad till:",
    HISTORY_FILE
)