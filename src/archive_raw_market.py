import pandas as pd
from pathlib import Path
from datetime import datetime, timezone


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CURRENT_FILE = (
    PROJECT_ROOT
    / "data"
    / "forward"
    / "market_raw.csv"
)

HISTORY_FILE = (
    PROJECT_ROOT
    / "data"
    / "forward"
    / "market_raw_history.csv"
)


# ==================================================
# LADDA AKTUELL SNAPSHOT
# ==================================================

current = pd.read_csv(
    CURRENT_FILE
)

snapshot_timestamp = (
    datetime.now(
        timezone.utc
    ).isoformat()
)

current[
    "snapshot_timestamp"
] = snapshot_timestamp


# ==================================================
# LADDA HISTORIK
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
print("RAW MARKET SNAPSHOT")
print("===================================")

print(
    "Timestamp:",
    snapshot_timestamp
)

print(
    "Bookmaker-rader:",
    len(current)
)

print(
    "Bookmakers:",
    current[
        "bookmaker"
    ].nunique()
)

print(
    "Fixtures:",
    current[
        "fixture_id"
    ].nunique()
)

print(
    "Totalt historikrader:",
    len(history)
)

print(
    "Snapshot-tider:",
    history[
        "snapshot_timestamp"
    ].nunique()
)

print()
print(
    "Sparad till:",
    HISTORY_FILE
)