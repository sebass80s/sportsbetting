import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FORWARD_DIR = PROJECT_ROOT / "data" / "forward"
PREDICTIONS_FILE = FORWARD_DIR / "v1_predictions.csv"
RAW_HISTORY_FILE = FORWARD_DIR / "market_raw_history.csv"

# Closing-only pipeline. Inga score-anrop här.
# Resultat uppdateras separat från Streamlit-sidan när användaren väljer det.
SNAPSHOT_SCRIPTS = [
    "build_market_consensus.py",
    "archive_market_snapshot.py",
    "archive_raw_market.py",
    "v1_forward_monitor.py",
    "ou_v1_forward_monitor.py",
    "evaluate_forward_clv.py",
    "evaluate_same_bookmaker_clv.py",
    "v1_dashboard.py",
]

# Budgetinställningar för eventuell automatisk closing-körning.
AUTO_WINDOW_MINUTES = 180
AUTO_MIN_SNAPSHOT_AGE_MINUTES = 90


def run_scripts(scripts):
    for script in scripts:
        script_path = PROJECT_ROOT / "src" / script

        print("-----------------------------------")
        print("Kör:", script)
        print("-----------------------------------")

        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT
        )

        if result.returncode != 0:
            print()
            print("FEL!")
            print(script, "returnerade kod", result.returncode)
            sys.exit(result.returncode)


def automatic_snapshot_needed():
    """Avgör om en closing-snapshot behövs utan något API-anrop."""

    if not PREDICTIONS_FILE.exists():
        return False, "v1_predictions.csv saknas"

    try:
        predictions = pd.read_csv(PREDICTIONS_FILE)
    except pd.errors.EmptyDataError:
        return False, "inga frysta V1-spel"

    if len(predictions) == 0:
        return False, "inga frysta V1-spel"

    predictions["start_time"] = pd.to_datetime(
        predictions["start_time"],
        utc=True,
        errors="coerce"
    )

    now = pd.Timestamp(datetime.now(timezone.utc))

    predictions["minutes_to_kickoff"] = (
        predictions["start_time"] - now
    ).dt.total_seconds() / 60

    candidates = predictions[
        (predictions["minutes_to_kickoff"] > 0)
        & (predictions["minutes_to_kickoff"] <= AUTO_WINDOW_MINUTES)
    ].copy()

    if len(candidates) == 0:
        return False, "ingen fryst V1-match inom 3 timmar"

    if not RAW_HISTORY_FILE.exists():
        return True, "match inom 3 timmar och ingen snapshot-historik"

    try:
        history = pd.read_csv(RAW_HISTORY_FILE)
    except pd.errors.EmptyDataError:
        return True, "match inom 3 timmar och tom snapshot-historik"

    if len(history) == 0 or "snapshot_timestamp" not in history.columns:
        return True, "match inom 3 timmar och ingen användbar historik"

    history["snapshot_timestamp"] = pd.to_datetime(
        history["snapshot_timestamp"],
        utc=True,
        errors="coerce"
    )

    candidate_ids = set(candidates["fixture_id"].astype(str))

    candidate_history = history[
        history["fixture_id"].astype(str).isin(candidate_ids)
    ].copy()

    if len(candidate_history) == 0:
        return True, "match inom 3 timmar utan tidigare snapshot"

    latest_snapshot = candidate_history["snapshot_timestamp"].max()

    if pd.isna(latest_snapshot):
        return True, "match inom 3 timmar utan giltig snapshot-tid"

    age_minutes = (now - latest_snapshot).total_seconds() / 60

    if age_minutes >= AUTO_MIN_SNAPSHOT_AGE_MINUTES:
        return True, f"senaste relevanta snapshot är {age_minutes:.0f} min gammal"

    return False, f"senaste relevanta snapshot är bara {age_minutes:.0f} min gammal"


manual = "--manual" in sys.argv

print()
print("===================================")

if manual:
    print("MANUAL MARKET SNAPSHOT")
    print("===================================")
    print("Start:", datetime.now().isoformat(timespec="seconds"))
    print()
    print("Budgetläge: inga score-anrop görs i denna pipeline.")
    print()
    run_scripts(SNAPSHOT_SCRIPTS)

else:
    print("AUTO CLOSING SNAPSHOT")
    print("===================================")

    needed, reason = automatic_snapshot_needed()

    print("Beslut:", "KÖR" if needed else "HOPPA ÖVER")
    print("Orsak:", reason)

    if not needed:
        raise SystemExit(0)

    print()
    print("Budgetläge: hämtar bara marknaden för closing-data.")
    print()
    run_scripts(SNAPSHOT_SCRIPTS)


print()
print("===================================")
print("SNAPSHOT KLAR")
print("===================================")
print("Slut:", datetime.now().isoformat(timespec="seconds"))
