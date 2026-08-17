import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_FILE = PROJECT_ROOT / "data/forward/ou_v1_predictions.csv"
RAW_HISTORY_FILE = PROJECT_ROOT / "data/forward/market_raw_history.csv"
OUTPUT_FILE = PROJECT_ROOT / "data/forward/ou_v1_forward_status.csv"

CLOSING_WINDOW_MINUTES = 180
ATTENTION_WINDOW_HOURS = 6

if not PREDICTIONS_FILE.exists():
    print("O/U V1-logg saknas. Kör generate_ou_v1_forward.py först.")
    raise SystemExit(0)

predictions = pd.read_csv(PREDICTIONS_FILE)
if len(predictions) == 0:
    print("O/U V1-loggen är tom.")
    raise SystemExit(0)

if not RAW_HISTORY_FILE.exists():
    print("Raw market history saknas.")
    raise SystemExit(0)

history = pd.read_csv(RAW_HISTORY_FILE)
predictions["start_time"] = pd.to_datetime(predictions["start_time"], utc=True)
history["snapshot_timestamp"] = pd.to_datetime(history["snapshot_timestamp"], utc=True)
now = pd.Timestamp(datetime.now(timezone.utc))

rows = []

for _, bet in predictions.iterrows():
    fixture_id = bet["fixture_id"]
    kickoff = bet["start_time"]
    bookmaker = bet["best_bookmaker"]
    side = bet["bet_side"]
    original_odds = bet["best_bet_odds"]
    minutes_to_kickoff = (kickoff - now).total_seconds() / 60

    snapshots = history[
        (history["fixture_id"] == fixture_id)
        & (history["bookmaker"] == bookmaker)
        & (history["snapshot_timestamp"] < kickoff)
    ].copy().sort_values("snapshot_timestamp")

    latest_snapshot = pd.NaT
    latest_odds = np.nan
    economic_clv = np.nan
    snapshot_minutes_before = np.nan

    if len(snapshots) > 0:
        latest = snapshots.iloc[-1]
        latest_snapshot = latest["snapshot_timestamp"]
        latest_odds = (
            latest["over_25"] if side == "OVER" else latest["under_25"]
        )
        snapshot_minutes_before = (
            kickoff - latest_snapshot
        ).total_seconds() / 60

        if pd.notna(original_odds) and pd.notna(latest_odds) and latest_odds > 1:
            economic_clv = original_odds / latest_odds - 1

    if minutes_to_kickoff > ATTENTION_WINDOW_HOURS * 60:
        status = "WAITING"
    elif minutes_to_kickoff > 0:
        if pd.notna(snapshot_minutes_before) and snapshot_minutes_before <= CLOSING_WINDOW_MINUTES:
            status = "CLOSING_SNAPSHOT_READY"
        else:
            status = "TAKE_SNAPSHOT"
    else:
        if pd.notna(snapshot_minutes_before) and snapshot_minutes_before <= CLOSING_WINDOW_MINUTES:
            status = "AWAITING_RESULT"
        else:
            status = "NO_CLOSE_SNAPSHOT"

    rows.append({
        "fixture_id": fixture_id,
        "start_time": kickoff,
        "home_team": bet["home_team"],
        "away_team": bet["away_team"],
        "bet_side": side,
        "bookmaker": bookmaker,
        "original_odds": original_odds,
        "predicted_movement": bet["predicted_movement"],
        "minutes_to_kickoff": minutes_to_kickoff,
        "latest_snapshot": latest_snapshot,
        "latest_odds": latest_odds,
        "snapshot_minutes_before_kickoff": snapshot_minutes_before,
        "current_economic_clv": economic_clv,
        "status": status,
    })

status_df = pd.DataFrame(rows)
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
status_df.to_csv(OUTPUT_FILE, index=False)

print()
print("===================================")
print("O/U 2.5 V1 FORWARD MONITOR")
print("===================================")
print("Signaler:", len(status_df))

view = status_df.copy()
view["hours_to_kickoff"] = view["minutes_to_kickoff"] / 60
view["clv_percent"] = view["current_economic_clv"] * 100
print(
    view[[
        "start_time",
        "home_team",
        "away_team",
        "bet_side",
        "bookmaker",
        "original_odds",
        "latest_odds",
        "clv_percent",
        "hours_to_kickoff",
        "status",
    ]].to_string(index=False)
)
print()
print("Sparad till:", OUTPUT_FILE)
