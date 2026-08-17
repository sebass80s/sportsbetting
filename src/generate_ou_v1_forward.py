import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ODDS_FILE = PROJECT_ROOT / "external_football_data/data/england/premier-league.csv"
FEATURE_FILE = PROJECT_ROOT / "data/processed/attack_defense_features.csv"
FORWARD_FEATURE_FILE = PROJECT_ROOT / "data/forward/forward_features.csv"
MARKET_FILE = PROJECT_ROOT / "data/forward/market_consensus.csv"
RAW_MARKET_FILE = PROJECT_ROOT / "data/forward/market_raw.csv"
LOG_FILE = PROJECT_ROOT / "data/forward/ou_v1_predictions.csv"
SNAPSHOT_FILE = PROJECT_ROOT / "data/forward/ou_v1_snapshot_predictions.csv"

SIGNAL_THRESHOLD = 0.005
MODEL_NAME = "OU_V1_MARKET_ATTACK_DEFENSE"

MARKET_FEATURES = [
    "over_open",
    "btts_open",
    "home_open_prob",
    "draw_open_prob",
    "balance",
]
FOOTBALL_FEATURES = [
    "home_attack",
    "home_defense",
    "away_attack",
    "away_defense",
]
FEATURES = MARKET_FEATURES + FOOTBALL_FEATURES

NAME_MAP = {
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Newcastle Utd": "Newcastle",
    "Nottingham": "Nott'm Forest",
    "Sheffield Utd": "Sheffield United",
    "Tottenham Hotspur": "Tottenham",
    "West Bromwich Albion": "West Brom",
    "Wolverhampton": "Wolves",
}


def novig_two_way(a, b):
    pa = 1 / a
    pb = 1 / b
    return pa / (pa + pb)


def build_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])


odds = pd.read_csv(ODDS_FILE)
football = pd.read_csv(FEATURE_FILE)

odds["Date"] = pd.to_datetime(odds["Date"]).dt.normalize()
football["date"] = pd.to_datetime(football["date"]).dt.normalize()

odds["over_open"] = novig_two_way(
    odds["over_2.5_open"], odds["under_2.5_open"]
)
odds["over_close"] = novig_two_way(
    odds["over_2.5_close"], odds["under_2.5_close"]
)
odds["btts_open"] = novig_two_way(
    odds["bts_yes_open"], odds["bts_no_open"]
)

home = 1 / odds["home_open"]
draw = 1 / odds["draw_open"]
away = 1 / odds["away_open"]
total = home + draw + away
odds["home_open_prob"] = home / total
odds["draw_open_prob"] = draw / total
odds["away_open_prob"] = away / total
odds["balance"] = 1 - abs(odds["home_open_prob"] - odds["away_open_prob"])
odds["movement"] = odds["over_close"] - odds["over_open"]
odds["home_team_merge"] = odds["HomeTeam"].replace(NAME_MAP)
odds["away_team_merge"] = odds["AwayTeam"].replace(NAME_MAP)

train = odds.merge(
    football,
    left_on=["Date", "home_team_merge", "away_team_merge"],
    right_on=["date", "home_team", "away_team"],
    how="left",
)
train = train.dropna(subset=FEATURES + ["movement"]).copy()
train = train[
    (train["home_matches_used"] >= 20)
    & (train["away_matches_used"] >= 20)
].copy()

model = build_model()
model.fit(train[FEATURES], train["movement"])

forward = pd.read_csv(FORWARD_FEATURE_FILE)
market = pd.read_csv(MARKET_FILE)
raw_market = pd.read_csv(RAW_MARKET_FILE)

forward = forward.merge(
    market,
    on=["fixture_id", "start_time", "home_team", "away_team"],
    how="inner",
)
forward = forward[
    (forward["status"] == "READY")
    & (forward["complete_v1_market"] == True)
].copy()

forward["over_open"] = forward["over_25_prob"]
forward["btts_open"] = forward["btts_yes_prob"]
forward["home_open_prob"] = forward["home_prob"]
forward["draw_open_prob"] = forward["draw_prob"]
forward["balance"] = 1 - abs(forward["home_prob"] - forward["away_prob"])

forward["predicted_movement"] = model.predict(forward[FEATURES])
forward["predicted_close_probability"] = (
    forward["over_open"] + forward["predicted_movement"]
).clip(0.01, 0.99)
forward["signal"] = forward["predicted_movement"].abs()
forward["bet_side"] = "NONE"
forward.loc[
    forward["predicted_movement"] >= SIGNAL_THRESHOLD, "bet_side"
] = "OVER"
forward.loc[
    forward["predicted_movement"] <= -SIGNAL_THRESHOLD, "bet_side"
] = "UNDER"

forward["consensus_bet_odds"] = np.where(
    forward["bet_side"] == "OVER",
    forward["avg_over_25_odds"],
    np.where(
        forward["bet_side"] == "UNDER",
        forward["avg_under_25_odds"],
        np.nan,
    ),
)

best_rows = []
for _, match in forward.iterrows():
    fixture_raw = raw_market[raw_market["fixture_id"] == match["fixture_id"]]
    side = match["bet_side"]
    best_odds = np.nan
    best_bookmaker = None

    if side == "OVER":
        valid = fixture_raw.dropna(subset=["over_25"])
        if len(valid) > 0:
            best = valid.loc[valid["over_25"].idxmax()]
            best_odds = best["over_25"]
            best_bookmaker = best["bookmaker"]
    elif side == "UNDER":
        valid = fixture_raw.dropna(subset=["under_25"])
        if len(valid) > 0:
            best = valid.loc[valid["under_25"].idxmax()]
            best_odds = best["under_25"]
            best_bookmaker = best["bookmaker"]

    best_rows.append({
        "fixture_id": match["fixture_id"],
        "best_bet_odds": best_odds,
        "best_bookmaker": best_bookmaker,
    })

forward = forward.merge(pd.DataFrame(best_rows), on="fixture_id", how="left")
forward["prediction_timestamp"] = datetime.now(timezone.utc).isoformat()
forward["model_name"] = MODEL_NAME

signals = forward[forward["bet_side"] != "NONE"].copy()

print()
print("===================================")
print("O/U 2.5 V1 FORWARD")
print("===================================")
print("Historiska träningsmatcher:", len(train))
print("Matcher med aktuella odds:", len(market))
print("V1 READY:", len(forward))
print("Signaler:", len(signals))

columns = [
    "start_time",
    "home_team",
    "away_team",
    "over_open",
    "predicted_movement",
    "predicted_close_probability",
    "bet_side",
    "signal",
    "consensus_bet_odds",
    "best_bet_odds",
    "best_bookmaker",
]
print()
print(forward[columns].to_string(index=False))

SNAPSHOT_FILE.parent.mkdir(parents=True, exist_ok=True)
forward.to_csv(SNAPSHOT_FILE, index=False)

log_columns = [
    "prediction_timestamp",
    "model_name",
    "fixture_id",
    "start_time",
    "home_team",
    "away_team",
    "over_open",
    "predicted_movement",
    "predicted_close_probability",
    "bet_side",
    "signal",
    "consensus_bet_odds",
    "best_bet_odds",
    "best_bookmaker",
    "home_attack",
    "home_defense",
    "away_attack",
    "away_defense",
    "bookmakers",
]
new_log = signals[log_columns].copy()

if LOG_FILE.exists():
    try:
        old_log = pd.read_csv(LOG_FILE)
    except pd.errors.EmptyDataError:
        old_log = pd.DataFrame()
else:
    old_log = pd.DataFrame()

if len(old_log) > 0:
    existing_ids = set(old_log["fixture_id"].astype(str))
    new_log = new_log[~new_log["fixture_id"].astype(str).isin(existing_ids)]

combined = pd.concat([old_log, new_log], ignore_index=True) if len(new_log) else old_log
if len(combined) > 0:
    combined.to_csv(LOG_FILE, index=False)

print()
print("Nya O/U-signaler loggade:", len(new_log))
print("Totalt i O/U-loggen:", len(combined))
print("Snapshot:", SNAPSHOT_FILE)
print("Permanent logg:", LOG_FILE)
