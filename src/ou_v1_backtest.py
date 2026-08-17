import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ODDS_FILE = PROJECT_ROOT / "external_football_data/data/england/premier-league.csv"
FEATURE_FILE = PROJECT_ROOT / "data/processed/attack_defense_features.csv"
OUTPUT_FILE = PROJECT_ROOT / "data/processed/ou_v1_backtest.csv"

TEST_SEASONS = ["2022-2023", "2023-2024", "2024-2025"]
SIGNAL_THRESHOLD = 0.005

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
odds["actual_over"] = ((odds["FTHG"] + odds["FTAG"]) > 2).astype(int)
odds["home_team_merge"] = odds["HomeTeam"].replace(NAME_MAP)
odds["away_team_merge"] = odds["AwayTeam"].replace(NAME_MAP)

data = odds.merge(
    football,
    left_on=["Date", "home_team_merge", "away_team_merge"],
    right_on=["date", "home_team", "away_team"],
    how="left",
)

data = data.dropna(
    subset=FEATURES + [
        "movement",
        "over_2.5_open",
        "under_2.5_open",
        "over_2.5_close",
        "under_2.5_close",
        "actual_over",
    ]
).copy()

data = data[
    (data["home_matches_used"] >= 20)
    & (data["away_matches_used"] >= 20)
].copy()

all_predictions = []

for test_season in TEST_SEASONS:
    train = data[data["Season"] < test_season].copy()
    test = data[data["Season"] == test_season].copy()

    if len(train) == 0 or len(test) == 0:
        continue

    model = build_model()
    model.fit(train[FEATURES], train["movement"])

    test["predicted_movement"] = model.predict(test[FEATURES])
    test["predicted_close_probability"] = (
        test["over_open"] + test["predicted_movement"]
    ).clip(0.01, 0.99)

    test["bet_side"] = "NONE"
    test.loc[test["predicted_movement"] >= SIGNAL_THRESHOLD, "bet_side"] = "OVER"
    test.loc[test["predicted_movement"] <= -SIGNAL_THRESHOLD, "bet_side"] = "UNDER"
    test["signal"] = test["predicted_movement"].abs()
    test["direction_correct"] = (
        np.sign(test["predicted_movement"]) == np.sign(test["movement"])
    )

    test["bet_odds"] = np.where(
        test["bet_side"] == "OVER",
        test["over_2.5_open"],
        np.where(test["bet_side"] == "UNDER", test["under_2.5_open"], np.nan),
    )
    test["close_odds_same_side"] = np.where(
        test["bet_side"] == "OVER",
        test["over_2.5_close"],
        np.where(test["bet_side"] == "UNDER", test["under_2.5_close"], np.nan),
    )
    test["won"] = np.where(
        test["bet_side"] == "OVER",
        test["actual_over"] == 1,
        np.where(test["bet_side"] == "UNDER", test["actual_over"] == 0, False),
    )
    test["profit"] = np.where(
        test["bet_side"] == "NONE",
        np.nan,
        np.where(test["won"], test["bet_odds"] - 1, -1),
    )
    test["economic_clv"] = np.where(
        test["bet_side"] == "NONE",
        np.nan,
        test["bet_odds"] / test["close_odds_same_side"] - 1,
    )
    test["test_season"] = test_season
    all_predictions.append(test)

if not all_predictions:
    raise RuntimeError("Ingen walk-forward-data kunde skapas.")

predictions = pd.concat(all_predictions, ignore_index=True)
signals = predictions[predictions["bet_side"] != "NONE"].copy()

baseline_mse = mean_squared_error(
    predictions["movement"], np.zeros(len(predictions))
)
model_mse = mean_squared_error(
    predictions["movement"], predictions["predicted_movement"]
)
model_mae = mean_absolute_error(
    predictions["movement"], predictions["predicted_movement"]
)
direction = predictions["direction_correct"].mean()

print()
print("===================================")
print("O/U 2.5 V1 WALK-FORWARD")
print("===================================")
print("Matcher:", len(predictions))
print("Signal threshold:", SIGNAL_THRESHOLD)
print("Baseline MSE:", round(baseline_mse, 8))
print("Model MSE:", round(model_mse, 8))
print("MSE improvement:", round(baseline_mse - model_mse, 8))
print("Model MAE:", round(model_mae, 6))
print("Direction accuracy:", round(direction, 4))

print()
print("===================================")
print("BETTING SIGNALS")
print("===================================")
print("Bets:", len(signals))
if len(signals) > 0:
    print("OVER:", int((signals["bet_side"] == "OVER").sum()))
    print("UNDER:", int((signals["bet_side"] == "UNDER").sum()))
    print("ROI:", round(signals["profit"].mean() * 100, 2), "%")
    print("Economic CLV:", round(signals["economic_clv"].mean() * 100, 3), "%")
    print("Beat close:", round((signals["economic_clv"] > 0).mean() * 100, 2), "%")

print()
print("===================================")
print("PER SÄSONG")
print("===================================")
for season, group in predictions.groupby("test_season"):
    season_signals = group[group["bet_side"] != "NONE"]
    print()
    print(season)
    print("Matcher:", len(group))
    print("Direction:", round(group["direction_correct"].mean() * 100, 2), "%")
    print("Bets:", len(season_signals))
    if len(season_signals) > 0:
        print("ROI:", round(season_signals["profit"].mean() * 100, 2), "%")
        print("CLV:", round(season_signals["economic_clv"].mean() * 100, 3), "%")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
predictions.to_csv(OUTPUT_FILE, index=False)
print()
print("Sparad till:", OUTPUT_FILE)
