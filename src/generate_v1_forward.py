import pandas as pd
import numpy as np

from pathlib import Path
from datetime import datetime, timezone

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ==================================================
# INSTÄLLNINGAR
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

HISTORICAL_ODDS_FILE = (
    PROJECT_ROOT
    / "external_football_data"
    / "data"
    / "england"
    / "premier-league.csv"
)

HISTORICAL_FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "attack_defense_features.csv"
)

FORWARD_FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "forward"
    / "forward_features.csv"
)

MARKET_FILE = (
    PROJECT_ROOT
    / "data"
    / "forward"
    / "market_consensus.csv"
)

RAW_MARKET_FILE = (
    PROJECT_ROOT
    / "data"
    / "forward"
    / "market_raw.csv"
)

LOG_FILE = (
    PROJECT_ROOT
    / "data"
    / "forward"
    / "v1_predictions.csv"
)


MODEL_NAME = "V1_MARKET_ATTACK_DEFENSE"
SIGNAL_THRESHOLD = 0.005


MARKET_FEATURES = [
    "btts_open",
    "over_open",
    "home_open_prob",
    "draw_open_prob",
    "balance"
]

FOOTBALL_FEATURES = [
    "home_attack",
    "home_defense",
    "away_attack",
    "away_defense"
]

ALL_FEATURES = (
    MARKET_FEATURES
    +
    FOOTBALL_FEATURES
)


# ==================================================
# HISTORISKA ODDS
# ==================================================

odds = pd.read_csv(
    HISTORICAL_ODDS_FILE
)

football = pd.read_csv(
    HISTORICAL_FEATURE_FILE
)

odds["Date"] = pd.to_datetime(
    odds["Date"]
).dt.normalize()

football["date"] = pd.to_datetime(
    football["date"]
).dt.normalize()


# ==================================================
# HISTORISK BTTS OPEN
# ==================================================

yes = 1 / odds["bts_yes_open"]
no = 1 / odds["bts_no_open"]

odds["btts_open"] = (
    yes / (yes + no)
)


# ==================================================
# HISTORISK BTTS CLOSE
# ==================================================

yes_close = 1 / odds["bts_yes_close"]
no_close = 1 / odds["bts_no_close"]

odds["btts_close"] = (
    yes_close
    /
    (yes_close + no_close)
)


# ==================================================
# O/U 2.5 OPEN
# ==================================================

over = 1 / odds["over_2.5_open"]
under = 1 / odds["under_2.5_open"]

odds["over_open"] = (
    over / (over + under)
)


# ==================================================
# 1X2 OPEN
# ==================================================

home = 1 / odds["home_open"]
draw = 1 / odds["draw_open"]
away = 1 / odds["away_open"]

total = (
    home
    + draw
    + away
)

odds["home_open_prob"] = (
    home / total
)

odds["draw_open_prob"] = (
    draw / total
)

odds["away_open_prob"] = (
    away / total
)

odds["balance"] = (
    1
    -
    abs(
        odds["home_open_prob"]
        -
        odds["away_open_prob"]
    )
)


# ==================================================
# TARGET
# ==================================================

odds["movement"] = (
    odds["btts_close"]
    -
    odds["btts_open"]
)


# ==================================================
# HISTORISKA LAGNAMN
# ==================================================

NAME_MAP = {
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Newcastle Utd": "Newcastle",
    "Nottingham": "Nott'm Forest",
    "Sheffield Utd": "Sheffield United",
    "Tottenham Hotspur": "Tottenham",
    "West Bromwich Albion": "West Brom",
    "Wolverhampton": "Wolves"
}

odds["home_team_merge"] = (
    odds["HomeTeam"]
    .replace(NAME_MAP)
)

odds["away_team_merge"] = (
    odds["AwayTeam"]
    .replace(NAME_MAP)
)


# ==================================================
# HISTORISK MERGE
# ==================================================

train = odds.merge(
    football,

    left_on=[
        "Date",
        "home_team_merge",
        "away_team_merge"
    ],

    right_on=[
        "date",
        "home_team",
        "away_team"
    ],

    how="left"
)


train = train.dropna(
    subset=(
        ALL_FEATURES
        +
        ["movement"]
    )
).copy()


train = train[
    (train["home_matches_used"] >= 20)
    &
    (train["away_matches_used"] >= 20)
].copy()


print()
print("===================================")
print("V1 FORWARD GENERATOR")
print("===================================")

print(
    "Historiska träningsmatcher:",
    len(train)
)


# ==================================================
# TRÄNA FRYST V1-ARKITEKTUR
# ==================================================

model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "ridge",
        Ridge(
            alpha=1.0
        )
    )
])


model.fit(
    train[ALL_FEATURES],
    train["movement"]
)


# ==================================================
# LADDA FORWARD DATA
# ==================================================

forward = pd.read_csv(
    FORWARD_FEATURE_FILE
)

market = pd.read_csv(
    MARKET_FILE
)

raw_market = pd.read_csv(
    RAW_MARKET_FILE
)


# ==================================================
# MERGE FORWARD FEATURES + MARKET
# ==================================================

forward = forward.merge(
    market,

    on=[
        "fixture_id",
        "start_time",
        "home_team",
        "away_team"
    ],

    how="inner"
)


# ==================================================
# ENDAST V1-READY
# ==================================================

forward = forward[
    (forward["status"] == "READY")
    &
    (
        forward[
            "complete_v1_market"
        ] == True
    )
].copy()


# ==================================================
# MAPPING LIVE → V1 FEATURENAMN
# ==================================================

forward["btts_open"] = (
    forward["btts_yes_prob"]
)

forward["over_open"] = (
    forward["over_25_prob"]
)

forward["home_open_prob"] = (
    forward["home_prob"]
)

forward["draw_open_prob"] = (
    forward["draw_prob"]
)

forward["balance"] = (
    1
    -
    abs(
        forward["home_prob"]
        -
        forward["away_prob"]
    )
)


# ==================================================
# PREDIKTERA MOVEMENT
# ==================================================

forward[
    "predicted_movement"
] = model.predict(
    forward[ALL_FEATURES]
)


forward[
    "predicted_close_probability"
] = (
    forward["btts_open"]
    +
    forward["predicted_movement"]
).clip(
    0.01,
    0.99
)


# ==================================================
# SIGNAL
# ==================================================

forward["bet_side"] = "NONE"

forward.loc[
    forward["predicted_movement"]
    >= SIGNAL_THRESHOLD,
    "bet_side"
] = "YES"

forward.loc[
    forward["predicted_movement"]
    <= -SIGNAL_THRESHOLD,
    "bet_side"
] = "NO"


forward["signal"] = abs(
    forward[
        "predicted_movement"
    ]
)


# ==================================================
# GENOMSNITTLIGT SNAPSHOT-ODDS
# ==================================================

forward[
    "consensus_bet_odds"
] = np.where(
    forward["bet_side"] == "YES",

    forward[
        "avg_btts_yes_odds"
    ],

    np.where(
        forward["bet_side"] == "NO",
        forward[
            "avg_btts_no_odds"
        ],
        np.nan
    )
)


# ==================================================
# BÄSTA TILLGÄNGLIGA ODDS AV 4 BOOKMAKERS
# ==================================================

best_rows = []


for _, match in forward.iterrows():

    fixture_raw = raw_market[
        raw_market["fixture_id"]
        ==
        match["fixture_id"]
    ]

    side = match["bet_side"]

    best_odds = np.nan
    best_bookmaker = None

    if side == "YES":

        valid = fixture_raw.dropna(
            subset=["btts_yes"]
        )

        if len(valid) > 0:

            best = valid.loc[
                valid[
                    "btts_yes"
                ].idxmax()
            ]

            best_odds = best[
                "btts_yes"
            ]

            best_bookmaker = best[
                "bookmaker"
            ]


    elif side == "NO":

        valid = fixture_raw.dropna(
            subset=["btts_no"]
        )

        if len(valid) > 0:

            best = valid.loc[
                valid[
                    "btts_no"
                ].idxmax()
            ]

            best_odds = best[
                "btts_no"
            ]

            best_bookmaker = best[
                "bookmaker"
            ]


    best_rows.append({
        "fixture_id":
            match["fixture_id"],

        "best_bet_odds":
            best_odds,

        "best_bookmaker":
            best_bookmaker
    })


best_df = pd.DataFrame(
    best_rows
)


forward = forward.merge(
    best_df,
    on="fixture_id",
    how="left"
)


# ==================================================
# TIMESTAMP
# ==================================================

prediction_timestamp = (
    datetime.now(
        timezone.utc
    ).isoformat()
)


forward[
    "prediction_timestamp"
] = prediction_timestamp


# ==================================================
# VISA RESULTAT
# ==================================================

print()
print(
    "Matcher med aktuella odds:",
    len(market)
)

print(
    "V1 READY:",
    len(forward)
)

print()

print("===================================")
print("V1 FORWARD PREDICTIONS")
print("===================================")

display_columns = [
    "start_time",
    "home_team",
    "away_team",
    "btts_open",
    "predicted_movement",
    "predicted_close_probability",
    "bet_side",
    "signal",
    "consensus_bet_odds",
    "best_bet_odds",
    "best_bookmaker"
]

print(
    forward[
        display_columns
    ].to_string(
        index=False
    )
)


# ==================================================
# ENDAST SIGNALER
# ==================================================

signals = forward[
    forward["bet_side"]
    != "NONE"
].copy()


print()
print("===================================")
print("V1 BET SIGNALS")
print("===================================")

if len(signals) == 0:

    print(
        "Inga signaler över 0.5 pp."
    )

else:

    print(
        signals[
            display_columns
        ].to_string(
            index=False
        )
    )


# ==================================================
# SPARA SNAPSHOT
# ==================================================

snapshot_file = (
    PROJECT_ROOT
    / "data"
    / "forward"
    / "v1_snapshot_predictions.csv"
)

forward.to_csv(
    snapshot_file,
    index=False
)


# ==================================================
# LÄGG TILL SIGNALER I PERMANENT LOGG
# ==================================================

log_columns = [
    "prediction_timestamp",
    "fixture_id",
    "start_time",
    "home_team",
    "away_team",

    "btts_open",
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

    "bookmakers"
]


new_log = signals[
    log_columns
].copy()


# Undvik dubbelloggning av samma fixture + modell
if LOG_FILE.exists():

    try:
        old_log = pd.read_csv(
            LOG_FILE
        )

    except pd.errors.EmptyDataError:
        old_log = pd.DataFrame()

else:
    old_log = pd.DataFrame()


if len(old_log) > 0:

    existing_ids = set(
        old_log.get(
            "fixture_id",
            pd.Series(
                dtype=str
            )
        ).astype(str)
    )

    new_log = new_log[
        ~new_log[
            "fixture_id"
        ].astype(str).isin(
            existing_ids
        )
    ]


if len(new_log) > 0:

    combined_log = pd.concat(
        [
            old_log,
            new_log
        ],
        ignore_index=True
    )

    combined_log.to_csv(
        LOG_FILE,
        index=False
    )

else:

    combined_log = old_log


print()
print(
    "Nya signaler loggade:",
    len(new_log)
)

print(
    "Totalt i forward-loggen:",
    len(combined_log)
)

print()
print(
    "Snapshot:",
    snapshot_file
)

print(
    "Permanent logg:",
    LOG_FILE
)