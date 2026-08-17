import pandas as pd
import numpy as np
from pathlib import Path


# ==================================================
# V1 - LÅST MODELL
# ==================================================

MODEL_NAME = "V1_MARKET_ATTACK_DEFENSE"
SIGNAL_THRESHOLD = 0.005

FORWARD_FILE = Path(
    "data/forward/v1_predictions.csv"
)


# ==================================================
# SKAPA FORWARD-LOGG OM DEN ÄR TOM
# ==================================================

COLUMNS = [
    "prediction_timestamp",
    "match_date",
    "home_team",
    "away_team",

    "btts_yes_open",
    "btts_no_open",

    "btts_open_probability",

    "predicted_movement",
    "predicted_close_probability",

    "bet_side",
    "bet_odds",

    "signal",

    "model_name",

    # Dessa ska INTE vara kända när prediction görs
    "btts_yes_close",
    "btts_no_close",
    "close_odds_same_side",

    "economic_clv",

    "home_goals",
    "away_goals",
    "actual_btts",

    "won",
    "profit"
]


if (
    not FORWARD_FILE.exists()
    or FORWARD_FILE.stat().st_size == 0
):

    pd.DataFrame(
        columns=COLUMNS
    ).to_csv(
        FORWARD_FILE,
        index=False
    )


# ==================================================
# LADDA LOGG
# ==================================================

log = pd.read_csv(
    FORWARD_FILE
)


print()
print("===================================")
print("V1 FORWARD TEST")
print("===================================")

print(
    "Model:",
    MODEL_NAME
)

print(
    "Signal threshold:",
    SIGNAL_THRESHOLD
)

print(
    "Loggfil:",
    FORWARD_FILE
)

print(
    "Befintliga predictioner:",
    len(log)
)


# ==================================================
# KONTROLLERA ATT UTFALLSFÄLT ÄR TOMMA
# ==================================================

future_columns = [
    "btts_yes_close",
    "btts_no_close",
    "close_odds_same_side",
    "economic_clv",
    "home_goals",
    "away_goals",
    "actual_btts",
    "won",
    "profit"
]


if len(log) > 0:

    filled = (
        log[future_columns]
        .notna()
        .sum()
    )

    print()
    print("Utfall redan registrerade:")
    print(filled)


print()
print("Forward-testmiljön är klar.")