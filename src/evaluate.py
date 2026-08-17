import pandas as pd
import numpy as np

# --------------------------------------------------
# LADDA PREDIKTIONER
# --------------------------------------------------

file_path = "data/processed/predictions.csv"

df = pd.read_csv(file_path)

# Säkerställ att actual_btts är 0/1
if df["actual_btts"].dtype == object:
    df["actual_btts"] = (
        df["actual_btts"]
        .astype(str)
        .str.lower()
        .map({
            "true": 1,
            "false": 0
        })
    )
else:
    df["actual_btts"] = df["actual_btts"].astype(int)


# --------------------------------------------------
# GRUNDMÅTT
# --------------------------------------------------

actual = df["actual_btts"]
probability = df["btts_probability"]

# Brier score
brier_score = np.mean(
    (probability - actual) ** 2
)

# Modellens klassificering med 50 % gräns
df["predicted_btts"] = (
    df["btts_probability"] >= 0.50
).astype(int)

accuracy = np.mean(
    df["predicted_btts"] == df["actual_btts"]
)

# Faktisk BTTS-andel
actual_btts_rate = actual.mean()

# Modellens genomsnittliga prediktion
average_prediction = probability.mean()


print()
print("===================================")
print("MODELLUTVÄRDERING")
print("===================================")

print()
print("Antal matcher:", len(df))

print(
    "Faktisk BTTS-andel:",
    round(actual_btts_rate, 4)
)

print(
    "Modellens genomsnitt:",
    round(average_prediction, 4)
)

print(
    "Accuracy:",
    round(accuracy, 4)
)

print(
    "Brier score:",
    round(brier_score, 4)
)


# --------------------------------------------------
# NAIV BASELINE
# --------------------------------------------------

baseline_probability = actual_btts_rate

baseline_brier = np.mean(
    (baseline_probability - actual) ** 2
)

print()
print("NAIV BASELINE")
print("-----------------------------------")

print(
    "Baseline-sannolikhet:",
    round(baseline_probability, 4)
)

print(
    "Baseline Brier score:",
    round(baseline_brier, 4)
)


# --------------------------------------------------
# KALIBRERING
# --------------------------------------------------

bins = [
    0.0,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    1.0
]

df["probability_bin"] = pd.cut(
    df["btts_probability"],
    bins=bins,
    include_lowest=True
)

calibration = (
    df.groupby(
        "probability_bin",
        observed=True
    )
    .agg(
        matches=("actual_btts", "size"),
        predicted_probability=(
            "btts_probability",
            "mean"
        ),
        actual_btts_rate=(
            "actual_btts",
            "mean"
        )
    )
)

print()
print("KALIBRERING")
print("-----------------------------------")

print(
    calibration.to_string()
)

# --------------------------------------------------
# RESULTAT PER SÄSONG
# --------------------------------------------------

season_results = []

for season, group in df.groupby("season"):

    actual = group["actual_btts"]
    probability = group["btts_probability"]

    season_brier = np.mean(
        (probability - actual) ** 2
    )

    season_btts_rate = actual.mean()

    naive_brier = np.mean(
        (season_btts_rate - actual) ** 2
    )

    season_results.append({
        "season": season,
        "matches": len(group),
        "actual_btts": season_btts_rate,
        "model_probability": probability.mean(),
        "model_brier": season_brier,
        "naive_brier": naive_brier
    })


season_results = pd.DataFrame(season_results)

season_results["difference"] = (
    season_results["naive_brier"]
    - season_results["model_brier"]
)


print()
print("RESULTAT PER SÄSONG")
print("-----------------------------------")

print(
    season_results.to_string(
        index=False
    )
)

# --------------------------------------------------
# RESULTAT PER SÄSONG
# --------------------------------------------------

season_results = []

for season, group in df.groupby("season"):

    actual = group["actual_btts"]
    probability = group["btts_probability"]

    # Modellens Brier score
    season_brier = np.mean(
        (probability - actual) ** 2
    )

    # Faktisk BTTS-andel för säsongen
    season_btts_rate = actual.mean()

    # Naiv modell:
    # samma sannolikhet för alla matcher under säsongen
    naive_brier = np.mean(
        (season_btts_rate - actual) ** 2
    )

    season_results.append({
        "season": season,
        "matches": len(group),
        "actual_btts": season_btts_rate,
        "model_probability": probability.mean(),
        "model_brier": season_brier,
        "naive_brier": naive_brier
    })


season_results = pd.DataFrame(season_results)

# Positivt värde = vår modell är bättre
# Negativt värde = baseline är bättre
season_results["difference"] = (
    season_results["naive_brier"]
    - season_results["model_brier"]
)


print()
print("RESULTAT PER SÄSONG")
print("-----------------------------------")

print(
    season_results.to_string(
        index=False
    )
)