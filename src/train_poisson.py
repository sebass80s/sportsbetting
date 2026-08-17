import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


# ==================================================
# INSTÄLLNINGAR
# ==================================================

DATA_FILE = "data/processed/poisson_data.csv"

DEVELOPMENT_SEASONS = [
    1516,
    1617,
    1718,
    1819,
    1920,
    2021,
    2122
]


# ==================================================
# LADDA DATA
# ==================================================

df = pd.read_csv(DATA_FILE)

df["date"] = pd.to_datetime(df["date"])

development = df[
    df["season"].isin(DEVELOPMENT_SEASONS)
].copy()


print()
print("===================================")
print("TRÄNINGSDATA")
print("===================================")

print(
    "Observationer:",
    len(development)
)

print(
    "Matcher:",
    len(development) // 2
)

print(
    "Lag:",
    development["team"].nunique()
)


# ==================================================
# POISSONREGRESSION
# ==================================================

print()
print("Tränar Poissonmodell...")

model = smf.glm(
    formula="goals ~ home + C(team) + C(opponent)",
    data=development,
    family=sm.families.Poisson()
).fit()


print("Klar!")


# ==================================================
# MODELLINFORMATION
# ==================================================

print()
print("===================================")
print("MODELL")
print("===================================")

print(
    "AIC:",
    round(model.aic, 2)
)

print(
    "Deviance:",
    round(model.deviance, 2)
)

print(
    "Home coefficient:",
    round(model.params["home"], 4)
)


# ==================================================
# HEMMAFÖRDEL
# ==================================================

home_multiplier = (
    float(
        __import__("math").exp(
            model.params["home"]
        )
    )
)

print(
    "Home multiplier:",
    round(home_multiplier, 4)
)


# ==================================================
# TESTPREDIKTION
# ==================================================

teams = sorted(
    development["team"].unique()
)

home_team = teams[0]
away_team = teams[1]

home_input = pd.DataFrame({
    "team": [home_team],
    "opponent": [away_team],
    "home": [1]
})

away_input = pd.DataFrame({
    "team": [away_team],
    "opponent": [home_team],
    "home": [0]
})


lambda_home = model.predict(
    home_input
).iloc[0]

lambda_away = model.predict(
    away_input
).iloc[0]


print()
print("===================================")
print("TESTPREDIKTION")
print("===================================")

print(
    home_team,
    "expected goals:",
    round(lambda_home, 3)
)

print(
    away_team,
    "expected goals:",
    round(lambda_away, 3)
)