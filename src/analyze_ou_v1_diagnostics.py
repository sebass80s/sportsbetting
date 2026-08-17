import pandas as pd
import numpy as np
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data/processed/ou_v1_backtest.csv"
OUTPUT_FILE = PROJECT_ROOT / "data/processed/ou_v1_diagnostics.csv"

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        "Backtestfil saknas. Kör först: python3 src/ou_v1_backtest.py"
    )


df = pd.read_csv(INPUT_FILE)
signals = df[df["bet_side"] != "NONE"].copy()

if len(signals) == 0:
    raise RuntimeError("Inga O/U-signaler finns i backtestet.")

signals["signal_pp"] = signals["signal"] * 100

bins = [0.5, 0.75, 1.0, 1.5, 2.0, np.inf]
labels = [
    "0.5-0.75 pp",
    "0.75-1.0 pp",
    "1.0-1.5 pp",
    "1.5-2.0 pp",
    "2.0+ pp",
]

signals["signal_bin"] = pd.cut(
    signals["signal_pp"],
    bins=bins,
    labels=labels,
    right=False,
    include_lowest=True,
)


def summarize(group):
    return pd.Series({
        "bets": len(group),
        "roi": group["profit"].mean() * 100,
        "economic_clv": group["economic_clv"].mean() * 100,
        "median_clv": group["economic_clv"].median() * 100,
        "beat_close": (group["economic_clv"] > 0).mean() * 100,
        "hit_rate": group["won"].mean() * 100,
        "avg_odds": group["bet_odds"].mean(),
        "direction": group["direction_correct"].mean() * 100,
    })


def print_section(title, table):
    print()
    print("===================================")
    print(title)
    print("===================================")
    print(table.to_string())


by_bucket = signals.groupby("signal_bin", observed=True).apply(summarize)
by_side = signals.groupby("bet_side").apply(summarize)
by_side_bucket = signals.groupby(["bet_side", "signal_bin"], observed=True).apply(summarize)
by_season_side = signals.groupby(["test_season", "bet_side"]).apply(summarize)

print_section("O/U SIGNAL BUCKETS", by_bucket)
print_section("OVER / UNDER", by_side)
print_section("OVER / UNDER PER SIGNAL BUCKET", by_side_bucket)
print_section("PER SÄSONG + SIDA", by_season_side)


# ==================================================
# THRESHOLD-SWEEP
# ==================================================

thresholds = [0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02]
threshold_rows = []

for threshold in thresholds:
    subset = df[df["signal"] >= threshold].copy()
    subset = subset[subset["bet_side"] != "NONE"]

    threshold_rows.append({
        "threshold_pp": threshold * 100,
        "bets": len(subset),
        "roi": subset["profit"].mean() * 100 if len(subset) else np.nan,
        "economic_clv": subset["economic_clv"].mean() * 100 if len(subset) else np.nan,
        "beat_close": (subset["economic_clv"] > 0).mean() * 100 if len(subset) else np.nan,
        "over_bets": int((subset["bet_side"] == "OVER").sum()),
        "under_bets": int((subset["bet_side"] == "UNDER").sum()),
    })

threshold_table = pd.DataFrame(threshold_rows)
print_section("THRESHOLD SWEEP", threshold_table.set_index("threshold_pp"))


# ==================================================
# VECKOBLOCK BOOTSTRAP CLV
# ==================================================

signals["Date"] = pd.to_datetime(signals["Date"], errors="coerce")
signals = signals.dropna(subset=["Date"])
signals["week_block"] = signals["Date"].dt.to_period("W").astype(str)

block_clv = signals.groupby("week_block")["economic_clv"].mean()

rng = np.random.default_rng(42)
bootstrap_means = []

if len(block_clv) > 1:
    values = block_clv.to_numpy()
    for _ in range(10000):
        sample = rng.choice(values, size=len(values), replace=True)
        bootstrap_means.append(sample.mean())

    bootstrap_means = np.array(bootstrap_means)
    low, high = np.percentile(bootstrap_means, [2.5, 97.5])
    p_le_zero = (bootstrap_means <= 0).mean()

    print()
    print("===================================")
    print("BLOCK BOOTSTRAP - ECONOMIC CLV")
    print("===================================")
    print("Veckoblock:", len(block_clv))
    print("Observerad CLV:", round(signals["economic_clv"].mean() * 100, 3), "%")
    print("95% interval:", round(low * 100, 3), "to", round(high * 100, 3), "%")
    print("P(CLV <= 0):", round(p_le_zero, 4))


# ==================================================
# SPARA SAMMANFATTNING
# ==================================================

rows = []
for (side, bucket), row in by_side_bucket.iterrows():
    item = row.to_dict()
    item["bet_side"] = side
    item["signal_bin"] = str(bucket)
    rows.append(item)

summary = pd.DataFrame(rows)
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(OUTPUT_FILE, index=False)

print()
print("Sparad till:", OUTPUT_FILE)
