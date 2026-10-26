"""Publish latest immutable V1 predictions without modifying historical results."""
import csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
src=ROOT/"data/forward/v1_predictions.csv"
dst=ROOT/"docs/v1_live_signals.csv"
fields=["fixture_id","start_time","home_team","away_team","bet_side","best_bet_odds","best_bookmaker","predicted_close_probability","prediction_timestamp"]
with src.open(newline="",encoding="utf-8") as f:
    rows=list(csv.DictReader(f))
seen={}
for row in rows:
    if row.get("fixture_id") and row.get("bet_side") in ("YES","NO"):
        seen.setdefault(row["fixture_id"],{k:row.get(k,"") for k in fields})
dst.parent.mkdir(parents=True,exist_ok=True)
with dst.open("w",newline="",encoding="utf-8") as f:
    writer=csv.DictWriter(f,fieldnames=fields)
    writer.writeheader()
    writer.writerows(seen.values())
print("Published",len(seen),"V1 signals")
