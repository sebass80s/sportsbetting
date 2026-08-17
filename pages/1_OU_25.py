import pandas as pd
import streamlit as st
from pathlib import Path
from datetime import datetime, timezone


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKTEST_FILE = PROJECT_ROOT / "data/processed/ou_v1_backtest.csv"
DIAGNOSTICS_FILE = PROJECT_ROOT / "data/processed/ou_v1_diagnostics.csv"
PREDICTIONS_FILE = PROJECT_ROOT / "data/forward/ou_v1_predictions.csv"
STATUS_FILE = PROJECT_ROOT / "data/forward/ou_v1_forward_status.csv"
SNAPSHOT_HISTORY_FILE = PROJECT_ROOT / "data/forward/market_snapshot_history.csv"

st.set_page_config(
    page_title="O/U 2.5 Forward Test",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ O/U 2.5 Forward Test")
st.caption("Separat O/U 2.5 V1 — market + attack/defense. BTTS-V1 lämnas oförändrad.")


def latest_snapshot_text():
    if not SNAPSHOT_HISTORY_FILE.exists():
        return "Senaste market snapshot: saknas"
    try:
        history = pd.read_csv(SNAPSHOT_HISTORY_FILE, usecols=["snapshot_timestamp"])
        if len(history) == 0:
            return "Senaste market snapshot: saknas"
        ts = pd.to_datetime(history["snapshot_timestamp"], utc=True, errors="coerce").max()
        if pd.isna(ts):
            return "Senaste market snapshot: saknas"
        local = ts.tz_convert("Europe/Stockholm")
        return f"Senaste market snapshot: {local:%Y-%m-%d %H:%M:%S} svensk tid"
    except Exception:
        return "Senaste market snapshot: kunde inte läsas"


st.caption(latest_snapshot_text())

forward_tab, backtest_tab, diagnostics_tab, info_tab = st.tabs([
    "Forward-signaler",
    "Backtest",
    "Diagnostik",
    "Om modellen",
])

with forward_tab:
    if not PREDICTIONS_FILE.exists():
        st.info(
            "Inga O/U-prediktioner finns ännu. Kör först `python3 src/ou_v1_backtest.py` "
            "och därefter `python3 src/generate_ou_v1_forward.py`."
        )
    else:
        predictions = pd.read_csv(PREDICTIONS_FILE)
        if len(predictions) == 0:
            st.info("O/U-modellen har ännu inte loggat några forward-signaler.")
        else:
            predictions["start_time"] = pd.to_datetime(
                predictions["start_time"], utc=True, errors="coerce"
            )

            if STATUS_FILE.exists():
                status = pd.read_csv(STATUS_FILE)
                keep = [
                    "fixture_id",
                    "latest_odds",
                    "current_economic_clv",
                    "minutes_to_kickoff",
                    "status",
                ]
                keep = [c for c in keep if c in status.columns]
                predictions = predictions.merge(
                    status[keep], on="fixture_id", how="left", suffixes=("", "_monitor")
                )

            if "minutes_to_kickoff" not in predictions.columns:
                now = pd.Timestamp(datetime.now(timezone.utc))
                predictions["minutes_to_kickoff"] = (
                    predictions["start_time"] - now
                ).dt.total_seconds() / 60

            predictions["hours_to_kickoff"] = predictions["minutes_to_kickoff"] / 60
            predictions["signal_pp"] = predictions["signal"] * 100
            predictions["predicted_move_pp"] = predictions["predicted_movement"] * 100

            if "current_economic_clv" in predictions.columns:
                predictions["current_clv_pct"] = predictions["current_economic_clv"] * 100
            else:
                predictions["current_clv_pct"] = pd.NA

            active = predictions[predictions["hours_to_kickoff"] > 0].copy()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("O/U-signaler", len(predictions))
            c2.metric("Aktiva", len(active))
            c3.metric("OVER", int((predictions["bet_side"] == "OVER").sum()))
            c4.metric("UNDER", int((predictions["bet_side"] == "UNDER").sum()))

            st.subheader("Aktiva O/U-signaler")
            if len(active) == 0:
                st.info("Inga aktiva O/U-signaler just nu.")
            else:
                display = active.rename(columns={
                    "start_time": "Kickoff",
                    "home_team": "Hemma",
                    "away_team": "Borta",
                    "bet_side": "Bet",
                    "best_bookmaker": "Bookmaker",
                    "best_bet_odds": "Odds",
                    "latest_odds": "Senaste odds",
                    "predicted_move_pp": "Förväntad rörelse (pp)",
                    "current_clv_pct": "Current CLV %",
                    "hours_to_kickoff": "Timmar till kickoff",
                    "status": "Status",
                })
                columns = [
                    "Kickoff",
                    "Hemma",
                    "Borta",
                    "Bet",
                    "Bookmaker",
                    "Odds",
                    "Senaste odds",
                    "Förväntad rörelse (pp)",
                    "Current CLV %",
                    "Timmar till kickoff",
                    "Status",
                ]
                columns = [c for c in columns if c in display.columns]
                st.dataframe(display[columns], use_container_width=True, hide_index=True)

            attention = predictions[
                predictions.get("status", pd.Series(index=predictions.index, dtype=str)).isin(
                    ["TAKE_SNAPSHOT", "NO_CLOSE_SNAPSHOT"]
                )
            ]
            st.subheader("Kräver uppmärksamhet")
            if len(attention) == 0:
                st.success("Inga O/U-matcher kräver åtgärd just nu.")
            else:
                for _, row in attention.iterrows():
                    st.warning(f"{row['home_team']} – {row['away_team']}: {row['status']}")

with backtest_tab:
    if not BACKTEST_FILE.exists():
        st.info("Backtest saknas. Kör `python3 src/ou_v1_backtest.py` på din Mac.")
    else:
        bt = pd.read_csv(BACKTEST_FILE)
        signals = bt[bt["bet_side"] != "NONE"].copy()

        baseline_mse = (bt["movement"] ** 2).mean()
        model_mse = ((bt["movement"] - bt["predicted_movement"]) ** 2).mean()
        direction = (
            (bt["predicted_movement"].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)))
            == (bt["movement"].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)))
        ).mean()

        roi = signals["profit"].mean() * 100 if len(signals) else 0
        clv = signals["economic_clv"].mean() * 100 if len(signals) else 0

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Matcher", len(bt))
        c2.metric("Bets", len(signals))
        c3.metric("Direction", f"{direction * 100:.1f}%")
        c4.metric("ROI", f"{roi:.2f}%")
        c5.metric("Economic CLV", f"{clv:.2f}%")

        st.caption(
            f"Movement MSE: {model_mse:.6f} · zero-movement baseline: {baseline_mse:.6f}"
        )

        if len(signals):
            season = signals.groupby("test_season").agg(
                bets=("bet_side", "size"),
                roi=("profit", lambda x: x.mean() * 100),
                clv=("economic_clv", lambda x: x.mean() * 100),
            ).reset_index()
            st.subheader("Backtest per säsong")
            st.dataframe(season, use_container_width=True, hide_index=True)

with diagnostics_tab:
    st.subheader("Signaldiagnostik")
    if not DIAGNOSTICS_FILE.exists():
        st.info(
            "Diagnostik saknas. Kör `python3 src/analyze_ou_v1_diagnostics.py` "
            "efter backtestet."
        )
    else:
        diag = pd.read_csv(DIAGNOSTICS_FILE)
        if len(diag) == 0:
            st.info("Ingen diagnostikdata hittades.")
        else:
            st.caption(
                "Jämför OVER/UNDER inom olika signalstyrkor. Positiv economic CLV är "
                "viktigare än kortsiktig ROI när vi bedömer om en bucket är lovande."
            )

            show = diag.copy()
            for col in ["roi", "economic_clv", "median_clv", "beat_close", "hit_rate", "direction"]:
                if col in show.columns:
                    show[col] = show[col].round(2)
            if "avg_odds" in show.columns:
                show["avg_odds"] = show["avg_odds"].round(3)

            st.dataframe(show, use_container_width=True, hide_index=True)

            positive = diag[
                (diag["economic_clv"] > 0)
                & (diag["beat_close"] > 50)
            ].copy()

            st.subheader("Buckets med positiv CLV")
            if len(positive) == 0:
                st.warning("Ingen bucket har både positiv mean CLV och >50% beat-close.")
            else:
                positive = positive.sort_values("economic_clv", ascending=False)
                st.dataframe(positive, use_container_width=True, hide_index=True)

with info_tab:
    st.markdown(
        """
### Vad O/U V1 försöker göra

Modellen förutsäger **förändringen i no-vig-sannolikheten för Over 2.5** från tidigt marknadspris mot closing.

- Positiv förväntad rörelse på minst 0,5 procentenheter → **OVER-signal**.
- Negativ förväntad rörelse på minst 0,5 procentenheter → **UNDER-signal**.
- Mindre rörelser → inget spel.

Features är O/U-marknaden, BTTS-marknaden, 1X2-marknaden samt samma attack/defense-features som används i BTTS-V1.

O/U-modellen har en **egen forward-logg** och ändrar inte BTTS-V1.
        """
    )
