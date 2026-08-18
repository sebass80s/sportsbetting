import pandas as pd
import streamlit as st
from pathlib import Path
from datetime import datetime, timezone


PROJECT_ROOT = Path(__file__).resolve().parent

STATUS_FILE = PROJECT_ROOT / "data/forward/v1_forward_status.csv"
RESULT_FILE = PROJECT_ROOT / "data/forward/v1_forward_results.csv"
SNAPSHOT_FILE = PROJECT_ROOT / "data/forward/market_snapshot_history.csv"
PREDICTION_FILE = PROJECT_ROOT / "data/forward/v1_snapshot_predictions.csv"


# ==================================================
# SIDINSTÄLLNINGAR
# ==================================================

st.set_page_config(
    page_title="V1 BTTS Forward Test",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ V1 BTTS Forward Test")
st.caption("Live forward-test av V1_MARKET_ATTACK_DEFENSE")


# ==================================================
# LADDA DATA
# ==================================================

@st.cache_data(ttl=60)
def load_data():
    status = pd.read_csv(STATUS_FILE)
    results = pd.read_csv(RESULT_FILE)
    status["start_time"] = pd.to_datetime(status["start_time"], utc=True)

    result_cols = [
        "fixture_id", "home_goals", "away_goals", "actual_btts",
        "won", "profit", "result_status"
    ]
    result_cols = [c for c in result_cols if c in results.columns]
    return status.merge(results[result_cols], on="fixture_id", how="left")


@st.cache_data(ttl=60)
def load_predictions():
    if not PREDICTION_FILE.exists():
        return pd.DataFrame()
    try:
        predictions = pd.read_csv(PREDICTION_FILE)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if "start_time" in predictions.columns:
        predictions["start_time"] = pd.to_datetime(
            predictions["start_time"], utc=True, errors="coerce"
        )
    return predictions


@st.cache_data(ttl=60)
def load_latest_snapshot_time():
    if not SNAPSHOT_FILE.exists():
        return None
    try:
        snapshots = pd.read_csv(SNAPSHOT_FILE, usecols=["snapshot_timestamp"])
    except (pd.errors.EmptyDataError, ValueError):
        return None
    timestamps = pd.to_datetime(
        snapshots["snapshot_timestamp"], utc=True, errors="coerce"
    ).dropna()
    return None if timestamps.empty else timestamps.max()


df = load_data()
predictions = load_predictions()
latest_snapshot_time = load_latest_snapshot_time()

if latest_snapshot_time is None:
    st.caption("Senaste market snapshot: ingen snapshot hittad")
else:
    latest_snapshot_local = latest_snapshot_time.tz_convert("Europe/Stockholm")
    st.caption(
        "Senaste market snapshot: "
        f"{latest_snapshot_local.strftime('%Y-%m-%d %H:%M:%S')} "
        "(Europe/Stockholm)"
    )


# ==================================================
# KPI
# ==================================================

finished = df[df["result_status"] == "FINISHED"].copy()
active = df[df["result_status"] != "FINISHED"].copy()

total_profit = finished["profit"].sum() if len(finished) else 0
roi = finished["profit"].mean() * 100 if len(finished) else 0
current_clv = df["current_economic_clv"].dropna().mean() * 100

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("V1 bets", len(df))
col2.metric("Aktiva", len(active))
col3.metric("Profit", f"{total_profit:.2f} u")
col4.metric("ROI", f"{roi:.2f}%")
col5.metric("Current CLV", f"{current_clv:.2f}%")


# ==================================================
# AKTIVA SIGNALER
# ==================================================

st.subheader("Aktiva signaler")

if len(active) == 0:
    st.success("Inga aktiva bets.")
else:
    active_display = active.copy()
    now = pd.Timestamp(datetime.now(timezone.utc))
    active_display["Timmar till kickoff"] = (
        active_display["start_time"] - now
    ).dt.total_seconds() / 3600
    active_display["CLV %"] = active_display["current_economic_clv"] * 100

    if "predicted_close_probability" in active_display.columns:
        predicted_close = pd.to_numeric(
            active_display["predicted_close_probability"], errors="coerce"
        )
        side_probability = pd.Series(
            float("nan"), index=active_display.index, dtype="float64"
        )
        yes_mask = active_display["bet_side"] == "YES"
        no_mask = active_display["bet_side"] == "NO"
        side_probability.loc[yes_mask] = predicted_close.loc[yes_mask]
        side_probability.loc[no_mask] = 1 - predicted_close.loc[no_mask]
        active_display["Spela om odds ≥"] = (1 / side_probability).round(2)
        latest_odds_numeric = pd.to_numeric(
            active_display["latest_odds"], errors="coerce"
        )
        playable_now = latest_odds_numeric >= active_display["Spela om odds ≥"]
        active_display["Spelvärde"] = "Vänta"
        active_display.loc[playable_now, "Spelvärde"] = "SPELVÄRDE NU"
    else:
        active_display["Spela om odds ≥"] = pd.NA
        active_display["Spelvärde"] = "Saknar V1 close-prob"

    active_display = active_display.rename(columns={
        "start_time": "Kickoff", "home_team": "Hemma",
        "away_team": "Borta", "bet_side": "Bet",
        "bookmaker": "Bookmaker", "original_odds": "Odds",
        "latest_odds": "Senaste odds", "status": "Status"
    })

    st.dataframe(
        active_display[[
            "Kickoff", "Hemma", "Borta", "Bet", "Bookmaker", "Odds",
            "Senaste odds", "Spela om odds ≥", "Spelvärde", "CLV %",
            "Timmar till kickoff", "Status"
        ]],
        use_container_width=True,
        hide_index=True
    )
    st.caption(
        "'Spela om odds ≥' bygger på V1:s predikterade closing probability. "
        "Det betyder att oddset förväntas slå modellens closing price; "
        "det är inte samma sak som en garanterad positiv förväntad avkastning."
    )


# ==================================================
# ALLA KOMMANDE V1-MATCHER
# ==================================================

st.subheader("Kommande matcher – V1 bevakning")
st.caption(
    "Visar alla matcher som klarar V1:s data- och marknadskrav, även när "
    "signalen ännu inte når ±0,5 procentenheter. Själva V1-testets "
    "signalgräns är oförändrad."
)

if predictions.empty:
    st.info("Inga aktuella V1-predictions hittades.")
else:
    watch = predictions.copy()
    now = pd.Timestamp(datetime.now(timezone.utc))
    if "start_time" in watch.columns:
        watch = watch[watch["start_time"] > now].copy()
        watch = watch.sort_values("start_time")

    watch["Signal pp"] = (
        pd.to_numeric(watch["predicted_movement"], errors="coerce") * 100
    ).round(2)

    predicted_close = pd.to_numeric(
        watch["predicted_close_probability"], errors="coerce"
    )
    yes_threshold = (1 / predicted_close).round(2)
    no_threshold = (1 / (1 - predicted_close)).round(2)

    watch["Bevakning"] = "Ingen signal ännu"
    watch.loc[watch["bet_side"] == "YES", "Bevakning"] = "YES-signal"
    watch.loc[watch["bet_side"] == "NO", "Bevakning"] = "NO-signal"

    # För NONE visar vi vilken sida modellen lutar åt och vilket odds som
    # motsvarar den predikterade closing probabilityn på den sidan.
    lean_yes = watch["predicted_movement"] >= 0
    watch["Lutar åt"] = "NO"
    watch.loc[lean_yes, "Lutar åt"] = "YES"
    watch.loc[watch["bet_side"] == "YES", "Lutar åt"] = "YES"
    watch.loc[watch["bet_side"] == "NO", "Lutar åt"] = "NO"

    watch["Modellodds"] = no_threshold
    watch.loc[watch["Lutar åt"] == "YES", "Modellodds"] = yes_threshold

    if "best_bet_odds" in watch.columns:
        watch["Bästa odds"] = pd.to_numeric(
            watch["best_bet_odds"], errors="coerce"
        )
    else:
        watch["Bästa odds"] = pd.NA

    watch["Timmar till kickoff"] = (
        watch["start_time"] - now
    ).dt.total_seconds() / 3600

    watch = watch.rename(columns={
        "start_time": "Kickoff", "home_team": "Hemma",
        "away_team": "Borta", "bookmakers": "Bookmakers"
    })

    columns = [
        "Kickoff", "Hemma", "Borta", "Lutar åt", "Signal pp",
        "Bevakning", "Modellodds", "Bästa odds", "Timmar till kickoff"
    ]
    if "Bookmakers" in watch.columns:
        columns.insert(3, "Bookmakers")

    st.dataframe(
        watch[columns],
        use_container_width=True,
        hide_index=True
    )


# ==================================================
# MATCHER SOM KRÄVER UPPMÄRKSAMHET
# ==================================================

attention = df[df["status"].isin(["TAKE_SNAPSHOT", "NO_CLOSE_SNAPSHOT"])]
st.subheader("Kräver uppmärksamhet")

if len(attention) == 0:
    st.success("Inga matcher kräver åtgärd just nu.")
else:
    for _, row in attention.iterrows():
        st.warning(f"{row['home_team']} – {row['away_team']}: {row['status']}")


# ==================================================
# FÄRDIGSPELADE
# ==================================================

st.subheader("Färdigspelade bets")

if len(finished) == 0:
    st.info("Inga V1-bets har avgjorts ännu.")
else:
    finished_display = finished.copy()
    finished_display["CLV %"] = finished_display["current_economic_clv"] * 100
    finished_display = finished_display.rename(columns={
        "home_team": "Hemma", "away_team": "Borta", "bet_side": "Bet",
        "original_odds": "Odds", "home_goals": "HG", "away_goals": "AG",
        "won": "Vinst", "profit": "Profit"
    })
    st.dataframe(
        finished_display[[
            "Hemma", "Borta", "Bet", "Odds", "HG", "AG",
            "Vinst", "Profit", "CLV %"
        ]],
        use_container_width=True,
        hide_index=True
    )


# ==================================================
# RESULTATSTATISTIK
# ==================================================

if len(finished) > 0:
    st.subheader("Forward-resultat")
    wins = finished["won"].sum()
    hit_rate = wins / len(finished) * 100
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avgjorda", len(finished))
    c2.metric("Wins", int(wins))
    c3.metric("Hit rate", f"{hit_rate:.1f}%")
    c4.metric("Profit", f"{total_profit:.2f} u")


# ==================================================
# INFORMATION
# ==================================================

st.divider()
st.caption(
    "Dashboarden läser endast lokala forward-testfiler. "
    "Den genererar inga nya predictions och hämtar inga odds."
)
