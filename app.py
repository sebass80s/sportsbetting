import pandas as pd
import streamlit as st
from pathlib import Path
from datetime import datetime, timezone


PROJECT_ROOT = Path(__file__).resolve().parent

STATUS_FILE = PROJECT_ROOT / "data/forward/v1_forward_status.csv"
RESULT_FILE = PROJECT_ROOT / "data/forward/v1_forward_results.csv"


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

    status["start_time"] = pd.to_datetime(
        status["start_time"],
        utc=True
    )

    result_cols = [
        "fixture_id",
        "home_goals",
        "away_goals",
        "actual_btts",
        "won",
        "profit",
        "result_status"
    ]

    result_cols = [
        c for c in result_cols
        if c in results.columns
    ]

    df = status.merge(
        results[result_cols],
        on="fixture_id",
        how="left"
    )

    return df


df = load_data()


# ==================================================
# KPI
# ==================================================

finished = df[
    df["result_status"] == "FINISHED"
].copy()

active = df[
    df["result_status"] != "FINISHED"
].copy()


total_profit = (
    finished["profit"].sum()
    if len(finished)
    else 0
)

roi = (
    finished["profit"].mean() * 100
    if len(finished)
    else 0
)

current_clv = (
    df["current_economic_clv"]
    .dropna()
    .mean()
    * 100
)


col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "V1 bets",
    len(df)
)

col2.metric(
    "Aktiva",
    len(active)
)

col3.metric(
    "Profit",
    f"{total_profit:.2f} u"
)

col4.metric(
    "ROI",
    f"{roi:.2f}%"
)

col5.metric(
    "Current CLV",
    f"{current_clv:.2f}%"
)


# ==================================================
# AKTIVA SIGNALER
# ==================================================

st.subheader("Aktiva signaler")

if len(active) == 0:

    st.success("Inga aktiva bets.")

else:

    active_display = active.copy()

    now = pd.Timestamp(
        datetime.now(timezone.utc)
    )

    active_display["Timmar till kickoff"] = (
        active_display["start_time"] - now
    ).dt.total_seconds() / 3600

    active_display["CLV %"] = (
        active_display[
            "current_economic_clv"
        ] * 100
    )

    active_display = active_display.rename(
        columns={
            "start_time": "Kickoff",
            "home_team": "Hemma",
            "away_team": "Borta",
            "bet_side": "Bet",
            "bookmaker": "Bookmaker",
            "original_odds": "Odds",
            "latest_odds": "Senaste odds",
            "status": "Status"
        }
    )

    st.dataframe(
        active_display[
            [
                "Kickoff",
                "Hemma",
                "Borta",
                "Bet",
                "Bookmaker",
                "Odds",
                "Senaste odds",
                "CLV %",
                "Timmar till kickoff",
                "Status"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )


# ==================================================
# MATCHER SOM KRÄVER UPPMÄRKSAMHET
# ==================================================

attention = df[
    df["status"].isin(
        [
            "TAKE_SNAPSHOT",
            "NO_CLOSE_SNAPSHOT"
        ]
    )
]


st.subheader("Kräver uppmärksamhet")

if len(attention) == 0:

    st.success(
        "Inga matcher kräver åtgärd just nu."
    )

else:

    for _, row in attention.iterrows():

        st.warning(
            f"{row['home_team']} – "
            f"{row['away_team']}: "
            f"{row['status']}"
        )


# ==================================================
# FÄRDIGSPELADE
# ==================================================

st.subheader("Färdigspelade bets")

if len(finished) == 0:

    st.info(
        "Inga V1-bets har avgjorts ännu."
    )

else:

    finished_display = finished.copy()

    finished_display["CLV %"] = (
        finished_display[
            "current_economic_clv"
        ] * 100
    )

    finished_display = finished_display.rename(
        columns={
            "home_team": "Hemma",
            "away_team": "Borta",
            "bet_side": "Bet",
            "original_odds": "Odds",
            "home_goals": "HG",
            "away_goals": "AG",
            "won": "Vinst",
            "profit": "Profit"
        }
    )

    st.dataframe(
        finished_display[
            [
                "Hemma",
                "Borta",
                "Bet",
                "Odds",
                "HG",
                "AG",
                "Vinst",
                "Profit",
                "CLV %"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )


# ==================================================
# RESULTATSTATISTIK
# ==================================================

if len(finished) > 0:

    st.subheader("Forward-resultat")

    wins = finished["won"].sum()

    hit_rate = (
        wins / len(finished) * 100
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Avgjorda",
        len(finished)
    )

    c2.metric(
        "Wins",
        int(wins)
    )

    c3.metric(
        "Hit rate",
        f"{hit_rate:.1f}%"
    )

    c4.metric(
        "Profit",
        f"{total_profit:.2f} u"
    )


# ==================================================
# INFORMATION
# ==================================================

st.divider()

st.caption(
    "Dashboarden läser endast lokala forward-testfiler. "
    "Den genererar inga nya predictions och hämtar inga odds."
)