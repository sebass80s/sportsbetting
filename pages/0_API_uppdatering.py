from pathlib import Path
import subprocess
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_SCRIPT = PROJECT_ROOT / "src" / "run_market_snapshot.py"
FIXTURES_SCRIPT = PROJECT_ROOT / "src" / "get_upcoming_pl.py"
RESULTS_SCRIPT = PROJECT_ROOT / "src" / "update_forward_results.py"
DASHBOARD_SCRIPT = PROJECT_ROOT / "src" / "v1_dashboard.py"
SNAPSHOT_FILE = PROJECT_ROOT / "data" / "forward" / "market_snapshot_history.csv"
UPCOMING_FILE = PROJECT_ROOT / "data" / "forward" / "upcoming_pl.csv"

st.set_page_config(
    page_title="API-uppdatering",
    page_icon="🔄",
    layout="wide",
)

st.title("🔄 API-uppdatering")
st.caption(
    "Manuellt budgetläge för OddsPapi. Sidan gör 0 API-anrop tills du själv trycker på knappen."
)

# Körloggen sparas i session_state så att den överlever Streamlit-reruns.
if "api_refresh_log" not in st.session_state:
    st.session_state.api_refresh_log = []
if "api_refresh_failed" not in st.session_state:
    st.session_state.api_refresh_failed = None
if "api_refresh_finished_at" not in st.session_state:
    st.session_state.api_refresh_finished_at = None

include_fixtures = st.checkbox(
    "Uppdatera även Premier League-matchlistan (+1 API-anrop)",
    value=False,
    help="Behövs främst när nya matcher har kommit in eller fixturelistan blivit gammal.",
)

include_results = st.checkbox(
    "Uppdatera även resultat (+1 fixture-anrop, score endast för nya färdiga bets)",
    value=True,
    help=(
        "Hämtar fixtures för de frysta V1-spelen. Redan sparade slutresultat återanvänds, "
        "så score-endpointen anropas bara för färdigspelade bets som ännu saknar resultat."
    ),
)

base_cost = 4 + (1 if include_fixtures else 0) + (1 if include_results else 0)

st.info(
    f"Den valda körningen kostar normalt minst **{base_cost} API-anrop** "
    f"(4 bookmakers"
    f"{', +1 fixturelista' if include_fixtures else ''}"
    f"{', +1 resultat-fixtures' if include_results else ''}). "
    "Om nya frysta bets har hunnit bli färdigspelade tillkommer ett score-anrop per sådan match."
)

if SNAPSHOT_FILE.exists():
    ts = pd.Timestamp(SNAPSHOT_FILE.stat().st_mtime, unit="s", tz="UTC").tz_convert("Europe/Stockholm")
    st.caption(f"Senaste lokala market snapshot-fil: {ts.strftime('%d/%m/%Y %H:%M:%S')}")

if UPCOMING_FILE.exists():
    ts = pd.Timestamp(UPCOMING_FILE.stat().st_mtime, unit="s", tz="UTC").tz_convert("Europe/Stockholm")
    st.caption(f"Senaste lokala fixturelista: {ts.strftime('%d/%m/%Y %H:%M:%S')}")

if st.button("🔄 Uppdatera odds/CLV nu", type="primary"):
    outputs = []
    failed = False

    with st.spinner("Kör manuell uppdatering…"):
        if include_fixtures:
            r = subprocess.run(
                [sys.executable, str(FIXTURES_SCRIPT)],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
            )
            outputs.append(("Fixturelista", r))
            if r.returncode != 0:
                failed = True

        if not failed:
            r = subprocess.run(
                [sys.executable, str(SNAPSHOT_SCRIPT), "--manual"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
            )
            outputs.append(("Odds + snapshots + CLV", r))
            if r.returncode != 0:
                failed = True

        if include_results and not failed:
            r = subprocess.run(
                [sys.executable, str(RESULTS_SCRIPT)],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
            )
            outputs.append(("Resultat", r))
            if r.returncode != 0:
                failed = True

        # Resultatuppdateraren körs efter snapshot-pipelinen, så dashboarden
        # byggs en gång till om resultat har uppdaterats.
        if include_results and not failed:
            r = subprocess.run(
                [sys.executable, str(DASHBOARD_SCRIPT)],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
            )
            outputs.append(("Dashboard", r))
            if r.returncode != 0:
                failed = True

    # Spara endast serialiserbar text i session_state, inte CompletedProcess-objekt.
    st.session_state.api_refresh_log = [
        {
            "name": name,
            "returncode": result.returncode,
            "text": (result.stdout or "")
            + ("\n" + result.stderr if result.stderr else ""),
        }
        for name, result in outputs
    ]
    st.session_state.api_refresh_failed = failed
    st.session_state.api_refresh_finished_at = pd.Timestamp.now(tz="Europe/Stockholm")

if st.session_state.api_refresh_failed is True:
    st.error("Senaste uppdateringen misslyckades. Inga automatiska omförsök görs.")
elif st.session_state.api_refresh_failed is False:
    st.success("Klart. Odds, snapshots, CLV och valda resultat är uppdaterade.")

if st.session_state.api_refresh_log:
    finished_at = st.session_state.api_refresh_finished_at
    if finished_at is not None:
        st.caption(
            "Senaste körning: "
            f"{finished_at.strftime('%d/%m/%Y %H:%M:%S')} (Europe/Stockholm)"
        )

    with st.expander("Visa senaste körlogg", expanded=False):
        for entry in st.session_state.api_refresh_log:
            status = "OK" if entry["returncode"] == 0 else f"FEL ({entry['returncode']})"
            st.markdown(f"**{entry['name']} — {status}**")
            st.code(entry["text"] or "(ingen output)")

        if st.button("Rensa körlogg"):
            st.session_state.api_refresh_log = []
            st.session_state.api_refresh_failed = None
            st.session_state.api_refresh_finished_at = None
            st.rerun()

st.divider()
st.subheader("Vad knappen gör")
st.markdown(
    "Den manuella körningen hämtar marknaden, arkiverar både consensus- och bookmaker-snapshots, "
    "uppdaterar V1-monitorn, räknar om CLV och bygger dashboarden. Resultatrutan är på som standard: "
    "redan färdiga resultat återanvänds och nya score-anrop görs bara när ett fryst bet har spelats klart."
)
