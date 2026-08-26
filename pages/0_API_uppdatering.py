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

include_fixtures = st.checkbox(
    "Uppdatera även Premier League-matchlistan (+1 API-anrop)",
    value=False,
    help="Behövs främst när nya matcher har kommit in eller fixturelistan blivit gammal.",
)

include_results = st.checkbox(
    "Uppdatera även resultat",
    value=False,
    help=(
        "Hämtar fixtures och score för färdigspelade frysta V1-spel. "
        "Detta använder extra API-anrop, så låt den vara av om du bara vill uppdatera odds och CLV."
    ),
)

base_cost = 4 + (1 if include_fixtures else 0)

st.info(
    f"Odds/CLV-uppdateringen kostar normalt **{base_cost} API-anrop** "
    f"({'1 fixtures + 4 bookmakers' if include_fixtures else '4 bookmakers'}). "
    "Resultatuppdatering kan använda ytterligare anrop beroende på hur många frysta matcher som är färdigspelade."
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

    if failed:
        st.error("Uppdateringen misslyckades. Inga automatiska omförsök görs.")
    else:
        st.success("Klart. Odds, snapshots och CLV är uppdaterade.")

    with st.expander("Visa körlogg"):
        for name, result in outputs:
            st.markdown(f"**{name}**")
            st.code((result.stdout or "") + ("\n" + result.stderr if result.stderr else ""))

st.divider()
st.subheader("Vad knappen gör")
st.markdown(
    "Den manuella odds/CLV-körningen hämtar marknaden, arkiverar både consensus- och "
    "bookmaker-snapshots, uppdaterar V1-monitorn, räknar om CLV och bygger om dashboarden. "
    "Resultat hämtas bara om du markerar resultatrutan ovan."
)
