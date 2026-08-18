import subprocess
import sys
from pathlib import Path
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SCRIPTS = [
    "build_market_consensus.py",
    "archive_market_snapshot.py",
    "archive_raw_market.py",
    "v1_forward_monitor.py",
    "ou_v1_forward_monitor.py",
    "update_forward_results.py",
]


# API-budgetskydd:
# En schemalagd/automatisk körning utan --manual avslutas innan några API-anrop görs.
# Dashboardens manuella uppdateringsknapp kör detta script med --manual.
if "--manual" not in sys.argv:
    print()
    print("===================================")
    print("MARKET SNAPSHOT HOPPAS ÖVER")
    print("===================================")
    print("Automatiska API-anrop är avstängda i budgetläge.")
    print("Kör med --manual eller använd Streamlit-sidan API-uppdatering.")
    raise SystemExit(0)


print()
print("===================================")
print("MANUAL MARKET SNAPSHOT")
print("===================================")

print("Start:", datetime.now().isoformat(timespec="seconds"))
print()


for script in SCRIPTS:

    script_path = (
        PROJECT_ROOT
        / "src"
        / script
    )

    print("-----------------------------------")
    print("Kör:", script)
    print("-----------------------------------")

    result = subprocess.run(
        [
            sys.executable,
            str(script_path)
        ],
        cwd=PROJECT_ROOT
    )

    if result.returncode != 0:

        print()
        print("FEL!")
        print(
            script,
            "returnerade kod",
            result.returncode
        )

        sys.exit(
            result.returncode
        )


print()
print("===================================")
print("SNAPSHOT KLAR")
print("===================================")

print(
    "Slut:",
    datetime.now().isoformat(
        timespec="seconds"
    )
)
