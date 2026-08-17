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


print()
print("===================================")
print("AUTOMATIC MARKET SNAPSHOT")
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
