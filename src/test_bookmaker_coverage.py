import requests
from pathlib import Path
import time


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

API_KEY = None

with open(ENV_FILE, "r", encoding="utf-8-sig") as f:
    for line in f:
        line = line.strip()

        if line.startswith("ODDSPAPI_API_KEY="):
            API_KEY = (
                line
                .split("=", 1)[1]
                .strip()
                .strip("'\"")
            )
            break

if not API_KEY:
    raise RuntimeError("API-nyckel saknas")


# Bookmakers vi testar
BOOKMAKERS = [
    "pinnacle",
    "bet365",
    "bet365.se",
    "unibet",
    "betfair",
    "betway"
]


URL = "https://api.oddspapi.io/v4/odds-by-tournaments"


print()
print("===================================")
print("BOOKMAKER COVERAGE")
print("===================================")


for bookmaker in BOOKMAKERS:

    params = {
        "apiKey": API_KEY,
        "tournamentIds": "17",
        "bookmaker": bookmaker,
        "language": "en",
        "verbosity": 3,
        "oddsFormat": "decimal"
    }

    try:

        response = requests.get(
            URL,
            params=params,
            timeout=30
        )

        if response.status_code != 200:

            print()
            print(bookmaker)
            print(
                "HTTP:",
                response.status_code
            )

            print(
                response.text[:300]
            )

            time.sleep(1.1)
            continue


        data = response.json()

        fixtures = len(data)

        print()
        print(
            bookmaker,
            "| fixtures:",
            fixtures
        )

    except Exception as e:

        print()
        print(
            bookmaker,
            "| FEL:",
            e
        )

    # API cooldown
    time.sleep(1.1)