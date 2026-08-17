import requests
from pathlib import Path


# ==================================================
# PROJEKTMAPP
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


# ==================================================
# LÄS API-NYCKEL DIREKT FRÅN FIL
# ==================================================

API_KEY = None

with open(ENV_FILE, "r", encoding="utf-8-sig") as f:
    for line in f:
        line = line.strip()

        if line.startswith("ODDSPAPI_API_KEY="):
            API_KEY = line.split("=", 1)[1].strip()

            # Ta bort eventuella citationstecken
            API_KEY = API_KEY.strip("'\"")
            break


if not API_KEY:
    raise RuntimeError(
        f"Kunde inte läsa ODDSPAPI_API_KEY från {ENV_FILE}"
    )


print()
print("===================================")
print("ODDSPAPI TEST")
print("===================================")

print("ENV-fil:", ENV_FILE)
print("API-nyckel hittad:", True)
print("Nyckellängd:", len(API_KEY))


# ==================================================
# TESTA API
# ==================================================

BASE_URL = "https://api.oddspapi.io/v4"

url = f"{BASE_URL}/markets"

params = {
    "apiKey": API_KEY,
    "sportId": 10
}


try:
    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    print("HTTP status:", response.status_code)
    print()

    if response.ok:
        print("Svar mottaget.")

        try:
            data = response.json()
            print(str(data)[:5000])

        except ValueError:
            print(response.text[:5000])

    else:
        print("API-anrop misslyckades.")
        print(response.text[:5000])

except requests.RequestException as e:
    print("Nätverksfel:")
    print(e)