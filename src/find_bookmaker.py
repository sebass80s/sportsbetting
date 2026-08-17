import requests
from pathlib import Path


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
    raise RuntimeError(
        "Kunde inte läsa API-nyckeln från .env"
    )


url = "https://api.oddspapi.io/v4/bookmakers"

params = {
    "apiKey": API_KEY
}


response = requests.get(
    url,
    params=params,
    timeout=30
)


print()
print("===================================")
print("BOOKMAKER SEARCH")
print("===================================")

print(
    "HTTP status:",
    response.status_code
)


if not response.ok:
    print(response.text[:5000])
    raise SystemExit


bookmakers = response.json()


search_terms = [
    "pinnacle",
    "bet365",
    "william hill"
]


for term in search_terms:

    print()
    print(
        "Söker:",
        term
    )

    print("-----------------------------------")

    found = []

    for bookmaker in bookmakers:

        text = str(bookmaker).lower()

        if term in text:
            found.append(bookmaker)

    for bookmaker in found:
        print(bookmaker)

    if not found:
        print("Ingen träff")