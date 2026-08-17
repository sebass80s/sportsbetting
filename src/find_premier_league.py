import requests
from pathlib import Path


# ==================================================
# API-NYCKEL
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

API_KEY = None

with open(ENV_FILE, "r", encoding="utf-8-sig") as f:
    for line in f:
        line = line.strip()

        if line.startswith("ODDSPAPI_API_KEY="):
            API_KEY = line.split("=", 1)[1].strip()
            API_KEY = API_KEY.strip("'\"")
            break


if not API_KEY:
    raise RuntimeError(
        "Kunde inte läsa API-nyckeln från .env"
    )


# ==================================================
# HÄMTA FOTBOLLSTURNERINGAR
# ==================================================

BASE_URL = "https://api.oddspapi.io/v4"

url = f"{BASE_URL}/tournaments"

params = {
    "apiKey": API_KEY,
    "sportId": 10,
    "language": "en"
}


response = requests.get(
    url,
    params=params,
    timeout=30
)

print()
print("===================================")
print("PREMIER LEAGUE SEARCH")
print("===================================")

print(
    "HTTP status:",
    response.status_code
)


if not response.ok:
    print(response.text[:3000])
    raise SystemExit


tournaments = response.json()


# ==================================================
# SÖK EFTER PREMIER LEAGUE
# ==================================================

matches = []

for tournament in tournaments:

    name = str(
        tournament.get(
            "tournamentName",
            ""
        )
    ).lower()

    slug = str(
        tournament.get(
            "tournamentSlug",
            ""
        )
    ).lower()

    category = str(
        tournament.get(
            "categoryName",
            ""
        )
    ).lower()

    if (
        "premier league" in name
        or
        "premier-league" in slug
    ):
        matches.append(
            tournament
        )


print()
print(
    "Antal träffar:",
    len(matches)
)

print()


for tournament in matches:

    print("-----------------------------------")

    print(
        "Tournament ID:",
        tournament.get(
            "tournamentId"
        )
    )

    print(
        "Name:",
        tournament.get(
            "tournamentName"
        )
    )

    print(
        "Slug:",
        tournament.get(
            "tournamentSlug"
        )
    )

    print(
        "Category:",
        tournament.get(
            "categoryName"
        )
    )

    print(
        "Future fixtures:",
        tournament.get(
            "futureFixtures"
        )
    )

    print(
        "Upcoming fixtures:",
        tournament.get(
            "upcomingFixtures"
        )
    )

    print()