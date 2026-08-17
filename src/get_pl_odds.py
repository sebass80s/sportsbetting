import requests
import pandas as pd
from pathlib import Path


# ==================================================
# INSTÄLLNINGAR
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

TOURNAMENT_ID = 17

MARKET_1X2 = "101"
MARKET_BTTS = "104"
MARKET_OU25 = "1010"


# ==================================================
# API-NYCKEL
# ==================================================

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


# ==================================================
# HÄMTA ODDS
# ==================================================

url = (
    "https://api.oddspapi.io/v4/"
    "odds-by-tournaments"
)

params = {
    "apiKey": API_KEY,
    "tournamentIds": str(TOURNAMENT_ID),
    "bookmaker": "pinnacle",
    "language": "en",
    "verbosity": 3,
    "oddsFormat": "decimal"
}


response = requests.get(
    url,
    params=params,
    timeout=60
)


print()
print("===================================")
print("PREMIER LEAGUE ODDS")
print("===================================")

print(
    "HTTP status:",
    response.status_code
)


if not response.ok:

    print(
        response.text[:5000]
    )

    raise SystemExit


data = response.json()


print(
    "Fixtures med odds:",
    len(data)
)


# ==================================================
# LADDA FIXTURE-NAMN
# ==================================================

fixture_file = (
    PROJECT_ROOT
    / "data"
    / "forward"
    / "upcoming_pl.csv"
)

fixtures = pd.read_csv(
    fixture_file
)

fixture_lookup = (
    fixtures
    .set_index("fixture_id")
    [
        [
            "start_time",
            "home_team",
            "away_team"
        ]
    ]
    .to_dict("index")
)


# ==================================================
# HJÄLPFUNKTION
# ==================================================

def get_price(
    bookmaker_data,
    market_id,
    outcome_id
):

    try:

        market = (
            bookmaker_data[
                "markets"
            ][market_id]
        )

        outcome = (
            market[
                "outcomes"
            ][str(outcome_id)]
        )

        players = outcome[
            "players"
        ]

        # Standardmarknader använder normalt player 0
        player = players.get(
            "0"
        )

        if player is None:
            return None

        if not player.get(
            "active",
            False
        ):
            return None

        return player.get(
            "price"
        )

    except (
        KeyError,
        TypeError
    ):
        return None


# ==================================================
# FLATTEN ODDS
# ==================================================

rows = []


for fixture in data:

    fixture_id = fixture.get(
        "fixtureId"
    )

    info = fixture_lookup.get(
        fixture_id,
        {}
    )

    bookmaker_odds = fixture.get(
        "bookmakerOdds",
        {}
    )


    for bookmaker, bookmaker_data in (
        bookmaker_odds.items()
    ):

        # ------------------------------------------
        # 1X2
        # ------------------------------------------

        home_odds = get_price(
            bookmaker_data,
            MARKET_1X2,
            101
        )

        draw_odds = get_price(
            bookmaker_data,
            MARKET_1X2,
            102
        )

        away_odds = get_price(
            bookmaker_data,
            MARKET_1X2,
            103
        )


        # ------------------------------------------
        # BTTS
        # ------------------------------------------

        btts_yes = get_price(
            bookmaker_data,
            MARKET_BTTS,
            104
        )

        btts_no = get_price(
            bookmaker_data,
            MARKET_BTTS,
            105
        )


        # ------------------------------------------
        # O/U 2.5
        # ------------------------------------------

        over_25 = get_price(
            bookmaker_data,
            MARKET_OU25,
            1010
        )

        under_25 = get_price(
            bookmaker_data,
            MARKET_OU25,
            1011
        )


        # ------------------------------------------
        # SPARA ENDAST OM NÅGOT FINNS
        # ------------------------------------------

        values = [
            home_odds,
            draw_odds,
            away_odds,
            btts_yes,
            btts_no,
            over_25,
            under_25
        ]

        if all(
            value is None
            for value in values
        ):
            continue


        rows.append({

            "fixture_id":
                fixture_id,

            "start_time":
                info.get(
                    "start_time"
                ),

            "home_team":
                info.get(
                    "home_team"
                ),

            "away_team":
                info.get(
                    "away_team"
                ),

            "bookmaker":
                bookmaker,

            "home_odds":
                home_odds,

            "draw_odds":
                draw_odds,

            "away_odds":
                away_odds,

            "btts_yes":
                btts_yes,

            "btts_no":
                btts_no,

            "over_25":
                over_25,

            "under_25":
                under_25
        })


# ==================================================
# DATAFRAME
# ==================================================

odds = pd.DataFrame(
    rows
)


print()
print(
    "Odds-rader:",
    len(odds)
)

print(
    "Bookmakers:",
    odds[
        "bookmaker"
    ].nunique()
    if len(odds) > 0
    else 0
)


# ==================================================
# KOMPLETTA MARKNADER
# ==================================================

if len(odds) > 0:

    odds[
        "complete_v1"
    ] = (

        odds[
            [
                "home_odds",
                "draw_odds",
                "away_odds",
                "btts_yes",
                "btts_no",
                "over_25",
                "under_25"
            ]
        ]
        .notna()
        .all(axis=1)
    )


    print(
        "Kompletta V1-rader:",
        odds[
            "complete_v1"
        ].sum()
    )


    print()
    print(
        odds[
            [
                "start_time",
                "home_team",
                "away_team",
                "bookmaker",
                "btts_yes",
                "btts_no",
                "over_25",
                "under_25"
            ]
        ]
        .head(30)
        .to_string(
            index=False
        )
    )


# ==================================================
# SPARA
# ==================================================

OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "forward"
    / "pl_odds_current.csv"
)

odds.to_csv(
    OUTPUT,
    index=False
)


print()
print(
    "Sparad till:",
    OUTPUT
)