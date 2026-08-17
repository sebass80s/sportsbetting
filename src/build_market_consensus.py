import requests
import pandas as pd
import numpy as np
from pathlib import Path
import time


# ==================================================
# INSTÄLLNINGAR
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

TOURNAMENT_ID = 17

BOOKMAKERS = [
    "pinnacle",
    "bet365",
    "unibet",
    "betway"
]

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
# FIXTURES
# ==================================================

fixture_file = (
    PROJECT_ROOT
    / "data"
    / "forward"
    / "upcoming_pl.csv"
)

fixtures = pd.read_csv(fixture_file)

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

        player = players.get("0")

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
# HÄMTA ALLA BOOKMAKERS
# ==================================================

rows = []

URL = (
    "https://api.oddspapi.io/v4/"
    "odds-by-tournaments"
)


for bookmaker in BOOKMAKERS:

    print(
        "Hämtar:",
        bookmaker
    )

    params = {
        "apiKey": API_KEY,
        "tournamentIds":
            str(TOURNAMENT_ID),
        "bookmaker":
            bookmaker,
        "language": "en",
        "verbosity": 3,
        "oddsFormat": "decimal"
    }

    response = requests.get(
        URL,
        params=params,
        timeout=60
    )

    if not response.ok:

        print(
            bookmaker,
            "fel:",
            response.status_code
        )

        print(
            response.text[:500]
        )

        time.sleep(1.1)
        continue


    data = response.json()


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

        bookmaker_data = (
            bookmaker_odds.get(
                bookmaker
            )
        )

        if bookmaker_data is None:
            continue


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

    time.sleep(1.1)


raw = pd.DataFrame(rows)


# ==================================================
# NO-VIG FUNKTIONER
# ==================================================

def two_way_novig(
    a,
    b
):

    if (
        pd.isna(a)
        or pd.isna(b)
        or a <= 1
        or b <= 1
    ):
        return (
            np.nan,
            np.nan
        )

    pa = 1 / a
    pb = 1 / b

    total = pa + pb

    return (
        pa / total,
        pb / total
    )


def three_way_novig(
    a,
    b,
    c
):

    if (
        pd.isna(a)
        or pd.isna(b)
        or pd.isna(c)
        or a <= 1
        or b <= 1
        or c <= 1
    ):
        return (
            np.nan,
            np.nan,
            np.nan
        )

    pa = 1 / a
    pb = 1 / b
    pc = 1 / c

    total = (
        pa + pb + pc
    )

    return (
        pa / total,
        pb / total,
        pc / total
    )


# ==================================================
# RÄKNA NO-VIG PER BOOKMAKER
# ==================================================

novig_rows = []


for _, row in raw.iterrows():

    (
        btts_yes_prob,
        btts_no_prob
    ) = two_way_novig(
        row["btts_yes"],
        row["btts_no"]
    )

    (
        over_prob,
        under_prob
    ) = two_way_novig(
        row["over_25"],
        row["under_25"]
    )

    (
        home_prob,
        draw_prob,
        away_prob
    ) = three_way_novig(
        row["home_odds"],
        row["draw_odds"],
        row["away_odds"]
    )


    novig_rows.append({

        **row.to_dict(),

        "btts_yes_prob":
            btts_yes_prob,

        "btts_no_prob":
            btts_no_prob,

        "over_25_prob":
            over_prob,

        "under_25_prob":
            under_prob,

        "home_prob":
            home_prob,

        "draw_prob":
            draw_prob,

        "away_prob":
            away_prob
    })


novig = pd.DataFrame(
    novig_rows
)


# ==================================================
# CONSENSUS PER FIXTURE
# ==================================================

consensus = (
    novig.groupby(
        [
            "fixture_id",
            "start_time",
            "home_team",
            "away_team"
        ],
        as_index=False
    )
    .agg(
        bookmakers=(
            "bookmaker",
            "nunique"
        ),

        btts_yes_prob=(
            "btts_yes_prob",
            "mean"
        ),

        btts_no_prob=(
            "btts_no_prob",
            "mean"
        ),

        over_25_prob=(
            "over_25_prob",
            "mean"
        ),

        under_25_prob=(
            "under_25_prob",
            "mean"
        ),

        home_prob=(
            "home_prob",
            "mean"
        ),

        draw_prob=(
            "draw_prob",
            "mean"
        ),

        away_prob=(
            "away_prob",
            "mean"
        ),

        avg_btts_yes_odds=(
            "btts_yes",
            "mean"
        ),

        avg_btts_no_odds=(
            "btts_no",
            "mean"
        ),

        avg_over_25_odds=(
            "over_25",
            "mean"
        ),

        avg_under_25_odds=(
            "under_25",
            "mean"
        )
    )
)


# ==================================================
# STATUS
# ==================================================

consensus[
    "complete_v1_market"
] = (
    consensus[
        [
            "btts_yes_prob",
            "btts_no_prob",
            "over_25_prob",
            "home_prob",
            "draw_prob",
            "away_prob"
        ]
    ]
    .notna()
    .all(axis=1)
)


# ==================================================
# SPARA
# ==================================================

raw_file = (
    PROJECT_ROOT
    / "data"
    / "forward"
    / "market_raw.csv"
)

consensus_file = (
    PROJECT_ROOT
    / "data"
    / "forward"
    / "market_consensus.csv"
)

novig.to_csv(
    raw_file,
    index=False
)

consensus.to_csv(
    consensus_file,
    index=False
)


# ==================================================
# VISA
# ==================================================

print()
print("===================================")
print("MARKET CONSENSUS")
print("===================================")

print(
    "Raw bookmaker rows:",
    len(raw)
)

print(
    "Fixtures:",
    len(consensus)
)

print(
    "Kompletta V1 markets:",
    consensus[
        "complete_v1_market"
    ].sum()
)

print()

print(
    consensus[
        [
            "start_time",
            "home_team",
            "away_team",
            "bookmakers",
            "btts_yes_prob",
            "over_25_prob",
            "home_prob",
            "draw_prob",
            "away_prob",
            "avg_btts_yes_odds",
            "complete_v1_market"
        ]
    ]
    .to_string(index=False)
)

print()
print(
    "Sparad till:",
    consensus_file
)