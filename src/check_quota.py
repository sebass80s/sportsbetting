import requests
from pathlib import Path


# ==================================================
# PROJEKTMAPP / API-NYCKEL
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

API_KEY = None

with open(
    ENV_FILE,
    "r",
    encoding="utf-8-sig"
) as f:

    for line in f:

        line = line.strip()

        if line.startswith(
            "ODDSPAPI_API_KEY="
        ):

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
# ACCOUNT ENDPOINT
# ==================================================

URL = (
    "https://api.oddspapi.io/v4/account"
)

params = {
    "apiKey": API_KEY
}


response = requests.get(
    URL,
    params=params,
    timeout=30
)


print()
print("===================================")
print("ODDSPAPI QUOTA")
print("===================================")

print(
    "HTTP status:",
    response.status_code
)


if not response.ok:

    print()
    print("API-anrop misslyckades:")
    print(
        response.text[:3000]
    )

    raise SystemExit


# ==================================================
# SVAR
# ==================================================

data = response.json()


# Först visar vi hela strukturen,
# men API-nyckeln ingår normalt inte i svaret.

print()
print("RAW ACCOUNT DATA")
print("-----------------------------------")
print(data)


# ==================================================
# FÖRSÖK HITTA VANLIGA FÄLT
# ==================================================

def get_value(
    source,
    names
):

    for name in names:

        if name in source:
            return source[name]

    return None


request_limit = get_value(
    data,
    [
        "request_limit",
        "requestLimit",
        "requests_limit",
        "requestsLimit"
    ]
)

request_count = get_value(
    data,
    [
        "request_count",
        "requestCount",
        "requests_used",
        "requestsUsed"
    ]
)

remaining = get_value(
    data,
    [
        "requests_remaining",
        "requestsRemaining",
        "remaining"
    ]
)

valid_from = get_value(
    data,
    [
        "valid_from",
        "validFrom"
    ]
)

valid_until = get_value(
    data,
    [
        "valid_until",
        "validUntil"
    ]
)


# ==================================================
# BERÄKNA ÅTERSTÅENDE OM DET BEHÖVS
# ==================================================

if (
    remaining is None
    and
    request_limit is not None
    and
    request_count is not None
):

    try:

        remaining = (
            int(request_limit)
            -
            int(request_count)
        )

    except (
        ValueError,
        TypeError
    ):

        remaining = None


# ==================================================
# VISA
# ==================================================

print()
print("===================================")
print("SAMMANFATTNING")
print("===================================")

print(
    "Request limit:",
    request_limit
)

print(
    "Used:",
    request_count
)

print(
    "Remaining:",
    remaining
)

print(
    "Valid from:",
    valid_from
)

print(
    "Valid until:",
    valid_until
)