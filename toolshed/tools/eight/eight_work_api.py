
import os, datetime as dt, requests, pandas as pd
from typing import List

API_KEY   = os.getenv("EIGHT_API_KEY")
USERNAME  = os.getenv("EIGHT_USERNAME")
PASSWORD  = os.getenv("EIGHT_PASSWORD")
PBX_ID    = os.getenv("EIGHT_PBX_ID", "allpbxes")
BASE_AUTH = "https://api.8x8.com/analytics/work/v1/oauth/token"      # :contentReference[oaicite:2]{index=2}
BASE_CDR  = "https://api.8x8.com/analytics/work/v2/call-records"     # :contentReference[oaicite:3]{index=3}
PAGE_SIZE = 50                                                       # API requires exactly 50

if not all((API_KEY, USERNAME, PASSWORD)):
    raise SystemExit("Set EIGHT_API_KEY, EIGHT_USERNAME, and EIGHT_PASSWORD env vars first.")

def get_token() -> str:
    """Return a bearer token that is valid for ~30 minutes."""
    hdrs = {"8x8-apikey": API_KEY,
            "Content-Type": "application/x-www-form-urlencoded"}
    resp = requests.post(BASE_AUTH, headers=hdrs,
                         data={"username": USERNAME, "password": PASSWORD},
                         timeout=15)
    resp.raise_for_status()
    return resp.json()["access_token"]

def fetch_cdrs(token: str,
               start: dt.datetime,
               end:   dt.datetime,
               tz: str = "UTC") -> List[dict]:
    """
    Download every Call Detail Record that falls within [start, end].

    Works around two quirks:
      • Subsequent pages STILL need startTime, endTime, timeZone.
      • When the server indicates 'No Data' we stop before sending one
        more request (that would get a 400).
    """
    hdrs = {"Authorization": f"Bearer {token}", "8x8-apikey": API_KEY}

    base_params = {                       # stays the same for every page
        "pbxId":    PBX_ID,
        "startTime": start.strftime("%Y-%m-%d %H:%M:%S"),
        "endTime":   end.strftime("%Y-%m-%d %H:%M:%S"),
        "timeZone":  tz,
        "pageSize":  PAGE_SIZE,           # must be 50
    }

    all_rows, next_scroll = [], None

    while True:
        params = dict(base_params)        # clone the template
        if next_scroll:
            params["scrollId"] = next_scroll

        r = requests.get(BASE_CDR, headers=hdrs, params=params, timeout=30)
        r.raise_for_status()
        body = r.json()

        all_rows.extend(body.get("data", []))

        # What scrollId should we send back?
        next_scroll = body.get("meta", {}).get("scrollId")

        # Stop when meta.scrollId is missing OR literally "No Data"
        if not next_scroll or next_scroll.lower() == "no data":
            break

    return all_rows