"""
sync_to_sharepoint.py — Push ROAD_Reason/results/val_metrics.csv to the
shared OneDrive Excel workbook at:
  https://ncaandt-my.sharepoint.com/personal/hmoradi_ncat_edu/...

Authentication: MSAL device-code flow (delegated, user-based).
First run:  prints a URL + code; you open it in a browser and log in once.
After that: token is cached in ~/.road_reason_sharepoint_token.json and
            silently refreshed (valid for ~1 year).

Usage:
  python results/sync_to_sharepoint.py              # push CSV → Excel
  python results/sync_to_sharepoint.py --dry-run    # print what would be written
  python results/sync_to_sharepoint.py --show       # print current sheet contents

Requirements:
  pip install msal requests

Setup:
  1. Register an Azure app (see README comment below).
  2. Set CLIENT_ID below to your app's Application (client) ID.
  3. Make sure Dr. Moradi has shared the Excel file with your NCAT email (edit perms).
  4. Check WORKSHEET_NAME matches the tab name in the Excel file.

README: Azure app registration
  - portal.azure.com → Azure Active Directory → App registrations → New registration
  - Name: "ROAD-Results-Sync", Accounts: "Accounts in any organizational directory"
  - After creating: note the Application (client) ID
  - API permissions → Add a permission → Microsoft Graph → Delegated:
      Files.ReadWrite
  - Grant admin consent (or ask IT; for personal delegated scope this is usually auto-granted)

IMPORTANT — sheet layout:
  The Excel sheet has static sections below the data table (Key, Literature/Gap/Plan).
  This script ONLY writes to the data table region (rows 1 through N_data+1).
  It does NOT touch anything below the last data row — static sections are safe.
"""

import csv
import sys
import argparse
from pathlib import Path

import msal
import requests

# ── Configuration ────────────────────────────────────────────────────────────

# TODO: paste your Azure app's Application (client) ID here after registration
CLIENT_ID = ""  # e.g. "a1b2c3d4-1234-5678-abcd-ef0123456789"

# NCAT tenant
TENANT_ID = "ncat.edu"   # or the full tenant GUID from portal.azure.com

# The sharing URL from Dr. Moradi's file — used to locate the drive item
SHARING_URL = (
    "https://ncaandt-my.sharepoint.com/:x:/r/personal/hmoradi_ncat_edu"
    "/_layouts/15/Doc.aspx?sourcedoc=%7B2E3575D3-9E40-417C-9D6E-FBCA6C640020%7D"
    "&file=ROAD-Waymo%20Results.xlsx&action=default&mobileredirect=true"
    "&DefaultItemOpen=1"
)

# Worksheet name inside the Excel file (check what the tab is called)
WORKSHEET_NAME = "Sheet1"   # update if the tab has a different name

# Local CSV to push
CSV_PATH = Path(__file__).parent / "val_metrics.csv"

# Token cache (persistent between runs)
TOKEN_CACHE_PATH = Path.home() / ".road_reason_sharepoint_token.json"

SCOPES = ["Files.ReadWrite", "User.Read"]
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"

# ── CSV column order — must match the Excel header row exactly ────────────────
# Excel columns: model | source | status | epoch | metric | split | iou |
#                agent_ness | agent | action | loc | duplex | triplet
CSV_FIELDS = [
    "model", "source", "status", "epoch", "metric", "split", "iou",
    "agent_ness", "agent", "action", "loc", "duplex", "triplet",
]

# ── Auth ─────────────────────────────────────────────────────────────────────

def _load_cache() -> msal.SerializableTokenCache:
    cache = msal.SerializableTokenCache()
    if TOKEN_CACHE_PATH.exists():
        cache.deserialize(TOKEN_CACHE_PATH.read_text())
    return cache


def _save_cache(cache: msal.SerializableTokenCache):
    if cache.has_state_changed:
        TOKEN_CACHE_PATH.write_text(cache.serialize())


def get_access_token() -> str:
    if not CLIENT_ID:
        sys.exit(
            "ERROR: CLIENT_ID is empty.\n"
            "Register an Azure app and paste its Application (client) ID into "
            "results/sync_to_sharepoint.py (CLIENT_ID variable at the top)."
        )

    cache = _load_cache()
    app = msal.PublicClientApplication(CLIENT_ID, authority=AUTHORITY, token_cache=cache)

    # Try silent first (uses cached refresh token)
    accounts = app.get_accounts()
    if accounts:
        result = app.acquire_token_silent(SCOPES, account=accounts[0])
        if result and "access_token" in result:
            _save_cache(cache)
            return result["access_token"]

    # Interactive device code flow (first run or token expired)
    flow = app.initiate_device_flow(scopes=SCOPES)
    if "user_code" not in flow:
        sys.exit(f"ERROR initiating device flow: {flow}")

    print("\n" + "=" * 60)
    print("First-time auth — open this URL in your browser and enter the code:")
    print(f"\n  URL:  {flow['verification_uri']}")
    print(f"  Code: {flow['user_code']}")
    print("\nWaiting for you to complete login...", flush=True)
    print("=" * 60 + "\n")

    result = app.acquire_token_by_device_flow(flow)
    if "access_token" not in result:
        sys.exit(f"Auth failed: {result.get('error_description', result)}")

    _save_cache(cache)
    print(f"Authenticated. Token cached at {TOKEN_CACHE_PATH}")
    return result["access_token"]


# ── Graph API helpers ─────────────────────────────────────────────────────────

def _headers(token: str, session_id: str | None = None) -> dict:
    h = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    if session_id:
        h["workbook-session-id"] = session_id
    return h


def _sharing_id(url: str) -> str:
    """Encode a sharing URL into the base64url format Graph API expects."""
    import base64
    encoded = base64.urlsafe_b64encode(url.encode()).rstrip(b"=").decode()
    return f"u!{encoded}"


def get_workbook_session(token: str, sharing_url: str) -> tuple[str, str]:
    """
    Resolve the sharing URL to a drive item, then create a persistent
    workbook session. Returns (drive_item_base_url, session_id).
    """
    share_id = _sharing_id(sharing_url)

    # Resolve shared item → drive item
    r = requests.get(
        f"https://graph.microsoft.com/v1.0/shares/{share_id}/driveItem",
        headers=_headers(token),
    )
    r.raise_for_status()
    item = r.json()

    drive_id = item["parentReference"]["driveId"]
    item_id  = item["id"]
    base_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{item_id}"

    # Create workbook session (enables batch edits without auto-save on each call)
    r = requests.post(
        f"{base_url}/workbook/createSession",
        headers=_headers(token),
        json={"persistChanges": True},
    )
    r.raise_for_status()
    session_id = r.json()["id"]
    return base_url, session_id


def read_sheet(token: str, base_url: str, session_id: str, sheet: str) -> list[list]:
    """Read the used range of a worksheet."""
    r = requests.get(
        f"{base_url}/workbook/worksheets/{sheet}/usedRange",
        headers=_headers(token, session_id),
    )
    r.raise_for_status()
    return r.json().get("values", [])


def write_range(
    token: str, base_url: str, session_id: str,
    sheet: str, address: str, values: list[list],
):
    """Write values to a specific worksheet range (e.g. 'A1:M8')."""
    r = requests.patch(
        f"{base_url}/workbook/worksheets/{sheet}/range(address='{address}')",
        headers=_headers(token, session_id),
        json={"values": values},
    )
    r.raise_for_status()


def close_session(token: str, base_url: str, session_id: str):
    requests.post(
        f"{base_url}/workbook/closeSession",
        headers=_headers(token, session_id),
    )


# ── Sync logic ────────────────────────────────────────────────────────────────

def col_letter(n: int) -> str:
    """0-indexed column number → Excel column letter (0→A, 12→M, 26→AA)."""
    s = ""
    n += 1
    while n:
        n, rem = divmod(n - 1, 26)
        s = chr(65 + rem) + s
    return s


def data_range_address(n_data_rows: int) -> str:
    """
    Returns the A1-style range for the data table only.
    e.g. 7 data rows + 1 header → A1:M8
    The Key and Literature sections sit further down in the sheet and are
    NOT touched because we write to a fixed range ending at row N+1.
    """
    n_cols = len(CSV_FIELDS)
    last_col = col_letter(n_cols - 1)   # "M" for 13 columns
    last_row = n_data_rows + 1          # +1 for header row
    return f"A1:{last_col}{last_row}"


def load_csv() -> list[dict]:
    with open(CSV_PATH, newline="") as f:
        return list(csv.DictReader(f))


def sync(dry_run: bool = False, show: bool = False):
    token = get_access_token()

    print("Connecting to workbook...", flush=True)
    base_url, session_id = get_workbook_session(token, SHARING_URL)

    try:
        if show:
            existing = read_sheet(token, base_url, session_id, WORKSHEET_NAME)
            print(f"\nCurrent contents of '{WORKSHEET_NAME}':")
            for i, row in enumerate(existing):
                print(f"  {i+1:3d}: {row}")
            return

        # Build the data table from CSV (header row + data rows)
        local_rows = load_csv()
        table = [CSV_FIELDS]   # header
        for r in local_rows:
            table.append([r.get(f, "") for f in CSV_FIELDS])

        address = data_range_address(len(local_rows))

        if dry_run:
            print(f"\nDry run — would write to {WORKSHEET_NAME}!{address}:")
            for row in table:
                print(" ", row)
            print(f"\n{len(local_rows)} data rows + 1 header.")
            print("Rows below this range (Key, Literature sections) are NOT touched.")
            return

        print(f"Writing {len(table)} rows to {WORKSHEET_NAME}!{address} ...", flush=True)
        write_range(token, base_url, session_id, WORKSHEET_NAME, address, table)
        print(f"Done. {len(local_rows)} result rows pushed to OneDrive.")
        print("Key and Literature sections below are unchanged.")

    finally:
        close_session(token, base_url, session_id)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Push val_metrics.csv to the shared OneDrive Excel workbook."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be written without modifying Excel")
    parser.add_argument("--show",    action="store_true",
                        help="Print current Excel sheet contents")
    args = parser.parse_args()

    sync(dry_run=args.dry_run, show=args.show)
