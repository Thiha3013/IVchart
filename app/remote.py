"""Writing the watchlist back to the repo from a deployed API.

Locally, "Track" appends a line to app/watchlist.txt and the next `python -m
app.snapshot` picks it up. Deployed, the API server and the GitHub Actions job
that takes snapshots are different machines with no shared disk. The thing they
do share is the repository -- so the deployed API commits the change there,
through the GitHub Contents API, and the Action reads the updated file on its
next run. GitHub is the database.

Configuration (environment variables on the API host):

    GITHUB_REPO    owner/name, e.g. Thiha3013/IVchart
    GITHUB_TOKEN   a fine-grained personal access token with Contents: read/write
                   on that one repository, nothing else
    GITHUB_BRANCH  optional, default "main"

If GITHUB_TOKEN is unset the API falls back to the local file, which is what you
want in development.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request

WATCHLIST_PATH = "app/watchlist.txt"
# The endpoint that calls this is public (a static frontend cannot keep a
# secret), so the blast radius of abuse is bounded here rather than by auth:
# at most this many tickers, each validated as a real optionable symbol.
WATCHLIST_MAX = int(os.environ.get("WATCHLIST_MAX", "40"))


def configured() -> bool:
    return bool(os.environ.get("GITHUB_TOKEN") and os.environ.get("GITHUB_REPO"))


def _request(method: str, url: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
        "User-Agent": "ivchart-api",
    })
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode())


def _contents_url() -> str:
    repo = os.environ["GITHUB_REPO"]
    branch = os.environ.get("GITHUB_BRANCH", "main")
    return f"https://api.github.com/repos/{repo}/contents/{WATCHLIST_PATH}?ref={branch}"


def read_watchlist_text() -> tuple[str, str]:
    """Current file text and its blob sha (needed to update it)."""
    j = _request("GET", _contents_url())
    return base64.b64decode(j["content"]).decode(), j["sha"]


def append_ticker(ticker: str) -> tuple[bool, str]:
    """Append `ticker` to the repo's watchlist in a single commit.

    Returns (added, detail). Idempotent: a ticker already present is not
    re-added. Race-safe enough for this use: the Contents API rejects an update
    whose sha is stale, and we report that rather than retrying blindly.
    """
    ticker = ticker.upper().strip()
    try:
        text, sha = read_watchlist_text()
    except urllib.error.HTTPError as e:
        return False, f"could not read watchlist from GitHub ({e.code})"

    present = {ln.split("#", 1)[0].strip().upper() for ln in text.splitlines()}
    present.discard("")
    if ticker in present:
        return False, f"{ticker} is already on the watchlist"
    if len(present) >= WATCHLIST_MAX:
        return False, f"watchlist is full ({WATCHLIST_MAX} tickers) -- remove one in the repo to add another"

    lines = text.splitlines()
    lines.append(ticker)
    new_text = "".join(ln + chr(10) for ln in lines)

    body = {
        "message": f"watchlist: track {ticker}",
        "content": base64.b64encode(new_text.encode()).decode(),
        "sha": sha,
        "branch": os.environ.get("GITHUB_BRANCH", "main"),
    }
    try:
        _request("PUT", _contents_url().split("?")[0], body)
    except urllib.error.HTTPError as e:
        if e.code == 409:
            return False, "watchlist changed underneath us -- try again"
        return False, f"GitHub rejected the update ({e.code})"
    return True, f"{ticker} added to the repo watchlist -- history starts with the next scheduled snapshot"
