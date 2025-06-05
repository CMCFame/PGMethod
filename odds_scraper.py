"""Lightweight odds fetchers with SofaScore primary source and
Caliente.mx fallback.  Use `fetch_portfolio_odds(fixtures)` to
get a full {match_no: {L,E,V}} mapping ready for the Toolkit.
"""
from __future__ import annotations
import json, re, time, logging, requests
from typing import Dict, Any, Optional
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

LOGGER = logging.getLogger("odds_scraper")
UA = UserAgent().random
HEADERS = {"User-Agent": UA, "Accept": "application/json"}
BASE_SOFA = "https://api.sofascore.com/api/v1"

# ─────────────────────────────────────────────
#  SofaScore helpers
# ─────────────────────────────────────────────

def _search_sofa(query: str) -> Optional[int]:
    """Return first eventId for `query` or None."""
    url = f"{BASE_SOFA}/search/multi/{query}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        for cat in data.get("categories", []):
            for ev in cat.get("events", []):
                return ev["id"]
    except Exception as exc:
        LOGGER.warning("SofaScore search error: %s", exc)
    return None


def _sofa_odds(event_id: int) -> Optional[Dict[str, float]]:
    url = f"{BASE_SOFA}/event/{event_id}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return None
        ev = r.json().get("event", {})
        odds = ev.get("markets", {}).get("fullTimeResult", {})
        if not odds:
            return None
        return {
            "L": float(odds.get("home", 0)),
            "E": float(odds.get("draw", 0)),
            "V": float(odds.get("away", 0)),
        }
    except Exception as exc:
        LOGGER.warning("SofaScore odds error: %s", exc)
    return None


def from_sofascore(home: str, away: str) -> Optional[Dict[str, float]]:
    q = f"{home} {away}".replace(" ", "%20")
    eid = _search_sofa(q)
    if not eid:
        return None
    return _sofa_odds(eid)

# ─────────────────────────────────────────────
#  Caliente fallback (requests + BS4)
# ─────────────────────────────────────────────

def from_caliente(event_url: str) -> Optional[Dict[str, float]]:
    try:
        r = requests.get(event_url, headers=HEADERS, timeout=10)
        if r.status_code != 200 or "market__row" not in r.text:
            return None
        soup = BeautifulSoup(r.text, "lxml")
        row = soup.select_one(".market__row")
        if not row:
            return None
        prices = [float(el.get_text()) for el in row.select(".price")][:3]
        if len(prices) == 3:
            return {"L": prices[0], "E": prices[1], "V": prices[2]}
    except Exception as exc:
        LOGGER.warning("Caliente parse error: %s", exc)
    return None

# ─────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────

def decimal_to_prob(odds: Dict[str, float]) -> Dict[str, float]:
    inv = {k: 1.0 / v for k, v in odds.items() if v > 0}
    s = sum(inv.values())
    return {k: v / s for k, v in inv.items()}


def fetch_portfolio_odds(fixtures: list[tuple[int, str, str]]) -> Dict[int, Dict[str, float]]:
    """Try SofaScore, fallback (if URL provided) to Caliente.
    `fixtures` is list of (match_no, home, away).
    """
    out: Dict[int, Dict[str, float]] = {}
    for no, home, away in fixtures:
        data = from_sofascore(home, away)
        if data:
            out[no] = data
            time.sleep(0.7)
    return out

# ————————————————————————————————————————————————