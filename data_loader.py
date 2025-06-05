# ===============  data_loader.py  ============================
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from . import config
from .odds_scraper import decimal_to_prob


def load_fixtures(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path).head(14)  # 14 regular matches
    return df[["match_no", "home", "away"]]


def load_data(csv_path: str | Path = None, odds: Dict[int, Dict[str, float]] = None) -> pd.DataFrame:
    if csv_path is None:
        csv_path = config.DEFAULT_PROGOL_CSV
    raw = pd.read_csv(csv_path)
    raw = raw.head(14)
    if odds is None:
        odds = config.DUMMY_ODDS
    raw["odds_dict"] = raw["match_no"].map(lambda m: odds.get(m, {}))
    raw[["prob_L", "prob_E", "prob_V"]] = raw["odds_dict"].apply(decimal_to_prob).apply(pd.Series)
    return raw.drop(columns=["odds_dict"])