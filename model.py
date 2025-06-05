# ===============  model.py  ==================================
import numpy as np
import pandas as pd


def stack_market_poisson(df: pd.DataFrame, w_raw: float = 0.6) -> pd.DataFrame:
    """Demo stack: use raw probs only (Poisson skipped for brevity)."""
    df["p_L"] = df["prob_L"] * w_raw + (1 - w_raw) * df["prob_L"]
    df["p_E"] = df["prob_E"] * w_raw + (1 - w_raw) * df["prob_E"]
    df["p_V"] = df["prob_V"] * w_raw + (1 - w_raw) * df["prob_V"]
    # Normalize each row
    s = df[["p_L", "p_E", "p_V"]].sum(axis=1)
    df[["p_L", "p_E", "p_V"]] = df[["p_L", "p_E", "p_V"]].div(s, axis=0)
    return df
