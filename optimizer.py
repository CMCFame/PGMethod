# ===============  optimizer.py  ==============================
import random
from typing import List
import numpy as np
import pandas as pd
from . import config

SIGNS = ["L", "E", "V"]


def greedy_core(df: pd.DataFrame) -> List[str]:
    """Return a 14‑char string (one quiniela) using argmax sign per match."""
    picks = []
    for _, row in df.iterrows():
        picks.append(max(SIGNS, key=lambda s: row[f"p_{s}"]))
    return picks


def generate_portfolio(df: pd.DataFrame, n: int = config.N_TICKETS) -> List[List[str]]:
    """Simplified: build n tickets by flipping random matches to maintain balance."""
    core = greedy_core(df)
    portfolio = [core.copy() for _ in range(n)]

    # introduce diversity
    for i in range(1, n):
        ticket = portfolio[i]
        flips = random.sample(range(len(ticket)), k=5)  # flip 5 positions
        for m in flips:
            ticket[m] = random.choice(SIGNS)
        portfolio[i] = ticket
    return portfolio