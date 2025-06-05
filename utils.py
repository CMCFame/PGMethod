# ===============  utils.py  ==================================
import numpy as np
from typing import List


def ticket_to_counts(ticket: List[str]):
    return {
        "L": ticket.count("L"),
        "E": ticket.count("E"),
        "V": ticket.count("V"),
    }