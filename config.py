# ===============  config.py  =================================  =================================  =================================
L_RANGE = (35, 41)   # percent
E_RANGE = (25, 33)
V_RANGE = (30, 36)
X_RANGE_PER_TICKET = (4, 6)
MAX_SIGN_GLOBAL = 0.70
MAX_SIGN_FIRST3 = 0.60
N_TICKETS = 30

DEFAULT_PROGOL_CSV = "/mnt/data/Progol.csv"

# Example odds to test UI without scraping
DUMMY_ODDS = {
    1: {"L": 1.95, "E": 3.50, "V": 3.90},
    2: {"L": 2.10, "E": 3.20, "V": 3.30},
}