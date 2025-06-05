# ===============  app.py  ===================================
import streamlit as st
import pandas as pd
from pathlib import Path
from io import StringIO
from . import data_loader, optimizer, config
from .odds_scraper import fetch_portfolio_odds, decimal_to_prob

st.set_page_config(page_title="Progol Toolkit", layout="wide")
st.title("⚽ Progol Quiniela Generator")

# Sidebar inputs
st.sidebar.header("Datos de Entrada")

csv_file = st.sidebar.file_uploader("Sube el Progol.csv", type=["csv"])
odds_json_text = st.sidebar.text_area(
    "Pega el JSON de momios (opcional)",
    placeholder="{\n  \"1\": {\"L\": 1.85, \"E\": 3.60, \"V\": 4.20},\n  ...\n}",
    height=150,
)

scrape_btn = st.sidebar.button("Raspar Momios 🌐 (SofaScore)")

uploaded_path: str | Path | None = None
if csv_file:
    uploaded_path = Path("/tmp/progol_upload.csv")
    uploaded_path.write_bytes(csv_file.getbuffer())

if scrape_btn and uploaded_path:
    fixt_df = data_loader.load_fixtures(uploaded_path)
    fixtures = fixt_df.to_records(index=False)
    with st.spinner("Raspando momios…"):
        odds_map = fetch_portfolio_odds(fixtures)
    st.sidebar.success(f"Momios obtenidos para {len(odds_map)}/14 partidos")
    odds_json_text = json.dumps(odds_map, indent=2)

# Parse odds JSON if present
try:
    odds_map = json.loads(odds_json_text) if odds_json_text.strip() else None
except Exception:
    st.sidebar.error("JSON de momios inválido – se usará dummy odds")
    odds_map = None

# Main action
if st.button("Generar Portafolio ✅"):
    if not uploaded_path:
        st.error("Debes subir un Progol.csv primero")
        st.stop()
    df = data_loader.load_data(uploaded_path, odds_map)
    st.subheader("Probabilidades por partido")
    st.dataframe(df[["match_no", "home", "away", "prob_L", "prob_E", "prob_V"]])

    port, metrics = optimizer.build_portfolio(df, config)  # assuming optimizer returns (matrix, dict)
    st.subheader("Matriz 14×N boletos")
    st.dataframe(pd.DataFrame(port))
    st.markdown(f"**Pr[≥11]:** {metrics['pr11']:.2%} – **ROI esp.:** {metrics['roi']:.1%}")
