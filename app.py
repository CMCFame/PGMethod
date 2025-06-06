import streamlit as st
import pandas as pd
import numpy as np
import base64
import json
import time
import math
from openai import OpenAI
import odds_scraper # Nuestro módulo de scraping

# --- Configuración de la App ---
st.set_page_config(page_title="Progol AI-Vision Optimizer", page_icon="🤖", layout="wide")
st.title("🤖 Progol AI-Vision Optimizer")
st.markdown("Una herramienta de vanguardia para la optimización de portafolios Progol, con dos modos de trabajo: **Automático (IA)** y **Manual (CSV)**.")

# --- CONSTANTES DE LA METODOLOGÍA ---
PROB_ANCLA = 0.60
DRAW_PROPENSITY_THRESHOLD = 0.08
L_SUM_RANGE = (5.0, 5.8)
E_SUM_RANGE = (3.5, 4.6)
V_SUM_RANGE = (4.2, 5.2)
MIN_DRAWS_PER_TICKET = 4
MAX_DRAWS_PER_TICKET = 6

# --- MÓDULO DE OCR CON OPENAI ---
def get_matches_from_image_with_ocr(image_bytes, api_key):
    st.info("Contactando a la IA de Visión... por favor espera.")
    try:
        client = OpenAI(api_key=api_key)
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        response = client.chat.completions.create(
            model="o4-mini-2025-04-16",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """
                            Analiza esta imagen de una quiniela de Progol. Extrae los partidos (principales y revancha).
                            Devuelve el resultado como un único array JSON.
                            Cada objeto debe tener las claves "home" y "away".
                            Limpia los nombres de los equipos (ej. "A. SAUD SUB" -> "Arabia Saudita Sub-23", "E.U.A." -> "USA").
                            Si no puedes analizar la imagen o no encuentras los partidos, devuelve un array JSON vacío: [].
                            Tu respuesta DEBE contener únicamente el array JSON y nada más.
                            """
                        },
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            max_completion_tokens=2000,
        )
        json_string = response.choices[0].message.content.strip()
        if json_string.startswith('[') and json_string.endswith(']'):
            json_string = json_string.replace("```json", "").replace("```", "")
            return json.loads(json_string)
        else:
            st.error("La respuesta de la IA no fue un JSON válido.")
            st.code(json_string)
            return None
    except Exception as e:
        st.error(f"Error con la API de OpenAI: {e}")
        return None

# --- MÓDULOS DE MODELADO Y OPTIMIZACIÓN ---
def get_most_probable_result(row):
    probs = {'L': row['p_L'], 'E': row['p_E'], 'V': row['p_V']}
    return max(probs, key=probs.get)

def apply_draw_propensity_rule(df):
    for i, row in df.iterrows():
        is_close = abs(row['p_L'] - row['p_V']) < DRAW_PROPENSITY_THRESHOLD
        is_draw_favored = row['p_E'] > max(row['p_L'], row['p_V'])
        if is_close and is_draw_favored:
            df.loc[i, 'p_E'] += 0.06; df.loc[i, 'p_L'] -= 0.03; df.loc[i, 'p_V'] -= 0.03
    return df

def apply_global_regularization(df):
    for col, (min_val, max_val) in zip(['p_L', 'p_E', 'p_V'], [L_SUM_RANGE, E_SUM_RANGE, V_SUM_RANGE]):
        col_sum = df[col].sum()
        if col_sum < min_val: df[col] *= (min_val / col_sum)
        elif col_sum > max_val: df[col] *= (max_val / col_sum)
    df[['p_L', 'p_E', 'p_V']] = df[['p_L', 'p_E', 'p_V']].div(df[['p_L', 'p_E', 'p_V']].sum(axis=1), axis=0)
    return df

@st.cache_data
def run_montecarlo_simulation(quiniela_tuple, probabilities_tuple, num_simulations):
    quiniela_indices = np.array([{'L': 0, 'E': 1, 'V': 2}[res] for res in quiniela_tuple])
    probabilities = np.array(probabilities_tuple)
    random_outcomes = np.array([np.random.choice(3, size=num_simulations, p=p_row) for p_row in probabilities]).T
    hits = np.sum(random_outcomes == quiniela_indices, axis=1)
    return np.sum(hits >= 11) / num_simulations

def calculate_portfolio_objective(portfolio, probabilities_tuple, num_simulations):
    if not portfolio: return 0
    probs_win = [run_montecarlo_simulation(tuple(q), probabilities_tuple, num_simulations) for q in portfolio]
    return 1 - np.prod([(1 - p) for p in probs_win])

def is_valid_quiniela(quiniela):
    draws = quiniela.count('E')
    return MIN_DRAWS_PER_TICKET <= draws <= MAX_DRAWS_PER_TICKET

def create_initial_portfolio(df, num_quinielas):
    portfolio = []
    base_results = df['result'].tolist()
    for _ in range(num_quinielas):
        quiniela = base_results.copy()
        while not is_valid_quiniela(quiniela):
            idx_to_change = np.random.randint(0, len(quiniela))
            quiniela[idx_to_change] = np.random.choice(['L', 'E', 'V'])
        portfolio.append(quiniela)
    return portfolio

def get_neighbor_portfolio(portfolio):
    new_portfolio = [q.copy() for q in portfolio]
    q_idx = np.random.randint(0, len(new_portfolio)); m_idx = np.random.randint(0, len(new_portfolio[q_idx]))
    original_quiniela = new_portfolio[q_idx].copy()
    for _ in range(10):
        new_quiniela = original_quiniela.copy()
        options = ['L', 'E', 'V']; options.remove(new_quiniela[m_idx])
        new_quiniela[m_idx] = np.random.choice(options)
        if is_valid_quiniela(new_quiniela):
            new_portfolio[q_idx] = new_quiniela
            return new_portfolio
    return portfolio

def run_simulated_annealing(df, num_quinielas, num_simulations, initial_temp, cooling_rate, iterations):
    probabilities_tuple = tuple(map(tuple, df[['p_L', 'p_E', 'p_V']].values))
    current_portfolio = create_initial_portfolio(df, num_quinielas)
    current_energy = calculate_portfolio_objective(current_portfolio, probabilities_tuple, num_simulations)
    best_portfolio, best_energy = current_portfolio, current_energy
    temp = initial_temp
    progress_bar = st.progress(0, text="Iniciando optimización...")
    for i in range(iterations):
        neighbor_portfolio = get_neighbor_portfolio(current_portfolio)
        neighbor_energy = calculate_portfolio_objective(neighbor_portfolio, probabilities_tuple, num_simulations)
        delta_energy = neighbor_energy - current_energy
        if delta_energy > 0 or np.random.rand() < math.exp(delta_energy / temp):
            current_portfolio, current_energy = neighbor_portfolio, neighbor_energy
        if current_energy > best_energy:
            best_portfolio, best_energy = current_portfolio, current_energy
        temp *= cooling_rate
        progress_bar.progress((i + 1) / iterations, text=f"Iteración {i+1}/{iterations} | Temp: {temp:.4f} | Mejor Score: {best_energy:.4f}")
    st.success("¡Optimización completada!")
    return best_portfolio

# --- FLUJO PRINCIPAL DE LA APP ---
if 'matches_df' not in st.session_state: st.session_state.matches_df = None
if 'final_df' not in st.session_state: st.session_state.final_df = None

st.sidebar.header("Elige tu modo de trabajo")

# --- MODO AUTOMÁTICO ---
with st.sidebar.expander("🤖 Modo Automático (IA y Scraping)", expanded=True):
    auto_uploaded_file = st.file_uploader("1. Sube la imagen de la quiniela", type=["png", "jpg"], key="auto_uploader")
    if auto_uploaded_file:
        if st.button("2. Leer Partidos con IA (OCR)", key="ocr_button"):
            api_key = st.secrets.get("OPENAI_API_KEY")
            if not api_key:
                st.error("Configura tu 'OPENAI_API_KEY' en los secretos de Streamlit Cloud.")
            else:
                image_bytes = auto_uploaded_file.getvalue()
                with st.spinner("La IA está analizando la imagen..."):
                    extracted_matches = get_matches_from_image_with_ocr(image_bytes, api_key)
                if extracted_matches:
                    st.session_state.matches_df = pd.DataFrame(extracted_matches)
                    st.session_state.final_df = None
                    st.success(f"¡IA extrajo {len(st.session_state.matches_df)} partidos!")
                else:
                    st.error("No se pudieron extraer los partidos.")

    # --- LÓGICA CORREGIDA ---
    # El botón de scraping solo aparece si la tabla de partidos ya fue creada por el OCR
    if isinstance(st.session_state.get('matches_df'), pd.DataFrame) and not st.session_state.matches_df.empty:
        if st.button("3. Buscar Momios en Tiempo Real", key="scrape_button"):
            all_odds = []
            progress_bar = st.progress(0, text="Buscando momios...")
            for i, row in st.session_state.matches_df.iterrows():
                progress_bar.progress((i + 1) / len(st.session_state.matches_df), text=f"Buscando: {row['home']} vs {row['away']}...")
                odds = odds_scraper.fetch_match_odds(row['home'], row['away'])
                all_odds.append(odds); time.sleep(0.5)
            
            temp_df = st.session_state.matches_df.copy()
            temp_df['odds'] = all_odds
            probs_df = temp_df['odds'].apply(odds_scraper.decimal_to_prob).apply(pd.Series)
            st.session_state.final_df = temp_df.join(probs_df)
            st.success("¡Búsqueda de momios completada!")

# --- MODO MANUAL ---
with st.sidebar.expander("✍️ Modo Manual (Subir CSV)"):
    manual_uploaded_file = st.file_uploader("Sube tu CSV con partidos y probabilidades", type=["csv"], key="manual_uploader")
    if manual_uploaded_file:
        df_manual = pd.read_csv(manual_uploaded_file)
        required_cols = ['home', 'away', 'p_L', 'p_E', 'p_V']
        if all(col in df_manual.columns for col in required_cols):
            st.session_state.final_df = df_manual
            st.success("CSV cargado y listo para optimizar.")
        else:
            st.error(f"El CSV debe contener las columnas: {', '.join(required_cols)}")

# --- PASO FINAL: OPTIMIZACIÓN (COMÚN A AMBOS MODOS) ---
if isinstance(st.session_state.get('final_df'), pd.DataFrame) and not st.session_state.final_df.empty:
    st.header("Datos Listos para Optimización")
    st.dataframe(st.session_state.final_df)
    
    st.header("Optimización del Portafolio")
    st.sidebar.header("Parámetros de Optimización")
    num_quinielas = st.sidebar.slider("Número de quinielas", 5, 30, 15, key="q_slider")
    iterations = st.sidebar.select_slider("Iteraciones", options=[500, 1000, 2000, 5000], value=1000, key="iter_slider")
    initial_temp = st.sidebar.slider("Temperatura inicial", 0.1, 1.0, 0.5, 0.05, key="temp_slider")
    cooling_rate = st.sidebar.select_slider("Tasa de enfriamiento", options=[0.99, 0.995, 0.999], value=0.995, key="cool_slider")
    num_simulations = st.sidebar.select_slider("Simulaciones Montecarlo", options=[1000, 2500, 5000], value=2500, key="sim_slider")

    if st.button("🔥 Iniciar Optimización Avanzada", type="primary"):
        with st.spinner("Aplicando reglas de modelado heurístico..."):
            df_modelado = st.session_state.final_df.copy()
            df_modelado = apply_draw_propensity_rule(df_modelado)
            df_modelado = apply_global_regularization(df_modelado)
            df_modelado['result'] = df_modelado.apply(get_most_probable_result, axis=1)
        st.success("Reglas de modelado aplicadas.")

        final_portfolio = run_simulated_annealing(df_modelado, num_quinielas, num_simulations, initial_temp, cooling_rate, iterations)
        
        st.header("Portafolio Óptimo Encontrado")
        probabilities_tuple = tuple(map(tuple, df_modelado[['p_L', 'p_E', 'p_V']].values))
        final_probs = [run_montecarlo_simulation(tuple(q), probabilities_tuple, num_simulations * 2) for q in final_portfolio]
        
        match_names = df_modelado.apply(lambda row: f"{row['home']} vs {row['away']}", axis=1).tolist()
        quiniela_names = [f"Quiniela {i+1}" for i in range(num_quinielas)]
        portfolio_dict = {name: data for name, data in zip(quiniela_names, final_portfolio)}
        portfolio_df = pd.DataFrame(portfolio_dict, index=match_names)
        
        prob_series = pd.Series({name: f"{prob:.2%}" for name, prob in zip(quiniela_names, final_probs)}, name="Pr[≥11]")
        portfolio_df.loc["**Pr[≥11]**"] = prob_series
        
        st.dataframe(portfolio_df)
        csv_output = portfolio_df.to_csv().encode('utf-8')
        st.download_button("📥 Descargar Portafolio Óptimo", csv_output, "portafolio_optimizado_pro.csv", "text/csv")