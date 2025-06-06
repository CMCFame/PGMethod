# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import time
import math

# --- Configuracion de la App ---
st.set_page_config(page_title="Progol Optimizer Pro (Definitive)", page_icon="??", layout="wide")
st.title("?? Progol Optimizer Pro (Version Definitiva)")
st.markdown("Una implementacion completa de la **Metodologia Definitiva Progol**, incluyendo todas las reglas de validacion y optimizacion estrategica.")

# --- CONSTANTES DE LA METODOLOGIA ---
NUM_MATCHES_PROGOL = 14
L_COUNT_RANGE = (round(0.35 * NUM_MATCHES_PROGOL), round(0.41 * NUM_MATCHES_PROGOL)) # 5-6
E_COUNT_RANGE = (4, 6) # Regla explicita
V_COUNT_RANGE = (round(0.30 * NUM_MATCHES_PROGOL), round(0.36 * NUM_MATCHES_PROGOL)) # 4-5
CONCENTRATION_LIMIT = 0.70 # Limite de concentracion para un solo signo

# Reglas de modelado
DRAW_PROPENSITY_THRESHOLD = 0.08
L_SUM_RANGE = (5.0, 5.8); E_SUM_RANGE = (3.5, 4.6); V_SUM_RANGE = (4.2, 5.2)

# --- FUNCION DE CARGA DE CSV MEJORADA ---
def load_csv_with_encoding_fallbacks(uploaded_file):
    """Intenta leer un CSV con diferentes codificaciones para evitar errores."""
    try:
        return pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        st.warning("La codificacion UTF-8 fallo. Intentando con 'latin-1'...")
        uploaded_file.seek(0)
        try:
            return pd.read_csv(uploaded_file, encoding='latin-1')
        except Exception as e:
            st.error(f"No se pudo leer el archivo CSV. Error: {e}")
            return None

# --- MODULOS DE MODELADO Y OPTIMIZACION ---
def classify_matches(df):
    prob_ancla = 0.60; prob_divisor_min = 0.40
    df['p_max'] = df[['p_L', 'p_E', 'p_V']].max(axis=1)
    conditions = [df['p_max'] >= prob_ancla, (df['p_max'] >= prob_divisor_min) & (df['p_max'] < prob_ancla)]
    df['classification'] = np.select(conditions, ['Ancla', 'Divisor'], default='Neutro')
    return df

# --- FUNCION CORREGIDA ---
def get_most_probable_result(row):
    """Obtiene el resultado mas probable (L, E, V) de una fila de partido."""
    probs = {'L': row['p_L'], 'E': row['p_E'], 'V': row['p_V']}
    # La clave para la funcion max() debe ser el metodo .get del diccionario 'probs'.
    # Esto le dice a max() que compare las LLAVES ('L','E','V') usando sus VALORES en el diccionario.
    return max(probs, key=probs.get)

def apply_draw_propensity_rule(df):
    for i, row in df.iterrows():
        if abs(row['p_L'] - row['p_V']) < DRAW_PROPENSITY_THRESHOLD and row['p_E'] > max(row['p_L'], row['p_V']):
            df.loc[i, 'p_E'] += 0.06; df.loc[i, 'p_L'] -= 0.03; df.loc[i, 'p_V'] -= 0.03
    return df

def apply_global_regularization(df):
    num_matches = len(df)
    l_range, e_range, v_range = (np.array(r) * num_matches / NUM_MATCHES_PROGOL for r in [L_SUM_RANGE, E_SUM_RANGE, V_SUM_RANGE])
    for col, (min_val, max_val) in zip(['p_L', 'p_E', 'p_V'], [l_range, e_range, v_range]):
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
    num_matches = len(quiniela)
    l_count, e_count, v_count = quiniela.count('L'), quiniela.count('E'), quiniela.count('V')
    l_min, l_max = [round(v) for v in np.array(L_COUNT_RANGE) * num_matches / NUM_MATCHES_PROGOL]
    v_min, v_max = [round(v) for v in np.array(V_COUNT_RANGE) * num_matches / NUM_MATCHES_PROGOL]
    if not E_COUNT_RANGE[0] <= e_count <= E_COUNT_RANGE[1]: return False
    if num_matches == NUM_MATCHES_PROGOL:
        if not l_min <= l_count <= l_max: return False
        if not v_min <= v_count <= v_max: return False
    if max(l_count, e_count, v_count) > CONCENTRATION_LIMIT * num_matches: return False
    return True

def create_initial_portfolio(df, num_quinielas):
    portfolio = []; core_quiniela = df['result'].tolist()
    while not is_valid_quiniela(core_quiniela):
        idx_to_change = np.random.randint(0, len(core_quiniela)); core_quiniela[idx_to_change] = np.random.choice(['L', 'E', 'V'])
    portfolio.append(core_quiniela)
    for _ in range(num_quinielas - 1):
        new_quiniela = core_quiniela.copy(); num_flips = np.random.randint(2, 5)
        for _ in range(200):
            temp_quiniela = new_quiniela.copy()
            indices_to_change = np.random.choice(len(temp_quiniela), num_flips, replace=False)
            for idx in indices_to_change:
                options = ['L', 'E', 'V']; options.remove(temp_quiniela[idx]); temp_quiniela[idx] = np.random.choice(options)
            if is_valid_quiniela(temp_quiniela): new_quiniela = temp_quiniela; break
        portfolio.append(new_quiniela)
    return portfolio

def get_neighbor_portfolio(portfolio, df):
    new_portfolio = [q.copy() for q in portfolio]
    q_idx = np.random.randint(0, len(new_portfolio))
    non_anchor_indices = df[df['classification'] != 'Ancla'].index.tolist()
    if not non_anchor_indices: non_anchor_indices = df.index.tolist()
    m_idx = np.random.choice(non_anchor_indices)
    original_quiniela = new_portfolio[q_idx].copy()
    for _ in range(10):
        new_q = original_quiniela.copy()
        options = ['L', 'E', 'V']; options.remove(new_q[m_idx]); new_q[m_idx] = np.random.choice(options)
        if is_valid_quiniela(new_q): new_portfolio[q_idx] = new_q; return new_portfolio
    return portfolio

def run_simulated_annealing(df, num_quinielas, num_simulations, initial_temp, cooling_rate, iterations):
    probabilities_tuple = tuple(map(tuple, df[['p_L', 'p_E', 'p_V']].values))
    current_portfolio = create_initial_portfolio(df, num_quinielas)
    current_energy = calculate_portfolio_objective(current_portfolio, probabilities_tuple, num_simulations)
    best_portfolio, best_energy = current_portfolio, current_energy; temp = initial_temp
    progress_bar = st.progress(0, text="Iniciando optimizacion...")
    for i in range(iterations):
        neighbor_portfolio = get_neighbor_portfolio(current_portfolio, df)
        neighbor_energy = calculate_portfolio_objective(neighbor_portfolio, probabilities_tuple, num_simulations)
        delta_energy = neighbor_energy - current_energy
        if delta_energy > 0 or np.random.rand() < math.exp(delta_energy / temp):
            current_portfolio, current_energy = neighbor_portfolio, neighbor_energy
        if current_energy > best_energy:
            best_portfolio, best_energy = current_portfolio, current_energy
        temp *= cooling_rate
        progress_bar.progress((i + 1) / iterations, text=f"Iteracion {i+1}/{iterations} | Score: {best_energy:.4f}")
    st.success("?Optimizacion completada!")
    return best_portfolio

# --- FLUJO PRINCIPAL DE LA APP ---
st.sidebar.header("Paso 1: Cargar Datos")
uploaded_file = st.sidebar.file_uploader("Sube tu CSV con partidos y probabilidades", type=["csv"])

if uploaded_file is not None:
    df = load_csv_with_encoding_fallbacks(uploaded_file)
    
    if df is not None:
        st.header("1. Datos Cargados")
        st.dataframe(df)

        st.sidebar.header("Paso 2: Parametros de Optimizacion")
        num_quinielas = st.sidebar.slider("Numero de quinielas", 5, 30, 15)
        iterations = st.sidebar.select_slider("Iteraciones del optimizador", options=[500, 1000, 2000, 5000], value=1000)
        initial_temp = st.sidebar.slider("Temperatura inicial", 0.1, 1.0, 0.5, 0.05)
        cooling_rate = st.sidebar.select_slider("Tasa de enfriamiento", options=[0.99, 0.995, 0.999], value=0.995)
        num_simulations = st.sidebar.select_slider("Simulaciones Montecarlo", options=[1000, 2500, 5000], value=1000)

        if st.sidebar.button("?? Iniciar Optimizacion Definitiva", type="primary"):
            st.header("2. Proceso de Optimizacion")
            with st.spinner("Aplicando reglas de modelado heuristico..."):
                df_modelado = df.copy()
                df_modelado = apply_draw_propensity_rule(df_modelado)
                df_modelado = apply_global_regularization(df_modelado)
                df_modelado = classify_matches(df_modelado)
                # Aqu¨ª se llama a la funci¨®n corregida
                df_modelado['result'] = df_modelado.apply(get_most_probable_result, axis=1)
            st.success("Reglas de modelado aplicadas.")
            st.write("Clasificacion de partidos para optimizacion:")
            st.dataframe(df_modelado[['home', 'away', 'classification', 'p_max']])

            final_portfolio = run_simulated_annealing(df_modelado, num_quinielas, num_simulations, initial_temp, cooling_rate, iterations)
            
            st.header("3. Portafolio Optimo Encontrado")
            probabilities_tuple = tuple(map(tuple, df_modelado[['p_L', 'p_E', 'p_V']].values))
            final_probs = [run_montecarlo_simulation(tuple(q), probabilities_tuple, num_simulations * 2) for q in final_portfolio]
            match_names = df_modelado.apply(lambda row: f"{row['home']} vs {row['away']}", axis=1).tolist()
            quiniela_names = [f"Quiniela {i+1}" for i in range(num_quinielas)]
            portfolio_dict = {name: data for name, data in zip(quiniela_names, final_portfolio)}
            portfolio_df = pd.DataFrame(portfolio_dict, index=match_names)
            prob_series = pd.Series({name: f"{prob:.2%}" for name, prob in zip(quiniela_names, final_probs)}, name="Pr[>=11]")
            portfolio_df.loc["**Pr[>=11]**"] = prob_series
            
            st.dataframe(portfolio_df)
            csv_output = portfolio_df.to_csv().encode('utf-8')
            st.download_button("?? Descargar Portafolio Optimo", csv_output, "portafolio_optimizado_pro.csv", "text/csv")
else:
    st.info("Bienvenido a Progol Optimizer Pro. Por favor, sube un archivo CSV para comenzar.")