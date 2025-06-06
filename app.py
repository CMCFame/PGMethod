import streamlit as st
import pandas as pd
import numpy as np
import math
import time

# --- App Configuration ---
st.set_page_config(page_title="Optimizador Progol Avanzado", page_icon="🔬", layout="wide")

# --- Constants from Methodology ---
# Probabilities for match classification
PROB_ANCLA = 0.60
PROB_DIVISOR_MIN = 0.40
# Draw propensity rule values 
DRAW_PROPENSITY_THRESHOLD = 0.08
# Historical distribution constraints 
L_SUM_RANGE = (5.0, 5.8)
E_SUM_RANGE = (3.5, 4.6)
V_SUM_RANGE = (4.2, 5.2)
# Portfolio constraints 
MIN_DRAWS_PER_TICKET = 4
MAX_DRAWS_PER_TICKET = 6


# --- Advanced Modeling Layer ---

def apply_draw_propensity_rule(df):
    """Aplica la regla para ajustar la probabilidad de empates en partidos cerrados. """
    for i, row in df.iterrows():
        is_close = abs(row['p_L'] - row['p_V']) < DRAW_PROPENSITY_THRESHOLD
        is_draw_favored = row['p_E'] > max(row['p_L'], row['p_V'])
        if is_close and is_draw_favored:
            df.loc[i, 'p_E'] += 0.06
            df.loc[i, 'p_L'] -= 0.03
            df.loc[i, 'p_V'] -= 0.03
    return df

def apply_global_regularization(df):
    """Ajusta las probabilidades de toda la quiniela para que se alineen con los rangos históricos. """
    # Esta es una implementación simplificada del concepto de proyección al simplex.
    for col, (min_val, max_val) in zip(['p_L', 'p_E', 'p_V'], [L_SUM_RANGE, E_SUM_RANGE, V_SUM_RANGE]):
        col_sum = df[col].sum()
        if col_sum < min_val:
            df[col] *= (min_val / col_sum)
        elif col_sum > max_val:
            df[col] *= (max_val / col_sum)
    # Re-normalizar para que cada fila sume 1
    df[['p_L', 'p_E', 'p_V']] = df[['p_L', 'p_E', 'p_V']].div(df[['p_L', 'p_E', 'p_V']].sum(axis=1), axis=0)
    return df

# --- Monte Carlo & Objective Function ---

@st.cache_data
def run_montecarlo_simulation(quiniela, probabilities, num_simulations=10000):
    """Calcula Pr[>=11] para una quiniela. """
    quiniela_indices = np.array([{'L': 0, 'E': 1, 'V': 2}[res] for res in quiniela])
    random_outcomes = np.array([np.random.choice(3, size=num_simulations, p=p_row) for p_row in probabilities]).T
    hits = np.sum(random_outcomes == quiniela_indices, axis=1)
    return np.sum(hits >= 11) / num_simulations

def calculate_portfolio_objective(portfolio, probabilities, _num_simulations):
    """Calcula el valor de la función objetivo F del portafolio. """
    if not portfolio:
        return 0
    # Usamos una caché para no recalcular repetidamente
    probs_win = [run_montecarlo_simulation(tuple(q), probabilities, _num_simulations) for q in portfolio]
    if any(p > 1 for p in probs_win): return 0 # sanity check
    
    return 1 - np.prod([(1 - p) for p in probs_win])

# --- Simulated Annealing Optimizer ---

def is_valid_quiniela(quiniela):
    """Verifica si una quiniela cumple las restricciones básicas. """
    draws = quiniela.count('E')
    return MIN_DRAWS_PER_TICKET <= draws <= MAX_DRAWS_PER_TICKET

def create_initial_portfolio(df, num_quinielas):
    """Crea un portafolio inicial válido y aleatorio."""
    portfolio = []
    base_results = df['result'].tolist()
    for _ in range(num_quinielas):
        quiniela = base_results.copy()
        while not is_valid_quiniela(quiniela):
            idx_to_change = np.random.randint(0, len(quiniela))
            quiniela[idx_to_change] = np.random.choice(['L', 'E', 'V'])
        portfolio.append(quiniela)
    return portfolio
    
def get_neighbor_portfolio(portfolio, df):
    """Crea una versión ligeramente modificada (vecina) del portafolio."""
    new_portfolio = [q.copy() for q in portfolio]
    # Elige una quiniela y un partido al azar para cambiar
    q_idx = np.random.randint(0, len(new_portfolio))
    m_idx = np.random.randint(0, len(new_portfolio[q_idx]))
    
    original_quiniela = new_portfolio[q_idx].copy()
    
    # Intenta hacer un cambio válido
    for _ in range(10): # Try 10 times to find a valid mutation
        new_quiniela = original_quiniela.copy()
        # Cambia el resultado a uno diferente
        options = ['L', 'E', 'V']
        options.remove(new_quiniela[m_idx])
        new_quiniela[m_idx] = np.random.choice(options)
        
        if is_valid_quiniela(new_quiniela):
            new_portfolio[q_idx] = new_quiniela
            return new_portfolio
            
    return portfolio # Return original if no valid neighbor found

def run_simulated_annealing(df, num_quinielas, num_simulations, initial_temp, cooling_rate, iterations):
    """
    Motor de optimización principal.
    Referencia: Conceptos de GRASP-Annealing del documento. 
    """
    probabilities = df[['p_L', 'p_E', 'p_V']].values
    
    current_portfolio = create_initial_portfolio(df, num_quinielas)
    current_energy = calculate_portfolio_objective(current_portfolio, probabilities, num_simulations)
    
    best_portfolio = current_portfolio
    best_energy = current_energy
    
    temp = initial_temp
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(iterations):
        # Genera un vecino
        neighbor_portfolio = get_neighbor_portfolio(current_portfolio, df)
        neighbor_energy = calculate_portfolio_objective(neighbor_portfolio, probabilities, num_simulations)
        
        # Decide si acepta el vecino
        delta_energy = neighbor_energy - current_energy
        if delta_energy > 0 or np.random.rand() < math.exp(delta_energy / temp):
            current_portfolio = neighbor_portfolio
            current_energy = neighbor_energy
            
        # Actualiza el mejor encontrado hasta ahora
        if current_energy > best_energy:
            best_portfolio = current_portfolio
            best_energy = current_energy
            
        # Enfría la temperatura
        temp *= cooling_rate
        
        status_text.text(f"Iteración {i+1}/{iterations} | Temp: {temp:.4f} | Mejor Score: {best_energy:.4f}")
        progress_bar.progress((i + 1) / iterations)
    
    status_text.text("¡Optimización completada!")
    return best_portfolio

# --- Main App UI ---

st.title("🔬 Optimizador Progol Avanzado (Paradigm Breaker Edition)")

# --- Sidebar Controls ---
st.sidebar.header("Parámetros de Optimización")
num_quinielas = st.sidebar.slider("Número de quinielas en el portafolio", 5, 30, 15)
st.sidebar.subheader("Recocido Simulado")
iterations = st.sidebar.select_slider("Iteraciones del optimizador", options=[500, 1000, 2000, 5000], value=1000)
initial_temp = st.sidebar.slider("Temperatura inicial", 0.1, 1.0, 0.5, 0.05)
cooling_rate = st.sidebar.select_slider("Tasa de enfriamiento", options=[0.99, 0.995, 0.999], value=0.995)
num_simulations = st.sidebar.select_slider("Simulaciones Montecarlo", options=[1000, 2500, 5000], value=2500)

# --- File Uploader ---
st.sidebar.header("Datos de Entrada")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo quiniela.csv (con p_L, p_E, p_V)", type="csv")

if uploaded_file is None:
    st.info("Sube tu archivo `quiniela.csv` para comenzar la optimización.")
    st.stop()

# --- Main Logic ---
try:
    df = pd.read_csv(uploaded_file)
    
    st.header("1. Probabilidades Iniciales")
    st.dataframe(df[['home', 'away', 'p_L', 'p_E', 'p_V']])

    # --- Pre-procesamiento y modelado ---
    st.header("2. Modelado Heurístico y Regularización")
    with st.spinner("Aplicando reglas de la metodología..."):
        # Normalizar por si acaso
        df[['p_L', 'p_E', 'p_V']] = df[['p_L', 'p_E', 'p_V']].div(df[['p_L', 'p_E', 'p_V']].sum(axis=1), axis=0)
        # Aplicar reglas
        df_modelado = apply_draw_propensity_rule(df.copy())
        df_modelado = apply_global_regularization(df_modelado)
        df_modelado['result'] = df_modelado.apply(lambda row: max({'L':row['p_L'], 'E':row['p_E'], 'V':row['p_V']}, key=lambda k: k), axis=1)

    st.write("Probabilidades ajustadas según la propensión al empate y los rangos históricos.")
    st.dataframe(df_modelado[['home', 'away', 'p_L', 'p_E', 'p_V']])

    # --- Optimización ---
    st.header("3. Optimización del Portafolio")
    if st.button("🔥 Iniciar Optimización Avanzada", type="primary"):
        with st.spinner("Ejecutando Recocido Simulado... Este proceso es intensivo y puede tardar varios minutos."):
            final_portfolio = run_simulated_annealing(df_modelado, num_quinielas, num_simulations, initial_temp, cooling_rate, iterations)
        
        st.success("Optimización finalizada.")

        # --- Display Results ---
        st.header("4. Portafolio Óptimo Encontrado")
        probabilities = df_modelado[['p_L', 'p_E', 'p_V']].values.tolist()
        # Convertir tuplas de probabilidades a listas para la caché
        probabilities_tuple = tuple(map(tuple, probabilities))

        # Recalcular probabilidades para el portafolio final
        final_probs = [run_montecarlo_simulation(tuple(q), probabilities_tuple, num_simulations * 2) for q in final_portfolio]
        
        match_names = df_modelado.apply(lambda row: f"{row['home']} vs {row['away']}", axis=1).tolist()
        quiniela_names = [f"Quiniela {i+1}" for i in range(num_quinielas)]
        
        portfolio_dict = {name: data for name, data in zip(quiniela_names, final_portfolio)}
        portfolio_df = pd.DataFrame(portfolio_dict, index=match_names)
        
        prob_series = pd.Series({name: f"{prob:.2%}" for name, prob in zip(quiniela_names, final_probs)}, name="Pr[≥11]")
        portfolio_df.loc["**Pr[≥11]**"] = prob_series
        
        st.dataframe(portfolio_df)
        
        csv_output = portfolio_df.to_csv().encode('utf-8')
        st.download_button(
            label="📥 Descargar Portafolio Óptimo",
            data=csv_output,
            file_name="portafolio_optimizado_pro.csv",
            mime="text/csv",
        )

except Exception as e:
    st.error(f"Ha ocurrido un error: {e}")
    st.exception(e)