import streamlit as st
import pandas as pd
import numpy as np
import time

# --- App Configuration ---
st.set_page_config(
    page_title="Optimizador Progol",
    page_icon="⚽",
    layout="wide"
)

# --- Constants and Rules ---
PROB_ANCLA = 0.60
PROB_DIVISOR_MIN = 0.40

# --- Helper Functions ---

def classify_matches(df):
    conditions = [
        df['p_max'] >= PROB_ANCLA,
        (df['p_max'] >= PROB_DIVISOR_MIN) & (df['p_max'] < PROB_ANCLA)
    ]
    choices = ['Ancla', 'Divisor']
    df['classification'] = np.select(conditions, choices, default='Neutro')
    return df

def get_most_probable_result(row):
    probs = {'L': row['p_L'], 'E': row['p_E'], 'V': row['p_V']}
    return max(probs, key=probs.get)

def get_second_most_probable_result(row):
    probs = {'L': row['p_L'], 'E': row['p_E'], 'V': row['p_V']}
    sorted_probs = sorted(probs.items(), key=lambda item: item[1], reverse=True)
    return sorted_probs[1][0]

# --- CORE NEW FUNCTIONS ---

def run_montecarlo_simulation(quiniela, probabilities, num_simulations):
    """
    Calcula la probabilidad de >= 11 aciertos para una quiniela dada.
    Referencia: Documento, Parte 4, "Distribución de hits para Q".
    """
    # Convertimos la quiniela (L,E,V) a números (0,1,2) para procesar con numpy
    quiniela_indices = np.array([{'L': 0, 'E': 1, 'V': 2}[res] for res in quiniela])
    
    # Creamos una matriz de resultados aleatorios basados en las probabilidades
    # Cada fila es una simulación, cada columna es un partido
    num_matches = len(probabilities)
    choices = [0, 1, 2] # L, E, V
    
    # Generamos los resultados de todas las simulaciones de una sola vez
    random_outcomes = np.array([
        np.random.choice(choices, size=num_simulations, p=prob_row)
        for prob_row in probabilities
    ]).T # Transponemos para que las filas sean las simulaciones

    # Comparamos los resultados simulados con nuestra quiniela para contar los aciertos
    hits = np.sum(random_outcomes == quiniela_indices, axis=1)
    
    # Calculamos la probabilidad de tener 11 o más aciertos
    prob_11_or_more = np.sum(hits >= 11) / num_simulations
    
    return prob_11_or_more

def generate_candidate_quinielas(base_quiniela, df, num_candidates=50):
    """
    Genera una lista de quinielas candidatas para el optimizador.
    Se basa en cambiar 1, 2 o 3 resultados de los partidos más inciertos.
    """
    candidates = []
    uncertain_indices = df[df['classification'] != 'Ancla'].index.tolist()
    if not uncertain_indices:
        uncertain_indices = df.index.tolist()

    for _ in range(num_candidates):
        candidate = base_quiniela.copy()
        num_flips = np.random.randint(1, 4) # Cambiar de 1 a 3 partidos
        
        indices_to_flip = np.random.choice(uncertain_indices, size=num_flips, replace=False)
        
        for index in indices_to_flip:
            # Cambia al segundo más probable para diversificar
            candidate[index] = get_second_most_probable_result(df.loc[index])
        
        candidates.append(candidate)
    return candidates

def get_portfolio_diversity(quiniela, portfolio):
    """
    Calcula qué tan diferente es una quiniela del portafolio existente.
    Una mayor puntuación significa mayor diversidad.
    """
    if not portfolio:
        return 0
    
    quiniela_arr = np.array(quiniela)
    total_diff = 0
    for existing_q in portfolio:
        total_diff += np.sum(quiniela_arr != np.array(existing_q))
        
    return total_diff / len(portfolio)


def generate_optimized_portfolio(df, num_quinielas, num_simulations, diversity_weight):
    """
    Función principal de optimización.
    Construye el portafolio seleccionando iterativamente la mejor quiniela
    basado en una puntuación que combina Pr[≥11] y diversidad.
    """
    probabilities = df[['p_L', 'p_E', 'p_V']].values
    
    # 1. Empezamos con la quiniela más probable como "Core"
    base_quiniela = [get_most_probable_result(row) for _, row in df.iterrows()]
    portfolio = [base_quiniela]
    portfolio_probs = [run_montecarlo_simulation(base_quiniela, probabilities, num_simulations)]

    # Barra de progreso
    progress_bar = st.progress(0)
    status_text = st.empty()

    # 2. Iteramos para generar el resto del portafolio (los "Satélites")
    for i in range(1, num_quinielas):
        status_text.text(f"Optimizando Quiniela {i+1}/{num_quinielas}...")
        
        # Generamos un set de candidatos
        candidates = generate_candidate_quinielas(base_quiniela, df, num_candidates=100)
        
        best_candidate = None
        best_score = -1

        # Evaluamos cada candidato
        for candidate in candidates:
            prob_win = run_montecarlo_simulation(candidate, probabilities, num_simulations)
            diversity = get_portfolio_diversity(candidate, portfolio)
            
            # Puntuación combinada: el corazón del optimizador
            score = (prob_win * (1 - diversity_weight)) + (diversity * diversity_weight / 10) # Se escala la diversidad

            if score > best_score:
                best_score = score
                best_candidate = candidate
                best_prob_win = prob_win
        
        portfolio.append(best_candidate)
        portfolio_probs.append(best_prob_win)
        progress_bar.progress((i + 1) / num_quinielas)

    status_text.text("¡Optimización completada!")
    return portfolio, portfolio_probs


# --- Main App UI ---
st.title("⚽ Optimizador de Portafolios Progol")
st.markdown("Esta versión utiliza **simulaciones de Montecarlo** y un **optimizador de portafolio** para generar quinielas diversificadas que buscan maximizar la probabilidad de obtener premios.")

# --- Sidebar Controls ---
st.sidebar.header("Parámetros de Optimización")
num_quinielas = st.sidebar.slider("Número total de quinielas a generar", 5, 30, 10)
num_simulations = st.sidebar.select_slider(
    "Número de simulaciones (más es más preciso pero lento)",
    options=[1_000, 5_000, 10_000, 20_000],
    value=5_000
)
diversity_weight = st.sidebar.slider(
    "Peso de la Diversificación (0=solo probabilidad, 1=solo diversidad)",
    0.0, 1.0, 0.3, 0.05
)

# --- File Uploader ---
st.sidebar.header("Datos de Entrada")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo quiniela.csv", type="csv")

if uploaded_file is None:
    st.info("Sube tu archivo `quiniela.csv` y configura los parámetros para comenzar.")
    st.stop()

# --- Main Logic ---
try:
    df = pd.read_csv(uploaded_file)
    df['p_max'] = df[['p_L', 'p_E', 'p_V']].max(axis=1)
    df = classify_matches(df)
    
    st.header("Análisis de Partidos")
    st.dataframe(df[['home', 'away', 'p_L', 'p_E', 'p_V', 'classification']])

    if st.button("🚀 Optimizar Portafolio", type="primary"):
        with st.spinner("Ejecutando simulaciones de Montecarlo... Esto puede tardar unos minutos."):
            start_time = time.time()
            portfolio, portfolio_probs = generate_optimized_portfolio(df, num_quinielas, num_simulations, diversity_weight)
            end_time = time.time()

        st.success(f"Portafolio optimizado en {end_time - start_time:.2f} segundos.")
        
        # --- Display Results ---
        match_names = df.apply(lambda row: f"{row['home']} vs {row['away']}", axis=1).tolist()
        quiniela_names = [f"Quiniela {i+1}" for i in range(num_quinielas)]
        
        portfolio_dict = {name: data for name, data in zip(quiniela_names, portfolio)}
        portfolio_df = pd.DataFrame(portfolio_dict, index=match_names)
        
        # Añadimos la fila con la probabilidad de ganar
        prob_series = pd.Series({name: f"{prob:.2%}" for name, prob in zip(quiniela_names, portfolio_probs)}, name="Pr[≥11]")
        
        # Usamos .loc para añadir la fila de forma segura
        portfolio_df.loc["Pr[≥11]"] = prob_series
        
        st.header("✅ Portafolio Optimizado")
        st.dataframe(portfolio_df)
        
        csv_output = portfolio_df.to_csv().encode('utf-8')
        st.download_button(
            label="📥 Descargar Portafolio en CSV",
            data=csv_output,
            file_name="portafolio_optimizado_progol.csv",
            mime="text/csv",
        )

except Exception as e:
    st.error(f"Ha ocurrido un error al procesar el archivo: {e}")
    st.exception(e)