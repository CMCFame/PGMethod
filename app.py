import streamlit as st
import pandas as pd
import numpy as np
import random

# --- App Configuration ---
st.set_page_config(
    page_title="Generador de Quinielas Progol",
    page_icon="⚽",
    layout="wide"
)

# --- Constants and Rules (inspired by config.py and the document) ---
MIN_DRAWS_PER_TICKET = 4
MAX_DRAWS_PER_TICKET = 6
PROB_ANCLA = 0.60  # Probabilidad mínima para ser "Ancla"
PROB_DIVISOR_MIN = 0.40 # Probabilidad para ser "Divisor"

# --- Helper Functions (inspired by utils.py and new logic) ---

def classify_matches(df):
    """
    Clasifica los partidos en 'Ancla', 'Divisor' y 'Neutro' basado en sus probabilidades.
    Esta es una pieza clave de la metodología.
    """
    conditions = [
        df['p_max'] >= PROB_ANCLA,
        (df['p_max'] >= PROB_DIVISOR_MIN) & (df['p_max'] < PROB_ANCLA)
    ]
    choices = ['Ancla', 'Divisor']
    df['classification'] = np.select(conditions, choices, default='Neutro')
    return df

def get_most_probable_result(row):
    """Obtiene el resultado más probable (L, E, V) de una fila de partido."""
    probs = {'L': row['p_L'], 'E': row['p_E'], 'V': row['p_V']}
    return max(probs, key=probs.get)

def get_second_most_probable_result(row):
    """Obtiene el segundo resultado más probable (L, E, V)."""
    probs = {'L': row['p_L'], 'E': row['p_E'], 'V': row['p_V']}
    # Ordena por probabilidad y toma el segundo
    sorted_probs = sorted(probs.items(), key=lambda item: item[1], reverse=True)
    return sorted_probs[1][0]

def generate_core_quiniela(df, min_draws, max_draws):
    """
    Genera la quiniela 'Core', que es la base de nuestro portafolio.
    Se asegura de cumplir con la regla de 4-6 empates.
    """
    core_quiniela = [get_most_probable_result(row) for _, row in df.iterrows()]
    
    # Ajustar número de empates (regla de oro de la metodología)
    num_draws = core_quiniela.count('E')
    
    # Si hay muy pocos empates, convierte los no-empates menos probables en empates.
    if num_draws < min_draws:
        # Partidos candidatos a cambiar a Empate (que no son 'E' y donde 'E' no es la peor opción)
        candidates_to_flip_to_e = df[df['result'] != 'E'].sort_values(by='p_E', ascending=False)
        for index in candidates_to_flip_to_e.index:
            if core_quiniela[index] != 'E':
                core_quiniela[index] = 'E'
                if core_quiniela.count('E') >= min_draws:
                    break # Salimos cuando cumplimos la cuota
    
    # Si hay demasiados empates, convierte los empates menos probables en su mejor alternativa.
    elif num_draws > max_draws:
        # Partidos candidatos a cambiar de Empate a otra cosa
        candidates_to_flip_from_e = df[df['result'] == 'E'].sort_values(by='p_E', ascending=True)
        for index in candidates_to_flip_from_e.index:
            if core_quiniela[index] == 'E':
                # Reemplaza 'E' con la mejor opción que no sea 'E'
                p_l, p_v = df.loc[index, 'p_L'], df.loc[index, 'p_V']
                core_quiniela[index] = 'L' if p_l > p_v else 'V'
                if core_quiniela.count('E') <= max_draws:
                    break # Salimos cuando cumplimos la cuota

    return core_quiniela


def generate_satellite_quinielas(df, core_quiniela, num_satellites):
    """
    Genera quinielas 'Satélite' creando variaciones en los partidos 'Divisor'.
    Esto crea la correlación negativa y diversificación que busca la metodología.
    """
    satellites = []
    divisor_indices = df[df['classification'] == 'Divisor'].index.tolist()

    if not divisor_indices:
        st.warning("No se encontraron partidos 'Divisor'. Los satélites serán aleatorios.")
        divisor_indices = df.index.tolist()

    for i in range(num_satellites):
        satellite = core_quiniela.copy()
        
        # Elegimos 1 o 2 partidos 'Divisor' al azar para cambiarlos
        num_flips = random.randint(1, min(2, len(divisor_indices)))
        indices_to_flip = random.sample(divisor_indices, num_flips)
        
        for index in indices_to_flip:
            # Cambiamos el resultado al segundo más probable
            satellite[index] = get_second_most_probable_result(df.loc[index])
            
        satellites.append(satellite)
        
    return satellites


# --- Main App UI ---
st.title("⚽ Generador de Portafolios para Progol")
st.markdown("""
Esta herramienta te ayuda a generar un portafolio de quinielas de Progol siguiendo la metodología **Core + Satélites**.
1.  **Prepara tu CSV**: Asegúrate de que tu archivo `quiniela.csv` tenga las columnas `home, away, p_L, p_E, p_V`.
2.  **Configura los parámetros**: Usa la barra lateral para definir cuántas quinielas quieres.
3.  **Genera y descarga**: Haz clic en el botón para crear tu portafolio y descárgalo.
""")

# --- Sidebar Controls ---
st.sidebar.header("Parámetros del Portafolio")
num_quinielas = st.sidebar.slider("Número total de quinielas a generar", 5, 30, 10)
min_draws, max_draws = st.sidebar.slider(
    "Rango de Empates por quiniela", 
    0, 14, 
    (MIN_DRAWS_PER_TICKET, MAX_DRAWS_PER_TICKET)
)

# --- File Uploader ---
st.sidebar.header("Datos de Entrada")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo quiniela.csv", type="csv")

if uploaded_file is None:
    st.info("Esperando a que subas tu archivo `quiniela.csv`...")
    st.stop()

# --- Main Logic ---
try:
    df = pd.read_csv(uploaded_file)
    # Data Validation
    required_cols = ['home', 'away', 'p_L', 'p_E', 'p_V']
    if not all(col in df.columns for col in required_cols):
        st.error(f"El archivo CSV debe contener las columnas: {', '.join(required_cols)}")
        st.stop()
    if not np.allclose(df[['p_L', 'p_E', 'p_V']].sum(axis=1), 1.0):
        st.warning("Algunas filas no suman 1.0. Las probabilidades serán normalizadas.")
        df[['p_L', 'p_E', 'p_V']] = df[['p_L', 'p_E', 'p_V']].div(df[['p_L', 'p_E', 'p_V']].sum(axis=1), axis=0)

    # Calculate max probability and best result
    df['p_max'] = df[['p_L', 'p_E', 'p_V']].max(axis=1)
    df['result'] = df.apply(get_most_probable_result, axis=1)
    
    # Classify matches
    df = classify_matches(df)
    
    st.header("Análisis de Partidos")
    st.dataframe(df[['home', 'away', 'p_L', 'p_E', 'p_V', 'p_max', 'result', 'classification']])

    if st.button("🚀 Generar Portafolio de Quinielas", type="primary"):
        # Generate Core Quiniela
        core_quiniela = generate_core_quiniela(df, min_draws, max_draws)
        
        # Generate Satellites
        num_satellites = num_quinielas - 1
        satellite_quinielas = generate_satellite_quinielas(df, core_quiniela, num_satellites)
        
        # Combine into final portfolio
        portfolio_list = [core_quiniela] + satellite_quinielas
        
        # --- LÍNEA CORREGIDA ---
        # Ahora el número de columnas se basa en el número de partidos en el archivo
        num_matches = len(df)
        portfolio_df = pd.DataFrame(
            portfolio_list,
            columns=[f"Partido {i+1}" for i in range(num_matches)], # <-- ESTA ES LA LÍNEA QUE CAMBIÓ
            index=[f"Quiniela Core"] + [f"Satélite {i+1}" for i in range(num_satellites)]
        )
        
        st.header("✅ Portafolio Generado")
        st.dataframe(portfolio_df)
        
        # Add download button
        csv_output = portfolio_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar Portafolio en CSV",
            data=csv_output,
            file_name="portafolio_progol.csv",
            mime="text/csv",
        )

except Exception as e:
    st.error(f"Ha ocurrido un error al procesar el archivo: {e}")