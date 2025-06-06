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
    sorted_probs = sorted(probs.items(), key=lambda item: item[1], reverse=True)
    return sorted_probs[1][0]

def generate_core_quiniela(df, min_draws, max_draws):
    """
    Genera la quiniela 'Core', asegurándose de cumplir con la regla de empates.
    """
    core_quiniela = [get_most_probable_result(row) for _, row in df.iterrows()]
    num_draws = core_quiniela.count('E')
    
    # Ajustar si hay muy pocos empates
    if num_draws < min_draws:
        candidates_to_flip_to_e = df[df['result'] != 'E'].sort_values(by='p_E', ascending=False)
        for index in candidates_to_flip_to_e.index:
            if core_quiniela[index] != 'E':
                core_quiniela[index] = 'E'
                if core_quiniela.count('E') >= min_draws:
                    break
    
    # Ajustar si hay demasiados empates
    elif num_draws > max_draws:
        candidates_to_flip_from_e = df[df['result'] == 'E'].sort_values(by='p_E', ascending=True)
        for index in candidates_to_flip_from_e.index:
            if core_quiniela[index] == 'E':
                p_l, p_v = df.loc[index, 'p_L'], df.loc[index, 'p_V']
                core_quiniela[index] = 'L' if p_l >= p_v else 'V'
                if core_quiniela.count('E') <= max_draws:
                    break
    return core_quiniela

def generate_satellite_quinielas(df, core_quiniela, num_satellites):
    """
    NUEVA LÓGICA: Genera satélites únicos y diversificados.
    Identifica los partidos más inciertos y los va cambiando sistemáticamente.
    """
    satellites = []
    # Priorizamos cambiar los partidos que no son anclas, ordenados del más incierto al menos incierto
    uncertain_matches = df[df['classification'] != 'Ancla'].sort_values(by='p_max', ascending=True)
    
    if uncertain_matches.empty:
        st.warning("No hay partidos inciertos para diversificar. Los satélites podrían ser idénticos.")
        return [core_quiniela.copy() for _ in range(num_satellites)]

    uncertain_indices = uncertain_matches.index.tolist()
    
    # Generamos satélites cambiando sistemáticamente los partidos más inciertos
    for i in range(num_satellites):
        satellite = core_quiniela.copy()
        
        # Usamos el operador de módulo para rotar a través de los partidos inciertos
        index_to_flip = uncertain_indices[i % len(uncertain_indices)]
        
        # Cambiamos el resultado al segundo más probable
        satellite[index_to_flip] = get_second_most_probable_result(df.loc[index_to_flip])
        
        # Verificación para asegurar que no sea idéntico al core si hay suficientes partidos para cambiar
        if satellite == core_quiniela and len(uncertain_indices) > i:
             # Si por casualidad el primer satélite es igual al core, prueba con el siguiente partido incierto
             next_index_to_flip = uncertain_indices[(i + 1) % len(uncertain_indices)]
             satellite[next_index_to_flip] = get_second_most_probable_result(df.loc[next_index_to_flip])

        satellites.append(satellite)
        
    return satellites

# --- Main App UI ---
st.title("⚽ Generador de Portafolios para Progol")
st.markdown("Esta herramienta te ayuda a generar un portafolio de quinielas de Progol siguiendo la metodología **Core + Satélites**.")

# --- Sidebar Controls ---
st.sidebar.header("Parámetros del Portafolio")
num_quinielas = st.sidebar.slider("Número total de quinielas a generar", 5, 30, 10)
min_draws, max_draws = st.sidebar.slider(
    "Rango de Empates por quiniela (Core)", 
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
    required_cols = ['home', 'away', 'p_L', 'p_E', 'p_V']
    if not all(col in df.columns for col in required_cols):
        st.error(f"El archivo CSV debe contener las columnas: {', '.join(required_cols)}")
        st.stop()
    if not np.allclose(df[['p_L', 'p_E', 'p_V']].sum(axis=1), 1.0, atol=0.01):
        st.warning("Algunas filas no suman 1.0. Las probabilidades serán normalizadas.")
        df[['p_L', 'p_E', 'p_V']] = df[['p_L', 'p_E', 'p_V']].div(df[['p_L', 'p_E', 'p_V']].sum(axis=1), axis=0)

    df['p_max'] = df[['p_L', 'p_E', 'p_V']].max(axis=1)
    df['result'] = df.apply(get_most_probable_result, axis=1)
    df = classify_matches(df)
    
    st.header("Análisis de Partidos")
    st.dataframe(df[['home', 'away', 'p_L', 'p_E', 'p_V', 'p_max', 'result', 'classification']])

    if st.button("🚀 Generar Portafolio de Quinielas", type="primary"):
        core_quiniela = generate_core_quiniela(df, min_draws, max_draws)
        num_satellites = num_quinielas - 1
        satellite_quinielas = generate_satellite_quinielas(df, core_quiniela, num_satellites)
        portfolio_list = [core_quiniela] + satellite_quinielas
        
        # --- LÓGICA DE VISUALIZACIÓN CORREGIDA ---
        # 1. Creamos un diccionario para el DataFrame
        match_names = df.apply(lambda row: f"{row['home']} vs {row['away']}", axis=1).tolist()
        quiniela_names = [f"Quiniela Core"] + [f"Satélite {i+1}" for i in range(num_satellites)]
        
        portfolio_dict = {name: data for name, data in zip(quiniela_names, portfolio_list)}
        
        # 2. Creamos el DataFrame con los partidos como índice
        portfolio_df = pd.DataFrame(portfolio_dict, index=match_names)
        
        st.header("✅ Portafolio Generado")
        st.dataframe(portfolio_df)
        
        # 3. Preparamos el CSV para descarga (con la orientación correcta)
        csv_output = portfolio_df.to_csv().encode('utf-8')
        st.download_button(
            label="📥 Descargar Portafolio en CSV",
            data=csv_output,
            file_name="portafolio_progol.csv",
            mime="text/csv",
        )

except Exception as e:
    st.error(f"Ha ocurrido un error al procesar el archivo: {e}")
    st.exception(e) # Muestra el detalle del error para depuración