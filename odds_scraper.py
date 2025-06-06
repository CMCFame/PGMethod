# odds_scraper.py

import requests
import json
import time
import logging
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

# --- Configuración ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
UA = UserAgent()
BASE_SOFA_URL = "https://api.sofascore.com/api/v1"

# --- Funciones de SofaScore (API) ---

def _search_sofascore_event_id(home_team, away_team):
    """Busca el ID de un evento en SofaScore."""
    headers = {'User-Agent': UA.random}
    search_query = f"{home_team} {away_team}"
    url = f"{BASE_SOFA_URL}/search/multi/{search_query.replace(' ', '%20')}"
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            # Busca en las categorías de resultados
            for category in data.get('categories', []):
                if category.get('name') == 'Events':
                    for event in category.get('events', []):
                        return event.get('id')
    except requests.RequestException as e:
        logging.error(f"Error en la búsqueda de SofaScore para '{search_query}': {e}")
    return None

def _get_sofascore_odds(event_id):
    """Obtiene los momios 1X2 para un ID de evento de SofaScore."""
    headers = {'User-Agent': UA.random}
    url = f"{BASE_SOFA_URL}/event/{event_id}/odds/1/all"
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            markets = response.json().get('markets', [])
            for market in markets:
                # 'fullTimeResult' es el mercado para 1X2
                if market.get('marketName') == 'Full time result':
                    choices = market.get('choices', [])
                    odds = {}
                    for choice in choices:
                        if choice['name'] == 'Home': odds['L'] = float(choice['decimalValue'])
                        if choice['name'] == 'Draw': odds['E'] = float(choice['decimalValue'])
                        if choice['name'] == 'Away': odds['V'] = float(choice['decimalValue'])
                    if 'L' in odds and 'E' in odds and 'V' in odds:
                        return odds
    except requests.RequestException as e:
        logging.error(f"Error obteniendo momios de SofaScore para el evento {event_id}: {e}")
    return None

# --- Funciones de Caliente (HTML Scraping) ---

def _get_caliente_odds(home_team, away_team):
    """Busca en Caliente.mx y extrae los momios (más frágil)."""
    headers = {'User-Agent': UA.random}
    # La búsqueda en Caliente es compleja, este es un placeholder para la lógica
    # Una implementación real requeriría una búsqueda o URLs directas.
    logging.warning("El scraper de Caliente.mx es un placeholder y no está implementado activamente.")
    return None # Placeholder

# --- Orquestador Principal ---

def fetch_match_odds(home_team, away_team):
    """
    Función principal para obtener momios de un partido.
    Intenta con SofaScore, si falla, intenta con otros.
    """
    logging.info(f"Buscando momios para: {home_team} vs {away_team}")
    
    # 1. Intentar con SofaScore
    event_id = _search_sofascore_event_id(home_team, away_team)
    if event_id:
        odds = _get_sofascore_odds(event_id)
        if odds:
            logging.info(f"Momios encontrados en SofaScore: {odds}")
            return odds
    
    # 2. Intentar con Caliente (Fallback)
    odds = _get_caliente_odds(home_team, away_team)
    if odds:
        logging.info(f"Momios encontrados en Caliente.mx: {odds}")
        return odds
        
    logging.warning(f"No se encontraron momios para {home_team} vs {away_team}")
    return None

def decimal_to_prob(odds_dict):
    """Convierte momios decimales a probabilidades, eliminando el margen de la casa."""
    if not odds_dict or any(v <= 0 for v in odds_dict.values()):
        return {'p_L': 0.34, 'p_E': 0.33, 'p_V': 0.33} # Fallback
        
    inv_odds = {k: 1/v for k, v in odds_dict.items()}
    total_inv = sum(inv_odds.values())
    
    probs = {f"p_{k}": v / total_inv for k, v in inv_odds.items()}
    return probs