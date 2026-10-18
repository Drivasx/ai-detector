"""Configuración central: rutas de caché de modelos, nombres de modelos y pesos de votación."""

# --- Segmentación -----------------------------------------------------------

# Por debajo de este número de palabras un bloque da señales poco fiables, así que
# los párrafos cortos (títulos, frases sueltas) se fusionan con los vecinos.
SEGMENT_MIN_WORDS = 40

# Límite superior por bloque. Los modelos tipo RoBERTa admiten 512 tokens y el
# español ronda 1,3-1,5 tokens por palabra: 300 palabras dejan margen suficiente.
SEGMENT_MAX_WORDS = 300

# --- Rasgos lingüísticos ------------------------------------------------------

# Cada rasgo se convierte en una "probabilidad de IA" con una rampa lineal entre
# un valor típico humano (0) y uno típico de IA (1). Son valores heurísticos de
# partida; el script de calibración permite ajustarlos a los textos del usuario.
# Formato: rasgo -> (valor_humano, valor_ia, peso).
LINGUISTIC_REFERENCES = {
    # Coeficiente de variación de la longitud de oraciones: los LLM escriben
    # oraciones de longitud muy pareja.
    "sentence_length_cv": (0.60, 0.25, 0.30),
    # Conectores y muletillas típicas de IA por cada 100 palabras.
    "connector_rate": (0.5, 3.0, 0.40),
    # TTR con ventana móvil de 50 palabras: menor diversidad léxica, más sospecha.
    "type_token_ratio": (0.85, 0.70, 0.15),
    # Proporción de palabras de contenido: el relleno baja la densidad informativa.
    "lexical_density": (0.55, 0.45, 0.15),
}

# Tamaño de la ventana del TTR móvil (MATTR); hace el TTR independiente de la longitud.
TTR_WINDOW = 50

