"""Configuración central: rutas de caché de modelos, nombres de modelos y pesos de votación."""

# --- Segmentación -----------------------------------------------------------

# Por debajo de este número de palabras un bloque da señales poco fiables, así que
# los párrafos cortos (títulos, frases sueltas) se fusionan con los vecinos.
SEGMENT_MIN_WORDS = 40

# Límite superior por bloque. Los modelos tipo RoBERTa admiten 512 tokens y el
# español ronda 1,3-1,5 tokens por palabra: 300 palabras dejan margen suficiente.
SEGMENT_MAX_WORDS = 300
