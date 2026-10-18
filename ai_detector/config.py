"""Configuración central: rutas de caché de modelos, nombres de modelos y pesos de votación."""

import os
from dataclasses import dataclass
from pathlib import Path

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

# --- Modelos ------------------------------------------------------------------

# Carpeta de la caché local de modelos (fuera del control de versiones). Se
# puede cambiar con la variable de entorno AI_DETECTOR_MODELS_DIR.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("AI_DETECTOR_MODELS_DIR", PROJECT_ROOT / "models"))


@dataclass(frozen=True)
class ModelSpec:
    """Modelo de Hugging Face fijado a una revisión concreta y a los archivos necesarios."""

    repo_id: str
    revision: str
    files: tuple[str, ...]


# GPT-2 small entrenado en español: modelo causal, da perplejidad real en una pasada.
PERPLEXITY_MODEL = ModelSpec(
    repo_id="DeepESP/gpt2-spanish",
    revision="1b935e39cf9893108bd2f4fb5317f48ae1c3ab5e",
    files=("config.json", "pytorch_model.bin", "vocab.json", "merges.txt", "tokenizer_config.json", "special_tokens_map.json"),
)

# Clasificador humano-vs-IA entrenado con AuTexTification 2023 (inglés y español).
# No se descargan sus archivos .py: la cabeza se reimplementa en analysis/classifier.py.
CLASSIFIER_MODEL = ModelSpec(
    repo_id="pandrei7/autextification-upb-mtl",
    revision="0da9bd7ae5df6267275ada6a749e972ac81f0795",
    files=("config.json", "model.safetensors", "sentencepiece.bpe.model", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"),
)

# Umbral de decisión que publicaron los autores del clasificador (campo "threshold"
# de su config.json en la revisión fijada). Sus probabilidades están muy sesgadas
# hacia 1, así que el evaluador las reescala alrededor de este valor.
CLASSIFIER_THRESHOLD = 0.9919

# Solo la configuración del codificador que usa el clasificador (sin pesos).
CLASSIFIER_ENCODER_CONFIG = ModelSpec(
    repo_id="FacebookAI/xlm-roberta-base",
    revision="e73636d4f797dec63c3081bb6ed5c7b0bb3f2089",
    files=("config.json",),
)

# Par de modelos de Binoculars (Hans et al., 2024): un modelo base ("observador")
# y su versión instruct ("ejecutor"). Comparten tokenizador, que es lo que exige
# el método. Qwen2.5 es multilingüe con buen español y cabe en CPU (~1 GB cada uno).
_QWEN_FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
BINOCULARS_OBSERVER_MODEL = ModelSpec(
    repo_id="Qwen/Qwen2.5-0.5B",
    revision="060db6499f32faf8b98477b0a26969ef7d8b9987",
    files=_QWEN_FILES,
)
BINOCULARS_PERFORMER_MODEL = ModelSpec(
    repo_id="Qwen/Qwen2.5-0.5B-Instruct",
    revision="7ae557604adf67be50417f59c2c2f167def9a775",
    files=_QWEN_FILES,
)

# Tokens máximos por bloque en Binoculars (el mismo límite que el método original).
BINOCULARS_MAX_TOKENS = 512

ALL_MODELS = (
    PERPLEXITY_MODEL,
    CLASSIFIER_MODEL,
    CLASSIFIER_ENCODER_CONFIG,
    BINOCULARS_OBSERVER_MODEL,
    BINOCULARS_PERFORMER_MODEL,
)

# --- Evaluación (votación ponderada) -----------------------------------------

# Peso de cada componente en el voto. La perplejidad es la señal que mejor separa
# texto humano de texto de IA; el clasificador apenas aporta (devuelve ~0,999 tanto
# para texto humano como de IA en español), así que pesa poco. Ver el diagnóstico
# en odd/tasks/detection-quality.md. scripts/calibrate.py --ai-folder los ajusta.
VOTING_WEIGHTS = {
    "perplexity": 0.30,
    "binoculars": 0.40,
    "classifier": 0.05,
    "linguistic": 0.25,
}

# Referencias de la perplejidad de DeepESP/gpt2-spanish: (valor_humano, valor_ia).
# La perplejidad se compara en escala logarítmica. Medidas reales: párrafos humanos
# con mediana ~144 (p25 ~95) y textos de IA de 17 a 69; burstiness humana ~3,6 frente
# a 2,6-3,1 en IA. scripts/calibrate.py --ai-folder las ajusta a los textos del usuario.
PERPLEXITY_REFERENCES = {
    "perplexity": (120.0, 25.0),
    "burstiness": (3.6, 2.6),
    # Parte de la puntuación de perplejidad que aporta la burstiness.
    "burstiness_weight": 0.3,
}

# Referencias de la puntuación Binoculars con Qwen2.5-0.5B: (valor_humano, valor_ia).
# Cuanto más baja, más probable que sea IA. Son las medianas medidas: párrafos
# humanos ~1,02 (p10 ~0,90) y textos de IA de 0,78 a 0,88 (AUC 0,97).
BINOCULARS_REFERENCES = (1.02, 0.81)

# Temperatura al reescalar el clasificador: >1 suaviza sus probabilidades extremas.
CLASSIFIER_TEMPERATURE = 2.0

# --- Informe ------------------------------------------------------------------

# Umbral de decisión sobre la probabilidad de IA. 0,5 es el punto neutro del voto;
# scripts/calibrate.py calcula uno propio a partir de textos humanos del usuario.
DETECTION_THRESHOLD = 0.5

# Número máximo de párrafos sospechosos que se destacan en el informe.
REPORT_MAX_HIGHLIGHTS = 5

# --- Calibración --------------------------------------------------------------

# Archivo donde scripts/calibrate.py guarda el umbral propio del usuario. Si
# existe, la interfaz lo usa en lugar de DETECTION_THRESHOLD. Es personal y no
# se versiona (está en .gitignore).
CALIBRATION_FILE = PROJECT_ROOT / "calibration.json"

# Proporción máxima de párrafos humanos que se aceptan por encima del umbral.
CALIBRATION_TARGET_FPR = 0.05
