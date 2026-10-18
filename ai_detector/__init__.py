"""Detector local de texto generado por IA (offline, CPU, español)."""

import os

# Modo offline por defecto: se fija antes de que se importe huggingface_hub o
# transformers, que leen estas variables al importarse. Así ningún análisis puede
# conectarse a internet. Solo scripts/download_models.py lo desactiva, de forma
# explícita, para la descarga inicial.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# Gradio envía telemetría de uso por defecto; se desactiva antes de importarlo.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
