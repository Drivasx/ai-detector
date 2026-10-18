"""Descarga una sola vez todos los modelos necesarios a la caché local.

Es el único punto del proyecto que se conecta a internet, y solo envía a
Hugging Face los nombres de los modelos: ningún texto del usuario. Después de
ejecutarlo, la aplicación funciona completamente offline.

Uso:
    python scripts/download_models.py
"""

import os
import sys
from pathlib import Path

# Se desactiva el modo offline ANTES de importar el paquete (que lo activa por
# defecto) y huggingface_hub (que lee la variable al importarse).
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_detector import config  # noqa: E402
from ai_detector.models import download_model  # noqa: E402


def main() -> int:
    print(f"Model cache: {config.MODELS_DIR}")
    for spec in config.ALL_MODELS:
        print(f"Downloading {spec.repo_id} ({len(spec.files)} files)...", flush=True)
        path = download_model(spec)
        print(f"  ok -> {path}")
    print("All models cached. Analysis now runs offline (HF_HUB_OFFLINE=1).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
