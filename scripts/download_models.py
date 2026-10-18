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

if __name__ == "__main__":
    # Se desactiva el modo offline ANTES de importar el paquete (que lo activa por
    # defecto) y huggingface_hub (que lee la variable al importarse). Solo al
    # ejecutar el script: importarlo (por ejemplo en las pruebas) no cambia nada.
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub.errors import HfHubHTTPError  # noqa: E402

from ai_detector import config  # noqa: E402
from ai_detector.models import ModelNotAvailableError, download_model  # noqa: E402


def main() -> int:
    print(f"Model cache: {config.MODELS_DIR}")
    failures = 0
    for spec in config.ALL_MODELS:
        print(f"Downloading {spec.repo_id} ({len(spec.files)} files)...", flush=True)
        try:
            path = download_model(spec)
        except (ModelNotAvailableError, HfHubHTTPError, OSError) as exc:
            failures += 1
            print(f"  FAILED: {exc}")
            continue
        print(f"  ok -> {path}")
    if failures:
        print(f"{failures} model(s) could not be downloaded. Fix the problem and run the script again.")
        return 1
    print("All models cached. Analysis now runs offline (HF_HUB_OFFLINE=1).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
