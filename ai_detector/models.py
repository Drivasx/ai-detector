"""Descarga única de modelos a la caché local y activación del modo offline (HF_HUB_OFFLINE=1).

Durante el análisis los modelos se resuelven siempre desde el disco
(`local_files_only=True`): si faltan, se lanza ModelNotAvailableError con la
instrucción para descargarlos, nunca se intenta una descarga silenciosa.
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

from ai_detector import config
from ai_detector.config import ModelSpec


class ModelNotAvailableError(RuntimeError):
    """El modelo no está en la caché local."""


def local_model_path(spec: ModelSpec) -> Path:
    """Ruta local de un modelo ya descargado, sin tocar la red."""
    try:
        path = snapshot_download(
            spec.repo_id,
            revision=spec.revision,
            cache_dir=config.MODELS_DIR,
            allow_patterns=list(spec.files),
            local_files_only=True,
        )
    except LocalEntryNotFoundError as exc:
        raise ModelNotAvailableError(
            f"Model '{spec.repo_id}' is not in the local cache ({config.MODELS_DIR}). "
            "Run `python scripts/download_models.py` once while online."
        ) from exc
    path = Path(path)
    missing = [name for name in spec.files if not (path / name).exists()]
    if missing:
        raise ModelNotAvailableError(
            f"Model '{spec.repo_id}' is incomplete (missing: {', '.join(missing)}). "
            "Run `python scripts/download_models.py` again."
        )
    return path


def download_model(spec: ModelSpec) -> Path:
    """Descarga (o completa) un modelo en la caché local. Requiere conexión."""
    path = snapshot_download(
        spec.repo_id,
        revision=spec.revision,
        cache_dir=config.MODELS_DIR,
        allow_patterns=list(spec.files),
    )
    return Path(path)
