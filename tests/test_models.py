"""Pruebas de la caché local de modelos y del modo offline."""

import os
import subprocess
import sys

import pytest

from ai_detector import config
from ai_detector.models import ModelNotAvailableError, local_model_path


def test_missing_model_raises_with_download_instructions(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)

    with pytest.raises(ModelNotAvailableError, match="scripts/download_models.py"):
        local_model_path(config.PERPLEXITY_MODEL)


def test_importing_the_package_enables_offline_mode_for_huggingface():
    env = {key: value for key, value in os.environ.items() if key not in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    code = "import ai_detector; from huggingface_hub import constants; print(constants.HF_HUB_OFFLINE)"

    result = subprocess.run(
        [sys.executable, "-c", code], env=env, cwd=config.PROJECT_ROOT, capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "True"
