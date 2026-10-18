"""Pruebas de la caché local de modelos y del modo offline."""

import os
import subprocess
import sys

import pytest

from ai_detector import config, models
from ai_detector.models import ModelNotAvailableError, local_model_path
from scripts import download_models


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


def _fake_snapshot(cache_dir, spec, files):
    """Crea una caché con la estructura de Hugging Face que contiene solo `files`."""
    snapshot = cache_dir / f"models--{spec.repo_id.replace('/', '--')}" / "snapshots" / spec.revision
    snapshot.mkdir(parents=True)
    for name in files:
        (snapshot / name).write_text("{}", encoding="utf-8")
    return snapshot


def test_incomplete_cached_model_raises_listing_missing_files(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    _fake_snapshot(tmp_path, config.PERPLEXITY_MODEL, ["config.json"])

    with pytest.raises(ModelNotAvailableError, match=r"incomplete.*pytorch_model\.bin"):
        local_model_path(config.PERPLEXITY_MODEL)


def test_download_fails_when_pinned_files_are_missing_upstream(tmp_path, monkeypatch):
    spec = config.PERPLEXITY_MODEL
    snapshot = _fake_snapshot(tmp_path, spec, ["config.json"])
    monkeypatch.setattr(models, "snapshot_download", lambda *args, **kwargs: str(snapshot))

    with pytest.raises(ModelNotAvailableError, match="pytorch_model.bin"):
        models.download_model(spec)


def test_download_script_reports_failures_with_nonzero_exit(monkeypatch, capsys):
    def failing_download(spec):
        raise ModelNotAvailableError(f"boom {spec.repo_id}")

    monkeypatch.setattr(download_models, "download_model", failing_download)

    assert download_models.main() == 1
    output = capsys.readouterr().out
    assert "FAILED" in output and "boom DeepESP/gpt2-spanish" in output
    assert "All models cached" not in output
