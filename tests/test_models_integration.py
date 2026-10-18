"""Pruebas de integración con los modelos reales de la caché local.

Se omiten si los modelos no se han descargado (`python scripts/download_models.py`).
"""

import os

import pytest
import torch

from ai_detector import config
from ai_detector.agents.analyzer import AnalyzerAgent
from ai_detector.agents.segmenter import Chunk
from ai_detector.analysis.classifier import AIClassifier
from ai_detector.analysis.perplexity import PerplexityScorer
from ai_detector.models import ModelNotAvailableError, local_model_path

FLUENT = (
    "El año pasado viajamos al norte del país para visitar a mis abuelos. "
    "Pasamos dos semanas en su casa, comiendo bien y caminando por el campo todas las tardes."
)


def _models_available() -> bool:
    try:
        for spec in config.ALL_MODELS:
            local_model_path(spec)
    except ModelNotAvailableError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _models_available(), reason="models not downloaded")


def test_analysis_runs_in_offline_mode():
    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_fluent_spanish_is_less_perplexing_than_shuffled_words():
    scorer = PerplexityScorer()
    words = FLUENT.split()
    shuffled = " ".join(words[::2] + words[1::2])

    fluent, scrambled = scorer.score(FLUENT), scorer.score(shuffled)

    assert fluent.perplexity < scrambled.perplexity
    assert fluent.token_count > len(words) // 2


def test_perplexity_model_runs_in_float32_on_cpu():
    # El checkpoint viene en float16, que en CPU es muchísimo más lento.
    scorer = PerplexityScorer()
    scorer.score("Hola.")

    assert scorer._model.dtype == torch.float32


def test_classifier_loads_published_weights_and_returns_a_probability():
    classifier = AIClassifier()

    probability = classifier.predict(FLUENT)

    assert 0.0 <= probability <= 1.0
    assert classifier.threshold == pytest.approx(0.9919)


def test_classifier_cache_contains_no_remote_code():
    # La arquitectura se reimplementa localmente: el código Python del repositorio nunca se descarga.
    assert not list(local_model_path(config.CLASSIFIER_MODEL).glob("*.py"))


def test_analyzer_with_local_models_produces_all_three_signals():
    results = AnalyzerAgent.with_local_models().analyze([Chunk(index=0, text=FLUENT)])

    (result,) = results
    assert result.perplexity.perplexity > 1.0
    assert 0.0 <= result.classifier_probability <= 1.0
    assert 0.0 <= result.linguistic.score <= 1.0
