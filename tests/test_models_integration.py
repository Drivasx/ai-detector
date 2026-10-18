"""Pruebas de integración con los modelos reales de la caché local.

Se omiten si los modelos no se han descargado (`python scripts/download_models.py`).
"""

import json
import os

import pytest
import torch

from ai_detector import config
from ai_detector.agents.analyzer import AnalyzerAgent
from ai_detector.agents.segmenter import Chunk
from ai_detector.analysis.binoculars import BinocularsScorer
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


def test_configured_classifier_threshold_matches_the_published_model():
    settings = json.loads((local_model_path(config.CLASSIFIER_MODEL) / "config.json").read_text(encoding="utf-8"))

    assert config.CLASSIFIER_THRESHOLD == settings["threshold"]


def test_classifier_cache_contains_no_remote_code():
    # La arquitectura se reimplementa localmente: el código Python del repositorio nunca se descarga.
    assert not list(local_model_path(config.CLASSIFIER_MODEL).glob("*.py"))


def test_binoculars_scores_fluent_spanish_lower_than_shuffled_words():
    scorer = BinocularsScorer()
    words = FLUENT.split()
    shuffled = " ".join(words[::2] + words[1::2])

    fluent, scrambled = scorer.score(FLUENT), scorer.score(shuffled)

    assert 0.0 < fluent < scrambled
    assert scorer._observer.dtype == scorer._performer.dtype == torch.float32


def test_binoculars_models_share_the_tokenizer():
    observer, performer = (
        local_model_path(spec) / "tokenizer.json"
        for spec in (config.BINOCULARS_OBSERVER_MODEL, config.BINOCULARS_PERFORMER_MODEL)
    )

    assert observer.read_bytes() == performer.read_bytes()


def test_analyzer_with_local_models_produces_all_signals():
    results = AnalyzerAgent.with_local_models().analyze([Chunk(index=0, text=FLUENT)])

    (result,) = results
    assert result.perplexity.perplexity > 1.0
    assert 0.0 <= result.classifier_probability <= 1.0
    assert 0.0 <= result.linguistic.score <= 1.0
    assert result.binoculars > 0.0
