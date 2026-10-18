"""Pruebas de la carga de modelos con dobles de prueba (sin descargar nada)."""

import json

import pytest
import torch
from transformers import XLMRobertaConfig

from ai_detector.analysis import classifier as classifier_module
from ai_detector.analysis.classifier import AIClassifier, AutextificationModel
from ai_detector.analysis.perplexity import PerplexityScorer

TINY_ENCODER = XLMRobertaConfig(
    vocab_size=40, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32, max_position_embeddings=40
)


def test_classifier_retries_full_load_after_a_failed_tokenizer_load(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps({"hidden_nodes": 8, "threshold": 0.9}), encoding="utf-8")
    state = AutextificationModel(TINY_ENCODER, hidden_nodes=8).state_dict()
    monkeypatch.setattr(classifier_module, "local_model_path", lambda spec: tmp_path)
    monkeypatch.setattr(classifier_module.XLMRobertaConfig, "from_pretrained", lambda *a, **k: TINY_ENCODER)
    monkeypatch.setattr(classifier_module, "load_file", lambda path: dict(state))

    def broken_tokenizer(*args, **kwargs):
        raise OSError("corrupt tokenizer.json")

    monkeypatch.setattr(classifier_module.XLMRobertaTokenizerFast, "from_pretrained", broken_tokenizer)
    classifier = AIClassifier()

    # Ambas llamadas deben fallar con el error real de carga, no con un estado a medias.
    for _ in range(2):
        with pytest.raises(OSError, match="corrupt tokenizer"):
            classifier.predict("Hola.")


class _FakeTokenizer:
    bos_token_id = 7

    def __call__(self, text, return_tensors, truncation, max_length):
        ids = list(range(10, 10 + len(text.split())))
        if truncation:
            ids = ids[:max_length]
        return type("Encoding", (), {"input_ids": torch.tensor([ids])})()


class _FakeCausalModel:
    def __init__(self, n_positions):
        self.config = type("Config", (), {"n_positions": n_positions})()
        self.seen = None

    def __call__(self, ids):
        self.seen = ids
        return type("Output", (), {"logits": torch.zeros(1, ids.shape[1], 50)})()


def test_long_input_is_truncated_to_fit_the_context_including_bos():
    scorer = PerplexityScorer()
    model = _FakeCausalModel(n_positions=8)
    scorer._tokenizer, scorer._model = _FakeTokenizer(), model

    result = scorer.score(" ".join(["palabra"] * 30))

    assert model.seen.shape[1] == 8
    assert model.seen[0, 0].item() == _FakeTokenizer.bos_token_id
    assert result.token_count == 7
