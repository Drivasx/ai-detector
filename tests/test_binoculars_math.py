"""Pruebas del cálculo de Binoculars a partir de logits (sin modelos reales)."""

import math

import pytest
import torch

from ai_detector.analysis.binoculars import BinocularsScorer, binoculars_from_logits


def test_score_is_log_perplexity_over_cross_perplexity():
    input_ids = torch.tensor([[0, 1, 0]])
    observer = torch.zeros(1, 3, 2)  # el observador reparte 50/50
    performer = torch.zeros(1, 3, 2)
    performer[0, 0] = torch.tensor([0.0, math.log(3.0)])  # p(1) = 0,75
    performer[0, 1] = torch.tensor([math.log(3.0), 0.0])  # p(0) = 0,75

    score = binoculars_from_logits(observer, performer, input_ids)

    # Log-perplejidad del ejecutor sobre los tokens reales (1 y luego 0).
    log_ppl = -math.log(0.75)
    # Entropía cruzada: distribución del observador frente a log-probabilidades del ejecutor.
    x_ppl = -(0.5 * math.log(0.25) + 0.5 * math.log(0.75))
    assert score == pytest.approx(log_ppl / x_ppl)


def test_text_that_surprises_more_than_expected_scores_higher():
    input_ids = torch.tensor([[0, 1, 1, 1]])
    observer = torch.zeros(1, 4, 2)
    predictable = torch.zeros(1, 4, 2)
    predictable[0, :, 1] = 3.0  # el ejecutor esperaba exactamente esos tokens
    surprising = torch.zeros(1, 4, 2)
    surprising[0, :, 0] = 3.0  # el ejecutor esperaba otros

    assert binoculars_from_logits(observer, predictable, input_ids) < binoculars_from_logits(
        observer, surprising, input_ids
    )


def test_single_token_has_no_score():
    assert binoculars_from_logits(torch.zeros(1, 1, 4), torch.zeros(1, 1, 4), torch.tensor([[2]])) is None


class _FakeTokenizer:
    bos_token_id = None

    def __call__(self, text, return_tensors, truncation, max_length):
        ids = list(range(10, 10 + len(text.split())))
        if truncation:
            ids = ids[:max_length]
        return type("Encoding", (), {"input_ids": torch.tensor([ids])})()


class _FakeCausalModel:
    def __init__(self):
        self.seen = None

    def __call__(self, ids):
        self.seen = ids
        return type("Output", (), {"logits": torch.zeros(1, ids.shape[1], 50)})()


def test_scorer_feeds_both_models_the_same_truncated_tokens():
    scorer = BinocularsScorer(max_tokens=6)
    observer, performer = _FakeCausalModel(), _FakeCausalModel()
    scorer._tokenizer, scorer._observer, scorer._performer = _FakeTokenizer(), observer, performer

    score = scorer.score(" ".join(["palabra"] * 30))

    assert observer.seen.shape[1] == 6
    assert torch.equal(observer.seen, performer.seen)
    # Logits iguales y uniformes: perplejidad y perplejidad cruzada coinciden.
    assert score == pytest.approx(1.0)
