"""Pruebas del cálculo de perplejidad a partir de logits (sin modelo real)."""

import math

import pytest
import torch

from ai_detector.analysis.perplexity import perplexity_from_logits


def test_uniform_predictions_give_perplexity_equal_to_vocabulary_size():
    vocabulary = 8
    logits = torch.zeros(1, 5, vocabulary)
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])

    result = perplexity_from_logits(logits, input_ids)

    assert result.perplexity == pytest.approx(vocabulary)
    assert result.burstiness == pytest.approx(0.0, abs=1e-6)
    assert result.token_count == 4  # el primer token no se predice


def test_confident_correct_predictions_give_perplexity_near_one():
    input_ids = torch.tensor([[0, 1, 2, 3]])
    logits = torch.full((1, 4, 4), -20.0)
    for position in range(3):
        logits[0, position, input_ids[0, position + 1]] = 20.0

    result = perplexity_from_logits(logits, input_ids)

    assert result.perplexity == pytest.approx(1.0, abs=1e-3)


def test_burstiness_is_standard_deviation_of_token_surprisal():
    input_ids = torch.tensor([[0, 1, 1]])
    logits = torch.zeros(1, 3, 2)
    logits[0, 0] = torch.tensor([0.0, 0.0])  # p(1) = 0,5
    logits[0, 1] = torch.tensor([0.0, math.log(3.0)])  # p(1) = 0,75

    result = perplexity_from_logits(logits, input_ids)

    surprisals = torch.tensor([-math.log(0.5), -math.log(0.75)])
    assert result.perplexity == pytest.approx(math.exp(surprisals.mean().item()))
    assert result.burstiness == pytest.approx(surprisals.std(correction=0).item())


def test_single_token_has_no_perplexity():
    result = perplexity_from_logits(torch.zeros(1, 1, 4), torch.tensor([[2]]))

    assert result.perplexity is None
    assert result.token_count == 0
