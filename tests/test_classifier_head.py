"""Pruebas de la arquitectura reimplementada del clasificador con un codificador diminuto."""

import torch
from transformers import XLMRobertaConfig

from ai_detector.analysis.classifier import AutextificationModel

TINY_ENCODER = XLMRobertaConfig(
    vocab_size=40,
    hidden_size=16,
    num_hidden_layers=1,
    num_attention_heads=2,
    intermediate_size=32,
    max_position_embeddings=40,
)


def test_outputs_one_probability_per_text():
    model = AutextificationModel(TINY_ENCODER, hidden_nodes=8).eval()
    input_ids = torch.tensor([[0, 5, 6, 2], [0, 7, 2, 1]])
    attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])

    with torch.inference_mode():
        probabilities = model(input_ids, attention_mask)

    assert probabilities.shape == (2,)
    assert torch.all((probabilities > 0) & (probabilities < 1))


def test_parameter_names_match_the_published_checkpoint_layout():
    names = set(AutextificationModel(TINY_ENCODER, hidden_nodes=8).state_dict())

    assert {"hidden.weight", "out_generated.weight", "out_language.weight"} <= names
    assert "encoder.pooler.dense.weight" in names
    assert all(name.split(".")[0] in {"encoder", "hidden", "out_generated", "out_language"} for name in names)
