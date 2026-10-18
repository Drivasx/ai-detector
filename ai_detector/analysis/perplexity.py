"""Calcula la perplejidad por bloque con un modelo multilingüe pequeño cacheado localmente.

Se usa un modelo causal (GPT-2 en español): cada token se predice a partir de
los anteriores, así que una sola pasada por el modelo da la probabilidad de
todo el bloque. Además de la perplejidad media se mide la "burstiness"
(desviación típica de la sorpresa por token): el texto humano alterna palabras
previsibles e inesperadas, mientras que el de un LLM es uniformemente previsible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ai_detector import config
from ai_detector.config import ModelSpec
from ai_detector.models import local_model_path


@dataclass(frozen=True)
class PerplexityResult:
    """Perplejidad de un bloque. Valores bajos y uniformes sugieren autoría de IA."""

    # None si el bloque tiene menos de dos tokens y no hay nada que predecir.
    perplexity: float | None
    burstiness: float | None
    token_count: int


def perplexity_from_logits(logits: torch.Tensor, input_ids: torch.Tensor) -> PerplexityResult:
    """Calcula perplejidad y burstiness a partir de la salida de un modelo causal.

    `logits` tiene forma [1, T, V] e `input_ids` [1, T]: la posición t predice el token t+1.
    """
    if input_ids.shape[1] < 2:
        return PerplexityResult(perplexity=None, burstiness=None, token_count=0)
    log_probs = torch.log_softmax(logits[0, :-1].float(), dim=-1)
    targets = input_ids[0, 1:]
    # Sorpresa de cada token real: -log p(token | contexto).
    surprisal = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    return PerplexityResult(
        perplexity=math.exp(surprisal.mean().item()),
        burstiness=surprisal.std(correction=0).item(),
        token_count=surprisal.numel(),
    )


class PerplexityScorer:
    """Puntúa bloques con el modelo causal configurado; lo carga del disco en el primer uso."""

    def __init__(self, spec: ModelSpec = config.PERPLEXITY_MODEL) -> None:
        self._spec = spec
        self._tokenizer = None
        self._model = None

    def score(self, text: str) -> PerplexityResult:
        tokenizer, model = self._load()
        # Se antepone el token de inicio para que también se prediga la primera palabra.
        max_tokens = model.config.n_positions - 1
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens).input_ids
        if tokenizer.bos_token_id is not None:
            ids = torch.cat([torch.tensor([[tokenizer.bos_token_id]]), ids], dim=1)
        with torch.inference_mode():
            logits = model(ids).logits
        return perplexity_from_logits(logits, ids)

    def _load(self):
        if self._model is None:
            path = local_model_path(self._spec)
            self._tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
            # float32 explícito: el checkpoint está guardado en float16, y en CPU las
            # operaciones en float16 son unas cien veces más lentas.
            self._model = AutoModelForCausalLM.from_pretrained(
                path, local_files_only=True, dtype=torch.float32
            ).eval()
        return self._tokenizer, self._model
