"""Puntuación Binoculars (Hans et al., 2024): perplejidad relativa entre dos modelos causales.

La perplejidad sola confunde "texto de IA" con "texto previsible" (un texto humano
sobre un tema muy común también es previsible). Binoculars la normaliza por lo
sorprendente que un modelo encuentra las predicciones de otro (perplejidad
cruzada), que mide cuánto de previsible es el texto *en sí*:

    puntuación = log-perplejidad(ejecutor) / entropía-cruzada(observador, ejecutor)

Ambos modelos deben compartir tokenizador. El texto de un LLM da puntuaciones
bajas (sorprende menos de lo esperable); el humano, altas. Se sigue la
implementación de referencia de los autores (github.com/ahans30/Binoculars).
"""

from __future__ import annotations

import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ai_detector import config
from ai_detector.config import ModelSpec
from ai_detector.models import ModelNotAvailableError, local_model_path


def binoculars_from_logits(
    observer_logits: torch.Tensor, performer_logits: torch.Tensor, input_ids: torch.Tensor
) -> float | None:
    """Puntuación Binoculars a partir de las salidas de los dos modelos sobre los mismos tokens.

    Los logits tienen forma [1, T, V] e `input_ids` [1, T]: la posición t predice el
    token t+1. Devuelve None si no hay al menos dos tokens.
    """
    if input_ids.shape[1] < 2:
        return None
    observer_log_probs = torch.log_softmax(observer_logits[0, :-1].float(), dim=-1)
    performer_log_probs = torch.log_softmax(performer_logits[0, :-1].float(), dim=-1)
    targets = input_ids[0, 1:]
    # Log-perplejidad: sorpresa media del ejecutor ante los tokens reales.
    log_ppl = -performer_log_probs.gather(1, targets.unsqueeze(1)).mean()
    # Perplejidad cruzada: sorpresa media del ejecutor ante lo que predice el observador.
    x_ppl = -(observer_log_probs.exp() * performer_log_probs).sum(dim=-1).mean()
    return (log_ppl / x_ppl).item()


class BinocularsScorer:
    """Puntúa bloques con el par observador/ejecutor; carga ambos modelos del disco en el primer uso."""

    def __init__(
        self,
        observer: ModelSpec = config.BINOCULARS_OBSERVER_MODEL,
        performer: ModelSpec = config.BINOCULARS_PERFORMER_MODEL,
        max_tokens: int = config.BINOCULARS_MAX_TOKENS,
    ) -> None:
        self._specs = (observer, performer)
        self._max_tokens = max_tokens
        self._tokenizer = None
        self._observer = None
        self._performer = None
        self._unavailable = False

    def score(self, text: str) -> float | None:
        """Puntuación del bloque; None si no hay tokens o faltan los modelos (entonces no vota)."""
        if self._unavailable:
            return None
        try:
            tokenizer, observer, performer = self._load()
        except ModelNotAvailableError as exc:
            # Es una señal más del voto: sin sus modelos el análisis sigue con las demás.
            self._unavailable = True
            warnings.warn(f"Binoculars is disabled: {exc}", stacklevel=2)
            return None
        max_tokens = self._max_tokens - (tokenizer.bos_token_id is not None)
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens).input_ids
        if tokenizer.bos_token_id is not None:
            ids = torch.cat([torch.tensor([[tokenizer.bos_token_id]]), ids], dim=1)
        with torch.inference_mode():
            observer_logits = observer(ids).logits
            performer_logits = performer(ids).logits
        return binoculars_from_logits(observer_logits, performer_logits, ids)

    def _load(self):
        if self._performer is None:
            observer_path, performer_path = (local_model_path(spec) for spec in self._specs)
            # El tokenizador es el mismo en los dos modelos: basta con cargar uno.
            tokenizer = AutoTokenizer.from_pretrained(observer_path, local_files_only=True)
            # float32 explícito: los checkpoints vienen en bfloat16, lento en muchas CPU.
            observer, performer = (
                AutoModelForCausalLM.from_pretrained(path, local_files_only=True, dtype=torch.float32).eval()
                for path in (observer_path, performer_path)
            )
            # Se asignan juntos al final: un fallo a medias no deja un estado inconsistente.
            self._tokenizer, self._observer, self._performer = tokenizer, observer, performer
        return self._tokenizer, self._observer, self._performer
