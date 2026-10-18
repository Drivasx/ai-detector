"""Clasificador preentrenado humano-vs-IA (HuggingFace) cargado desde la caché local.

Usa los pesos de `pandrei7/autextification-upb-mtl`, entrenado con unos 32 000
textos en español (y otros tantos en inglés) de la tarea AuTexTification 2023.

Ese repositorio define su arquitectura en código Python propio, que normalmente
se cargaría con `trust_remote_code=True`, es decir, ejecutando código de un
tercero. Para evitarlo, la arquitectura (un codificador XLM-RoBERTa más una
cabeza multitarea de dos capas) se reimplementa aquí con los mismos nombres de
parámetros, y los pesos se cargan directamente del archivo safetensors.
"""

from __future__ import annotations

import json

import torch
from safetensors.torch import load_file
from torch import nn
from transformers import XLMRobertaConfig, XLMRobertaModel, XLMRobertaTokenizerFast

from ai_detector import config
from ai_detector.config import ModelSpec
from ai_detector.models import local_model_path


class AutextificationModel(nn.Module):
    """Reimplementación de `AutextificationMTLModel` (los nombres deben coincidir con el checkpoint)."""

    def __init__(self, encoder_config: XLMRobertaConfig, hidden_nodes: int) -> None:
        super().__init__()
        self.encoder = XLMRobertaModel(encoder_config)
        self.hidden = nn.Linear(encoder_config.hidden_size, hidden_nodes)
        # Dos salidas: probabilidad de texto generado y probabilidad de que esté en inglés.
        # Solo se usa la primera, pero la segunda se conserva para cargar el checkpoint completo.
        self.out_generated = nn.Linear(hidden_nodes, 1)
        self.out_language = nn.Linear(hidden_nodes, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Devuelve la probabilidad de que cada texto del lote haya sido generado por IA."""
        pooled = self.encoder(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        hidden = torch.relu(self.hidden(pooled))
        return torch.sigmoid(self.out_generated(hidden)).squeeze(-1)


class AIClassifier:
    """Predice la probabilidad de autoría de IA de un bloque; carga el modelo en el primer uso."""

    def __init__(
        self,
        spec: ModelSpec = config.CLASSIFIER_MODEL,
        encoder_spec: ModelSpec = config.CLASSIFIER_ENCODER_CONFIG,
    ) -> None:
        self._spec = spec
        self._encoder_spec = encoder_spec
        self._tokenizer = None
        self._model = None

    def predict(self, text: str) -> float:
        tokenizer, model = self._load()
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.inference_mode():
            probability = model(inputs["input_ids"], inputs["attention_mask"])
        return probability.item()

    def _load(self):
        if self._model is None:
            path = local_model_path(self._spec)
            encoder_config = XLMRobertaConfig.from_pretrained(
                local_model_path(self._encoder_spec), local_files_only=True
            )
            settings = json.loads((path / "config.json").read_text(encoding="utf-8"))
            model = AutextificationModel(encoder_config, hidden_nodes=settings["hidden_nodes"])
            model.load_state_dict(_compatible_state_dict(load_file(path / "model.safetensors")), strict=True)
            # Clase concreta en lugar de AutoTokenizer: este leería el `auto_map` del
            # config.json y ofrecería ejecutar el código remoto del repositorio.
            tokenizer = XLMRobertaTokenizerFast.from_pretrained(path, local_files_only=True)
            # Se asignan juntos al final: si algo falla, el siguiente uso reintenta
            # la carga completa en lugar de quedar con un estado a medias.
            self._tokenizer, self._model = tokenizer, model.eval()
        return self._tokenizer, self._model

def _compatible_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Quita los buffers `position_ids` que guardaban versiones antiguas de transformers."""
    return {key: value for key, value in state.items() if not key.endswith("position_ids")}
