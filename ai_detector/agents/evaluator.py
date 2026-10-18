"""Agente evaluador: combina las tres puntuaciones de cada bloque mediante votación ponderada.

Cada componente del análisis se traduce primero a una probabilidad de IA en [0, 1]:

- Perplejidad: rampa en escala logarítmica entre una perplejidad típica humana y
  una típica de IA, combinada con la burstiness (variación de la sorpresa por token).
- Clasificador: sus probabilidades vienen sesgadas hacia 1, así que se reescalan
  en el espacio logit alrededor de su umbral de decisión (el umbral pasa a ser 0,5).
- Rasgos lingüísticos: su puntuación ya es una probabilidad.

Después se aplica un voto ponderado "suave": la probabilidad final es la media
ponderada de las probabilidades de los componentes disponibles. Cada componente
emite además un voto (IA si su probabilidad es >= 0,5), y `agreement` indica qué
parte de los votos coincide con la decisión final.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ai_detector import config
from ai_detector.agents.analyzer import ChunkAnalysis
from ai_detector.agents.segmenter import Chunk
from ai_detector.analysis.linguistic import ramp

COMPONENTS = ("perplexity", "classifier", "linguistic")

# Evita logits infinitos cuando el clasificador devuelve exactamente 0 o 1.
_EPSILON = 1e-6


@dataclass(frozen=True)
class ComponentScore:
    """Probabilidad de IA que aporta un componente y su voto."""

    name: str
    probability: float
    weight: float

    @property
    def votes_ai(self) -> bool:
        return self.probability >= 0.5


@dataclass(frozen=True)
class ChunkEvaluation:
    """Resultado del voto para un bloque."""

    analysis: ChunkAnalysis
    ai_probability: float
    components: tuple[ComponentScore, ...]
    # Fracción de componentes cuyo voto coincide con la decisión final (>= 0,5 = IA).
    agreement: float

    @property
    def chunk(self) -> Chunk:
        return self.analysis.chunk


class EvaluatorAgent:
    """Convierte las señales de cada bloque en una probabilidad de autoría de IA."""

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        perplexity_references: dict | None = None,
        classifier_threshold: float = config.CLASSIFIER_THRESHOLD,
        classifier_temperature: float = config.CLASSIFIER_TEMPERATURE,
    ) -> None:
        self.weights = dict(weights if weights is not None else config.VOTING_WEIGHTS)
        _validate_weights(self.weights)
        self.perplexity_references = perplexity_references or config.PERPLEXITY_REFERENCES
        self.classifier_threshold = classifier_threshold
        self.classifier_temperature = classifier_temperature

    def evaluate(self, analyses: list[ChunkAnalysis]) -> list[ChunkEvaluation]:
        """Evalúa cada bloque en orden."""
        return [self._evaluate(analysis) for analysis in analyses]

    def _evaluate(self, analysis: ChunkAnalysis) -> ChunkEvaluation:
        probabilities = {
            "perplexity": self._perplexity_probability(analysis),
            "classifier": self._classifier_probability(analysis.classifier_probability),
            "linguistic": analysis.linguistic.score,
        }
        # Solo votan los componentes medibles en este bloque y con peso positivo.
        components = tuple(
            ComponentScore(name=name, probability=probability, weight=self.weights[name])
            for name, probability in probabilities.items()
            if probability is not None and self.weights[name] > 0
        )
        total_weight = sum(component.weight for component in components)
        # Sin ningún voto no hay evidencia en ningún sentido: probabilidad neutral.
        combined = (
            sum(component.weight * component.probability for component in components) / total_weight
            if total_weight
            else 0.5
        )

        # Un bloque más corto que el mínimo del segmentador aporta menos evidencia:
        # su probabilidad se acerca a 0,5 en proporción a lo que le falta.
        reliability = min(1.0, analysis.chunk.word_count / config.SEGMENT_MIN_WORDS)
        ai_probability = 0.5 + (combined - 0.5) * reliability

        final_vote = ai_probability >= 0.5
        agreement = (
            sum(component.votes_ai == final_vote for component in components) / len(components) if components else 0.0
        )
        return ChunkEvaluation(
            analysis=analysis, ai_probability=ai_probability, components=components, agreement=agreement
        )

    def _perplexity_probability(self, analysis: ChunkAnalysis) -> float | None:
        result = analysis.perplexity
        if result.perplexity is None:
            return None  # Bloque sin tokens que predecir: el componente no vota.
        human_ppl, ai_ppl = self.perplexity_references["perplexity"]
        perplexity_part = ramp(math.log(result.perplexity), math.log(human_ppl), math.log(ai_ppl))
        if result.burstiness is None:
            return perplexity_part
        human_burst, ai_burst = self.perplexity_references["burstiness"]
        burstiness_weight = self.perplexity_references["burstiness_weight"]
        burstiness_part = ramp(result.burstiness, human_burst, ai_burst)
        return (1 - burstiness_weight) * perplexity_part + burstiness_weight * burstiness_part

    def _classifier_probability(self, probability: float) -> float:
        # Desplazamiento en el espacio logit: el umbral de los autores queda en 0,5.
        shifted = (_logit(probability) - _logit(self.classifier_threshold)) / self.classifier_temperature
        return 1 / (1 + math.exp(-shifted))


def _logit(probability: float) -> float:
    probability = min(1 - _EPSILON, max(_EPSILON, probability))
    return math.log(probability / (1 - probability))


def _validate_weights(weights: dict[str, float]) -> None:
    if set(weights) != set(COMPONENTS):
        raise ValueError(f"Weights must define exactly {', '.join(COMPONENTS)}; got {', '.join(sorted(weights))}.")
    if any(weight < 0 for weight in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("Weights must be non-negative and not all zero.")
