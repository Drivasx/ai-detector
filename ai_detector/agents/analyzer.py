"""Agente analizador: ejecuta perplejidad, Binoculars, clasificador preentrenado y rasgos lingüísticos sobre cada bloque.

El agente solo mide: devuelve las señales en bruto de cada bloque y deja
la combinación en una probabilidad final al agente evaluador. Los componentes se
inyectan en el constructor, así que en las pruebas se sustituyen por versiones
falsas y no hace falta cargar ningún modelo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from ai_detector.agents.segmenter import Chunk
from ai_detector.analysis.binoculars import BinocularsScorer
from ai_detector.analysis.classifier import AIClassifier
from ai_detector.analysis.linguistic import LinguisticAnalyzer, LinguisticFeatures
from ai_detector.analysis.perplexity import PerplexityResult, PerplexityScorer


class PerplexityModel(Protocol):
    def score(self, text: str) -> PerplexityResult: ...


class BinocularsModel(Protocol):
    def score(self, text: str) -> float | None: ...


class ClassifierModel(Protocol):
    def predict(self, text: str) -> float: ...


@dataclass(frozen=True)
class ChunkAnalysis:
    """Las señales medidas sobre un bloque."""

    chunk: Chunk
    perplexity: PerplexityResult
    classifier_probability: float
    linguistic: LinguisticFeatures
    # Puntuación Binoculars; None si no se midió (sin modelos o bloque sin tokens).
    binoculars: float | None = None


class AnalyzerAgent:
    """Aplica los componentes de análisis a cada bloque."""

    def __init__(
        self,
        perplexity: PerplexityModel,
        classifier: ClassifierModel,
        linguistic: LinguisticAnalyzer,
        binoculars: BinocularsModel | None = None,
    ) -> None:
        self._perplexity = perplexity
        self._classifier = classifier
        self._linguistic = linguistic
        self._binoculars = binoculars

    @classmethod
    def with_local_models(cls) -> AnalyzerAgent:
        """Construye el agente con los modelos reales de la caché local (se cargan al primer bloque)."""
        return cls(
            perplexity=PerplexityScorer(),
            classifier=AIClassifier(),
            linguistic=LinguisticAnalyzer(),
            binoculars=BinocularsScorer(),
        )

    def analyze(
        self,
        chunks: list[Chunk],
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[ChunkAnalysis]:
        """Analiza los bloques en orden; `on_progress(hechos, total)` se llama tras cada uno."""
        results: list[ChunkAnalysis] = []
        for done, chunk in enumerate(chunks, start=1):
            results.append(
                ChunkAnalysis(
                    chunk=chunk,
                    perplexity=self._perplexity.score(chunk.text),
                    classifier_probability=self._classifier.predict(chunk.text),
                    linguistic=self._linguistic.analyze(chunk.text),
                    binoculars=self._binoculars.score(chunk.text) if self._binoculars else None,
                )
            )
            if on_progress is not None:
                on_progress(done, len(chunks))
        return results
