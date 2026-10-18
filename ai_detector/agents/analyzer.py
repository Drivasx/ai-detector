"""Agente analizador: ejecuta perplejidad, clasificador preentrenado y rasgos lingüísticos sobre cada bloque.

El agente solo mide: devuelve las tres señales en bruto de cada bloque y deja
la combinación en una probabilidad final al agente evaluador. Los componentes se
inyectan en el constructor, así que en las pruebas se sustituyen por versiones
falsas y no hace falta cargar ningún modelo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from ai_detector.agents.segmenter import Chunk
from ai_detector.analysis.classifier import AIClassifier
from ai_detector.analysis.linguistic import LinguisticAnalyzer, LinguisticFeatures
from ai_detector.analysis.perplexity import PerplexityResult, PerplexityScorer


class PerplexityModel(Protocol):
    def score(self, text: str) -> PerplexityResult: ...


class ClassifierModel(Protocol):
    def predict(self, text: str) -> float: ...


@dataclass(frozen=True)
class ChunkAnalysis:
    """Las tres señales medidas sobre un bloque."""

    chunk: Chunk
    perplexity: PerplexityResult
    classifier_probability: float
    linguistic: LinguisticFeatures


class AnalyzerAgent:
    """Aplica los tres componentes de análisis a cada bloque."""

    def __init__(self, perplexity: PerplexityModel, classifier: ClassifierModel, linguistic: LinguisticAnalyzer) -> None:
        self._perplexity = perplexity
        self._classifier = classifier
        self._linguistic = linguistic

    @classmethod
    def with_local_models(cls) -> AnalyzerAgent:
        """Construye el agente con los modelos reales de la caché local (se cargan al primer bloque)."""
        return cls(perplexity=PerplexityScorer(), classifier=AIClassifier(), linguistic=LinguisticAnalyzer())

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
                )
            )
            if on_progress is not None:
                on_progress(done, len(chunks))
        return results
