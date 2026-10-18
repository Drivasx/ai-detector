"""Orquesta el flujo completo: Reader -> Segmenter -> Analyzer -> Evaluator -> Reporter.

Es el único punto de entrada que necesitan la interfaz y el script de
calibración. Los agentes se pueden inyectar (las pruebas usan modelos falsos);
por defecto el analizador se construye con los modelos locales la primera vez
que se analiza un documento, así que crear el pipeline no carga nada.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ai_detector.agents.analyzer import AnalyzerAgent
from ai_detector.agents.evaluator import ChunkEvaluation, EvaluatorAgent
from ai_detector.agents.reader import EmptyDocumentError, ReaderAgent, clean_text
from ai_detector.agents.reporter import Report, ReporterAgent
from ai_detector.agents.segmenter import SegmenterAgent

ProgressCallback = Callable[[int, int], None]


class DetectionPipeline:
    """Encadena los cinco agentes para producir el informe de un documento."""

    def __init__(
        self,
        reader: ReaderAgent | None = None,
        segmenter: SegmenterAgent | None = None,
        analyzer: AnalyzerAgent | None = None,
        evaluator: EvaluatorAgent | None = None,
        reporter: ReporterAgent | None = None,
    ) -> None:
        self._reader = reader or ReaderAgent()
        self._segmenter = segmenter or SegmenterAgent()
        self._analyzer = analyzer
        self._evaluator = evaluator or EvaluatorAgent()
        self.reporter = reporter or ReporterAgent()

    def analyze_file(self, path: str | Path, on_progress: ProgressCallback | None = None) -> Report:
        """Analiza un archivo .txt, .pdf o .docx."""
        path = Path(path)
        text = self._reader.read(path)
        return self._report(text, source_name=path.name, on_progress=on_progress)

    def analyze_text(
        self, text: str, source_name: str | None = None, on_progress: ProgressCallback | None = None
    ) -> Report:
        """Analiza texto pegado directamente, con la misma limpieza que el agente lector."""
        cleaned = clean_text(text)
        if not cleaned:
            raise EmptyDocumentError("The text is empty.")
        return self._report(cleaned, source_name=source_name, on_progress=on_progress)

    def evaluate_text(self, text: str, on_progress: ProgressCallback | None = None) -> list[ChunkEvaluation]:
        """Evaluaciones por bloque sin construir el informe (lo usa la calibración)."""
        chunks = self._segmenter.segment(text)
        return self._evaluator.evaluate(self._get_analyzer().analyze(chunks, on_progress=on_progress))

    def _report(self, text: str, source_name: str | None, on_progress: ProgressCallback | None) -> Report:
        return self.reporter.build(self.evaluate_text(text, on_progress=on_progress), source_name=source_name)

    def _get_analyzer(self) -> AnalyzerAgent:
        if self._analyzer is None:
            self._analyzer = AnalyzerAgent.with_local_models()
        return self._analyzer
