"""Pruebas del pipeline completo con componentes de modelo falsos."""

import pytest

from ai_detector.agents.analyzer import AnalyzerAgent
from ai_detector.agents.reader import EmptyDocumentError
from ai_detector.agents.reporter import Verdict
from ai_detector.analysis.linguistic import LinguisticAnalyzer
from ai_detector.analysis.perplexity import PerplexityResult
from ai_detector.pipeline import DetectionPipeline

PARAGRAPH = (
    "Ayer fui al mercado y compré tomates, cebollas y un poco de pan. El verdulero me regaló dos limones "
    "porque decía que tenía cara de resfriado, y volví caminando bajo la lluvia sin paraguas. Después preparé "
    "una ensalada enorme con todo lo que había comprado y la comimos en el patio con mi hermana."
)


class FakePerplexity:
    def score(self, text):
        return PerplexityResult(perplexity=50.0, burstiness=3.0, token_count=len(text.split()))


class FakeClassifier:
    def predict(self, text):
        return 0.5


@pytest.fixture
def pipeline():
    analyzer = AnalyzerAgent(perplexity=FakePerplexity(), classifier=FakeClassifier(), linguistic=LinguisticAnalyzer())
    return DetectionPipeline(analyzer=analyzer)


def test_analyzes_a_file_through_every_agent(pipeline, tmp_path):
    path = tmp_path / "ensayo.txt"
    path.write_text(f"{PARAGRAPH}\n\n{PARAGRAPH}", encoding="utf-8")

    report = pipeline.analyze_file(path)

    assert report.source_name == "ensayo.txt"
    assert report.chunk_count == 2
    assert report.verdict is Verdict.LIKELY_HUMAN


def test_analyzes_pasted_text_after_cleaning_it(pipeline):
    report = pipeline.analyze_text(f"  {PARAGRAPH}\r\n\r\n\r\n{PARAGRAPH}  ", source_name="pasted text")

    assert report.source_name == "pasted text"
    assert [e.chunk.text for e in report.evaluations] == [PARAGRAPH, PARAGRAPH]


def test_blank_pasted_text_raises_empty_document(pipeline):
    with pytest.raises(EmptyDocumentError):
        pipeline.analyze_text(" \n\t ")


def test_reports_analysis_progress(pipeline):
    progress = []

    pipeline.analyze_text(f"{PARAGRAPH}\n\n{PARAGRAPH}", on_progress=lambda done, total: progress.append((done, total)))

    assert progress == [(1, 2), (2, 2)]


def test_default_pipeline_builds_the_analyzer_lazily():
    pipeline = DetectionPipeline()

    # Construirlo no carga modelos: solo se crean al analizar el primer documento.
    assert pipeline._analyzer is None
