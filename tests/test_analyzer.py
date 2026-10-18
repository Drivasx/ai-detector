"""Pruebas del agente analizador con componentes falsos (sin descargar modelos)."""

from ai_detector.agents.analyzer import AnalyzerAgent, ChunkAnalysis
from ai_detector.agents.segmenter import Chunk
from ai_detector.analysis.linguistic import LinguisticAnalyzer
from ai_detector.analysis.perplexity import PerplexityResult


class FakePerplexity:
    def __init__(self):
        self.calls = []

    def score(self, text):
        self.calls.append(text)
        return PerplexityResult(perplexity=float(len(text)), burstiness=1.0, token_count=len(text.split()))


class FakeClassifier:
    def __init__(self):
        self.calls = []

    def predict(self, text):
        self.calls.append(text)
        return 0.25


def test_runs_every_component_once_per_chunk_in_order():
    perplexity, classifier = FakePerplexity(), FakeClassifier()
    agent = AnalyzerAgent(perplexity=perplexity, classifier=classifier, linguistic=LinguisticAnalyzer())
    chunks = [Chunk(index=0, text="Primer bloque."), Chunk(index=1, text="Segundo bloque más largo.")]

    results = agent.analyze(chunks)

    assert [r.chunk for r in results] == chunks
    assert perplexity.calls == classifier.calls == ["Primer bloque.", "Segundo bloque más largo."]
    assert all(isinstance(r, ChunkAnalysis) for r in results)
    assert results[1].perplexity.perplexity == float(len("Segundo bloque más largo."))
    assert results[1].classifier_probability == 0.25
    assert results[1].linguistic.word_count == 4


def test_reports_progress_after_each_chunk():
    agent = AnalyzerAgent(perplexity=FakePerplexity(), classifier=FakeClassifier(), linguistic=LinguisticAnalyzer())
    chunks = [Chunk(index=i, text=f"Bloque {i}.") for i in range(3)]
    progress = []

    agent.analyze(chunks, on_progress=lambda done, total: progress.append((done, total)))

    assert progress == [(1, 3), (2, 3), (3, 3)]


def test_no_chunks_returns_empty_list():
    agent = AnalyzerAgent(perplexity=FakePerplexity(), classifier=FakeClassifier(), linguistic=LinguisticAnalyzer())

    assert agent.analyze([]) == []
