"""Constructores de resultados intermedios para las pruebas (sin modelos)."""

from ai_detector.agents.analyzer import ChunkAnalysis
from ai_detector.agents.evaluator import ChunkEvaluation, ComponentScore
from ai_detector.agents.segmenter import Chunk
from ai_detector.analysis.linguistic import LinguisticFeatures
from ai_detector.analysis.perplexity import PerplexityResult

HUMAN_PPL, AI_PPL = 60.0, 15.0
HUMAN_BURST, AI_BURST = 3.2, 2.2
THRESHOLD = 0.9


def linguistic(score: float, connector_hits: tuple[str, ...] = ()) -> LinguisticFeatures:
    return LinguisticFeatures(
        word_count=50,
        sentence_count=3,
        sentence_length_mean=16.0,
        sentence_length_variance=4.0,
        sentence_length_cv=0.125,
        type_token_ratio=0.8,
        connector_rate=1.0,
        connector_hits=connector_hits,
        lexical_density=0.5,
        score=score,
    )


def analysis(
    perplexity: float | None = HUMAN_PPL,
    burstiness: float | None = HUMAN_BURST,
    classifier: float = THRESHOLD,
    linguistic_score: float = 0.5,
    words: int = 60,
    index: int = 0,
    text: str | None = None,
    connector_hits: tuple[str, ...] = (),
) -> ChunkAnalysis:
    chunk = Chunk(index=index, text=text if text is not None else " ".join(["palabra"] * words))
    result = PerplexityResult(perplexity=perplexity, burstiness=burstiness, token_count=0 if perplexity is None else 80)
    return ChunkAnalysis(
        chunk=chunk,
        perplexity=result,
        classifier_probability=classifier,
        linguistic=linguistic(linguistic_score, connector_hits),
    )


def evaluation(
    ai_probability: float,
    index: int = 0,
    words: int = 60,
    perplexity: float | None = HUMAN_PPL,
    text: str | None = None,
    connector_hits: tuple[str, ...] = (),
    components: dict[str, float] | None = None,
) -> ChunkEvaluation:
    """Evaluación de un bloque con la probabilidad final fijada a mano."""
    components = components or {"perplexity": ai_probability, "classifier": ai_probability, "linguistic": ai_probability}
    return ChunkEvaluation(
        analysis=analysis(perplexity=perplexity, words=words, index=index, text=text, connector_hits=connector_hits),
        ai_probability=ai_probability,
        components=tuple(ComponentScore(name=n, probability=p, weight=1 / len(components)) for n, p in components.items()),
        agreement=1.0,
    )
