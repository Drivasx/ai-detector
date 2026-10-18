"""Pruebas del agente evaluador con análisis construidos a mano (sin modelos)."""

import math

import pytest

from ai_detector import config
from ai_detector.agents.analyzer import ChunkAnalysis
from ai_detector.agents.evaluator import EvaluatorAgent
from ai_detector.agents.segmenter import Chunk
from ai_detector.analysis.linguistic import LinguisticFeatures
from ai_detector.analysis.perplexity import PerplexityResult

HUMAN_PPL, AI_PPL = 60.0, 15.0
HUMAN_BURST, AI_BURST = 3.2, 2.2
THRESHOLD = 0.9


def linguistic(score: float) -> LinguisticFeatures:
    return LinguisticFeatures(
        word_count=50,
        sentence_count=3,
        sentence_length_mean=16.0,
        sentence_length_variance=4.0,
        sentence_length_cv=0.125,
        type_token_ratio=0.8,
        connector_rate=1.0,
        connector_hits=(),
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
) -> ChunkAnalysis:
    chunk = Chunk(index=index, text=" ".join(["palabra"] * words))
    result = PerplexityResult(perplexity=perplexity, burstiness=burstiness, token_count=0 if perplexity is None else 80)
    return ChunkAnalysis(chunk=chunk, perplexity=result, classifier_probability=classifier, linguistic=linguistic(linguistic_score))


@pytest.fixture
def evaluator():
    return EvaluatorAgent(
        weights={"perplexity": 0.5, "classifier": 0.2, "linguistic": 0.3},
        perplexity_references={"perplexity": (HUMAN_PPL, AI_PPL), "burstiness": (HUMAN_BURST, AI_BURST), "burstiness_weight": 0.25},
        classifier_threshold=THRESHOLD,
        classifier_temperature=1.0,
    )


def components(evaluation) -> dict[str, float]:
    return {component.name: component.probability for component in evaluation.components}


def test_uses_config_defaults():
    agent = EvaluatorAgent()

    assert agent.weights == config.VOTING_WEIGHTS
    assert agent.classifier_threshold == config.CLASSIFIER_THRESHOLD


def test_config_voting_weights_sum_to_one():
    assert sum(config.VOTING_WEIGHTS.values()) == pytest.approx(1.0)


def test_ai_probability_is_the_weighted_average_of_component_probabilities(evaluator):
    # Perplejidad en el valor de IA (1), clasificador en su umbral (0,5), lingüístico 0,8.
    (evaluation,) = evaluator.evaluate([analysis(perplexity=AI_PPL, burstiness=AI_BURST, linguistic_score=0.8)])

    assert components(evaluation) == pytest.approx({"perplexity": 1.0, "classifier": 0.5, "linguistic": 0.8})
    assert evaluation.ai_probability == pytest.approx(0.5 * 1.0 + 0.2 * 0.5 + 0.3 * 0.8)


def test_perplexity_component_combines_log_perplexity_and_burstiness(evaluator):
    geometric_middle = math.sqrt(HUMAN_PPL * AI_PPL)  # mitad del camino en escala logarítmica

    (evaluation,) = evaluator.evaluate([analysis(perplexity=geometric_middle, burstiness=AI_BURST)])

    assert components(evaluation)["perplexity"] == pytest.approx(0.75 * 0.5 + 0.25 * 1.0)


def test_lower_perplexity_means_higher_ai_probability(evaluator):
    low, high = evaluator.evaluate([analysis(perplexity=20.0, index=0), analysis(perplexity=50.0, index=1)])

    assert components(low)["perplexity"] > components(high)["perplexity"]


def test_classifier_is_rescaled_around_its_decision_threshold(evaluator):
    below, at, above = evaluator.evaluate(
        [analysis(classifier=0.5, index=0), analysis(classifier=THRESHOLD, index=1), analysis(classifier=0.99, index=2)]
    )

    assert components(at)["classifier"] == pytest.approx(0.5)
    assert components(below)["classifier"] < 0.5 < components(above)["classifier"]
    # Con temperatura 1: sigmoide(logit(p) - logit(umbral)).
    expected = 1 / (1 + math.exp(-(math.log(0.99 / 0.01) - math.log(0.9 / 0.1))))
    assert components(above)["classifier"] == pytest.approx(expected)


def test_extreme_classifier_outputs_stay_finite(evaluator):
    zero, one = evaluator.evaluate([analysis(classifier=0.0, index=0), analysis(classifier=1.0, index=1)])

    assert 0.0 <= components(zero)["classifier"] < 0.5 < components(one)["classifier"] <= 1.0


def test_missing_perplexity_is_excluded_and_weights_are_renormalized(evaluator):
    (evaluation,) = evaluator.evaluate([analysis(perplexity=None, burstiness=None, linguistic_score=1.0)])

    assert "perplexity" not in components(evaluation)
    assert evaluation.ai_probability == pytest.approx((0.2 * 0.5 + 0.3 * 1.0) / 0.5)


def test_each_component_votes_and_agreement_is_the_share_matching_the_final_call(evaluator):
    # Perplejidad vota IA (1,0), clasificador vota IA (0,5 cuenta como IA), lingüístico vota humano (0,0).
    (evaluation,) = evaluator.evaluate([analysis(perplexity=AI_PPL, burstiness=AI_BURST, linguistic_score=0.0)])

    votes = {component.name: component.votes_ai for component in evaluation.components}
    assert votes == {"perplexity": True, "classifier": True, "linguistic": False}
    assert evaluation.ai_probability >= 0.5
    assert evaluation.agreement == pytest.approx(2 / 3)


def test_short_chunks_are_pulled_towards_uncertainty(evaluator):
    full, short = evaluator.evaluate(
        [
            analysis(perplexity=AI_PPL, burstiness=AI_BURST, classifier=0.99, linguistic_score=1.0, words=60, index=0),
            analysis(perplexity=AI_PPL, burstiness=AI_BURST, classifier=0.99, linguistic_score=1.0, words=10, index=1),
        ]
    )

    reliability = 10 / config.SEGMENT_MIN_WORDS
    assert short.ai_probability == pytest.approx(0.5 + (full.ai_probability - 0.5) * reliability)


def test_keeps_chunk_order_and_original_analysis(evaluator):
    analyses = [analysis(index=i) for i in range(3)]

    evaluations = evaluator.evaluate(analyses)

    assert [e.analysis for e in evaluations] == analyses
    assert [e.chunk.index for e in evaluations] == [0, 1, 2]


def test_no_analyses_returns_empty_list(evaluator):
    assert evaluator.evaluate([]) == []


@pytest.mark.parametrize(
    "weights",
    [
        {"perplexity": -0.1, "classifier": 0.5, "linguistic": 0.6},
        {"perplexity": 0.0, "classifier": 0.0, "linguistic": 0.0},
        {"perplexity": 0.5, "classifier": 0.5},
    ],
)
def test_invalid_weights_raise_value_error(weights):
    with pytest.raises(ValueError):
        EvaluatorAgent(weights=weights)


def test_zero_weight_components_do_not_vote_and_no_evidence_is_neutral():
    agent = EvaluatorAgent(weights={"perplexity": 1.0, "classifier": 0.0, "linguistic": 0.0})

    (evaluation,) = agent.evaluate([analysis(perplexity=None, burstiness=None, linguistic_score=1.0)])

    assert evaluation.components == ()
    assert evaluation.ai_probability == pytest.approx(0.5)
