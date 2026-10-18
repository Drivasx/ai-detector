"""Pruebas del agente reportero con evaluaciones construidas a mano."""

import statistics

import pytest

from ai_detector import config
from ai_detector.agents.reporter import ReporterAgent, Verdict, render_markdown
from tests.builders import evaluation


@pytest.fixture
def reporter():
    return ReporterAgent(threshold=0.6, max_highlights=2)


def test_uses_config_defaults():
    agent = ReporterAgent()

    assert (agent.threshold, agent.max_highlights) == (config.DETECTION_THRESHOLD, config.REPORT_MAX_HIGHLIGHTS)


def test_document_probability_is_the_word_weighted_mean_of_chunks(reporter):
    report = reporter.build([evaluation(0.2, index=0, words=100), evaluation(0.8, index=1, words=50)])

    assert report.ai_probability == pytest.approx((0.2 * 100 + 0.8 * 50) / 150)
    assert (report.chunk_count, report.word_count) == (2, 150)


def test_verdict_is_likely_ai_when_document_probability_reaches_threshold(reporter):
    report = reporter.build([evaluation(0.7, index=0), evaluation(0.6, index=1)])

    assert report.verdict is Verdict.LIKELY_AI


def test_verdict_is_mixed_when_only_some_chunks_reach_threshold(reporter):
    report = reporter.build([evaluation(0.9, index=0), evaluation(0.1, index=1), evaluation(0.2, index=2)])

    assert report.verdict is Verdict.MIXED
    assert report.flagged_count == 1


def test_verdict_is_likely_human_when_no_chunk_reaches_threshold(reporter):
    report = reporter.build([evaluation(0.3, index=0), evaluation(0.59, index=1)])

    assert report.verdict is Verdict.LIKELY_HUMAN
    assert report.highlights == ()


@pytest.mark.parametrize("evaluations", [[], [evaluation(0.95, words=config.SEGMENT_MIN_WORDS - 1)]])
def test_verdict_is_insufficient_text_for_empty_or_very_short_documents(reporter, evaluations):
    report = reporter.build(evaluations)

    assert report.verdict is Verdict.INSUFFICIENT_TEXT
    assert report.confidence is None
    assert report.highlights == ()


@pytest.mark.parametrize(("probability", "confidence"), [(0.65, "low"), (0.8, "medium"), (0.95, "high"), (0.3, "high")])
def test_confidence_grows_with_distance_from_threshold(reporter, probability, confidence):
    assert reporter.build([evaluation(probability)]).confidence == confidence


def test_highlights_only_flagged_chunks_most_suspicious_first_up_to_the_limit(reporter):
    report = reporter.build(
        [evaluation(0.7, index=0), evaluation(0.95, index=1), evaluation(0.2, index=2), evaluation(0.8, index=3)]
    )

    assert [h.index for h in report.highlights] == [1, 3]
    assert [h.ai_probability for h in report.highlights] == [0.95, 0.8]


def test_highlight_explains_components_and_lists_unique_connectors(reporter):
    flagged = evaluation(
        0.9,
        components={"perplexity": 0.95, "classifier": 0.7, "linguistic": 0.9},
        connector_hits=("además", "cabe destacar", "además"),
    )

    (highlight,) = reporter.build([flagged]).highlights

    assert highlight.components == {"perplexity": 0.95, "classifier": 0.7, "linguistic": 0.9}
    assert highlight.connector_hits == ("además", "cabe destacar")
    assert highlight.text == flagged.chunk.text


def test_perplexity_variation_across_chunks_is_reported_with_three_or_more_chunks(reporter):
    values = [20.0, 22.0, 30.0]
    report = reporter.build([evaluation(0.5, index=i, perplexity=p) for i, p in enumerate(values)])

    assert report.perplexity_variation == pytest.approx(statistics.pstdev(values) / statistics.fmean(values))
    assert reporter.build([evaluation(0.5, index=i) for i in range(2)]).perplexity_variation is None


def test_keeps_source_name_and_all_evaluations(reporter):
    evaluations = [evaluation(0.5, index=i) for i in range(3)]

    report = reporter.build(evaluations, source_name="ensayo.docx")

    assert report.source_name == "ensayo.docx"
    assert report.evaluations == tuple(evaluations)


# --- render_markdown ----------------------------------------------------


def test_markdown_report_shows_verdict_probability_and_highlights(reporter):
    long_text = "Además, la tecnología avanza. " * 30
    report = reporter.build(
        [evaluation(0.9, index=0, text=long_text, connector_hits=("además",)), evaluation(0.1, index=1, words=120)],
        source_name="ensayo.docx",
    )

    markdown = render_markdown(report)

    assert "ensayo.docx" in markdown
    assert "Mixed" in markdown
    assert "50%" in markdown  # (0,9 + 0,1) / 2 con bloques del mismo tamaño
    assert "Paragraph 1" in markdown and "90%" in markdown
    assert "además" in markdown
    assert "…" in markdown and long_text not in markdown  # el fragmento se recorta
    assert "not proof" in markdown


def test_markdown_report_for_insufficient_text_has_no_highlights_section(reporter):
    markdown = render_markdown(reporter.build([]))

    assert "Not enough text" in markdown
    assert "Most suspicious" not in markdown


def test_markdown_escapes_document_text_so_it_cannot_inject_links_images_or_html(reporter):
    hostile = "Mirá esto ![x](http://example.com/track.png) y [clic](http://example.com) <img src=x> **negrita** " * 6
    report = reporter.build([evaluation(0.9, text=hostile)], source_name="[doc](http://example.com).txt")

    markdown = render_markdown(report)

    assert "Paragraph 1" in markdown  # el fragmento hostil sí se muestra, pero escapado
    assert "![x](" not in markdown
    assert "[clic](" not in markdown
    assert "<img" not in markdown
    assert "[doc](" not in markdown
    assert "**negrita**" not in markdown
