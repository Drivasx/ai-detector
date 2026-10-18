"""Pruebas del agente segmentador a través de SegmenterAgent.segment()."""

import pytest

from ai_detector import config
from ai_detector.agents.segmenter import Chunk, SegmenterAgent


def words(n: int, word: str = "palabra") -> str:
    """Genera un texto de `n` palabras sin puntuación."""
    return " ".join([word] * n)


def sentence(n: int) -> str:
    """Genera una oración de `n` palabras que empieza en mayúscula y termina en punto."""
    return "Esta " + words(n - 1) + "."


def test_uses_config_defaults():
    segmenter = SegmenterAgent()

    assert (segmenter.min_words, segmenter.max_words) == (config.SEGMENT_MIN_WORDS, config.SEGMENT_MAX_WORDS)


def test_each_paragraph_becomes_a_chunk():
    segmenter = SegmenterAgent(min_words=3, max_words=50)
    text = f"{words(5, 'uno')}\n\n{words(5, 'dos')}"

    chunks = segmenter.segment(text)

    assert chunks == [Chunk(index=0, text=words(5, "uno")), Chunk(index=1, text=words(5, "dos"))]


def test_unwraps_single_line_breaks_inside_a_paragraph():
    segmenter = SegmenterAgent(min_words=3, max_words=50)

    chunks = segmenter.segment("La inteligencia\nartificial avanza\nrápido.")

    assert [chunk.text for chunk in chunks] == ["La inteligencia artificial avanza rápido."]


def test_short_paragraphs_merge_with_following_until_min_words():
    segmenter = SegmenterAgent(min_words=10, max_words=50)
    text = f"Introducción\n\n{words(4, 'a')}\n\n{words(8, 'b')}\n\n{words(12, 'c')}"

    chunks = segmenter.segment(text)

    assert [chunk.text for chunk in chunks] == [
        f"Introducción\n\n{words(4, 'a')}\n\n{words(8, 'b')}",
        words(12, "c"),
    ]


def test_trailing_short_paragraph_merges_into_previous_chunk():
    segmenter = SegmenterAgent(min_words=10, max_words=50)
    text = f"{words(12, 'a')}\n\n{words(3, 'b')}"

    chunks = segmenter.segment(text)

    assert [chunk.text for chunk in chunks] == [f"{words(12, 'a')}\n\n{words(3, 'b')}"]


def test_trailing_short_paragraph_stays_alone_when_merge_would_exceed_max():
    segmenter = SegmenterAgent(min_words=10, max_words=20)
    text = f"{words(18, 'a')}\n\n{words(5, 'b')}"

    chunks = segmenter.segment(text)

    assert [chunk.text for chunk in chunks] == [words(18, "a"), words(5, "b")]


def test_document_shorter_than_min_words_is_a_single_chunk():
    segmenter = SegmenterAgent(min_words=40, max_words=300)

    chunks = segmenter.segment("Texto breve.")

    assert chunks == [Chunk(index=0, text="Texto breve.")]


def test_long_paragraph_is_split_at_sentence_boundaries_within_max_words():
    segmenter = SegmenterAgent(min_words=5, max_words=25)
    sentences = [sentence(10) for _ in range(6)]  # 60 palabras en un solo párrafo

    chunks = segmenter.segment(" ".join(sentences))

    assert len(chunks) > 1
    assert all(chunk.word_count <= 25 for chunk in chunks)
    # Ninguna oración queda partida: cada bloque termina en punto.
    assert all(chunk.text.endswith(".") for chunk in chunks)


def test_sentence_longer_than_max_words_is_split_by_words():
    segmenter = SegmenterAgent(min_words=5, max_words=20)

    chunks = segmenter.segment(words(45))

    assert [chunk.word_count for chunk in chunks] == [20, 20, 5]


def test_no_words_are_lost_or_reordered():
    segmenter = SegmenterAgent(min_words=8, max_words=20)
    text = "\n\n".join(
        ["Título", " ".join(sentence(7) for _ in range(5)), "Corto aquí.", words(30, "largo"), "Fin."]
    )

    chunks = segmenter.segment(text)

    assert " ".join(chunk.text for chunk in chunks).split() == text.split()


def test_chunk_indices_are_sequential_from_zero():
    segmenter = SegmenterAgent(min_words=3, max_words=10)

    chunks = segmenter.segment("\n\n".join(words(5) for _ in range(4)))

    assert [chunk.index for chunk in chunks] == [0, 1, 2, 3]


def test_empty_text_returns_no_chunks():
    assert SegmenterAgent().segment(" \n\n ") == []


@pytest.mark.parametrize(("min_words", "max_words"), [(0, 10), (10, 10), (20, 10)])
def test_invalid_limits_raise_value_error(min_words, max_words):
    with pytest.raises(ValueError):
        SegmenterAgent(min_words=min_words, max_words=max_words)
