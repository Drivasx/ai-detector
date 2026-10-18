"""Pruebas del analizador de rasgos lingüísticos (sin modelos)."""

import pytest

from ai_detector.analysis.linguistic import LinguisticAnalyzer

HUMAN_LIKE = (
    "Ayer fui al mercado. Llovía a cántaros y el colectivo tardó casi una hora en pasar, "
    "así que llegué empapado, con las zapatillas hechas sopa y de muy mal humor. "
    "Compré tomates. El verdulero, que me conoce desde chico, me regaló dos limones "
    "porque decía que tenía cara de resfriado. ¡Qué tipo! Volví caminando."
)

AI_LIKE = (
    "En la actualidad, la tecnología desempeña un papel fundamental en la sociedad. "
    "Además, es importante destacar que la innovación permite abordar nuevos desafíos. "
    "Asimismo, cabe destacar que la educación juega un papel crucial en este proceso. "
    "Por otro lado, es fundamental fomentar el pensamiento crítico en los estudiantes. "
    "En conclusión, la tecnología no solo transforma la sociedad, sino también la educación."
)


@pytest.fixture
def analyzer():
    return LinguisticAnalyzer()


def test_sentence_length_statistics(analyzer):
    features = analyzer.analyze("Uno dos. Uno dos tres cuatro. Uno dos tres cuatro cinco seis.")

    assert features.sentence_count == 3
    assert features.sentence_length_mean == pytest.approx(4.0)
    assert features.sentence_length_variance == pytest.approx(8 / 3)
    assert features.sentence_length_cv == pytest.approx((8 / 3) ** 0.5 / 4)


def test_sentence_length_cv_is_none_with_fewer_than_three_sentences(analyzer):
    features = analyzer.analyze("Una sola oración aquí. Y otra más.")

    assert features.sentence_length_cv is None


def test_type_token_ratio_is_case_insensitive(analyzer):
    features = analyzer.analyze("Casa casa CASA perro.")

    assert features.type_token_ratio == pytest.approx(2 / 4)


def test_type_token_ratio_uses_a_moving_window_on_long_texts(analyzer):
    # 100 palabras distintas repetidas dos veces: el TTR clásico sería 0,5, pero
    # cada ventana de 50 palabras contiene 50 palabras distintas.
    vocabulary = [f"palabra{chr(97 + i // 26)}{chr(97 + i % 26)}" for i in range(100)]
    features = analyzer.analyze(" ".join(vocabulary * 2))

    assert features.type_token_ratio == pytest.approx(1.0)


def test_counts_ai_connectors_as_whole_phrases_per_hundred_words(analyzer):
    text = "Además, el plan avanza. Cabe destacar que nadie lo vio. Ademásx no cuenta."

    features = analyzer.analyze(text)

    assert sorted(features.connector_hits) == ["además", "cabe destacar"]
    assert features.connector_rate == pytest.approx(2 / features.word_count * 100)


def test_overlapping_connector_phrases_are_counted_once(analyzer):
    features = analyzer.analyze("La educación desempeña un papel fundamental en la vida.")

    assert features.connector_hits == ("un papel fundamental",)


def test_detects_split_correlative_no_solo_sino_tambien(analyzer):
    features = analyzer.analyze("El cambio no solo afecta a la ciudad, sino también al campo.")

    assert "no solo… sino también" in features.connector_hits


def test_lexical_density_is_share_of_content_words(analyzer):
    # 11 palabras. Funcionales: el, de, la, y, en, el. De contenido: perro, casa, corre, rápido, parque.
    features = analyzer.analyze("El perro de la casa corre rápido y en el parque.")

    assert features.lexical_density == pytest.approx(5 / 11)


def test_score_is_a_probability(analyzer):
    for text in (HUMAN_LIKE, AI_LIKE, "Hola."):
        assert 0.0 <= analyzer.analyze(text).score <= 1.0


def test_ai_like_text_scores_higher_than_human_like_text(analyzer):
    assert analyzer.analyze(AI_LIKE).score > analyzer.analyze(HUMAN_LIKE).score


def test_empty_text_yields_neutral_features(analyzer):
    features = analyzer.analyze("")

    assert features.word_count == 0
    assert features.connector_hits == ()
    assert features.score == pytest.approx(0.5)
