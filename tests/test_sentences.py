"""Pruebas del divisor de oraciones en español."""

from ai_detector.sentences import split_sentences


def test_splits_on_terminal_punctuation_including_spanish_openers():
    text = "Hoy llueve. ¿Vas a salir? ¡No lo creo! Mañana veremos."

    assert split_sentences(text) == ["Hoy llueve.", "¿Vas a salir?", "¡No lo creo!", "Mañana veremos."]


def test_does_not_split_after_common_abbreviations():
    text = "El Dr. Pérez y la Sra. Gómez viven en EE. UU. desde 2010. Les va bien."

    assert split_sentences(text) == [
        "El Dr. Pérez y la Sra. Gómez viven en EE. UU. desde 2010.",
        "Les va bien.",
    ]


def test_does_not_split_after_initials():
    text = "Leí a J. R. R. Tolkien. Me gustó."

    assert split_sentences(text) == ["Leí a J. R. R. Tolkien.", "Me gustó."]


def test_splits_after_ellipsis_and_closing_quotes():
    text = "Esperó... Nadie vino. Dijo «basta.» Luego se fue… Fin."

    assert split_sentences(text) == ["Esperó...", "Nadie vino.", "Dijo «basta.»", "Luego se fue…", "Fin."]


def test_keeps_decimal_numbers_and_lowercase_continuations_together():
    text = "Mide 3.5 metros. y sigue aquí"

    assert split_sentences(text) == ["Mide 3.5 metros. y sigue aquí"]


def test_empty_text_returns_no_sentences():
    assert split_sentences("   ") == []
