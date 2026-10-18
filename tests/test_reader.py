"""Pruebas del agente lector a través de su interfaz pública ReaderAgent.read()."""

import pytest

from ai_detector.agents.reader import (
    EmptyDocumentError,
    ReaderAgent,
    ReaderError,
    UnsupportedFormatError,
)


@pytest.fixture
def reader():
    return ReaderAgent()


# --- .txt ---------------------------------------------------------------


def test_reads_utf8_txt_with_spanish_characters(reader, tmp_path):
    path = tmp_path / "nota.txt"
    path.write_text("¿Qué opinás del año pasado? ¡Ñandú!", encoding="utf-8")

    assert reader.read(path) == "¿Qué opinás del año pasado? ¡Ñandú!"


def test_strips_utf8_byte_order_mark(reader, tmp_path):
    path = tmp_path / "bom.txt"
    path.write_bytes("﻿Canción".encode("utf-8"))

    assert reader.read(path) == "Canción"


def test_falls_back_to_cp1252_for_legacy_windows_files(reader, tmp_path):
    path = tmp_path / "legacy.txt"
    path.write_bytes("El niño comió “dulces”.".encode("cp1252"))

    assert reader.read(path) == "El niño comió “dulces”."


def test_extension_is_case_insensitive(reader, tmp_path):
    path = tmp_path / "MAYUS.TXT"
    path.write_text("Hola", encoding="utf-8")

    assert reader.read(path) == "Hola"


def test_accepts_path_as_string(reader, tmp_path):
    path = tmp_path / "str.txt"
    path.write_text("Hola", encoding="utf-8")

    assert reader.read(str(path)) == "Hola"


# --- limpieza -----------------------------------------------------------


def test_normalizes_line_endings_and_whitespace(reader, tmp_path):
    path = tmp_path / "sucio.txt"
    path.write_bytes(
        "  Primer   párrafo\tcon espacios.  \r\n\r\n\r\n\r\nSegundo\x07 párrafo.\r\n".encode("utf-8")
    )

    assert reader.read(path) == "Primer párrafo con espacios.\n\nSegundo párrafo."


def test_removes_soft_hyphens_and_rejoins_hyphenated_line_breaks(reader, tmp_path):
    path = tmp_path / "guiones.txt"
    path.write_text("La inteli­gencia artifi-\ncial avanza.\nMadrid-Barcelona.", encoding="utf-8")

    assert reader.read(path) == "La inteligencia artificial avanza.\nMadrid-Barcelona."


def test_composes_unicode_to_nfc(reader, tmp_path):
    path = tmp_path / "nfd.txt"
    path.write_text("Canción", encoding="utf-8")  # "ó" descompuesta

    assert reader.read(path) == "Canción"


# --- .docx --------------------------------------------------------------


def test_reads_docx_paragraphs_separated_by_blank_lines(reader, make_docx):
    path = make_docx(["Primer párrafo.", "", "Segundo párrafo con ñ."])

    assert reader.read(path) == "Primer párrafo.\n\nSegundo párrafo con ñ."


def test_docx_tables_are_excluded(reader, make_docx):
    path = make_docx(["Texto en prosa."], table_rows=[["celda", "otra"]])

    assert reader.read(path) == "Texto en prosa."


# --- .pdf ---------------------------------------------------------------


def test_reads_pdf_text_across_pages(reader, make_pdf):
    path = make_pdf([["Página uno con acción."], ["Página dos."]])

    text = reader.read(path)

    assert "Página uno con acción." in text
    assert "Página dos." in text
    assert text.index("Página uno") < text.index("Página dos")


def test_rejoins_hyphenated_words_split_across_pdf_lines(reader, make_pdf):
    path = make_pdf([["La inteligencia artifi-", "cial avanza."]])

    assert reader.read(path) == "La inteligencia artificial avanza."


# --- errores ------------------------------------------------------------


def test_unsupported_extension_raises(reader, tmp_path):
    path = tmp_path / "hoja.xlsx"
    path.write_bytes(b"irrelevant")

    with pytest.raises(UnsupportedFormatError, match=r"\.xlsx"):
        reader.read(path)


def test_missing_file_raises_file_not_found(reader, tmp_path):
    with pytest.raises(FileNotFoundError):
        reader.read(tmp_path / "no_existe.txt")


def test_whitespace_only_txt_raises_empty_document(reader, tmp_path):
    path = tmp_path / "vacio.txt"
    path.write_text("  \n\n\t ", encoding="utf-8")

    with pytest.raises(EmptyDocumentError):
        reader.read(path)


def test_pdf_without_text_layer_raises_empty_document(reader, make_pdf):
    path = make_pdf([[]])

    with pytest.raises(EmptyDocumentError, match="scanned"):
        reader.read(path)


def test_corrupt_pdf_raises_reader_error(reader, tmp_path):
    path = tmp_path / "roto.pdf"
    path.write_bytes(b"this is not a pdf")

    with pytest.raises(ReaderError):
        reader.read(path)


def test_reader_errors_share_a_common_base(reader):
    assert issubclass(UnsupportedFormatError, ReaderError)
    assert issubclass(EmptyDocumentError, ReaderError)


def test_capitalised_word_after_a_line_break_hyphen_is_not_joined(reader, tmp_path):
    path = tmp_path / "ciudades.txt"
    path.write_text("Ruta Madrid-\nBarcelona.", encoding="utf-8")

    assert reader.read(path) == "Ruta Madrid-\nBarcelona."
