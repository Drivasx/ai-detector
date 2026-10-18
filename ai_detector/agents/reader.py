"""Agente lector: lee archivos .txt, .pdf y .docx y devuelve texto plano limpio.

Es el primer eslabón del pipeline. No interpreta el contenido: solo extrae el
texto y lo normaliza para que el resto de agentes trabajen siempre con la misma
forma (Unicode NFC, saltos de línea "\\n", párrafos separados por una línea en
blanco y sin caracteres de control).
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Callable

from docx import Document
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = (".txt", ".pdf", ".docx")

# Codificaciones probadas en orden para archivos .txt. "utf-8-sig" elimina el BOM;
# "cp1252" cubre los archivos antiguos de Windows en español; "latin-1" nunca
# falla, así que actúa como último recurso.
_TXT_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")

# Espacios Unicode "raros" (no separables, finos, etc.) que se convierten en un espacio normal.
_UNICODE_SPACES = re.compile(r"[\t   -   　]")
# Caracteres invisibles que solo ensucian el texto: guion blando y marcas de ancho cero.
_INVISIBLE_CHARS = re.compile(r"[­​‌‍⁠﻿]")
# Palabra cortada con guion al final de una línea (típico en PDFs maquetados).
_HYPHENATED_BREAK = re.compile(r"(\w)-[ ]*\n[ ]*(\w)")
_MULTIPLE_SPACES = re.compile(r" {2,}")
_EXTRA_BLANK_LINES = re.compile(r"\n{3,}")


class ReaderError(Exception):
    """Error base del agente lector."""


class UnsupportedFormatError(ReaderError):
    """La extensión del archivo no está entre las soportadas."""


class EmptyDocumentError(ReaderError):
    """El documento no contiene texto aprovechable tras la limpieza."""


class ReaderAgent:
    """Lee un documento del disco y devuelve su texto plano limpio."""

    def __init__(self) -> None:
        # Tabla de despacho: extensión -> función de extracción en bruto.
        self._extractors: dict[str, Callable[[Path], str]] = {
            ".txt": self._extract_txt,
            ".pdf": self._extract_pdf,
            ".docx": self._extract_docx,
        }

    def read(self, path: str | Path) -> str:
        """Extrae y limpia el texto de `path`.

        Lanza FileNotFoundError si el archivo no existe, UnsupportedFormatError
        si la extensión no está soportada, EmptyDocumentError si no queda texto
        y ReaderError si el archivo está dañado o no se puede leer.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        extension = path.suffix.lower()
        extractor = self._extractors.get(extension)
        if extractor is None:
            supported = ", ".join(SUPPORTED_EXTENSIONS)
            raise UnsupportedFormatError(
                f"Unsupported file format '{extension or path.name}'. Supported formats: {supported}."
            )

        text = clean_text(extractor(path))
        if not text:
            hint = (
                " It may be a scanned document without a text layer (OCR is not supported)."
                if extension == ".pdf"
                else ""
            )
            raise EmptyDocumentError(f"No text could be extracted from '{path.name}'.{hint}")
        return text

    # --- Extractores por formato -------------------------------------------

    @staticmethod
    def _extract_txt(path: Path) -> str:
        raw = path.read_bytes()
        for encoding in _TXT_ENCODINGS:
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                continue
        # Inalcanzable: latin-1 decodifica cualquier secuencia de bytes.
        raise ReaderError(f"Could not decode '{path.name}'.")

    @staticmethod
    def _extract_pdf(path: Path) -> str:
        # pypdf lanza excepciones muy variadas ante PDFs dañados; todas se
        # traducen a ReaderError para que el llamador maneje un único tipo.
        try:
            pdf = PdfReader(path)
            if pdf.is_encrypted and not pdf.decrypt(""):
                raise ReaderError(f"'{path.name}' is password-protected.")
            # Las páginas se unen con un salto simple: un cambio de página no
            # implica un cambio de párrafo, eso lo decide el segmentador.
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
        except ReaderError:
            raise
        except Exception as exc:
            raise ReaderError(f"Could not read PDF '{path.name}': {exc}") from exc

    @staticmethod
    def _extract_docx(path: Path) -> str:
        try:
            document = Document(str(path))
        except Exception as exc:
            raise ReaderError(f"Could not read DOCX '{path.name}': {exc}") from exc
        # Solo se toman los párrafos del cuerpo; las tablas se excluyen a
        # propósito porque no son prosa y distorsionarían el análisis.
        paragraphs = (paragraph.text for paragraph in document.paragraphs)
        return "\n\n".join(text for text in paragraphs if text.strip())


def clean_text(text: str) -> str:
    """Normaliza texto extraído para que todos los formatos queden homogéneos."""
    # 1. Forma Unicode compuesta: "o" + tilde combinada pasa a ser "ó".
    text = unicodedata.normalize("NFC", text)
    # 2. Saltos de línea de Windows y Mac antiguo a "\n".
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 3. Fuera caracteres invisibles; los espacios Unicode pasan a ser espacios.
    text = _INVISIBLE_CHARS.sub("", text)
    text = _UNICODE_SPACES.sub(" ", text)
    # 4. Fuera caracteres de control (campanas, nulos...) salvo el salto de línea.
    text = "".join(ch for ch in text if ch == "\n" or unicodedata.category(ch) != "Cc")
    # 5. Reunir palabras cortadas con guion al final de línea ("artifi-\ncial").
    #    Solo si la siguiente letra es minúscula, para respetar "Madrid-\nBarcelona".
    text = _HYPHENATED_BREAK.sub(_join_hyphenated, text)
    # 6. Espacios repetidos dentro de cada línea y espacios en los extremos.
    lines = (_MULTIPLE_SPACES.sub(" ", line).strip() for line in text.split("\n"))
    text = "\n".join(lines)
    # 7. Como máximo una línea en blanco entre párrafos.
    text = _EXTRA_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def _join_hyphenated(match: re.Match[str]) -> str:
    before, after = match.group(1), match.group(2)
    if after.islower():
        return before + after
    return match.group(0)
