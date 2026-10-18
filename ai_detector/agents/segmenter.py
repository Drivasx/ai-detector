"""Agente segmentador: divide el texto en párrafos o bloques lógicos para analizarlos por separado.

Estrategia, partiendo del texto limpio que entrega el agente lector:

1. Los párrafos se separan por líneas en blanco; los saltos simples dentro de un
   párrafo (líneas cortadas de un PDF) se convierten en espacios.
2. Un párrafo más largo que `max_words` se divide en oraciones y estas se
   reparten en partes de tamaño parecido, sin partir ninguna oración (salvo que
   una sola oración supere el máximo, en cuyo caso se corta por palabras).
3. Las unidades resultantes se agrupan en orden: un bloque se cierra en cuanto
   alcanza `min_words`, y nunca supera `max_words`.
4. Un resto final demasiado corto se une al bloque anterior si cabe.

Nunca se pierde ni se reordena ninguna palabra.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from ai_detector import config
from ai_detector.sentences import split_sentences

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class Chunk:
    """Bloque de texto que se analiza de forma independiente."""

    index: int
    text: str

    @property
    def word_count(self) -> int:
        return _count_words(self.text)


@dataclass(frozen=True)
class _Unit:
    """Fragmento indivisible: un párrafo entero o una parte de un párrafo largo."""

    paragraph: int
    text: str
    words: int


class SegmenterAgent:
    """Divide un texto en bloques de entre `min_words` y `max_words` palabras."""

    def __init__(self, min_words: int = config.SEGMENT_MIN_WORDS, max_words: int = config.SEGMENT_MAX_WORDS) -> None:
        if min_words < 1 or min_words >= max_words:
            raise ValueError(f"Expected 1 <= min_words < max_words, got min_words={min_words}, max_words={max_words}.")
        self.min_words = min_words
        self.max_words = max_words

    def segment(self, text: str) -> list[Chunk]:
        """Devuelve los bloques de `text` en orden; lista vacía si no hay texto."""
        units = [unit for i, paragraph in enumerate(_paragraphs(text)) for unit in self._units(i, paragraph)]
        groups = self._group(units)
        return [Chunk(index=i, text=_render(group)) for i, group in enumerate(groups)]

    # --- Paso 2: unidades ------------------------------------------------------

    def _units(self, paragraph_index: int, paragraph: str) -> list[_Unit]:
        """Devuelve el párrafo como una unidad, o repartido en varias si es largo."""
        total = _count_words(paragraph)
        if total <= self.max_words:
            return [_Unit(paragraph_index, paragraph, total)]

        pieces = [piece for sentence in split_sentences(paragraph) for piece in self._fit(sentence)]
        # Objetivo por parte: repartir el párrafo en el mínimo número de partes
        # de tamaño similar, para no dejar un resto diminuto al final.
        target = math.ceil(total / math.ceil(total / self.max_words))
        units: list[_Unit] = []
        current: list[str] = []
        current_words = 0
        for piece in pieces:
            piece_words = _count_words(piece)
            if current and (current_words >= target or current_words + piece_words > self.max_words):
                units.append(_Unit(paragraph_index, " ".join(current), current_words))
                current, current_words = [], 0
            current.append(piece)
            current_words += piece_words
        units.append(_Unit(paragraph_index, " ".join(current), current_words))
        return units

    def _fit(self, sentence: str) -> list[str]:
        """Corta por palabras una oración que por sí sola supera el máximo."""
        tokens = sentence.split()
        return [" ".join(tokens[i : i + self.max_words]) for i in range(0, len(tokens), self.max_words)]

    # --- Pasos 3 y 4: agrupación ---------------------------------------------

    def _group(self, units: list[_Unit]) -> list[list[_Unit]]:
        groups: list[list[_Unit]] = []
        buffer: list[_Unit] = []
        buffer_words = 0
        for unit in units:
            if buffer and buffer_words + unit.words > self.max_words:
                groups.append(buffer)
                buffer, buffer_words = [], 0
            buffer.append(unit)
            buffer_words += unit.words
            if buffer_words >= self.min_words:
                groups.append(buffer)
                buffer, buffer_words = [], 0

        if buffer:
            if groups and sum(u.words for u in groups[-1]) + buffer_words <= self.max_words:
                groups[-1].extend(buffer)
            else:
                groups.append(buffer)
        return groups


def _paragraphs(text: str) -> list[str]:
    """Párrafos del texto, con los saltos de línea internos convertidos en espacios."""
    paragraphs = (" ".join(block.split()) for block in _PARAGRAPH_BREAK.split(text))
    return [paragraph for paragraph in paragraphs if paragraph]


def _render(group: list[_Unit]) -> str:
    """Une las unidades de un bloque: espacio dentro del mismo párrafo, línea en blanco entre párrafos."""
    text = group[0].text
    for previous, unit in zip(group, group[1:]):
        separator = " " if unit.paragraph == previous.paragraph else "\n\n"
        text += separator + unit.text
    return text


def _count_words(text: str) -> int:
    return len(text.split())
