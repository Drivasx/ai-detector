"""División de texto en oraciones adaptada al español.

La usan el agente segmentador (para cortar párrafos largos sin partir oraciones)
y el analizador lingüístico (para medir la varianza de longitud de oraciones).
Se basa en reglas, sin modelos, para que funcione offline y sea rápida en CPU.
"""

from __future__ import annotations

import re

# Posible fin de oración: puntuación terminal, cierres opcionales (comillas,
# paréntesis), espacio y un carácter que puede abrir oración (mayúscula, dígito,
# signos de apertura españoles o comillas).
_BOUNDARY = re.compile(
    r"[.!?…]+"
    r"[\"”»')\]]*"
    r"(?P<space>\s+)"
    r"(?=[¿¡«\"“(\[]?[A-ZÁÉÍÓÚÜÑ0-9])"
)

# Abreviaturas frecuentes que terminan en punto sin cerrar la oración.
# "etc." se omite a propósito: casi siempre aparece al final de una oración.
_ABBREVIATIONS = frozenset(
    {
        "sr", "sra", "srta", "sres", "dr", "dra", "lic", "ing", "prof", "arq",
        "ud", "uds", "vd", "vds", "d", "dña",
        "pág", "págs", "núm", "art", "cap", "vol", "fig", "ej", "aprox",
        "av", "avda", "dpto", "depto", "tel", "ee", "uu",
    }
)


def split_sentences(text: str) -> list[str]:
    """Devuelve las oraciones de `text`, con los espacios internos normalizados."""
    sentences: list[str] = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        if _ends_with_abbreviation(text, match):
            continue
        _append(sentences, text[start : match.start("space")])
        start = match.end()
    _append(sentences, text[start:])
    return sentences


def _ends_with_abbreviation(text: str, match: re.Match[str]) -> bool:
    """Indica si el punto del límite pertenece a una abreviatura o a una inicial."""
    if text[match.start()] != ".":
        return False
    preceding = text[: match.start()].split()
    if not preceding:
        return False
    word = preceding[-1].lstrip("(«\"“¿¡").lower()
    # Una sola letra se trata como inicial ("J. R. R. Tolkien"). Contrapartida:
    # una oración que termine en una letra suelta ("vitamina C.") no se corta.
    return (len(word) == 1 and word.isalpha()) or word in _ABBREVIATIONS


def _append(sentences: list[str], fragment: str) -> None:
    normalized = " ".join(fragment.split())
    if normalized:
        sentences.append(normalized)
