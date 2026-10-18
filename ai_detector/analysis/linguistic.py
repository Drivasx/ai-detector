"""Rasgos lingüísticos basados en reglas: varianza de longitud de oraciones, TTR, conectores típicos de IA y densidad informativa.

No usa modelos: todo se calcula con expresiones regulares y listas de palabras,
así que es instantáneo en CPU. Cada rasgo se traduce a una "probabilidad de IA"
con las referencias de `config.LINGUISTIC_REFERENCES` y se combinan en `score`.
"""

from __future__ import annotations

import re
import statistics
import unicodedata
from dataclasses import dataclass

from ai_detector import config
from ai_detector.sentences import split_sentences

_WORD = re.compile(r"[a-záéíóúüñ]+(?:-[a-záéíóúüñ]+)*")

# Conectores, fórmulas y muletillas que los LLM usan en español con mucha más
# frecuencia que las personas. Se buscan como frases completas, sin distinguir
# mayúsculas. La lista es deliberadamente conservadora: palabras muy comunes en
# textos humanos (como "pero" o "también") no están.
AI_CONNECTORS: tuple[str, ...] = (
    # Estructuradores del discurso
    "en primer lugar", "en segundo lugar", "en tercer lugar", "por último", "finalmente",
    "en conclusión", "en resumen", "en definitiva", "en síntesis", "en última instancia",
    "por otro lado", "por otra parte", "además", "asimismo", "de igual manera", "del mismo modo",
    "en este sentido", "en este contexto", "en otras palabras", "es decir",
    "por lo tanto", "por consiguiente", "no obstante", "sin embargo",
    # Fórmulas de énfasis típicas
    "cabe destacar", "cabe mencionar", "cabe señalar", "cabe resaltar",
    "es importante destacar", "es importante señalar", "es importante mencionar", "es importante tener en cuenta",
    "es fundamental", "es crucial", "es esencial", "vale la pena",
    # Sin variantes con verbo ("juega/desempeña un papel..."): se solaparían y
    # contarían dos veces la misma expresión.
    "un papel crucial", "un papel fundamental", "un papel clave",
    # Muletillas de contexto y vocabulario "de folleto"
    "en la actualidad", "hoy en día", "en el mundo actual", "en la era digital",
    "en constante evolución", "en constante cambio", "a lo largo de la historia",
    "en el ámbito de", "una amplia gama", "un amplio abanico",
    "fomentar", "abordar", "potenciar", "profundizar", "sumergirse", "panorama", "sinergia",
    "de manera significativa", "de manera efectiva", "de manera integral",
)

# Correlativas partidas: se detectan con una expresión propia y se informan con esta etiqueta.
_NO_SOLO_SINO = re.compile(r"(?<!\w)no solo(?!\w).{0,120}?(?<!\w)sino también(?!\w)", re.DOTALL)
_NO_SOLO_SINO_LABEL = "no solo… sino también"

_CONNECTOR_PATTERNS = tuple(
    (phrase, re.compile(r"(?<!\w)" + r"\s+".join(map(re.escape, phrase.split())) + r"(?!\w)"))
    for phrase in AI_CONNECTORS
)

# Palabras funcionales (artículos, preposiciones, pronombres, auxiliares...). Lo
# que no está aquí se considera palabra de contenido para la densidad informativa.
SPANISH_STOPWORDS = frozenset(
    """
    a al algo algunas algunos ante antes aquel aquella aquellas aquellos aquí así aun aún
    bajo bien cada como cómo con contra cual cuál cuales cuando cuándo cuanto de del desde
    donde dónde durante e el él ella ellas ellos en entre era eran eres es esa esas ese eso
    esos esta está estaba estado están estar estas este esto estos fue fueron ha había han
    hasta hay la las le les lo los más me mi mis mucho muchos muy nada ni no nos nosotros
    o os otra otras otro otros para pero poco por porque que qué quien quién quienes se
    sea ser si sí siempre sin sino sobre son su sus también tan tanto te tiene tienen todo
    todos tras tu tú tus u un una unas uno unos usted ustedes vosotros y ya yo
    """.split()
)


@dataclass(frozen=True)
class LinguisticFeatures:
    """Rasgos de un bloque y su puntuación combinada (0 = humano, 1 = IA)."""

    word_count: int
    sentence_count: int
    sentence_length_mean: float
    sentence_length_variance: float
    # None con menos de tres oraciones: con tan pocas la variación no significa nada.
    sentence_length_cv: float | None
    type_token_ratio: float
    connector_rate: float
    connector_hits: tuple[str, ...]
    lexical_density: float
    score: float


class LinguisticAnalyzer:
    """Calcula los rasgos lingüísticos de un texto."""

    def __init__(self, references: dict[str, tuple[float, float, float]] | None = None) -> None:
        self.references = references or config.LINGUISTIC_REFERENCES

    def analyze(self, text: str) -> LinguisticFeatures:
        lowered = unicodedata.normalize("NFC", text).lower()
        words = _WORD.findall(lowered)
        sentence_lengths = [len(_WORD.findall(s.lower())) for s in split_sentences(text)]
        sentence_lengths = [n for n in sentence_lengths if n > 0]

        mean = statistics.fmean(sentence_lengths) if sentence_lengths else 0.0
        variance = statistics.pvariance(sentence_lengths) if sentence_lengths else 0.0
        cv = variance**0.5 / mean if len(sentence_lengths) >= 3 and mean > 0 else None

        hits = _find_connectors(lowered)
        signals = {
            "sentence_length_cv": cv,
            "connector_rate": len(hits) / len(words) * 100 if words else None,
            "type_token_ratio": _moving_ttr(words) if words else None,
            "lexical_density": _lexical_density(words) if words else None,
        }
        return LinguisticFeatures(
            word_count=len(words),
            sentence_count=len(sentence_lengths),
            sentence_length_mean=mean,
            sentence_length_variance=variance,
            sentence_length_cv=cv,
            type_token_ratio=signals["type_token_ratio"] or 0.0,
            connector_rate=signals["connector_rate"] or 0.0,
            connector_hits=tuple(hits),
            lexical_density=signals["lexical_density"] or 0.0,
            score=self._score(signals),
        )

    def _score(self, signals: dict[str, float | None]) -> float:
        """Media ponderada de las probabilidades de IA de cada rasgo disponible."""
        weighted = 0.0
        total_weight = 0.0
        for name, value in signals.items():
            if value is None:
                continue  # Rasgo no medible en este bloque: se reparte su peso.
            human_value, ai_value, weight = self.references[name]
            weighted += weight * ramp(value, human_value, ai_value)
            total_weight += weight
        # Sin ningún rasgo medible no hay evidencia en ningún sentido: neutral.
        return weighted / total_weight if total_weight else 0.5


def ramp(value: float, human_value: float, ai_value: float) -> float:
    """Proyecta `value` a [0, 1]: 0 en el valor humano, 1 en el de IA (sirve en ambos sentidos)."""
    position = (value - human_value) / (ai_value - human_value)
    return min(1.0, max(0.0, position))


def _find_connectors(lowered: str) -> list[str]:
    hits = [phrase for phrase, pattern in _CONNECTOR_PATTERNS for _ in pattern.finditer(lowered)]
    hits.extend(_NO_SOLO_SINO_LABEL for _ in _NO_SOLO_SINO.finditer(lowered))
    return hits


def _moving_ttr(words: list[str]) -> float:
    """TTR medio en ventanas móviles (MATTR); TTR clásico si el texto es más corto que la ventana."""
    window = config.TTR_WINDOW
    if len(words) <= window:
        return len(set(words)) / len(words)
    ratios = (len(set(words[i : i + window])) / window for i in range(len(words) - window + 1))
    return statistics.fmean(ratios)


def _lexical_density(words: list[str]) -> float:
    content = sum(1 for word in words if word not in SPANISH_STOPWORDS)
    return content / len(words)
