"""Agente reportero: agrega las puntuaciones, resalta los párrafos más sospechosos y genera el informe.

A partir de la evaluación de cada bloque calcula:

- La probabilidad de IA del documento: media de los bloques ponderada por su
  número de palabras, para que un título no pese lo mismo que un párrafo largo.
- Un veredicto: IA probable, mixto (solo algunos pasajes superan el umbral),
  humano probable, o texto insuficiente para decidir.
- Una confianza (baja, media o alta) según la distancia al umbral.
- Los bloques más sospechosos, con lo que aportó cada componente y los
  conectores típicos de IA encontrados.
- La variación de la perplejidad entre bloques: una perplejidad muy uniforme
  a lo largo de todo el documento es otra pista de autoría de IA.

`render_markdown` convierte el informe en texto legible para la interfaz.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from enum import Enum

from ai_detector import config
from ai_detector.agents.evaluator import ChunkEvaluation

# Distancias al umbral a partir de las cuales la confianza pasa a media y a alta.
_CONFIDENCE_MARGINS = ((0.25, "high"), (0.10, "medium"))
_EXCERPT_CHARS = 280

# Caracteres con significado en Markdown dentro de una línea. El texto del
# documento se escapa antes de mostrarlo: así no puede inyectar enlaces, imágenes
# externas (que el navegador descargaría, rompiendo el funcionamiento offline) ni
# HTML. `_escape` además lo deja en una sola línea y neutraliza los marcadores de
# bloque al inicio (listas, reglas), porque tras "> " también se interpretan.
_MARKDOWN_SPECIAL = re.compile(r"([\\`*_{}\[\]()!|~#])")
# Marcador de bloque al inicio del texto: "-", "+", "=" o una lista numerada ("1.", "2)").
_LEADING_BLOCK_MARKER = re.compile(r"^(?:[-+=]|\d+(?=[.)]))")


class Verdict(Enum):
    LIKELY_AI = "Likely AI-generated"
    MIXED = "Mixed: some passages look AI-generated"
    LIKELY_HUMAN = "Likely human-written"
    INSUFFICIENT_TEXT = "Not enough text to decide"


@dataclass(frozen=True)
class Highlight:
    """Bloque destacado por superar el umbral."""

    index: int
    ai_probability: float
    text: str
    components: dict[str, float]
    connector_hits: tuple[str, ...]


@dataclass(frozen=True)
class Report:
    source_name: str | None
    verdict: Verdict
    ai_probability: float
    threshold: float
    # None cuando no hay texto suficiente para dar un veredicto.
    confidence: str | None
    chunk_count: int
    word_count: int
    flagged_count: int
    # Coeficiente de variación de la perplejidad entre bloques; None con menos de tres.
    perplexity_variation: float | None
    highlights: tuple[Highlight, ...]
    evaluations: tuple[ChunkEvaluation, ...]


class ReporterAgent:
    """Construye el informe del documento a partir de las evaluaciones por bloque."""

    def __init__(
        self,
        threshold: float = config.DETECTION_THRESHOLD,
        max_highlights: int = config.REPORT_MAX_HIGHLIGHTS,
    ) -> None:
        self.threshold = threshold
        self.max_highlights = max_highlights

    def build(self, evaluations: list[ChunkEvaluation], source_name: str | None = None) -> Report:
        word_count = sum(e.chunk.word_count for e in evaluations)
        flagged = [e for e in evaluations if e.ai_probability >= self.threshold]
        enough_text = word_count >= config.SEGMENT_MIN_WORDS

        ai_probability = (
            sum(e.ai_probability * e.chunk.word_count for e in evaluations) / word_count if word_count else 0.5
        )
        if not enough_text:
            verdict = Verdict.INSUFFICIENT_TEXT
        elif ai_probability >= self.threshold:
            verdict = Verdict.LIKELY_AI
        elif flagged:
            verdict = Verdict.MIXED
        else:
            verdict = Verdict.LIKELY_HUMAN

        most_suspicious = sorted(flagged, key=lambda e: e.ai_probability, reverse=True)[: self.max_highlights]
        return Report(
            source_name=source_name,
            verdict=verdict,
            ai_probability=ai_probability,
            threshold=self.threshold,
            confidence=self._confidence(ai_probability) if enough_text else None,
            chunk_count=len(evaluations),
            word_count=word_count,
            flagged_count=len(flagged),
            perplexity_variation=_perplexity_variation(evaluations),
            highlights=tuple(_highlight(e) for e in most_suspicious) if enough_text else (),
            evaluations=tuple(evaluations),
        )

    def _confidence(self, ai_probability: float) -> str:
        # Redondeo: 0,6 - 0,5 da 0,0999... en coma flotante y caería en "low".
        margin = round(abs(ai_probability - self.threshold), 9)
        for minimum, label in _CONFIDENCE_MARGINS:
            if margin >= minimum:
                return label
        return "low"


def _highlight(evaluation: ChunkEvaluation) -> Highlight:
    hits = evaluation.analysis.linguistic.connector_hits
    return Highlight(
        index=evaluation.chunk.index,
        ai_probability=evaluation.ai_probability,
        text=evaluation.chunk.text,
        components={c.name: c.probability for c in evaluation.components},
        # Sin repetidos, en el orden en que los informa el analizador lingüístico.
        connector_hits=tuple(dict.fromkeys(hits)),
    )


def _perplexity_variation(evaluations: list[ChunkEvaluation]) -> float | None:
    values = [e.analysis.perplexity.perplexity for e in evaluations if e.analysis.perplexity.perplexity is not None]
    if len(values) < 3:
        return None
    return statistics.pstdev(values) / statistics.fmean(values)


# --- Presentación ---------------------------------------------------------------


def render_markdown(report: Report) -> str:
    """Informe legible en Markdown (lo muestra la interfaz Gradio)."""
    title = f"# AI-text analysis: {_escape(report.source_name)}" if report.source_name else "# AI-text analysis"
    lines = [title, "", f"## Verdict: {report.verdict.value}", ""]

    if report.verdict is Verdict.INSUFFICIENT_TEXT:
        lines.append(
            f"The document has {report.word_count} words; at least {config.SEGMENT_MIN_WORDS} are needed for a "
            "meaningful analysis."
        )
    else:
        lines += [
            f"- **AI probability:** {_percent(report.ai_probability)} "
            f"(threshold {_percent(report.threshold)}, confidence {report.confidence})",
            f"- **Analyzed:** {report.word_count} words in {report.chunk_count} paragraph(s); "
            f"{report.flagged_count} above the threshold",
        ]
        if report.perplexity_variation is not None:
            lines.append(
                f"- **Perplexity variation across paragraphs:** {report.perplexity_variation:.2f} "
                "(lower = more uniform, which is typical of AI text)"
            )
        if report.highlights:
            lines += ["", "## Most suspicious paragraphs", ""]
            for highlight in report.highlights:
                lines += _render_highlight(highlight)

    lines += [
        "",
        "---",
        "_This is a statistical estimate, not proof of authorship. Detectors make mistakes, especially with "
        "short, formal or non-native texts; do not use it as the only evidence._",
    ]
    return "\n".join(lines)


def _render_highlight(highlight: Highlight) -> list[str]:
    signals = ", ".join(f"{name} {_percent(p)}" for name, p in highlight.components.items())
    lines = [
        f"### Paragraph {highlight.index + 1} — {_percent(highlight.ai_probability)}",
        "",
        f"> {_escape(_excerpt(highlight.text))}",
        "",
        f"- Signals: {signals}",
    ]
    if highlight.connector_hits:
        lines.append(f"- Typical AI connectors: {_escape(', '.join(highlight.connector_hits))}")
    lines.append("")
    return lines


def _excerpt(text: str) -> str:
    """Primeros caracteres del bloque, cortados en un límite de palabra y en una sola línea."""
    flat = " ".join(text.split())
    if len(flat) <= _EXCERPT_CHARS:
        return flat
    return flat[:_EXCERPT_CHARS].rsplit(" ", 1)[0] + " …"


def _escape(text: str) -> str:
    # Una sola línea: un salto dentro del texto abriría bloques nuevos (títulos,
    # código) e inyectaría estructura en el informe.
    text = " ".join(text.split())
    # Primero las entidades HTML, que se muestran literalmente; después Markdown.
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = _MARKDOWN_SPECIAL.sub(r"\\\1", text)
    return _LEADING_BLOCK_MARKER.sub(_escape_marker, text, count=1)


def _escape_marker(match: re.Match[str]) -> str:
    marker = match.group(0)
    # En una lista numerada se escapa el signo que sigue al número ("1\.").
    return marker + "\\" if marker[0].isdigit() else "\\" + marker


def _percent(value: float) -> str:
    return f"{value * 100:.0f}%"
