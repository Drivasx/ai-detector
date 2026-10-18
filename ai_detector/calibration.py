"""Calibración del umbral de detección con textos humanos del propio usuario.

Con textos que se sabe que son humanos (escritos antes de 2022, cuando no había
LLM de uso masivo) solo se puede medir una clase de error: los falsos positivos.
Por eso el umbral se elige como el más bajo que deja como mucho `target_fpr` de
los párrafos humanos por encima. Se calibra por párrafo y no por documento
porque un texto humano marcado como "mixto" ya es un falso positivo parcial, y
el umbral por párrafo también acota los falsos positivos por documento.

El umbral nunca baja del valor por defecto (bajarlo solo añadiría falsos
positivos) ni supera MAX_THRESHOLD (por encima ya no detectaría nada).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from ai_detector import config
from ai_detector.agents.evaluator import ChunkEvaluation
from ai_detector.agents.reader import SUPPORTED_EXTENSIONS, ReaderAgent, ReaderError

MAX_THRESHOLD = 0.99
# Paso de redondeo del umbral: queda justo por encima de la puntuación de corte.
_STEP = 0.001


class CalibrationError(RuntimeError):
    """No se pudo calibrar (carpeta inexistente o sin textos utilizables)."""


class EvaluatingPipeline(Protocol):
    def evaluate_text(self, text: str, on_progress=None) -> list[ChunkEvaluation]: ...


@dataclass(frozen=True)
class ThresholdChoice:
    threshold: float
    # True si los textos humanos puntúan tan alto que el umbral tocó el máximo.
    capped: bool


@dataclass(frozen=True)
class CalibrationResult:
    threshold: float
    capped: bool
    target_false_positive_rate: float
    file_count: int
    chunk_count: int
    # Proporciones de párrafos humanos marcados y de documentos humanos con al
    # menos un párrafo marcado (veredicto "IA" o "mixto").
    chunk_false_positive_rate: float
    document_false_positive_rate: float
    skipped: list[tuple[str, str]] = field(default_factory=list)


def compute_threshold(
    scores: list[float],
    target_fpr: float,
    floor: float = config.DETECTION_THRESHOLD,
    ceiling: float = MAX_THRESHOLD,
) -> ThresholdChoice:
    """Umbral mínimo que deja como mucho `target_fpr` de `scores` en o por encima de él."""
    if not scores:
        raise ValueError("At least one human score is needed to calibrate.")
    if not 0.0 <= target_fpr < 1.0:
        raise ValueError(f"target_fpr must be in [0, 1), got {target_fpr}.")

    ranked = sorted(scores, reverse=True)
    allowed_above = math.floor(target_fpr * len(ranked))
    # Justo por encima de la primera puntuación que ya no se puede dejar pasar.
    cutoff = ranked[allowed_above]
    threshold = math.ceil((cutoff + 1e-9) / _STEP) * _STEP
    threshold = round(max(floor, threshold), 3)
    if threshold > ceiling:
        return ThresholdChoice(threshold=ceiling, capped=True)
    return ThresholdChoice(threshold=threshold, capped=False)


def discover_texts(folder: str | Path, recursive: bool = False) -> list[Path]:
    """Archivos soportados de la carpeta, en orden alfabético."""
    folder = Path(folder)
    if not folder.is_dir():
        raise CalibrationError(f"'{folder}' is not a folder.")
    candidates = folder.rglob("*") if recursive else folder.iterdir()
    files = [p for p in candidates if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    return sorted(files, key=lambda p: (p.name.lower(), str(p)))


def calibrate(
    paths: list[Path],
    pipeline: EvaluatingPipeline,
    reader: ReaderAgent,
    target_fpr: float = config.CALIBRATION_TARGET_FPR,
) -> CalibrationResult:
    """Analiza cada texto humano y elige el umbral."""
    documents: list[list[ChunkEvaluation]] = []
    skipped: list[tuple[str, str]] = []
    for path in paths:
        try:
            evaluations = pipeline.evaluate_text(reader.read(path))
        except (ReaderError, FileNotFoundError) as exc:
            skipped.append((path.name, str(exc)))
            continue
        if evaluations:
            documents.append(evaluations)
    if not documents:
        raise CalibrationError("No usable human texts were found to calibrate with.")

    scores = [e.ai_probability for evaluations in documents for e in evaluations]
    choice = compute_threshold(scores, target_fpr)
    # Un documento cuenta como falso positivo si el informe lo marcaría como IA o
    # como mixto, es decir, si al menos uno de sus párrafos alcanza el umbral.
    flagged_documents = sum(
        any(e.ai_probability >= choice.threshold for e in evaluations) for evaluations in documents
    )
    return CalibrationResult(
        threshold=choice.threshold,
        capped=choice.capped,
        target_false_positive_rate=target_fpr,
        file_count=len(documents),
        chunk_count=len(scores),
        chunk_false_positive_rate=sum(s >= choice.threshold for s in scores) / len(scores),
        document_false_positive_rate=flagged_documents / len(documents),
        skipped=skipped,
    )


def save_calibration(result: CalibrationResult, path: str | Path = config.CALIBRATION_FILE) -> None:
    data = asdict(result)
    data["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    # Los modelos fijados condicionan las puntuaciones: si cambian, hay que recalibrar.
    data["models"] = {spec.repo_id: spec.revision for spec in config.ALL_MODELS}
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_threshold(path: str | Path = config.CALIBRATION_FILE) -> float | None:
    """Umbral calibrado, o None si el archivo no existe, no es válido o está fuera de [0,5, 0,99]."""
    try:
        threshold = json.loads(Path(path).read_text(encoding="utf-8"))["threshold"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    # Solo valores que la propia calibración puede producir: un archivo editado a
    # mano con un umbral más bajo añadiría falsos positivos sin avisar.
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        return None
    if not config.DETECTION_THRESHOLD <= threshold <= MAX_THRESHOLD:
        return None
    return float(threshold)


def resolve_threshold(path: str | Path = config.CALIBRATION_FILE) -> float:
    """Umbral que debe usar la aplicación: el calibrado si existe, si no el de config."""
    calibrated = load_threshold(path)
    return calibrated if calibrated is not None else config.DETECTION_THRESHOLD
