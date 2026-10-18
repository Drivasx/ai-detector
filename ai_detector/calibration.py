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
import statistics
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from ai_detector import config
from ai_detector.agents.analyzer import ChunkAnalysis
from ai_detector.agents.evaluator import COMPONENTS, ChunkEvaluation, EvaluatorAgent
from ai_detector.agents.reader import SUPPORTED_EXTENSIONS, ReaderAgent, ReaderError

MAX_THRESHOLD = 0.99
# Paso de redondeo del umbral: queda justo por encima de la puntuación de corte.
_STEP = 0.001


class CalibrationError(RuntimeError):
    """No se pudo calibrar (carpeta inexistente o sin textos utilizables)."""


class EvaluatingPipeline(Protocol):
    def evaluate_text(self, text: str, on_progress=None) -> list[ChunkEvaluation]: ...

    def analyze_chunks(self, text: str, on_progress=None) -> list[ChunkAnalysis]: ...


@dataclass(frozen=True)
class FittedParameters:
    """Referencias y pesos del voto ajustados con textos humanos y de IA del usuario."""

    perplexity_references: dict
    binoculars_references: tuple[float, float]
    weights: dict[str, float]
    # Área bajo la curva ROC de cada componente: 0,5 = no separa, 1 = separa del todo.
    auc: dict[str, float]
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CalibrationSettings:
    """Lo que la aplicación toma de calibration.json."""

    threshold: float
    # Argumentos para EvaluatorAgent (pesos y referencias ajustados); vacío si no hay.
    evaluator_kwargs: dict


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
    # Solo con calibración supervisada (carpeta de textos IA).
    ai_file_count: int = 0
    ai_chunk_count: int = 0
    # Proporción de párrafos de IA que superan el umbral elegido.
    ai_detection_rate: float | None = None
    weights: dict[str, float] | None = None
    perplexity_references: dict | None = None
    binoculars_references: tuple[float, float] | None = None
    auc: dict[str, float] | None = None
    warnings: list[str] = field(default_factory=list)


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
    ai_paths: list[Path] | None = None,
) -> CalibrationResult:
    """Analiza los textos humanos (y, si se dan, los de IA) y elige el umbral.

    Sin `ai_paths` solo se ajusta el umbral. Con `ai_paths` se ajustan además las
    referencias y los pesos del voto, y se informa qué parte de la IA se detecta.
    """
    skipped: list[tuple[str, str]] = []
    if not ai_paths:
        documents = _read_all(paths, reader, pipeline.evaluate_text, skipped)
        if not documents:
            raise CalibrationError("No usable human texts were found to calibrate with.")
        return _threshold_result(documents, target_fpr, skipped)

    human_docs = _read_all(paths, reader, pipeline.analyze_chunks, skipped)
    ai_docs = _read_all(ai_paths, reader, pipeline.analyze_chunks, skipped)
    if not human_docs:
        raise CalibrationError("No usable human texts were found to calibrate with.")
    if not ai_docs:
        raise CalibrationError("No usable AI texts were found to calibrate with.")

    fitted = fit_parameters([a for doc in human_docs for a in doc], [a for doc in ai_docs for a in doc])
    evaluator = EvaluatorAgent(**_evaluator_kwargs(fitted))
    result = _threshold_result([evaluator.evaluate(doc) for doc in human_docs], target_fpr, skipped)
    ai_scores = [e.ai_probability for doc in ai_docs for e in evaluator.evaluate(doc)]
    return replace(
        result,
        ai_file_count=len(ai_docs),
        ai_chunk_count=len(ai_scores),
        ai_detection_rate=sum(s >= result.threshold for s in ai_scores) / len(ai_scores),
        weights=fitted.weights,
        perplexity_references=fitted.perplexity_references,
        binoculars_references=fitted.binoculars_references,
        auc=fitted.auc,
        warnings=fitted.warnings,
    )


def _read_all(paths, reader, analyze, skipped) -> list[list]:
    """Aplica `analyze` al texto de cada archivo; los ilegibles se anotan en `skipped`."""
    documents = []
    for path in paths:
        try:
            chunks = analyze(reader.read(path))
        except (ReaderError, FileNotFoundError) as exc:
            skipped.append((path.name, str(exc)))
            continue
        if chunks:
            documents.append(chunks)
    return documents


def _threshold_result(documents: list[list[ChunkEvaluation]], target_fpr: float, skipped) -> CalibrationResult:
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


def fit_parameters(human: list[ChunkAnalysis], ai: list[ChunkAnalysis]) -> FittedParameters:
    """Ajusta referencias (medianas de cada clase) y pesos (según cuánto separa cada señal)."""
    warnings: list[str] = []
    references = _fit_references(human, ai, warnings)
    if references is None:
        return FittedParameters(
            config.PERPLEXITY_REFERENCES, config.BINOCULARS_REFERENCES, dict(config.VOTING_WEIGHTS), {}, warnings
        )
    binoculars_references = _fit_binoculars_references(human, ai, warnings)

    # Peso uniforme solo para obtener la probabilidad de cada componente por separado.
    probe = EvaluatorAgent(
        weights={name: 1.0 for name in COMPONENTS},
        perplexity_references=references,
        binoculars_references=binoculars_references,
    )
    human_probs, ai_probs = _component_probabilities(probe, human), _component_probabilities(probe, ai)
    auc = {name: _auc(human_probs[name], ai_probs[name]) for name in COMPONENTS}
    # Una señal que no separa mejor que el azar (AUC <= 0,5) no vota.
    separation = {name: max(0.0, value - 0.5) for name, value in auc.items()}
    total = sum(separation.values())
    if total == 0:
        warnings.append("No signal separates your human texts from the AI texts; default weights kept.")
        weights = dict(config.VOTING_WEIGHTS)
    else:
        weights = {name: value / total for name, value in separation.items()}
    return FittedParameters(references, binoculars_references, weights, auc, warnings)


def _evaluator_kwargs(fitted: FittedParameters) -> dict:
    return {
        "weights": fitted.weights,
        "perplexity_references": fitted.perplexity_references,
        "binoculars_references": fitted.binoculars_references,
    }


def _fit_references(human, ai, warnings) -> dict | None:
    def median(analyses, attribute):
        values = [getattr(a.perplexity, attribute) for a in analyses if getattr(a.perplexity, attribute) is not None]
        return statistics.median(values) if values else None

    human_ppl, ai_ppl = median(human, "perplexity"), median(ai, "perplexity")
    human_burst, ai_burst = median(human, "burstiness"), median(ai, "burstiness")
    if None in (human_ppl, ai_ppl, human_burst, ai_burst) or human_ppl <= ai_ppl or human_burst <= ai_burst:
        warnings.append(
            "Your AI texts are not more predictable than your human texts; default references and weights kept."
        )
        return None
    return {
        "perplexity": (human_ppl, ai_ppl),
        "burstiness": (human_burst, ai_burst),
        "burstiness_weight": config.PERPLEXITY_REFERENCES["burstiness_weight"],
    }


def _fit_binoculars_references(human, ai, warnings) -> tuple[float, float]:
    """Medianas de Binoculars de cada clase; las de config si no hay datos o salen invertidas."""
    human_scores = [a.binoculars for a in human if a.binoculars is not None]
    ai_scores = [a.binoculars for a in ai if a.binoculars is not None]
    if not human_scores or not ai_scores:
        return config.BINOCULARS_REFERENCES  # Sin medidas: su AUC será 0,5 y no votará.
    human_median, ai_median = statistics.median(human_scores), statistics.median(ai_scores)
    if human_median <= ai_median:
        warnings.append("Binoculars does not score your AI texts lower than your human texts; its default references kept.")
        return config.BINOCULARS_REFERENCES
    return (human_median, ai_median)


def _component_probabilities(evaluator: EvaluatorAgent, analyses: list[ChunkAnalysis]) -> dict[str, list[float]]:
    probabilities: dict[str, list[float]] = {name: [] for name in COMPONENTS}
    for evaluation in evaluator.evaluate(analyses):
        scores = {c.name: c.probability for c in evaluation.components}
        for name in COMPONENTS:
            # Un componente que no pudo medirse cuenta como neutral.
            probabilities[name].append(scores.get(name, 0.5))
    return probabilities


def _auc(human: list[float], ai: list[float]) -> float:
    """Probabilidad de que un bloque de IA puntúe más que uno humano (empates cuentan la mitad)."""
    wins = sum((a > h) + 0.5 * (a == h) for a in ai for h in human)
    return wins / (len(ai) * len(human))


def save_calibration(result: CalibrationResult, path: str | Path = config.CALIBRATION_FILE) -> None:
    data = asdict(result)
    data["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    # Los modelos fijados condicionan las puntuaciones: si cambian, hay que recalibrar.
    data["models"] = {spec.repo_id: spec.revision for spec in config.ALL_MODELS}
    # Parámetros con los que se puntuó al elegir el umbral: si al cargar no coinciden
    # con los que se usarán, el umbral ya no vale y la calibración se ignora.
    data["scoring"] = _scoring(result.weights, result.perplexity_references, result.binoculars_references)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_calibration(path: str | Path = config.CALIBRATION_FILE) -> CalibrationSettings | None:
    """Ajustes guardados, o None si el archivo no existe o ya no es aplicable.

    Se ignora entero si el umbral no es válido, si los pesos o referencias ajustados
    están dañados, o si se calculó con otros parámetros de puntuación (por ejemplo,
    valores por defecto anteriores): en todos esos casos hay que recalibrar.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        threshold = data["threshold"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    # Solo valores que la propia calibración puede producir: un archivo editado a
    # mano con un umbral más bajo añadiría falsos positivos sin avisar.
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        return None
    if not config.DETECTION_THRESHOLD <= threshold <= MAX_THRESHOLD:
        return None
    weights, references = data.get("weights"), data.get("perplexity_references")
    binoculars = data.get("binoculars_references")
    evaluator_kwargs = {}
    if weights is not None or references is not None or binoculars is not None:
        valid = _valid_weights(weights) and _valid_references(references) and _valid_binoculars(binoculars)
        if not valid:
            return None  # Ajuste supervisado dañado o anterior a Binoculars: su umbral no sirve sin él.
        evaluator_kwargs = {
            "weights": {name: float(weights[name]) for name in COMPONENTS},
            "perplexity_references": {
                "perplexity": tuple(references["perplexity"]),
                "burstiness": tuple(references["burstiness"]),
                "burstiness_weight": float(references["burstiness_weight"]),
            },
            "binoculars_references": (float(binoculars[0]), float(binoculars[1])),
        }
    expected = _scoring(
        evaluator_kwargs.get("weights"),
        evaluator_kwargs.get("perplexity_references"),
        evaluator_kwargs.get("binoculars_references"),
    )
    if data.get("scoring") != expected:
        return None  # Umbral elegido con otros pesos/referencias (p. ej. valores por defecto anteriores).
    return CalibrationSettings(threshold=float(threshold), evaluator_kwargs=evaluator_kwargs)


def _scoring(weights: dict | None, references: dict | None, binoculars: tuple | None) -> dict:
    """Pesos y referencias efectivos, en forma comparable con lo guardado en JSON."""
    used = {
        "weights": weights or config.VOTING_WEIGHTS,
        "perplexity_references": references or config.PERPLEXITY_REFERENCES,
        "binoculars_references": binoculars or config.BINOCULARS_REFERENCES,
    }
    return json.loads(json.dumps(used))


def _valid_weights(weights) -> bool:
    if not isinstance(weights, dict) or set(weights) != set(COMPONENTS):
        return False
    values = list(weights.values())
    return all(_is_number(v) and v >= 0 for v in values) and sum(values) > 0


def _valid_references(references) -> bool:
    try:
        human_ppl, ai_ppl = references["perplexity"]
        human_burst, ai_burst = references["burstiness"]
        burstiness_weight = references["burstiness_weight"]
    except (TypeError, KeyError, ValueError):
        return False
    numbers = (human_ppl, ai_ppl, human_burst, ai_burst, burstiness_weight)
    if not all(_is_number(v) for v in numbers):
        return False
    return human_ppl > ai_ppl > 0 and human_burst > ai_burst > 0 and 0 <= burstiness_weight <= 1


def _valid_binoculars(references) -> bool:
    if not isinstance(references, (list, tuple)) or len(references) != 2:
        return False
    human_score, ai_score = references
    return _is_number(human_score) and _is_number(ai_score) and human_score > ai_score > 0


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def load_threshold(path: str | Path = config.CALIBRATION_FILE) -> float | None:
    """Umbral calibrado, o None si el archivo no existe, no es válido o está fuera de [0,5, 0,99]."""
    settings = load_calibration(path)
    return settings.threshold if settings else None


def resolve_settings(path: str | Path = config.CALIBRATION_FILE) -> CalibrationSettings:
    """Ajustes que debe usar la aplicación: los calibrados si existen, si no los de config."""
    return load_calibration(path) or CalibrationSettings(threshold=config.DETECTION_THRESHOLD, evaluator_kwargs={})


def resolve_threshold(path: str | Path = config.CALIBRATION_FILE) -> float:
    """Umbral que debe usar la aplicación: el calibrado si existe, si no el de config."""
    return resolve_settings(path).threshold
