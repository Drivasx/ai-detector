"""Pruebas de la calibración del umbral con textos humanos (sin modelos)."""

import json

import pytest

from ai_detector import calibration, config
from ai_detector.agents.reader import ReaderAgent
from ai_detector.calibration import (
    CalibrationError,
    calibrate,
    compute_threshold,
    discover_texts,
    load_threshold,
    resolve_threshold,
    save_calibration,
)
from tests.builders import evaluation

# --- compute_threshold ----------------------------------------------------


def test_zero_target_puts_threshold_just_above_every_human_score():
    result = compute_threshold([0.55, 0.72, 0.61], target_fpr=0.0)

    assert result.threshold == pytest.approx(0.721)
    assert result.capped is False


def test_target_allows_that_share_of_human_scores_above_threshold():
    scores = [0.51, 0.52, 0.53, 0.54, 0.55, 0.80, 0.85, 0.90]  # 25% -> se permiten 2 por encima

    result = compute_threshold(scores, target_fpr=0.25)

    assert result.threshold == pytest.approx(0.801)
    assert sum(s >= result.threshold for s in scores) / len(scores) <= 0.25


@pytest.mark.parametrize("target", [0.0, 0.05, 0.1, 0.3])
def test_empirical_false_positive_rate_never_exceeds_target(target):
    scores = [i / 37 for i in range(37)]

    result = compute_threshold(scores, target_fpr=target)

    assert sum(s >= result.threshold for s in scores) / len(scores) <= target


def test_threshold_is_never_lowered_below_the_default():
    result = compute_threshold([0.1, 0.2, 0.3], target_fpr=0.0)

    assert result.threshold == config.DETECTION_THRESHOLD


def test_threshold_is_capped_when_human_texts_score_extremely_high():
    result = compute_threshold([0.995, 0.999], target_fpr=0.0)

    assert result.threshold == calibration.MAX_THRESHOLD
    assert result.capped is True


@pytest.mark.parametrize(("scores", "target"), [([], 0.05), ([0.5], -0.1), ([0.5], 1.0)])
def test_invalid_inputs_raise_value_error(scores, target):
    with pytest.raises(ValueError):
        compute_threshold(scores, target_fpr=target)


# --- discover_texts -------------------------------------------------------


def test_discovers_supported_files_sorted_and_optionally_recursive(tmp_path):
    for name in ("b.txt", "a.pdf", "c.docx", "notas.md", "sub/d.txt"):
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("x", encoding="utf-8")

    assert [p.name for p in discover_texts(tmp_path)] == ["a.pdf", "b.txt", "c.docx"]
    assert [p.name for p in discover_texts(tmp_path, recursive=True)] == ["a.pdf", "b.txt", "c.docx", "d.txt"]


def test_missing_folder_raises_calibration_error(tmp_path):
    with pytest.raises(CalibrationError, match="not a folder"):
        discover_texts(tmp_path / "no_existe")


# --- calibrate --------------------------------------------------------------


class FakePipeline:
    """Devuelve, para cada texto, las probabilidades por bloque indicadas en el propio texto."""

    def evaluate_text(self, text, on_progress=None):
        return [evaluation(float(p), index=i, words=60) for i, p in enumerate(text.split())]


def write(folder, name, probabilities):
    path = folder / name
    path.write_text(" ".join(str(p) for p in probabilities), encoding="utf-8")
    return path


def test_calibrates_on_chunk_scores_of_every_readable_file(tmp_path):
    write(tmp_path, "a.txt", [0.2, 0.3])
    write(tmp_path, "b.txt", [0.4, 0.7])
    (tmp_path / "vacio.txt").write_text("   ", encoding="utf-8")

    result = calibrate(discover_texts(tmp_path), FakePipeline(), ReaderAgent(), target_fpr=0.0)

    assert result.threshold == pytest.approx(0.701)
    assert (result.file_count, result.chunk_count) == (2, 4)
    assert [name for name, _ in result.skipped] == ["vacio.txt"]
    # Con el umbral elegido, ningún bloque ni documento humano queda marcado.
    assert result.chunk_false_positive_rate == 0.0
    assert result.document_false_positive_rate == 0.0


def test_reports_false_positive_rates_at_the_chosen_threshold(tmp_path):
    write(tmp_path, "a.txt", [0.9, 0.9])  # documento claramente por encima
    write(tmp_path, "b.txt", [0.1, 0.1])
    write(tmp_path, "c.txt", [0.2, 0.2])
    write(tmp_path, "d.txt", [0.3, 0.3])

    result = calibrate(discover_texts(tmp_path), FakePipeline(), ReaderAgent(), target_fpr=0.25)

    assert result.chunk_false_positive_rate == pytest.approx(0.25)
    assert result.document_false_positive_rate == pytest.approx(0.25)


def test_no_usable_texts_raises_calibration_error(tmp_path):
    (tmp_path / "vacio.txt").write_text("", encoding="utf-8")

    with pytest.raises(CalibrationError, match="No usable"):
        calibrate(discover_texts(tmp_path), FakePipeline(), ReaderAgent(), target_fpr=0.05)


# --- persistencia ---------------------------------------------------------


def test_saved_calibration_is_loaded_back(tmp_path):
    write(tmp_path, "a.txt", [0.6, 0.65])
    result = calibrate(discover_texts(tmp_path), FakePipeline(), ReaderAgent(), target_fpr=0.0)
    path = tmp_path / "calibration.json"

    save_calibration(result, path)

    assert load_threshold(path) == pytest.approx(0.651)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["file_count"] == 1 and data["target_false_positive_rate"] == 0.0


@pytest.mark.parametrize(
    "content", [None, "not json", '{"threshold": 7}', '{"other": 1}', '{"threshold": 0.3}', '{"threshold": 0.995}']
)
def test_missing_or_invalid_calibration_file_is_ignored(tmp_path, content):
    path = tmp_path / "calibration.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")

    assert load_threshold(path) is None


def test_resolve_threshold_prefers_calibration_and_falls_back_to_default(tmp_path):
    path = tmp_path / "calibration.json"
    assert resolve_threshold(path) == config.DETECTION_THRESHOLD

    path.write_text('{"threshold": 0.73}', encoding="utf-8")
    assert resolve_threshold(path) == 0.73


def test_document_rate_counts_documents_with_any_flagged_paragraph(tmp_path):
    write(tmp_path, "a.txt", [0.1, 0.1, 0.1, 0.95])  # media baja, pero el informe diría "mixto"
    write(tmp_path, "b.txt", [0.1, 0.2])
    write(tmp_path, "c.txt", [0.2, 0.3])

    result = calibrate(discover_texts(tmp_path), FakePipeline(), ReaderAgent(), target_fpr=0.15)

    assert result.threshold < 0.95
    assert result.document_false_positive_rate == pytest.approx(1 / 3)
