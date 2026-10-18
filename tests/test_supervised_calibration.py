"""Pruebas de la calibración supervisada (textos humanos + textos IA), sin modelos."""

import json

import pytest

from ai_detector import config
from ai_detector.agents.reader import ReaderAgent
from ai_detector.calibration import (
    calibrate,
    discover_texts,
    fit_parameters,
    load_calibration,
    resolve_settings,
    save_calibration,
)
from tests.builders import analysis


def chunk(ppl, burst=3.0, clf=0.999, ling=0.3, index=0):
    return analysis(perplexity=ppl, burstiness=burst, classifier=clf, linguistic_score=ling, words=60, index=index)


HUMAN = [chunk(p, b, ling=l, index=i) for i, (p, b, l) in enumerate([(150, 3.7, 0.3), (110, 3.5, 0.2), (200, 3.9, 0.4), (130, 3.6, 0.3)])]
AI = [chunk(p, b, ling=l, index=i) for i, (p, b, l) in enumerate([(20, 2.5, 0.3), (35, 2.7, 0.2), (25, 2.6, 0.4), (30, 2.8, 0.3)])]


def test_references_are_the_median_of_each_class():
    fitted = fit_parameters(HUMAN, AI)

    assert fitted.perplexity_references["perplexity"] == pytest.approx((140.0, 27.5))
    assert fitted.perplexity_references["burstiness"] == pytest.approx((3.65, 2.65))


def test_weights_follow_how_well_each_signal_separates_the_classes():
    fitted = fit_parameters(HUMAN, AI)

    # La perplejidad separa por completo; clasificador y rasgos son idénticos en ambas clases.
    assert fitted.auc["perplexity"] == pytest.approx(1.0)
    assert fitted.auc["classifier"] == pytest.approx(0.5)
    assert fitted.weights == pytest.approx({"perplexity": 1.0, "classifier": 0.0, "linguistic": 0.0})


def test_inverted_references_fall_back_to_defaults():
    fitted = fit_parameters(AI, HUMAN)  # clases al revés: "humano" con menor perplejidad

    assert fitted.perplexity_references == config.PERPLEXITY_REFERENCES
    assert fitted.weights == config.VOTING_WEIGHTS
    assert fitted.warnings


class FakeAnalyzingPipeline:
    """Cada palabra del texto es un bloque "perplejidad/burstiness"."""

    def analyze_chunks(self, text, on_progress=None):
        values = [token.split("/") for token in text.split()]
        return [chunk(float(p), float(b), index=i) for i, (p, b) in enumerate(values)]


def write(folder, name, values):
    folder.mkdir(exist_ok=True)
    (folder / name).write_text(" ".join(f"{p}/{b}" for p, b in values), encoding="utf-8")


def test_calibrate_with_ai_folder_fits_parameters_and_reports_detection(tmp_path):
    write(tmp_path / "human", "a.txt", [(150, 3.7), (110, 3.5)])
    write(tmp_path / "human", "b.txt", [(200, 3.9), (130, 3.6)])
    write(tmp_path / "ai", "x.txt", [(20, 2.5), (35, 2.7)])
    write(tmp_path / "ai", "y.txt", [(25, 2.6), (30, 2.8)])

    result = calibrate(
        discover_texts(tmp_path / "human"),
        FakeAnalyzingPipeline(),
        ReaderAgent(),
        target_fpr=0.0,
        ai_paths=discover_texts(tmp_path / "ai"),
    )

    assert (result.ai_file_count, result.ai_chunk_count) == (2, 4)
    assert result.weights["perplexity"] == pytest.approx(1.0)
    assert result.chunk_false_positive_rate == 0.0
    assert result.ai_detection_rate == pytest.approx(1.0)


def test_supervised_settings_round_trip_and_resolve(tmp_path):
    write(tmp_path / "human", "a.txt", [(150, 3.7), (110, 3.5)])
    write(tmp_path / "ai", "x.txt", [(20, 2.5), (35, 2.7)])
    result = calibrate(
        discover_texts(tmp_path / "human"), FakeAnalyzingPipeline(), ReaderAgent(), 0.0, discover_texts(tmp_path / "ai")
    )
    path = tmp_path / "calibration.json"

    save_calibration(result, path)
    settings = resolve_settings(path)

    assert settings.threshold == result.threshold
    assert settings.evaluator_kwargs["weights"] == pytest.approx(result.weights)
    assert settings.evaluator_kwargs["perplexity_references"]["perplexity"] == pytest.approx((130.0, 27.5))


@pytest.mark.parametrize(
    "extra",
    [
        {"weights": {"perplexity": -1, "classifier": 1, "linguistic": 1}},
        {"weights": {"perplexity": 1}},
        {"perplexity_references": {"perplexity": [20, 120], "burstiness": [3.6, 2.6], "burstiness_weight": 0.3}},
    ],
)
def test_invalid_fitted_parameters_are_ignored_but_threshold_is_kept(tmp_path, extra):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"threshold": 0.6, **extra}), encoding="utf-8")

    settings = load_calibration(path)

    assert settings.threshold == 0.6
    assert settings.evaluator_kwargs == {}


def test_without_calibration_file_settings_are_the_defaults(tmp_path):
    settings = resolve_settings(tmp_path / "none.json")

    assert settings.threshold == config.DETECTION_THRESHOLD
    assert settings.evaluator_kwargs == {}
