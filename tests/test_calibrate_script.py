"""Pruebas del script scripts/calibrate.py con un pipeline falso."""

import json

import pytest

from scripts import calibrate as script
from tests.test_calibration import FakePipeline, write


@pytest.fixture(autouse=True)
def fake_pipeline(monkeypatch):
    monkeypatch.setattr(script, "DetectionPipeline", FakePipeline)


def test_writes_calibration_and_prints_summary(tmp_path, capsys):
    write(tmp_path, "a.txt", [0.6, 0.62])
    write(tmp_path, "b.txt", [0.3, 0.4])
    output = tmp_path / "out" / "calibration.json"

    code = script.main([str(tmp_path), "--target-fpr", "0", "--output", str(output)])

    assert code == 0
    assert json.loads(output.read_text(encoding="utf-8"))["threshold"] == pytest.approx(0.621)
    printed = capsys.readouterr().out
    assert "Threshold: 0.621" in printed
    assert "2 files, 4 paragraphs" in printed
    assert "before 2022" in printed


def test_lists_skipped_files(tmp_path, capsys):
    write(tmp_path, "a.txt", [0.6])
    (tmp_path / "vacio.txt").write_text("", encoding="utf-8")

    script.main([str(tmp_path), "--output", str(tmp_path / "c.json")])

    assert "Skipped vacio.txt" in capsys.readouterr().out


def test_warns_when_threshold_is_capped(tmp_path, capsys):
    write(tmp_path, "a.txt", [0.999])

    script.main([str(tmp_path), "--output", str(tmp_path / "c.json")])

    assert "WARNING" in capsys.readouterr().out


def test_missing_folder_exits_with_error_and_writes_nothing(tmp_path, capsys):
    output = tmp_path / "c.json"

    code = script.main([str(tmp_path / "no_existe"), "--output", str(output)])

    assert code == 1
    assert "not a folder" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("value", ["-0.1", "1", "abc"])
def test_invalid_target_fpr_is_rejected(tmp_path, value):
    with pytest.raises(SystemExit) as exit_info:
        script.main([str(tmp_path), "--target-fpr", value])

    assert exit_info.value.code == 2


def test_ai_folder_runs_supervised_calibration_and_reports_detection(tmp_path, capsys, monkeypatch):
    from tests.test_supervised_calibration import FakeAnalyzingPipeline
    from tests.test_supervised_calibration import write as write_pairs

    monkeypatch.setattr(script, "DetectionPipeline", FakeAnalyzingPipeline)
    write_pairs(tmp_path / "human", "a.txt", [(150, 3.7), (110, 3.5)])
    write_pairs(tmp_path / "ai", "x.txt", [(20, 2.5), (35, 2.7)])
    output = tmp_path / "c.json"

    code = script.main([str(tmp_path / "human"), "--ai-folder", str(tmp_path / "ai"), "--output", str(output)])

    assert code == 0
    printed = capsys.readouterr().out
    assert "100.0% of the AI paragraphs" in printed
    assert "Weights:" in printed
    assert json.loads(output.read_text(encoding="utf-8"))["weights"]["perplexity"] == pytest.approx(1.0)


def test_missing_ai_folder_exits_with_error(tmp_path, capsys):
    write(tmp_path, "a.txt", [0.6])

    assert script.main([str(tmp_path), "--ai-folder", str(tmp_path / "nope"), "--output", str(tmp_path / "c.json")]) == 1
    assert "not a folder" in capsys.readouterr().err
