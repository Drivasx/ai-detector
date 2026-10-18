"""Pruebas de la interfaz Gradio sin lanzar el servidor."""

import os
import re
import subprocess
import sys

import gradio as gr
import pytest

from ai_detector import config
from ai_detector.agents.reader import UnsupportedFormatError
from ai_detector.agents.reporter import ReporterAgent
from ai_detector.models import ModelNotAvailableError
from ai_detector.ui import app
from tests.builders import evaluation


class FakePipeline:
    """Pipeline que devuelve un informe fijo y registra cómo se le llamó."""

    def __init__(self, error: Exception | None = None):
        self.calls = []
        self.error = error
        self.reporter = ReporterAgent(threshold=0.5)

    def _report(self, source_name):
        if self.error:
            raise self.error
        evaluations = [evaluation(0.9, index=0, text="Primer párrafo."), evaluation(0.2, index=1, text="Segundo.")]
        return self.reporter.build(evaluations, source_name=source_name)

    def analyze_file(self, path, on_progress=None):
        self.calls.append(("file", str(path)))
        if on_progress:
            on_progress(1, 1)
        return self._report(os.path.basename(str(path)))

    def analyze_text(self, text, source_name=None, on_progress=None):
        self.calls.append(("text", text))
        if on_progress:
            on_progress(1, 1)
        return self._report(source_name)


def test_analyzes_uploaded_file_and_returns_report_and_highlights(tmp_path):
    pipeline = FakePipeline()
    path = tmp_path / "ensayo.txt"

    markdown, highlights = app.make_handler(pipeline)(str(path), "")

    assert pipeline.calls == [("file", str(path))]
    assert "ensayo" in markdown
    # Cada bloque con su puntuación en [-1, 1]: -1 humano, +1 IA, 0 neutral.
    assert highlights == [("Primer párrafo.", pytest.approx(0.8)), ("\n\n", None), ("Segundo.", pytest.approx(-0.6))]


def test_analyzes_pasted_text_when_no_file_is_given():
    pipeline = FakePipeline()

    markdown, _ = app.make_handler(pipeline)(None, "Texto pegado.")

    assert pipeline.calls == [("text", "Texto pegado.")]
    assert "Pasted text" in markdown


def test_uploaded_file_takes_priority_over_pasted_text(tmp_path):
    pipeline = FakePipeline()

    app.make_handler(pipeline)(str(tmp_path / "a.txt"), "Texto pegado.")

    assert [kind for kind, _ in pipeline.calls] == ["file"]


def test_missing_input_shows_an_error():
    with pytest.raises(gr.Error, match="Upload a file or paste"):
        app.make_handler(FakePipeline())(None, "   ")


@pytest.mark.parametrize(
    "error",
    [UnsupportedFormatError("Unsupported file format '.xlsx'."), ModelNotAvailableError("Run `python scripts/download_models.py`")],
)
def test_reader_and_model_errors_are_shown_to_the_user(error):
    with pytest.raises(gr.Error, match=re.escape(str(error))):
        app.make_handler(FakePipeline(error=error))(None, "Hola.")


def test_app_is_built_with_analytics_disabled():
    blocks = app.build_app(FakePipeline())

    assert isinstance(blocks, gr.Blocks)
    assert blocks.analytics_enabled is False


def test_importing_the_package_disables_gradio_analytics():
    env = {key: value for key, value in os.environ.items() if key != "GRADIO_ANALYTICS_ENABLED"}
    code = "import ai_detector, os; print(os.environ['GRADIO_ANALYTICS_ENABLED'])"

    result = subprocess.run(
        [sys.executable, "-c", code], env=env, cwd=config.PROJECT_ROOT, capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "False"


def test_theme_uses_only_local_fonts():
    fonts = [font for attr in ("font", "font_mono") for font in getattr(app.THEME, "_" + attr, [])]

    assert fonts
    assert not any(isinstance(font, gr.themes.GoogleFont) for font in fonts)


def test_server_listens_only_on_localhost():
    assert app.SERVER_NAME == "127.0.0.1"


def test_user_exported_gradio_overrides_cannot_reenable_telemetry_or_exposure():
    env = dict(os.environ, GRADIO_ANALYTICS_ENABLED="True", GRADIO_SERVER_NAME="0.0.0.0", GRADIO_SHARE="True")
    code = (
        "import os, ai_detector.ui.app as app; "
        "print(os.environ['GRADIO_ANALYTICS_ENABLED'], os.environ.get('GRADIO_SERVER_NAME'), os.environ.get('GRADIO_SHARE'))"
    )

    result = subprocess.run(
        [sys.executable, "-c", code], env=env, cwd=config.PROJECT_ROOT, capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "False None None"


def test_uploaded_document_is_deleted_from_gradio_cache_after_analysis(tmp_path):
    uploaded = tmp_path / "privado.txt"
    uploaded.write_text("contenido privado", encoding="utf-8")

    app.make_handler(FakePipeline())(str(uploaded), "")

    assert not uploaded.exists()


def test_uploaded_document_is_deleted_even_when_analysis_fails(tmp_path):
    uploaded = tmp_path / "privado.xlsx"
    uploaded.write_text("x", encoding="utf-8")

    with pytest.raises(gr.Error):
        app.make_handler(FakePipeline(error=UnsupportedFormatError("Unsupported")))(str(uploaded), "")

    assert not uploaded.exists()


def test_app_periodically_clears_gradio_cache():
    blocks = app.build_app(FakePipeline())

    assert blocks.delete_cache == app.DELETE_CACHE


def test_progress_is_forwarded_to_gradio():
    seen = []

    app.make_handler(FakePipeline())(None, "Texto.", progress=lambda value, desc=None: seen.append((value, desc)))

    assert seen == [((1, 1), "Analyzing paragraphs")]


def test_main_launches_on_localhost_without_share_on_the_requested_port(monkeypatch):
    captured = {}
    monkeypatch.setattr(app, "DetectionPipeline", FakePipeline)
    monkeypatch.setattr(gr.Blocks, "launch", lambda self, **kwargs: captured.update(kwargs))

    app.main(["--port", "8123"])

    assert captured["server_name"] == "127.0.0.1"
    assert captured["share"] is False
    assert captured["server_port"] == 8123
    assert captured["theme"] is app.THEME
