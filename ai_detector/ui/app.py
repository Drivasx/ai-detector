"""Interfaz Gradio que se sirve solo en localhost.

Uso:
    python -m ai_detector.ui.app [--port 7860]

Garantías de funcionamiento offline de la interfaz:
- El servidor escucha solo en 127.0.0.1 y nunca crea enlaces públicos (share=False).
- La telemetría de Gradio se desactiva (ai_detector/__init__.py y analytics_enabled=False).
- El tema usa fuentes del sistema: el navegador no descarga nada de internet.
- El informe Markdown llega ya escapado (ver agents/reporter.py) y se desactiva
  el renderizado de LaTeX, así que el texto del documento no puede inyectar contenido.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Callable

# Gradio lee estas variables al lanzar y anularían server_name/share: se eliminan
# para que la interfaz escuche siempre solo en 127.0.0.1 y nunca se publique.
for _variable in ("GRADIO_SERVER_NAME", "GRADIO_SHARE"):
    os.environ.pop(_variable, None)

import gradio as gr  # noqa: E402

from ai_detector.agents.reader import SUPPORTED_EXTENSIONS, ReaderError  # noqa: E402
from ai_detector.agents.reporter import Report, ReporterAgent, render_markdown  # noqa: E402
from ai_detector.agents.evaluator import EvaluatorAgent  # noqa: E402
from ai_detector import config  # noqa: E402
from ai_detector.calibration import load_calibration, resolve_settings  # noqa: E402
from ai_detector.models import ModelNotAvailableError  # noqa: E402
from ai_detector.pipeline import DetectionPipeline  # noqa: E402

SERVER_NAME = "127.0.0.1"
DEFAULT_PORT = 7860

THEME = gr.themes.Default(font=["system-ui", "sans-serif"], font_mono=["ui-monospace", "monospace"])

PASTED_TEXT_NAME = "Pasted text"

# Gradio copia cada archivo subido a su caché temporal. El manejador borra la
# copia en cuanto termina el análisis; además, cada 10 minutos (y al cerrar el
# servidor) se borra lo que tenga más de 10 minutos, por si quedara algún resto.
DELETE_CACHE = (600, 600)

# Fragmento del documento con su puntuación, o separador sin etiqueta.
Highlight = tuple[str, float | None]


def make_handler(pipeline: DetectionPipeline) -> Callable[..., tuple[str, list[Highlight], None]]:
    """Crea la función que Gradio ejecuta al pulsar "Analyze".

    Devuelve el informe, el documento coloreado y None para vaciar el componente
    de archivo, cuya copia en caché se borra al terminar.
    """

    def handle(file_path: str | None, pasted_text: str | None, progress: gr.Progress = gr.Progress()):
        def on_progress(done: int, total: int) -> None:
            progress((done, total), desc="Analyzing paragraphs")

        try:
            # Si hay archivo y texto pegado a la vez, manda el archivo.
            if file_path:
                report = pipeline.analyze_file(file_path, on_progress=on_progress)
            elif pasted_text and pasted_text.strip():
                report = pipeline.analyze_text(pasted_text, source_name=PASTED_TEXT_NAME, on_progress=on_progress)
            else:
                raise gr.Error("Upload a file or paste some text to analyze.")
        except FileNotFoundError as exc:
            # Solo es una subida caducada si falta justo el archivo subido; cualquier
            # otro archivo ausente (p. ej. de un modelo) se muestra tal cual.
            if file_path and not Path(file_path).exists():
                raise gr.Error("The uploaded file is no longer available; please upload it again.") from exc
            raise gr.Error(str(exc)) from exc
        except (ReaderError, ModelNotAvailableError) as exc:
            raise gr.Error(str(exc)) from exc
        finally:
            if file_path:
                _delete_cached_upload(file_path)
        return render_markdown(report), _highlights(report), None

    return handle


def _delete_cached_upload(file_path: str) -> None:
    """Borra la copia del archivo subido para no dejar documentos privados en disco.

    `file_path` es siempre la copia en la caché de Gradio (Gradio rechaza rutas
    del cliente fuera de sus carpetas permitidas), nunca el original del usuario.
    Un fallo al borrar no debe ocultar el resultado: el barrido periódico de
    DELETE_CACHE lo eliminará más tarde.
    """
    try:
        Path(file_path).unlink(missing_ok=True)
    except OSError:
        pass


def _highlights(report: Report) -> list[Highlight]:
    """Documento completo coloreado por bloque: -1 humano, 0 neutral, +1 IA."""
    highlights: list[Highlight] = []
    for evaluation in report.evaluations:
        if highlights:
            highlights.append(("\n\n", None))
        highlights.append((evaluation.chunk.text, 2 * evaluation.ai_probability - 1))
    return highlights


def build_app(pipeline: DetectionPipeline) -> gr.Blocks:
    """Construye la interfaz sin lanzarla."""
    with gr.Blocks(title="AI-text detector", analytics_enabled=False, delete_cache=DELETE_CACHE) as blocks:
        gr.Markdown(
            "# AI-text detector\n"
            "Runs entirely on this computer: the text never leaves it. "
            "Upload a document or paste text written in Spanish."
        )
        with gr.Row():
            with gr.Column():
                file_input = gr.File(
                    label="Document", file_types=list(SUPPORTED_EXTENSIONS), type="filepath"
                )
                text_input = gr.Textbox(label="Or paste text", lines=10)
                analyze_button = gr.Button("Analyze", variant="primary")
            with gr.Column():
                report_output = gr.Markdown(label="Report", latex_delimiters=[], sanitize_html=True)
        highlights_output = gr.HighlightedText(
            label="Paragraphs coloured by AI likelihood (red = AI, blue = human)",
            show_legend=True,
            combine_adjacent=False,
        )
        analyze_button.click(
            make_handler(pipeline),
            inputs=[file_input, text_input],
            outputs=[report_output, highlights_output, file_input],
        )
    return blocks


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Offline AI-text detector (local web UI).")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port on 127.0.0.1 (default {DEFAULT_PORT})")
    args = parser.parse_args(argv)

    # Umbral (y pesos/referencias si la calibración fue supervisada) de calibration.json.
    if config.CALIBRATION_FILE.exists() and load_calibration(config.CALIBRATION_FILE) is None:
        print(
            f"Ignoring {config.CALIBRATION_FILE.name}: it is outdated or invalid for the current scoring; "
            "run scripts/calibrate.py again. Using the default threshold."
        )
    settings = resolve_settings(config.CALIBRATION_FILE)
    pipeline = DetectionPipeline(
        evaluator=EvaluatorAgent(**settings.evaluator_kwargs), reporter=ReporterAgent(threshold=settings.threshold)
    )
    app = build_app(pipeline)
    # En Gradio 6 el tema se pasa a launch() (por eso requirements.txt exige gradio>=6.0);
    # es el que garantiza que el navegador no descarga fuentes de internet.
    app.queue().launch(server_name=SERVER_NAME, server_port=args.port, share=False, theme=THEME)


if __name__ == "__main__":
    main()
