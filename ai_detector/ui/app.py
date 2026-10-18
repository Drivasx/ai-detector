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
from typing import Callable

import gradio as gr

from ai_detector.agents.reader import SUPPORTED_EXTENSIONS, ReaderError
from ai_detector.agents.reporter import Report, render_markdown
from ai_detector.models import ModelNotAvailableError
from ai_detector.pipeline import DetectionPipeline

SERVER_NAME = "127.0.0.1"
DEFAULT_PORT = 7860

THEME = gr.themes.Default(font=["system-ui", "sans-serif"], font_mono=["ui-monospace", "monospace"])

PASTED_TEXT_NAME = "Pasted text"

# Fragmento del documento con su puntuación, o separador sin etiqueta.
Highlight = tuple[str, float | None]


def make_handler(pipeline: DetectionPipeline) -> Callable[..., tuple[str, list[Highlight]]]:
    """Crea la función que Gradio ejecuta al pulsar "Analyze"."""

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
        except (ReaderError, FileNotFoundError, ModelNotAvailableError) as exc:
            raise gr.Error(str(exc)) from exc
        return render_markdown(report), _highlights(report)

    return handle


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
    with gr.Blocks(title="AI-text detector", analytics_enabled=False) as blocks:
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
            outputs=[report_output, highlights_output],
        )
    return blocks


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Offline AI-text detector (local web UI).")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port on 127.0.0.1 (default {DEFAULT_PORT})")
    args = parser.parse_args(argv)

    app = build_app(DetectionPipeline())
    app.queue().launch(server_name=SERVER_NAME, server_port=args.port, share=False, theme=THEME)


if __name__ == "__main__":
    main()
