"""Calibra el umbral de detección con una carpeta de textos humanos (anteriores a 2022) para minimizar falsos positivos.

Analiza cada texto (.txt, .pdf, .docx) con el pipeline completo, offline, y
elige el umbral más bajo que deja como mucho `--target-fpr` de los párrafos
humanos por encima. El resultado se guarda en calibration.json y la interfaz lo
usa automáticamente a partir de entonces.

Con `--ai-folder` (textos generados por IA sobre temas parecidos) la calibración
es supervisada: además del umbral ajusta las referencias y los pesos del voto a
los textos del usuario e informa qué parte de la IA se detecta.

Uso:
    python scripts/calibrate.py CARPETA [--ai-folder CARPETA_IA] [--target-fpr 0.05] [--recursive] [--output calibration.json]
"""

import argparse
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_detector import config  # noqa: E402
from ai_detector.agents.reader import ReaderAgent  # noqa: E402
from ai_detector.calibration import CalibrationError, calibrate, discover_texts, save_calibration  # noqa: E402
from ai_detector.models import ModelNotAvailableError  # noqa: E402
from ai_detector.pipeline import DetectionPipeline  # noqa: E402


def _target_fpr(value: str) -> float:
    try:
        rate = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a number: {value!r}") from exc
    if not 0.0 <= rate < 1.0:
        raise argparse.ArgumentTypeError("must be between 0 (inclusive) and 1 (exclusive)")
    return rate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate the AI-detection threshold with your own human-written texts.")
    parser.add_argument("folder", help="folder with texts you wrote yourself before 2022 (.txt, .pdf, .docx)")
    parser.add_argument(
        "--target-fpr",
        type=_target_fpr,
        default=config.CALIBRATION_TARGET_FPR,
        help=f"maximum share of your paragraphs allowed above the threshold (default {config.CALIBRATION_TARGET_FPR})",
    )
    parser.add_argument(
        "--ai-folder", help="folder with AI-generated texts on similar topics: also fits the voting weights and references"
    )
    parser.add_argument("--recursive", action="store_true", help="also read texts in subfolders")
    parser.add_argument("--output", type=Path, default=config.CALIBRATION_FILE, help="where to save the calibration")
    args = parser.parse_args(argv)

    try:
        paths = discover_texts(args.folder, recursive=args.recursive)
        ai_paths = discover_texts(args.ai_folder, recursive=args.recursive) if args.ai_folder else None
        total = len(paths) + len(ai_paths or [])
        print(f"Analyzing {total} file(s) offline. This can take a while on CPU...", flush=True)
        result = calibrate(paths, DetectionPipeline(), ReaderAgent(), target_fpr=args.target_fpr, ai_paths=ai_paths)
    except (CalibrationError, ModelNotAvailableError) as exc:
        print(f"Calibration failed: {exc}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_calibration(result, args.output)

    for name, reason in result.skipped:
        print(f"Skipped {name}: {reason}")
    print(f"Calibrated on {result.file_count} files, {result.chunk_count} paragraphs.")
    print(f"Threshold: {result.threshold:.3f} (default {config.DETECTION_THRESHOLD})")
    print(
        f"With this threshold, {result.chunk_false_positive_rate:.1%} of your paragraphs and "
        f"{result.document_false_positive_rate:.1%} of your documents would be flagged as AI or mixed."
    )
    if result.ai_detection_rate is not None:
        print(
            f"AI texts: {result.ai_file_count} files, {result.ai_chunk_count} paragraphs; "
            f"{result.ai_detection_rate:.1%} of the AI paragraphs reach the threshold."
        )
        print("Weights: " + ", ".join(f"{name} {weight:.2f}" for name, weight in result.weights.items()))
        print("Separation (AUC, 0.5 = none, 1 = perfect): " + ", ".join(f"{n} {v:.2f}" for n, v in result.auc.items()))
        if result.ai_chunk_count < 20:
            print("Note: fewer than 20 AI paragraphs; add more AI texts for a more reliable fit.")
    for warning in result.warnings:
        print(f"WARNING: {warning}")
    if result.capped:
        print(
            "WARNING: your texts score very high even though they are human-written. The threshold was capped at "
            f"{result.threshold}; the detector is not reliable for your writing style."
        )
    print("Remember: only texts you wrote before 2022 are guaranteed to be free of AI assistance.")
    print(f"Saved to {args.output}. The UI uses it automatically.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
