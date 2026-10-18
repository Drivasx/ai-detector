# ai-detector

Offline, CPU-only detector of AI-generated text, focused on Spanish documents.
No text ever leaves the machine: models are downloaded once, cached in `models/`,
and every analysis runs with `HF_HUB_OFFLINE=1`.

## Pipeline

| Agent | Module | Responsibility |
|-------|--------|----------------|
| Reader | `ai_detector/agents/reader.py` | Read `.txt`, `.pdf`, `.docx` into clean plain text |
| Segmenter | `ai_detector/agents/segmenter.py` | Split text into paragraphs / logical chunks |
| Analyzer | `ai_detector/agents/analyzer.py` | Perplexity, Binoculars, pretrained classifier, linguistic features |
| Evaluator | `ai_detector/agents/evaluator.py` | Weighted vote into a per-chunk AI probability |
| Reporter | `ai_detector/agents/reporter.py` | Document verdict and most suspicious paragraphs |

Supporting code: `ai_detector/analysis/` (scoring components), `ai_detector/models.py`
(local model cache), `ai_detector/ui/app.py` (Gradio on localhost),
`scripts/calibrate.py` (threshold calibration), `scripts/download_models.py`.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
python scripts/download_models.py   # once, online (~3.4 GB); only model names are sent
```

After the download every analysis runs offline: importing `ai_detector` sets
`HF_HUB_OFFLINE=1` and models are loaded with `local_files_only=True`.

| Component | Model | Notes |
|-----------|-------|-------|
| Perplexity | `DeepESP/gpt2-spanish` | Causal GPT-2 small in Spanish, run in float32 |
| Binoculars | `Qwen/Qwen2.5-0.5B` + `Qwen/Qwen2.5-0.5B-Instruct` | [Binoculars](https://arxiv.org/abs/2401.12070) score: perplexity divided by cross-perplexity between the base and instruct models; ~4 GB of RAM, a few seconds per paragraph on CPU. If these models are missing, a warning is printed and the other signals vote without it |
| Classifier | `pandrei7/autextification-upb-mtl` | AuTexTification 2023 (es/en); architecture re-implemented locally, no remote code |

## Run

```bash
python -m ai_detector.ui.app            # http://127.0.0.1:7860 (localhost only)
python -m ai_detector.ui.app --port 8000
```

Upload a `.txt`, `.pdf` or `.docx` file or paste text. The report shows the
verdict, the AI probability and the most suspicious paragraphs; below it, every
paragraph is coloured by AI likelihood.

The UI never leaves the machine: it listens on `127.0.0.1` only, never creates a
public share link, has Gradio telemetry disabled, uses system fonts (no web
fonts) and renders the report with document text escaped and LaTeX disabled.

## Calibration

False positives depend on each person's writing style. Calibrate the threshold
with texts **you wrote yourself before 2022** (before mass-market LLMs):

```bash
python scripts/calibrate.py ~/my-old-texts             # .txt, .pdf, .docx
python scripts/calibrate.py ~/my-old-texts --recursive --target-fpr 0.02
```

For a much better fit, also give it AI-generated texts on similar topics
(for example, ask an online assistant to write about the same subjects as your
own texts; 20+ paragraphs recommended):

```bash
python scripts/calibrate.py ~/my-old-texts --ai-folder ~/ai-texts
```

With `--ai-folder` the calibration is supervised: it also fits the perplexity
and Binoculars references (median of each class) and the voting weights (by how well each
signal separates your texts from the AI ones, AUC) and reports the share of AI
paragraphs detected.

The script analyzes every paragraph offline and picks the lowest threshold that
flags at most `--target-fpr` (default 5%) of your paragraphs. The threshold is
never lowered below the default (0.5) and is capped at 0.99, with a warning when
your texts score that high. The result is saved to `calibration.json`
(git-ignored) and the UI picks it up automatically; delete the file to go back
to the default.

## Tests

```bash
pytest
```

## Status

- [x] Project scaffold
- [x] Reader Agent
- [x] Segmenter Agent
- [x] Analyzer Agent
- [x] Evaluator Agent
- [x] Reporter Agent
- [x] Gradio UI
- [x] Calibration script
