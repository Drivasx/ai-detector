# ai-detector

Offline, CPU-only detector of AI-generated text, focused on Spanish documents.
No text ever leaves the machine: models are downloaded once, cached in `models/`,
and every analysis runs with `HF_HUB_OFFLINE=1`.

## Pipeline

| Agent | Module | Responsibility |
|-------|--------|----------------|
| Reader | `ai_detector/agents/reader.py` | Read `.txt`, `.pdf`, `.docx` into clean plain text |
| Segmenter | `ai_detector/agents/segmenter.py` | Split text into paragraphs / logical chunks |
| Analyzer | `ai_detector/agents/analyzer.py` | Perplexity, pretrained classifier, linguistic features |
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
```

## Tests

```bash
pytest
```

## Status

- [x] Project scaffold
- [ ] Reader Agent
- [ ] Segmenter, Analyzer, Evaluator, Reporter, UI, calibration
