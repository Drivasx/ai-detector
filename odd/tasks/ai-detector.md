# Feature: ai-detector — offline multi-agent AI-text detector (Spanish)

Objective: local, CPU-only AI-text detector built as an agent pipeline with a Gradio UI. Branch: `feat/ai-detector`. Delivery strategy: `ask-on-risk`. Test runner: `pytest` (in `.venv`).

## Specs

- **S1 — Offline, CPU-only.** "runs entirely offline on a CPU-only machine (Intel Core i5 7200U, 16 GB DDR3 RAM, no dedicated GPU)". "must analyze texts written primarily in Spanish and must never send any text to external services or APIs."
- **S2 — Scaffold.** "Start by scaffolding the full project structure with folders, empty modules, and a requirements.txt."
- **S3 — Reader Agent.** "Reader Agent – reads .txt, .pdf, and .docx files and returns clean plain text."
- **S4 — Segmenter Agent.** "Segmenter Agent – splits the text into paragraphs or logical chunks for individual analysis."
- **S5 — Analyzer Agent.** "Analyzer Agent – runs three lightweight analysis components on each chunk:"
  - "Perplexity scorer: uses a small multilingual model (e.g. xlm-roberta-base or similar, max ~500 MB) to compute per-chunk perplexity. Low, uniform perplexity suggests AI authorship."
  - "Pretrained classifier: loads a small HuggingFace classifier already trained to distinguish human vs. AI text, preferring models with Spanish support."
  - "Linguistic feature analyzer: computes rule-based signals with no model required — sentence length variance, type-token ratio (TTR), frequency of typical AI connectors and filler phrases in Spanish, and information density."
- **S6 — Evaluator Agent.** "Evaluator Agent – combines the three scores per chunk using a weighted voting system and produces a per-chunk AI probability score."
- **S7 — Reporter Agent.** "Reporter Agent – aggregates chunk scores into a document-level verdict, highlights the most suspicious paragraphs, and outputs a human-readable report."
- **S8 — Local model cache.** "All models must be downloaded once and cached locally. Set HF_HUB_OFFLINE=1 after the first run so nothing is ever sent to the internet during analysis."
- **S9 — UI.** "The UI must run on localhost using Gradio."
- **S10 — Calibration.** "Include a calibration script that accepts a folder of known-human texts (written before 2022) and computes the optimal detection threshold to minimize false positives for this specific user."
- **S11 — Code quality.** "The codebase must be clean, modular, and well-commented in Spanish."
- **S12 — Pacing.** "implement one agent at a time, starting with the Reader Agent. Ask me before moving to the next agent."

## Tasks

| ID | Specs | Route | Status | Evidence |
|----|-------|-------|--------|----------|
| T1 | S2, S11 | inline | done | 7e4cab5 |
| T2 | S3, S1, S11 | inline | done | 09aeeaf; RED (ImportError) -> GREEN 18 passed |
| T3 | S4 | inline | done | d19bbcd; RED (ImportError) -> GREEN 39 passed |
| T4 | S5, S1, S8 | inline | done | 8bb39de (linguistic), 26a8e2a (model cache/offline), 68cf28e (perplexity, classifier, agent); 67 passed incl. 7 integration |
| T5 | S6 | inline | pending — waits for user OK (S12) | |
| T6 | S7 | inline | pending | |
| T7 | S9 | inline | pending | |
| T8 | S10 | inline | pending | |
| T9 | S8 | inline | core done in 26a8e2a (download script, offline default); UI/first-run wiring pending | |

Delivery: chain strategy `stacked-to-main` (user choice). Slices: (1) 7e4cab5 scaffold; (2) 09aeeaf reader; (3) d19bbcd segmenter; (4) 8bb39de linguistic; (5) 26a8e2a model cache; (6) 68cf28e analyzer models + agent.

Route evidence: no Writer trigger (agents are sequential by S12, not parallel units); work stays inline.

## Log

- **L1 (2026-10-17, user request verbatim):** "sí, inicializá el repo git" + pasted prompt: "You are an expert Python developer and NLP engineer. Help me build a local AI-text detector application that runs entirely offline on a CPU-only machine (Intel Core i5 7200U, 16 GB DDR3 RAM, no dedicated GPU). The tool must analyze texts written primarily in Spanish and must never send any text to external services or APIs. [...] Start by scaffolding the full project structure with folders, empty modules, and a requirements.txt. Then implement one agent at a time, starting with the Reader Agent. Ask me before moving to the next agent." (full requirement fragments quoted in S1–S12).
- **L2 decisions:** identifiers, messages and docs in English; comments and docstrings in Spanish (S11). Only Python 3.14 is available on the machine; torch/transformers installation on 3.14 is not yet verified (checked in T4). Install torch from the CPU wheel index to avoid CUDA builds.
- **L3 open question for T4:** xlm-roberta-base is a masked LM, so it yields pseudo-perplexity (one forward pass per masked token, slow on CPU); a small causal LM may be cheaper and more faithful. Decide with the user before T4.
- **L4 (2026-10-17) T1+T2 done.** Scaffold 7e4cab5; Reader 09aeeaf. RED: collection ImportError; GREEN: `pytest` 18 passed. Smoke test on a real PDF works (pypdf may insert kerning spaces, e.g. "T est", a known extraction limitation). Native assess (base ae1976c, committed-only): risk medium, review_due false, reason under_budget — slice stays open. Next: T3 Segmenter after user OK (S12).
- **L5 (2026-10-17) user: "sí, seguí con el Segmenter".** T3 done in d19bbcd: `ai_detector/sentences.py` (rule-based Spanish sentence splitter, reused later by the linguistic analyzer) and `SegmenterAgent` (paragraphs, merge below `SEGMENT_MIN_WORDS`=40, balanced split above `SEGMENT_MAX_WORDS`=300 at sentence boundaries, no words lost). RED: collection ImportError; GREEN: 39 passed. Reader->Segmenter smoke test OK. Assess (base ae1976c): medium, review_due false, under_budget; branch at ~934 changed lines, so the PR chain strategy (stacked-to-main / feature-branch-chain) must be decided before the T4 commit. Next: T4 Analyzer after user OK; settle L3 first.
- **L6 (2026-10-17) user: "si sigue".** Decisions for T4 (via question UI, verified against the Hugging Face API):
  - Perplexity model: "DeepESP/gpt2-spanish (Recommended)" — causal GPT-2 small in Spanish, 261 MB weights, MIT. Resolves L3 (xlm-roberta-base is 1.1 GB fp32 and only gives pseudo-perplexity).
  - Classifier: "pandrei7, cabeza propia (Recommended)" — pandrei7/autextification-upb-mtl (xlm-roberta-base encoder + 64-unit MTL head, trained on ~32k Spanish docs, macro F1 65.01 Spanish). Head re-implemented locally and weights loaded from safetensors: no `trust_remote_code`. Its own threshold is 0.9919 on `bot_prob` (skewed scores; handle in T5/T8).
- **L7 (2026-10-17) T4.** User chose PR strategy "Stacked a main (Recommended)". Evidence: torch 2.14.1+cpu and transformers 5.20.1 install on Python 3.14 (resolves L2 risk); models downloaded (1.4 GB in `models/`). Linguistic RED (ImportError) -> GREEN; overlap bug found ("desempeña un papel fundamental" counted twice) -> RED -> fixed. `tests/test_models.py` written after code (no RED); mutation check: removing both offline env vars fails the test (only one is not enough: huggingface_hub also honours TRANSFORMERS_OFFLINE). Found and fixed: (a) AutoTokenizer read the classifier's `auto_map` and prompted to run remote code -> concrete XLMRobertaTokenizerFast; (b) DeepESP weights are float16 -> 13.7 s/chunk on CPU; forced float32 -> ~0.5 s/chunk (RED by mutation: float16 != float32). Offline probe under `unshare -rn` (network unreachable) analyzes successfully. Signals on two sample texts (human vs AI-like): perplexity 42.4 vs 17.0, burstiness 3.14 vs 2.32, linguistic 0.00 vs 0.84, classifier 0.9980 vs 0.9990 (classifier barely separates; scores skewed near its 0.9919 threshold -> T5 must rescale it). Harmless load notice: GPT-2 `attn.masked_bias` UNEXPECTED (legacy buffer).
- **L8 (2026-10-17) independent verifier (item 2) on 26a8e2a + 68cf28e: PASS** on offline guarantee (probes under `unshare -rn` and a socket guard: zero network attempts, even with `HF_HUB_OFFLINE=0` exported, thanks to `local_files_only=True`), no remote code (no `.py` cached, no `transformers_modules` loaded), pinned revisions (only the exact shas resolve), and 67 passed. Advisories applied: `local_files_only=True` on the encoder config load; test count corrected. Advisories kept: `setdefault` does not override a user-exported `HF_HUB_OFFLINE=0` (no egress anyway); `MODELS_DIR` read once at import.
- **L9 (2026-10-17) native review of 26a8e2a (base 8bb39de, lenses risk+reliability): user granted consent; APPROVED and acknowledged (lineage review-88611f26466199bf, authority burned).** 5 non-blocking findings. Addressed now: R3-download-unverified (download_model now fails when pinned files are missing upstream; RED->GREEN) and R3-incomplete-branch-untested (test added; behaviour already existed, so no RED), plus R3-download-no-error-handling (script reports FAILED per model and exits 1; RED->GREEN). Side fix: the script only flips HF_HUB_OFFLINE=0 when run as `__main__`, so importing it never disables offline mode. Kept as later work: R1-001 (DeepESP ships only pickle `pytorch_model.bin`; mitigated by pinned sha + transformers weights_only loading), R3-transformers-bump-unexercised (now exercised by the integration tests). 70 passed.
