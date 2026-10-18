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
| T3 | S4 | inline | pending — waits for user OK (S12) | |
| T4 | S5, S1, S8 | inline | pending | |
| T5 | S6 | inline | pending | |
| T6 | S7 | inline | pending | |
| T7 | S9 | inline | pending | |
| T8 | S10 | inline | pending | |
| T9 | S8 | inline | pending | |

Route evidence: no Writer trigger (agents are sequential by S12, not parallel units); work stays inline.

## Log

- **L1 (2026-10-17, user request verbatim):** "sí, inicializá el repo git" + pasted prompt: "You are an expert Python developer and NLP engineer. Help me build a local AI-text detector application that runs entirely offline on a CPU-only machine (Intel Core i5 7200U, 16 GB DDR3 RAM, no dedicated GPU). The tool must analyze texts written primarily in Spanish and must never send any text to external services or APIs. [...] Start by scaffolding the full project structure with folders, empty modules, and a requirements.txt. Then implement one agent at a time, starting with the Reader Agent. Ask me before moving to the next agent." (full requirement fragments quoted in S1–S12).
- **L2 decisions:** identifiers, messages and docs in English; comments and docstrings in Spanish (S11). Only Python 3.14 is available on the machine; torch/transformers installation on 3.14 is not yet verified (checked in T4). Install torch from the CPU wheel index to avoid CUDA builds.
- **L3 open question for T4:** xlm-roberta-base is a masked LM, so it yields pseudo-perplexity (one forward pass per masked token, slow on CPU); a small causal LM may be cheaper and more faithful. Decide with the user before T4.
- **L4 (2026-10-17) T1+T2 done.** Scaffold 7e4cab5; Reader 09aeeaf. RED: collection ImportError; GREEN: `pytest` 18 passed. Smoke test on a real PDF works (pypdf may insert kerning spaces, e.g. "T est", a known extraction limitation). Native assess (base ae1976c, committed-only): risk medium, review_due false, reason under_budget — slice stays open. Next: T3 Segmenter after user OK (S12).
