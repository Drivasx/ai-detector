# Feature: detection-quality — fix under-detection of AI text

Objective: AI-generated Spanish text must score clearly above the threshold without raising false positives. Branch: `feat/detection-quality` (from `main` 21efe09). Delivery: single PR (forecast < 400 lines per task). Test runner: `pytest` (`.venv`).

## Specs

- **S1 — Reported failure.** User: "Probe pasandole un texto de 100 palabras generado por una IA y me dio una probabilidad de 41% generado por IA".
- **S2 — Retuned defaults (quick fix).** User chose "Ambas (Recommended)": "Ahora: nuevas referencias (perplejidad 120→25, burstiness 3,6→2,6) y clasificador al 10% como valores por defecto."
- **S3 — Supervised calibration.** Same choice: "Después: calibración supervisada con una carpeta de textos tuyos y otra de textos IA, que ajusta pesos y referencias a tu caso."
- **S4 — Binoculars.** User: "implementa binoculars" (after asking to improve precision further).
- Assumptions (mine, not requested): weights 0.60/0.10/0.30 for perplexity/classifier/linguistic; supervised fitting uses per-chunk medians for the perplexity/burstiness references and AUC-based weights (no new dependency); calibrations without an AI folder keep working as before.

## Tasks

| ID | Specs | Route | Status | Evidence |
|----|-------|-------|--------|----------|
| T1 | S1, S2 | inline | done | 4e0a7dc; 181 passed |
| T2 | S1, S3 | inline | done | 192 passed |
| T3 | S4 | inline | done | Qwen2.5-0.5B base/instruct pair; new `binoculars` vote component |

Route evidence: sequential, dependent units (T2 builds on T1 defaults); no Writer trigger.

## Log

- **L1 (2026-10-17, user verbatim):** "ya hice merge del #13, y ya estoy probando el sistema. Probe pasandole un texto de 100 palabras generado por una IA y me dio una probabilidad de 41% generado por IA"
- **L2 diagnosis (reproduced):** six AI-written ~100-word Spanish samples scored 0.29-0.66 (3/6 >= 0.5); 144 human GNOME help paragraphs: median 0.26, FP@0.5 2.8%. Perplexity separates (human median 144, p25 95; AI 17-69) but the reference 60 -> 15 was a guess from two samples; burstiness separates (human median 3.63; AI 2.57-3.08) but reference 3.2 too low; classifier near-constant (raw ~0.999 for both); linguistic weak (AI 0.27-0.44 vs human median 0.32). Simulation with S2 values: AI 0.43-0.81 (4/6), human median 0.18, FP@0.5 5.6%; formal AI texts (perplexity 59-69) still missed. Caveat: small, unrepresentative samples (one model family; translated technical help) -> overfitting risk, hence S3.
- **L3 (user choice verbatim):** "Ambas (Recommended)".
- **L4 (2026-10-18) user: "Ya se reinicio mi uso, puedes seguir con la T2".** T2 done: `fit_parameters` (class medians for perplexity/burstiness references, AUC-based weights, fallback to defaults with a warning when classes are inverted or nothing separates), `calibrate(..., ai_paths=)`, `calibration.json` stores weights/references/AUC/AI detection rate, `load_calibration`/`resolve_settings` validate them (invalid -> only threshold kept), UI builds the evaluator from them, `scripts/calibrate.py --ai-folder`. RED -> GREEN; 192 passed. Real offline run (15 GNOME human texts, 6 AI samples): threshold 0.610, 4.9% human paragraphs / 40% human documents flagged (documents have ~10 paragraphs, so a 5% paragraph rate flags many documents as "mixed"), 66.7% AI paragraphs detected; AUC perplexity 0.97, classifier 0.87, linguistic 0.69. Only 6 AI paragraphs: not reliable, the script says so. T1 native review declined by the user ("Omitir esta vez").
- **L5 (2026-10-18) native review of 21efe09..05e0669 (lens reliability): user granted consent; APPROVED and acknowledged (lineage review-d2b89c62039fc9bb).** Fixed afterwards (RED 5 failed -> GREEN): a saved threshold depends on the scoring parameters it was computed with, so `calibration.json` now records `scoring` (effective weights + references) and `load_calibration` ignores the whole file when they differ from what will be used (stale defaults, pre-T1 files, or damaged supervised parameters); the UI prints a warning to recalibrate when it ignores an existing file. Dropped: a fallback test that did not reach the `total == 0` branch (that branch stays untested). 195 passed.
- **L6 (2026-10-18) user: "implementa binoculars"; later "continua con lo que estabas haciendo y termina la implementacion de binoculars".** T3: `ai_detector/analysis/binoculars.py` (reference formula log-ppl / cross-ppl, observer Qwen2.5-0.5B, performer Qwen2.5-0.5B-Instruct, 512 tokens), analyzer passes an optional `binoculars` signal, evaluator votes it with a linear ramp between references (1.02 human, 0.81 AI; measured AUC 0.97 on the local samples), weights 0.30/0.40/0.05/0.25, supervised calibration fits its references and stores them in `scoring` (older calibration files are ignored and must be recomputed).
  209 passed (integration with real Qwen models included).
- **L7 (2026-10-18) native review of 21efe09..a2734fb (lens reliability): user granted; APPROVED and acknowledged (lineage review-d5ad35b829c8b3a5).** Non-blocking follow-ups: (W) analysis fails outright when the Qwen models are missing instead of dropping Binoculars; (S) `fit_parameters` returns all defaults when perplexity/burstiness medians are inverted, without fitting Binoculars; (S) the `total == 0` branch is untested.
- **L8 (2026-10-18) user: "sí, arreglá el punto 1".** `BinocularsScorer.score` catches `ModelNotAvailableError` once, warns "Binoculars is disabled: ..." and returns None from then on, so the other signals vote. RED 1 failed -> GREEN; 210 passed.
- **L9 (2026-10-18) native review of 21efe09..0f0b220 (lens reliability): user granted; APPROVED and acknowledged (lineage review-1c1c29d2be2d4536).** Non-blocking follow-ups: (W) `calibrate` treats an empty `ai_paths` list as unsupervised; (W) `BinocularsScorer.score` runs the models before checking for zero tokens and only catches `ModelNotAvailableError`; (S) same `fit_parameters` notes as L7.
- **L10 (2026-10-18) user: "si, arreglalo ahora".** `calibrate` only treats `ai_paths=None` as unsupervised (an empty AI folder now raises "No usable AI texts..."); `BinocularsScorer.score` skips the models when there are fewer than two tokens and disables itself on any load failure (corrupt checkpoint included), not only on missing models. RED 3 failed -> GREEN; 213 passed.
