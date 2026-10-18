# Feature: detection-quality — fix under-detection of AI text

Objective: AI-generated Spanish text must score clearly above the threshold without raising false positives. Branch: `feat/detection-quality` (from `main` 21efe09). Delivery: single PR (forecast < 400 lines per task). Test runner: `pytest` (`.venv`).

## Specs

- **S1 — Reported failure.** User: "Probe pasandole un texto de 100 palabras generado por una IA y me dio una probabilidad de 41% generado por IA".
- **S2 — Retuned defaults (quick fix).** User chose "Ambas (Recommended)": "Ahora: nuevas referencias (perplejidad 120→25, burstiness 3,6→2,6) y clasificador al 10% como valores por defecto."
- **S3 — Supervised calibration.** Same choice: "Después: calibración supervisada con una carpeta de textos tuyos y otra de textos IA, que ajusta pesos y referencias a tu caso."
- Assumptions (mine, not requested): weights 0.60/0.10/0.30 for perplexity/classifier/linguistic; supervised fitting uses per-chunk medians for the perplexity/burstiness references and AUC-based weights (no new dependency); calibrations without an AI folder keep working as before.

## Tasks

| ID | Specs | Route | Status | Evidence |
|----|-------|-------|--------|----------|
| T1 | S1, S2 | inline | done | 181 passed |
| T2 | S1, S3 | inline | pending | |

Route evidence: sequential, dependent units (T2 builds on T1 defaults); no Writer trigger.

## Log

- **L1 (2026-10-17, user verbatim):** "ya hice merge del #13, y ya estoy probando el sistema. Probe pasandole un texto de 100 palabras generado por una IA y me dio una probabilidad de 41% generado por IA"
- **L2 diagnosis (reproduced):** six AI-written ~100-word Spanish samples scored 0.29-0.66 (3/6 >= 0.5); 144 human GNOME help paragraphs: median 0.26, FP@0.5 2.8%. Perplexity separates (human median 144, p25 95; AI 17-69) but the reference 60 -> 15 was a guess from two samples; burstiness separates (human median 3.63; AI 2.57-3.08) but reference 3.2 too low; classifier near-constant (raw ~0.999 for both); linguistic weak (AI 0.27-0.44 vs human median 0.32). Simulation with S2 values: AI 0.43-0.81 (4/6), human median 0.18, FP@0.5 5.6%; formal AI texts (perplexity 59-69) still missed. Caveat: small, unrepresentative samples (one model family; translated technical help) -> overfitting risk, hence S3.
- **L3 (user choice verbatim):** "Ambas (Recommended)".
