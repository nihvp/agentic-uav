# Paper reproduction results

## Paper metrics

| metric | rule | agentic_qwen3-vl-8b | agentic_qwen3-vl-8b_nothink |
|---|---|---|---|
| samples | 44 | 44 | 44 |
| processing time, mean (SD) s | 0.018 (0.006) | 21.638 (10.102) | 23.183 (23.325) |
| processing time, median s | 0.016 | 19.645 | 18.762 |
| detection confidence, mean (SD) | 0.899 (0.020) | 0.940 (0.023) | 0.937 (0.022) |
| persons detected | 44/44 (100.0%) | 44/44 (100.0%) | 43/44 (97.7%) |
| ARR (any action recommended) | 0/44 (0.0%) | 44/44 (100.0%) | 43/44 (97.7%) |
| CAR (context > 50 chars) | 0/44 (0.0%) | 44/44 (100.0%) | 43/44 (97.7%) |

## Correctness against ground truth (not in the paper)

| metric | rule | agentic_qwen3-vl-8b | agentic_qwen3-vl-8b_nothink |
|---|---|---|---|
| situation accuracy | n/a | 44/44 (100.0%) | n/a |
| emergency recall | n/a | 22/22 (100.0%) | n/a |
| false-alarm rate (normal scenes) | n/a | 0/22 (0.0%) | n/a |
| correct response | 22/44 (50.0%) | 44/44 (100.0%) | 43/44 (97.7%) |
| person count, mean abs error | 0.36 | 0.20 | 0.21 |
| person count, mean error (bias) | -0.36 | 0.02 | -0.02 |
| YOLO count, mean abs error | 0.36 | 0.36 | 0.36 |
| unparseable replies | 0 | 0 | 1 |

## LLM diagnostics

| metric | rule | agentic_qwen3-vl-8b | agentic_qwen3-vl-8b_nothink |
|---|---|---|---|
| LLM time, mean s | n/a | 21.60 | 23.15 |
| completion tokens, mean | n/a | 2139 | 2261 |
| thinking text, mean chars | n/a | 8290 | 8817 |

## Tests (alpha 0.05, * = significant)

**processing time**
- Shapiro-Wilk p: rule <0.001 *, agentic_qwen3-vl-8b 0.026 *, agentic_qwen3-vl-8b_nothink <0.001 *; Levene p <0.001 *; ANOVA p <0.001 *
- rule vs agentic_qwen3-vl-8b: Tukey p <0.001 *, Mann-Whitney p <0.001 * (used: mann_whitney), Cohen's d -3.03, rank-biserial r -1.00
- rule vs agentic_qwen3-vl-8b_nothink: Tukey p <0.001 *, Mann-Whitney p <0.001 * (used: mann_whitney), Cohen's d -1.40, rank-biserial r -1.00
- agentic_qwen3-vl-8b vs agentic_qwen3-vl-8b_nothink: Tukey p 0.874, Mann-Whitney p 0.465 (used: mann_whitney), Cohen's d -0.09, rank-biserial r 0.09

**detection confidence**
- Shapiro-Wilk p: rule 0.344, agentic_qwen3-vl-8b <0.001 *, agentic_qwen3-vl-8b_nothink n/a; Levene p 0.714; ANOVA p <0.001 *
- rule vs agentic_qwen3-vl-8b: Tukey p <0.001 *, Mann-Whitney p <0.001 * (used: mann_whitney), Cohen's d -1.92, rank-biserial r -0.76
- rule vs agentic_qwen3-vl-8b_nothink: Tukey p <0.001 *, Mann-Whitney p <0.001 * (used: mann_whitney), Cohen's d -1.84, rank-biserial r -0.73
- agentic_qwen3-vl-8b vs agentic_qwen3-vl-8b_nothink: Tukey p 0.846, Mann-Whitney p 0.632 (used: mann_whitney), Cohen's d 0.11, rank-biserial r 0.05

**persons detected**
- chi-square across systems p 0.365
- rule vs agentic_qwen3-vl-8b: identical p 1.000
- rule vs agentic_qwen3-vl-8b_nothink: fisher p 1.000
- agentic_qwen3-vl-8b vs agentic_qwen3-vl-8b_nothink: fisher p 1.000

**ARR**
- chi-square across systems p <0.001 *
- rule vs agentic_qwen3-vl-8b: chi2 p <0.001 *
- rule vs agentic_qwen3-vl-8b_nothink: chi2 p <0.001 *
- agentic_qwen3-vl-8b vs agentic_qwen3-vl-8b_nothink: fisher p 1.000

**CAR**
- chi-square across systems p <0.001 *
- rule vs agentic_qwen3-vl-8b: chi2 p <0.001 *
- rule vs agentic_qwen3-vl-8b_nothink: chi2 p <0.001 *
- agentic_qwen3-vl-8b vs agentic_qwen3-vl-8b_nothink: fisher p 1.000

**correct response**
- chi-square across systems p <0.001 *
- rule vs agentic_qwen3-vl-8b: chi2 p <0.001 *
- rule vs agentic_qwen3-vl-8b_nothink: chi2 p <0.001 *
- agentic_qwen3-vl-8b vs agentic_qwen3-vl-8b_nothink: fisher p 1.000

