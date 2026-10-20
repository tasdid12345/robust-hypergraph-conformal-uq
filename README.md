# Conformal Coverage Under Hypergraph Structure Corruption

Conformal prediction gives distribution-free coverage guarantees, but those guarantees assume the calibration data match the test data. When the relational structure of a hypergraph is corrupted at test time, a conformal predictor calibrated on clean structure can lose coverage. The paper "Learning Robust Hypergraph Embeddings for Distribution-Free Uncertainty Quantification" (2026) studies exactly this tension between robust hypergraph representations and distribution-free uncertainty quantification. This experiment asks whether a simple version of that idea, calibrating on noise-perturbed structure, helps in a small, controlled setting.

**Question:** Can split conformal prediction calibrated on noise-perturbed hypergraph structure keep its nominal coverage better than standard clean-calibrated conformal prediction when the test hypergraph is corrupted?

**Dataset:** scikit-learn's built-in digits dataset (loaded with `load_digits`, no download). This is not the paper's own dataset.

**Baseline vs. improved:**
- Baseline: an MLP on hypergraph-smoothed features, with split conformal prediction (LAC score) calibrated on the clean hypergraph.
- Improved: a histogram gradient-boosting classifier trained on clean plus structure-corrupted copies, with conformal calibration pooled over clean and corrupted calibration structures.

The main metric is the absolute gap between coverage on the corrupted test structure and the target coverage. Average prediction-set size, clean-structure coverage and macro F1 are reported as secondary metrics.

## How to run

```bash
pip install -r requirements.txt
python analysis.py
```

## Results

Results are written to `results.json` (and a chart to `figures/coverage_corrupted.png`).

## Results (latest run)

On scikit-learn digits, the clean-calibrated MLP with split conformal reached 80.2% coverage on corrupted test structure (gap 9.8 points), versus 89.0% (gap 1.0 points) for my noise-augmented gradient-boosting variant over 3 seeds, at the cost of larger prediction sets (0.90 vs 0.81).

| Metric | Value |
|---|---|
| baseline_score | 0.09778 |
| improved_score | 0.01037 |
| improvement | 0.08741 |
| baseline_score_std | 0.0137 |
| improved_score_std | 0.01313 |
| baseline_corrupted_coverage | 0.8022 |
| improved_corrupted_coverage | 0.8896 |
| baseline_clean_coverage | 0.8978 |
| improved_clean_coverage | 0.9319 |
| baseline_set_size | 0.8059 |
| improved_set_size | 0.903 |
| baseline_f1 | 0.9626 |
| improved_f1 | 0.9508 |

Run on 2026-10-20. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Single stratified 50/25/25 split; test set used once per seed for final evaluation. No hyper-parameter search was run, so no CV-based selection occurred. Corruption is uniform random, not adversarial. Coverage is averaged over 3 seeds which vary model init and corruption draws; the std is over these seeds. Hypergraph is built transductively over all nodes using features only (no labels).

## Reference

This project is motivated by the following paper:

> A Choudhuri, B Adhikari (2026). *Learning Robust Hypergraph Embeddings for Distribution-Free Uncertainty Quantification*. Proceedings of the 32nd ACM SIGKDD. https://dl.acm.org/doi/abs/10.1145/3770855.3818095

Paper author: Prof. Bijaya Adhikari, University of Iowa (bijaya-adhikari@uiowa.edu)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
