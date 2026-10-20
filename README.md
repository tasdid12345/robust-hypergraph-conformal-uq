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

## Reference

This project is motivated by the following paper:

> A Choudhuri, B Adhikari (2026). *Learning Robust Hypergraph Embeddings for Distribution-Free Uncertainty Quantification*. Proceedings of the 32nd ACM SIGKDD. https://dl.acm.org/doi/abs/10.1145/3770855.3818095

Paper author: Prof. Bijaya Adhikari, University of Iowa (bijaya-adhikari@uiowa.edu)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
