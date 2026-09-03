# Guided Differential Evolution (GDE)
GDE in python .



> **A bimodal Differential Evolution framework with historical population guidance for global numerical optimization.**

This repository contains the official Python implementation of **Guided Differential Evolution (GDE)**, a novel variant of Differential Evolution (DE) that addresses the exploration–exploitation trade-off through two core mechanisms:

* **Bimodal Phase Scheduling:** A deterministic alternation between an exploration-oriented stage (`DE/rand/1` with high scaling factor) and a guided exploitation stage that leverages a sliding-window history of the population.
* **Historical Centroid Guidance:** A lightweight mutation operator that steers candidate solutions toward the population centroid computed over the last $\omega$ generations, reducing stochastic fluctuations and stabilizing convergence.

---

## Key Features

* **Modular Architecture:** Clean, modular implementation of GDE alongside baseline algorithms (`DE/rand/1`, `PSO`, `JADE`, `SaDE`).
* **Benchmarking Ready:** Ready-to-run comparison scripts on standard benchmark functions (e.g., Griewank, CEC'2017 suite).
* **Reproducible Setup:** Pre-configured for 50-dimensional problems across 30 independent runs, with configurable population sizes and generation budgets.

---

## Quick Start

```bash
git clone https://github.com/espartan0007/GDE.git
cd GDE
python main.py

```

---

## Citation

If you use this code in your research, please cite:

> Valdivia, A., et al. *Guided Differential Evolution through history sliding population.* (CEC'2017 benchmark evaluation).
