# GDE_Guided_Differential_Evolution
GDE in python .


Guided Differential Evolution (GDE)
A bimodal DE framework with historical population guidance for global numerical optimization.
This repository contains the official Python implementation of Guided Differential Evolution (GDE), a novel variant of Differential Evolution (DE) that addresses the exploration–exploitation trade-off through two core mechanisms:
Bimodal Phase Scheduling — A deterministic alternation between an exploration-oriented stage (DE/rand/1 with high scaling factor) and a guided exploitation stage that leverages a sliding-window history of the population.
Historical Centroid Guidance — A lightweight mutation operator that steers candidate solutions toward the population centroid computed over the last ω generations, reducing stochastic fluctuations and stabilizing convergence.
Key Features
Clean, modular implementation of GDE alongside baseline algorithms (DE/rand/1, PSO, JADE, SaDE).
Ready-to-run comparison scripts on standard benchmark functions (e.g., Griewank, CEC'2017 suite).
Reproducible experimental setup: 50-dimensional problems, 30 independent runs, configurable population size and generation budget.
Quick Start
bash
git clone https://github.com/espartan0007/GDE.git
cd GDE
python main.py
Citation
If you use this code, please cite:
Valdivia, A., et al. Guided Differential Evolution through history sliding population. (CEC'2017 benchmark evaluation).
