# Graph Neural Networks for Uncertainty Quantification in Micromagnetic Modeling

This repository contains research code for studying **uncertainty quantification (UQ)** in **graph neural networks (GNNs)** applied to **micromagnetic simulations**.

The goal of this project is to develop GNN-based surrogate models that predict magnetic properties, such as the **coercivity** of complex microstructures, while simultaneously providing **disentangled uncertainty estimates**, including both **aleatoric** and **epistemic** uncertainty.

The framework enables the analysis of prediction reliability and model confidence in micromagnetic modeling tasks. By executing `main.py`, the code trains the models, evaluates predictive performance, computes uncertainty estimates, and generates corresponding **confidence curves** for assessing the quality of the uncertainty quantification.
