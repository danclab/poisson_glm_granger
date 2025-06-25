# GLM Granger

Python implementation and generalization of Granger causality for spike train data using Generalized Linear Models (GLMs), adapted from:

> Kim et al. (2011).  
> A Granger Causality Measure for Point Process Models of Ensemble Neural Spiking Activity  
> [PLOS Comput Biol, 7(3):e1001110](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1001110)

## Features

- Supports any `statsmodels` GLM family (e.g. Poisson, Negative Binomial, Binomial)
- Optimal lag selection via K-fold cross-validation
- Optional filtering of indirect connections via causal pathway analysis
- Permutation-based significance testing with FDR correction
- Parallelized computation with `joblib`

## Installation

This package depends on:

- Python 3.8+
- `scikit-learn`
- `joblib`
- `numpy`
- `statsmodels >= 0.14.2`

You can install the required packages using:

```bash
pip install scikit-learn joblib numpy statsmodels
