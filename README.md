# Poisson GLM Granger
Python implementation and adaptation of Poisson GLM Granger causality
from Kim et al. (2011) A Granger Causality Measure for Point Process 
Models of Ensemble Neural Spiking Activity

<https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1001110>

Here the optimal lag is determined via cross-validation, redundant predictors are
(optionally) removed based on VIF, indirect connections are filtered out, and 
permutation tests are used to assess significance

## Requirements
python version: scikit-learn, joblib, numpy, statsmodels (>=0.14.2)