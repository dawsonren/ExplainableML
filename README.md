# Explainable ML Experiments

Examine methods for explainable machine learning.

Currently looking at new methods related to Accumulated Local Effects plots.

## Building the Cython extension

This repository ships a small Cython module that speeds up some of the
ALE calculations. To build it, install the requirements and run:

```bash
pip install cython numpy
python setup.py build_ext --inplace
```

The build is optional. If the compiled module is unavailable, the
notebook will automatically fall back to the pure Python implementation.
