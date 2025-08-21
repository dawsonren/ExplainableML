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

Make sure to drag the `.so` output into `ale` after compilation.

## Running Tests
Run
```bash
python -m unittest discover tests
```