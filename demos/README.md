# cd-dynamax Python demos

This directory contains Python demos showcasing the usage of the `cd-dynamax` library for continuous-discrete state space models.

The demos illustrate various functionalities, including model creation, filtering, smoothing, and parameter estimation.

We provide both python scripts and Jupyter notebooks for users to explore the examples interactively.

The following directory structure is used:

```
demos/
├── README.md                  # This file
├── python/                    # Python demos
│   ├── configs/                   # Configuration files for demos
│   ├── notebooks/                 # Jupyter notebooks for interactive demos
│   └── scripts/                   # Python scripts for running demos   
├── numpyro/                   # Numpyro demos (Bayesian parameter inference)
│   ├── notebooks/                 # Jupyter notebooks for interactive demos
│   └── scripts/                   # Python scripts for running demos   
```

## Getting Started

To run the demos, ensure you have the `cd-dynamax` library installed in your Python environment.

Then navigate to the `demos/python/scripts` or `demos/numpyro/scripts` directory and execute the desired script using Python ---see details in [python/scripts/README.md](python/scripts/README.md) and [numpyro/scripts/README.md](numpyro/scripts/README.md).

If you prefer Jupyter notebooks, navigate to the `demos/python/notebooks` or `demos/numpyro/notebooks` directory and launch Jupyter Notebook or JupyterLab to open and run the notebooks ---see details in [python/notebooks/README.md](python/notebooks/README.md) and [numpyro/notebooks/README.md](numpyro/notebooks/README.md).

## Important USER NOTICE:

This README file was written during a transition towards a more user-friendly way to interact with the cd-dynamax library. The new recommended way to use cd-dynamax is used throughout `demos/numpyro` and is also illustrated (without numpyro) in [python/notebooks/lorenz63_filter_based_likelihood_tutorial_newAPI.ipynb](python/notebooks/lorenz63_filter_based_likelihood_tutorial_newAPI.ipynb).

## Highlighted Tutorials

- [SGD-based Neural Network drift fitting tutorial](./numpyro/notebooks/lorenz63_nndrift_sgd_fit_to_data_tutorial_newAPI.ipynb) on how to learn a continuous-discrete SDE drift function using Neural Networks, to fit model to observed data.

- [Filtering-based likelihood tutorial (using new `.build_params` API)](./python/notebooks/lorenz63_filter_based_likelihood_tutorial_newAPI.ipynb) on computing filtering-based likelihoods (and their gradients) for continuous-discrete SDEs.
