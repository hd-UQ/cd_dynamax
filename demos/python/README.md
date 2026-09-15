# cd-dynamax Python demos

This directory contains Python demos showcasing the usage of the `cd-dynamax` library for continuous-discrete state space models.

The demos illustrate various functionalities, including model creation, filtering, smoothing, and parameter estimation.

We provide both python scripts and Jupyter notebooks for users to explore the examples interactively.

The following directory structure is used:

```
demos/python/
├── README.md                  # This file
├── configs/                   # Configuration files for demos
├── notebooks/                 # Jupyter notebooks for interactive demos
└── scripts/                   # Python scripts for running demos
```

## Getting Started

To run the demos, ensure you have the `cd-dynamax` library installed in your Python environment. 

We offer two complementary ways to exercise the framework: notebooks teach the API, scripts+configs run repeatable experiments.
    
### cd-dynamax demos: notebooks vs. scripts

- Jupyter notebooks in [demos/python/notebooks](./notebooks) 
    - If you prefer interactive exploration, navigate to the [./notebooks](./notebooks) directory and launch Jupyter Notebook or JupyterLab to open and run the notebooks
    - These are code-first, self-contained tutorials. Model, data, filter, and fit are all defined inline in Python (no config files).
    - Good for learning the API or prototyping a new idea.
    - See details in the [./notebooks/README.md](./notebooks/README.md).

- Python [scripts](./scripts/) + [configs](./configs/)
    - If you prefer a more structured, reproducible approach, use the Python scripts in `demos/python/scripts` along with the configuration files in `demos/python/configs`.
    - This is a config-file driven experiment harness.
    - Each experiment component (data, model, filter, etc.) is its own config file; scripts assemble them at runtime.
    - No code changes needed to try a new model/filter/solver/optimizer combination.
    - Navigate to the [./scripts](./scripts) directory and execute the desired script using Python
        - Full details in the [./scripts/README.md](./scripts/README.md).
