# Configuration Files for cd-dynamax

This directory contains configuration files used for the cd-dynamax codebase in general, and for the demos in particular.

These configuration files define various parameters and settings for continuous-discrete state space models, filtering and smoothing algorithms, and other functionalities provided by the cd-dynamax library.

Specifically, there are configuration files provided according to the following structure:

```demos/python/configs/
├── README.md                  # This file
├── data/                    # Configuration files for data generation
├── model/                   # Configuration files for model definitions
├── solver/                  # Configuration files for SDE solver settings, as per difrax
├── filter/                  # Configuration files for filtering algorithms
├── fitting/                 # Configuration files for fitting (parameter learning) algorithms
├── prior/                   # Configuration files for prior settings   
```

Users can modify these configuration files to customize the behavior of the cd-dynamax library for their specific use cases.

When running demos or scripts, the appropriate configuration files can be loaded to set up the desired models and algorithms.