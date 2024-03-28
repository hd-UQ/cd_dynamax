# cd-dynamax

Codebase to extend dynamax to deal with irregular sampling, via continuous-discrete dynamics modeling

## Continuous-discrete state-space models

[TODO: motivate problem set- up]

## Codebase status

- We have implemented [continuous-discrete linear and non-linear models](./src/README.md), along with filtering and smoothing algorithms.

- We are leveraging [dynamax](https://github.com/probml/dynamax) code
    - Currently, based on a [dynamax pull at version '0.1.1+147.g3ad2ac5'](./dynamax)
        - Synching and updates to new dynamax version is PENDING

## Conda environment

- We provide a working conda environment
    - with dependencies installed using the pip-based requirements file

```bash
$ conda create --name hduq_nodynamax python=3.11.4
$ conda activate hduq_nodynamax
$ conda install pip
$ pip install -r hduq_pip_nodynamax_requirements.txt
```
