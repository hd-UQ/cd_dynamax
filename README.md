# hybrid_dynamics_uq
Work on hybrid modeling of dynamical systems and their uncertainty quantification

# Codebase

- We are leveraing [dynamax](https://github.com/probml/dynamax) code

- Instead of forking it, we just try by creating a mirror copy of the repository at commit [43f8d3d](https://github.com/probml/dynamax/commit/43f8d3d52bdd4a946e7a504b12a9ddc97e19803b)
```bash
$ git clone --bare git@github.com:probml/dynamax.git
$ cd dynamax.git/
$ git push --mirror git@github.com:iurteaga/hybrid_dynamics_uq.git
$ rm -rf dynamax.git
$ git pull --allow-unrelated-histories
```

- Then we rename dynamax' README to [README_dynamax.md](./README_dynamax.md), and create our own (i.e., this document)

# Working environment

## Conda environment using conda install (UNSUCCESSFUL)

Environment creation (based on Python 3.10.6)

```bash
$ conda create -n hduq
$ conda activate hduq
```

Installation of packages

- Based on [dynamax requirements](https://github.com/probml/dynamax/blob/main/setup.cfg#L21)
```bash
$ conda install jax jaxlib optax
$ conda install scikit-learn matplotlib
$ conda install fastprogress
$ conda install -c conda-forge tensorflow-probability
$ conda install -c conda-forge typing-extensions
```

$ conda install -c conda-forge tensorflow-probability='0.19.0'

- To run the notebooks
```bash
$ conda install jupyter
```


## Conda environment with pip install

- Main conda environment
```bash
$ conda create -n hduq_pip
$ conda activate hduq_pip
```

- Main installation via pip
```bash
$ conda install pip
$ pip install dynamax
conda install typing_extensions
```

- Pip install to run the notebooks 
```bash
$ pip install jupyter matplotlib seaborn flax blackjax graphviz scipy
```

- To replicate
```bash
$ conda list --export > hduq_requirements.txt
```
