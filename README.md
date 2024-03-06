# hybrid_dynamics_uq
Work on hybrid modeling of dynamical systems and their uncertainty quantification

# Codebase

- We are leveraging [dynamax](https://github.com/probml/dynamax) code

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

## Conda environment with pip install

### Definition

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

- Pip install the [diffrax]() library for 
```bash
$ pip install diffrax
```

- To actually use the code within the repo we want to change, then we
```bash
$ pip uninstall dynamax
```

### Replication

- Replicate active conda environment, based on pip
```bash
$ pip freeze > hduq_pip_requirements.txt
$ gedit hduq_pip_requirements.txt
    Remove line referring to certifi
```

- Create copy of conda environment using the pip-based requirements file

```bash
$ conda create --name hduq python=3.11.4
$ conda activate hduq
$ conda install pip
$ pip install -r hduq_pip_requirements.txt
```

- If you have issues with running the notebooks (due to jupyter not knowing about the conda environment), try running:
```bash
$ conda install jupyter
$ pip install -U "jupyter-server<2.0.0"
```
