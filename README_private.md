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

- Later, we simply downloaded to local directory with [Dynamax release 0.1.5](https://github.com/probml/dynamax/releases/tag/0.1.5)

## Github repositories

- We work with 2 separate repositories

- The main, private one [https://github.com/iurteaga/hybrid_dynamics_uq](https://github.com/iurteaga/hybrid_dynamics_uq)
    - When working with private work, simply make sure branches point to such private repository
        ```$ git push --set-upstream origin name_of_private_branch```
    
- The public repository [https://github.com/hd-UQ/cd_dynamax](https://github.com/hd-UQ/cd_dynamax)
    - The first time around, set up the public remote to point to such repo    
        ```$ git remote add public git@github.com:hd-UQ/cd_dynamax.git```
    
    - Then, locally create a public branch, and make sure it points to the public repo
        ```
        $ git checkout public
        $ git pull public
        $ git push --set-upstream public
        ```
    
    - It is better to have a public branch on our private repo, and then push to the public repo.
        ```
        $ git checkout public # This is the local private branch
        $ git pull origin public # Pull from the remote private branch
        $ # Make commits that you want...
        $ git push origin public # Push to the private branch
        $ # Make sure everyone agrees...
        $ git push public public # Push to the public branch
        ```

# Conda environments

## Original conda environment with pip install

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

## Updated definition (for latest packages on November 2024)

- Environment
```bash
$ conda create -n hduq_latest_packages python=3.12.4 
$ conda activate hduq_latest_packages
$ conda install pip typing_extensions
$ pip install jupyter matplotlib seaborn flax graphviz scipy scikit-learn
```

- [Becasue of issues with numpy >2 raised by tensorflow-probability](https://github.com/tensorflow/probability/issues/1814)
```bash
$ pip install "numpy<2.0" tensorflow-probability
```

### Jax in CPU

- Jax and related packages

```
$ pip install -U jax
$ pip install blackjax diffrax optax 
```

- Now, we don't really need to install dynamax, as we are using the codebase, but it helps figure out dependencies, so

```
$ pip install dynamax
$ pip uninstall dynamax
```

Get requirements

```
$ pip freeze > hduq_latest_packages_cpu.txt
```

### Jax with NVIDIA GPU + CUDA

- After making sure Nvidia GPU driver and CUDA is installed in ubuntu

- Jax and related packages
    - in general we would
    
    ```
    $ pip install -U "jax[cuda12]"
    $ pip install blackjax diffrax optax 
    ```

- Now, we don't really need to install dynamax, as we are using the codebase, but it helps figure out dependencies, so
    
    ```
    $ pip install dynamax
    $ pip uninstall dynamax
    ```

    Get requirements
    
    ```
    $ pip freeze > hduq_latest_packages_cuda.txt
    ```

### In MacOS 14.7, with CPU for latest version of packages
```bash
$ conda create --name hduq_latest python=3.12.4
$ conda activate hduq_latest
$ conda install pip
$ pip install -r hduq_latest_packages_mac.txt
```
