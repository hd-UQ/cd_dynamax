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

Conda
