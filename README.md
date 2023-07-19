# hybrid_dynamics_uq
Work on hybrid modeling of dynamical systems and their uncertainty quantification

# Codebase

- We are leveraing [dynamax](https://github.com/probml/dynamax) code: instead of forking, just by creating a mirror copy of the repository
```bash
$ git clone --bare git@github.com:probml/dynamax.git
$ cd dynamax.git/
$ git push --mirror git@github.com:iurteaga/hybrid_dynamics_uq.git
$ rm -rf dynamax.git
$ git pull --allow-unrelated-histories
```

- Then we rename dynamax' README to [README_dynamax.md](./README_dynamax.md), and create our own (i.e., this document)
