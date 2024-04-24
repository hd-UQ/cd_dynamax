[MS Data-Driven Learning of Dynamical Systems from Partial Observations](https://meetings.siam.org/sess/dsp_programsess.cfm?sessioncode=80523)

- [Submission page](https://meetings.siam.org/Showenter.cfm)


# Hierarchical learning of partially observed interrelated dynamical systems

We present a novel hierarchical Bayesian inference toolbox that leverages differentiable data-assimilation to learn a set of interconnected, partially observed stochastic dynamical systems.

Contrary to the common assumption that each observed data trajectory represents an independent realization of a different (unknown) dynamical system, we study the case where each observed trajectory is unique, yet related to a set of observed, akin trajectories. This set-up underpins many real-world time-series data (e.g., in biomedicine and engineering), as it accommodates the idiosyncrasies of each individual time-series, yet incorporates the similarities across a group of interrelated time-series. Namely, each dynamical system trajectory is idiosyncratic (the individual), yet shares commonalities with other observed trajectories (the population).

We merge hierarchical Bayesian modeling with differentiable data-assimilation to efficiently tackle the challenges of filtering, smoothing, predicting, and system identification in these scenarios. We show how the proposed modeling framework and the presented toolbox enables efficient and accurate Bayesian inference of interrelated, linear and/or nonlinear continuous-discrete, partially observed dynamical systems.
