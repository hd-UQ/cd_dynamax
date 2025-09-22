## 1) Lorenz–63 via polynomial dictionary + Laplace prior  
**File:** `demos/l63_LaplaceDict.py`  
**Goal:** Learn a sparse polynomial drift for L63 from a single trajectory.

### Generative model
Let $x(t)\in\mathbb{R}^3$ be the latent state and $y_k\in\mathbb{R}^{E}$ the observation (default $E=3$).  
Build a dictionary $\Phi(x)\in\mathbb{R}^{P}$ (monomials up to degree `--poly_degree`, default 2).  
With weights $W\in\mathbb{R}^{3\times P}$:

$$
\begin{aligned}
\text{Drift:} &\quad f_W(x) = W \Phi(x) \\
\text{Dynamics:} &\quad dx = f_W(x)\,dt + L\,dw(t), \quad L = \sigma I \\
\text{Obs:} &\quad y_k \sim \mathcal{N}(x(t_k), \sigma_{\text{obs}}^2 I)
\end{aligned}
$$

### Priors
$$
W_{ij} \sim \text{Laplace}(0 \ \lambda) \\
\sigma \sim \text{Uniform}(\sigma_{\min} \ \sigma_{\max}) \\
\sigma_{\text{obs}} \sim \text{Uniform}(\sigma_{\text{obs,min}} \ \sigma_{\text{obs,max}})
$$

### Inference
- Filtering likelihood: **EnKF**  
- Inference: **MAP** via SVI with `AutoDelta`.  

### Example
```bash
python ./cd_dynamax/demos/l63_LaplaceDict.py --poly_degree 2 --num_epochs 2000
```

---

## 2) Lorenz–96 via polynomial dictionary + Laplace prior  
**File:** `demos/L96_LaplaceDict.py`  
**Goal:** Learn a sparse polynomial drift for L96 from a single trajectory.

### Generative model
Let $x(t)\in\mathbb{R}^D$ be the latent state and $y_k\in\mathbb{R}^{E}$ the observation.  
Build a dictionary $\Phi(x)\in\mathbb{R}^{P}$ (monomials up to degree `--poly_degree`).  
With weights $W\in\mathbb{R}^{D\times P}$:

$$
\begin{aligned}
\text{Drift:} &\quad f_W(x) = W \Phi(x) \\
\text{Dynamics:} &\quad dx = f_W(x)\,dt + L\,dw(t), \quad L = \sigma I \\
\text{Obs:} &\quad y_k \sim \mathcal{N}(x(t_k), \sigma_{\text{obs}}^2 I)
\end{aligned}
$$

### Priors
$$
W_{ij} \sim \text{Laplace}(0 \ \lambda) \\
\sigma \sim \text{Uniform}(\sigma_{\min} \ \sigma_{\max}) \\
\sigma_{\text{obs}} \sim \text{Uniform}(\sigma_{\text{obs,min}} \ \sigma_{\text{obs,max}})
$$

### Inference
- Filtering likelihood: **EnKF**  
- Inference: **MAP** via SVI with `AutoDelta`.  

### Example
```bash
python ./cd_dynamax/demos/L96_LaplaceDict.py --poly_degree 2 --num_epochs 2000
```

---

## 3) Linear SDE from multiple i.i.d. trajectories  
**File:** `demos/LinearGaussian_MultiTraj.py`  
**Goal:** Learn a single linear drift matrix from many repeated noisy experiments.

### Generative model
$$
dx(t) = A x(t)\,dt + L\,dw(t), \quad L = \sigma I \\
y_k \sim \mathcal{N}(H x(t_k), \sigma_{\text{obs}}^2 I)
$$

### Priors
$$
A_{ij} \sim \text{Uniform}(-a \ a) \\
\sigma \sim \text{Uniform}(\sigma_{\min} \ \sigma_{\max}) \\
\sigma_{\text{obs}} \sim \text{Uniform}(\sigma_{\text{obs,min}} \ \sigma_{\text{obs,max}})
$$

### Inference
- Filtering likelihood: **Kalman filter**  
- Inference: **MAP** (SVI + AutoDelta) or **posterior** (NUTS).  

### Example
```bash
python ./cd_dynamax/demos/LinearGaussian_MultiTraj.py --emission_dim 5 --state_dim 5 --N_trajectories 30
```

---

## 4) Hierarchical learning of linear SDEs  
**File:** `demos/LinearGaussian_MultiTraj_Hier.py`  
**Goal:** Model heterogeneity across trajectories by giving each one its own bias term $b_i$.

### Generative model
For trajectory $i$:
$$
dx_i(t) = A (x_i(t) - b_i)\,dt + L\,dw_i(t), \quad L = \sigma I \\
y_{i,k} \sim \mathcal{N}(H x_i(t_k), \sigma_{\text{obs}}^2 I)
$$

Hierarchical priors on biases:
$$
\mu_d \sim \text{TruncNormal}(m_\mu \ s_\mu \ \text{low}=\epsilon) \\
\sigma_d \sim \text{TruncNormal}(m_\sigma \ s_\sigma \ \text{low}=\epsilon) \\
b_i \sim \text{TruncNormal}(\mu \ \sigma \ \text{low}=\epsilon)
$$

### Inference
- Filtering likelihood: **Kalman filter**  
- Inference: **SVI** with AutoMVN over global params and individual $b_i$  
- Supports Empirical Bayes: first fit population-level $(\mu \ \sigma)$ then refit individuals  

### Example
```bash
python ./cd_dynamax/demos/LinearGaussian_MultiTraj_Hier.py --emission_dim 5 --state_dim 5 --N_trajectories 30
```
