*Remark*: This repository is being re-organized. The version before 08/01/2022 can be found in the branch early.
---

### 1. Introduction

We study a non-equilbrium importance sampling (NEIS) method based on ODE flows $\dot X_t = \boldsymbol{b}(X_t)$ to estimate high-dimensional integration $\mathcal Z_1 = \int_{\Omega} e^{-U_1}$ where $\Omega$ is the domain and $U_1$ is a potential function. This NEIS method was initially proposed by [Rotskoff and Vanden-Eijnden (2019)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.122.150602). We study how to achieve the largest variance reduction via optimizing the ODE flows $\boldsymbol{b}$
$$\min_{\boldsymbol{b}} \text{Var}(\boldsymbol{b}),$$
where $\text{Var}(\boldsymbol{b})$ is the variance of Monte Carlo estimators associated with the dynamics $\boldsymbol{b}$.

### 2. What is included in this repository:
- src: contains all source codes (in the form of modules).
- example: training examples.
- result: codes to produce figures and all other results
  - result/assets: figures;
  - result/assets_gen: figures.
- README.md: this file.
- install_pkgs.jl: used to resolve dependence.

#### Code dependence
All depended packages are listed in Project.toml. You may install all packages via 
```
using Pkg; Pkg.instantiate();
```

#### To reproduce training results
For instance, go to the folder "example/test2" and run
```
julia test2_nn.jl
```
The training results are saved as jld2 data files.

#### To reproduce figures
Go to the folder "result" and run two bash scripts:
- "runall_test.sh" helps to obtain all training related results.
- "runall.sh" helps to produce all other figures.

### 3. Training

#### Ansatz type
1. Generic parameterization via neural networks
2. Gradient form
3. Divergence-free form
4. Underdamped Langevin form (the implementation is available but related results are not included in the manuscript)
5. Other self-customizable form

#### Current implements: two approximation methods

| Methods | $T_{-} = 0$ | $T_{-}=-1/2$ |
|---------|-------------|-------------|
| Integration-based   | Supported | Supported |
| ODE-based | Supported | Not supported |

The ODE-based approximation is not yet supported for $T_{-} < 0$, as it is harder to implement and it is expected to have larger computational costs. See the manuscript for details about these methods.

### 4. Anonymous version
- The code is posted via [Anonymous Github](https://anonymous.4open.science/r/NEIS-4DC0).
- Tip: To *clone* the code from Anonymous Github, please see the approach by [Gabin An](https://github.com/agb94/download-anonymous-github).
