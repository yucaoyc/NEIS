### 1. Introduction

#### About this method

We study a non-equilbrium importance sampling (NEIS) method based on ODE flows $\dot X_t = \boldsymbol{b}(X_t)$ to estimate high-dimensional integration $\mathcal Z_1 = \int_{\Omega} e^{-U_1}$ where $\Omega$ is the domain and $U_1$ is a potential function. This NEIS method was initially proposed by [Rotskoff and Vanden-Eijnden (2019)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.122.150602). We study how to achieve the largest variance reduction via optimizing the ODE flows $\boldsymbol{b}$
$$\min_{\boldsymbol{b}} \text{Var}(\boldsymbol{b}),$$
where $\text{Var}(\boldsymbol{b})$ is the variance of Monte Carlo estimators associated with the dynamics $\boldsymbol{b}$.

#### Overview of this package
This Julia package **NEIS** provides implementation of potentials, parameterization of flows, functions to estimate normalizing constants via NEIS, as well as the training scheme to find optimized flow. See documentation (coming soon) for more details.

This package includes:
- README.md: this file.
- src: contains all source codes.
- eg_train: training examples.
- eg_generator: examples to validate that NEIS can be used as a generative model.
- tests: codes for unit testing.
- Project.toml: package dependency.


All dependent packages are listed in *Project.toml* and you may install all via
```Julia
using Pkg
Pkg.activate("./")
Pkg.instantiate()
```
- We assumed that you have matplotlib and python3 installed in your system already.
- Though not necessary, it would be nice if *unbuffer* and *tee* are available (in Linux system),
if you want to run some bash script files (see below).

### 2. How to produce results in the maniscript?
Assuming that you are currently in the main folder of this package:

#### To run all test files
```bash
cd tests; sh test.sh
```

#### To produce training results
```bash
cd eg_train; sh runall.sh
```
This bash script automatically allocates 75% available CPU threads;
you may want to change configurations before running this script.
It took around 12 hours to run on a pc (i7-12700H, 15 threads are being used at maximum) to produce *everything* (both training and comparison with AIS).
It will also automatically creates two sub-folders "assets" and "data" to store images and data files.

#### To produce results about generative models
```bash
cd eg_generator; sh runall.sh
```
These examples are simple 1D or 2D cases to validate that NEIS can be used as a generative model;
see manuscript for details.

### 3. Documentation
More detailed explanation will be included in the folder "docs" (coming later).
