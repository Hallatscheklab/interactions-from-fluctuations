# interactions-from-fluctuations

This is a Python 3 package to infer interaction matrices from count data via different inference methods that assume different kinds of sampling noise.
The functionalities of the package can be accessed via the Python API or via command line. Please refer to the `test/` directory for an example of a basic implementation.

## References
[1] [Inferring Migration Networks with Time-Lagged F2 Statistics, Isacchini et al, 2026](https://www.biorxiv.org/content/10.64898/2026.03.12.710875v1)

[2] [Uncovering heterogeneous intercommunity disease transmission from neutral allele frequency time series, Okada et al, 2025](https://www.pnas.org/doi/10.1073/pnas.2500663122)

## Overview

The `influ` package infers a transition/interaction matrix **A** from allele frequency time-series data. Given observations of allele counts across multiple demes (dimensions) and lineages over time, the package estimates the matrix **A** such that:

```
x(t+1) ≈ A · x(t)
```

where `x(t)` is the vector of allele frequencies across demes at time `t`. The package offers three inference methods, each handling sampling noise differently.

## Input data format

All methods share the same input format. Both `counts` and `totcounts` must be NumPy arrays of shape `(ND, Ntraj, T)`, where:

| Dimension | Description |
|-----------|-------------|
| `ND`    | Number of demes / dimensions |
| `Ntraj` | Number of independent lineages / trajectories |
| `T`     | Number of time points |

- `counts[i, l, t]`: number of observed allele copies in deme `i`, lineage `l`, at time `t`
- `totcounts[i, l, t]`: total number of sequenced alleles (non-missing counts) in deme `i`, lineage `l`, at time `t`

The allele frequencies are computed internally as `freq = counts / totcounts`.

## Inference methods

### F2 — fluctuation-based estimator

**Class:** `influ.F2.F2`

Builds F2 statistics (mean squared allele-frequency differences between pairs of demes) at consecutive time points and solves a constrained least-squares problem via [CVXPY](https://www.cvxpy.org/). Sampling noise is corrected using the Patterson et al. correction factor. The inference is repeated `n` times on random subsets of the data (fraction `fraction`) for robustness, and the results are averaged.

```python
from influ.F2 import F2

infer_F2 = F2(counts, totcounts)
A = infer_F2.infer(chunks_size=1000, n=5, fraction=0.8)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunks_size` | 1000 | Block size for genome partitioning |
| `n` | 5 | Number of bootstrap iterations |
| `fraction` | 0.8 | Fraction of blocks used per iteration |

---

### DMD — Dynamic Mode Decomposition

**Class:** `influ.DMD.DMD`

Fits the linear dynamical system directly on allele frequencies. Two solvers are available:

- **`LS` (Least Squares, default):** Ordinary least squares with non-negativity and row-sum-to-one constraints, solved via `cvxopt`.
- **`TLS` (Total Least Squares):** Accounts for errors in both input and output via orthogonal distance regression (`scipy.odr`).

```python
from influ.DMD import DMD

# Least squares
infer_LS = DMD(counts, totcounts)
A = infer_LS.infer()           # we=1 → LS mode

# Total least squares
infer_TLS = DMD(counts, totcounts)
A = infer_TLS.infer(we=0.9)    # we<1 → TLS mode
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `we` | 1 | Weight for TLS (1 = ordinary LS; <1 = TLS) |
| `lam` | 0 | LASSO regularization parameter |

---

### Kalman — Expectation-Maximization with Kalman filter

**Class:** `influ.Kalman.Kalman`

It models allele frequencies as a latent linear dynamical system observed through noisy count data. Uses an Expectation-Maximization (EM) algorithm with a Kalman filter (E-step) and a constrained quadratic program (M-step) to simultaneously infer **A**, the effective population size `Ne` (controlling genetic drift noise), and the sampling noise amplitude `Csn`.

```python
from influ.Kalman import Kalman

infer_EM = Kalman(counts + 1, totcounts)   # +1 pseudocount recommended
A = infer_EM.infer(
    frac=0.5,
    Ne_old=1000,
    em_step_max=100,
    terminate_th=0.001,
    infer_samplenoise=True,
    noisemode=2,
    ridge=0.0,
    penalty_mode='L2'
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `frac` | 0.5 | Mixing fraction between LS and random initial guess for **A** |
| `Ne_old` | 1000 | Initial effective population size |
| `em_step_max` | 100 | Maximum number of EM iterations |
| `terminate_th` | 0.001 | Convergence threshold on max change in **A** |
| `infer_samplenoise` | True | Whether to infer the sampling noise amplitude |
| `noisemode` | 2 | Heteroscedasticity model (0: local freq, 1: global mean, 2: per-deme mean) |
| `ridge` | 0.0 | Ridge (L2) regularization strength |
| `penalty_mode` | `'L2'` | Regularization type: `'L2'` or `'L1'` |

## Simulation utility

A Wright-Fisher simulator is provided in `influ.utils` to generate synthetic count data for testing:

```python
from influ.utils import WF_sim
import numpy as np

ND = 3           # number of demes
T = 20           # number of time points
Ntraj = 200      # number of lineages

A_true = np.array([[0.8, 0.1, 0.1],
                   [0.1, 0.8, 0.1],
                   [0.1, 0.1, 0.8]])

A, counts, freqs = WF_sim(
    Npop=500,
    counts_per_demeweek=200,
    Csn=[1, 1, 1],
    ND=ND,
    T=T,
    A=A_true,
    Ntraj=Ntraj
)
totcounts = np.ones_like(counts) * 200
```

## Installation

Install from source using pip (recommended):

```sh
pip install .
```

Or using the legacy setup script:

```sh
python setup.py install
```

**Requirements:** Python >= 3.9, `numpy`, `scipy`, `cvxpy`, `cvxopt`

## Installation time

Installing from a clean environment typically takes **under a minute**
(~20–30 s measured on Python 3.12, Linux x86-64), since `influ` itself is
pure Python and all dependencies install as prebuilt wheels.

On platforms where wheels are unavailable (older Python, uncommon
architectures, or no C/Fortran toolchain), `cvxpy`, `cvxopt`, and `scipy`
may build from source, which can extend install time to 10–20+ minutes.
In that case, installing the heavy dependencies via conda first is
recommended:

    conda install -c conda-forge numpy scipy cvxpy cvxopt
    pip install .

**Tested with:** Python 3.12, pip (wheels) — numpy 2.4, scipy 1.17,
cvxpy 1.9, cvxopt 1.3.

## Command-line interface

The package exposes an `influ` command after installation. It reads a CSV file and writes the inferred matrix to stdout.

```sh
influ --method LS -i data.csv
```

**Supported methods:** `LS`, `TLS`, `F2`, `EM`

The input CSV must contain the following columns:

| Column | Description |
|--------|-------------|
| `time` | Time index (0-indexed integer) |
| `lineage` | Lineage index (0-indexed integer) |
| `dimension` | Deme/dimension index (0-indexed integer) |
| `counts` | Observed allele counts |
| `tot_counts` | Total sequenced alleles |

**Options:**

| Flag | Description |
|------|-------------|
| `--method` | Inference method: `LS`, `TLS`, `F2`, `EM` |
| `-i` / `--infile` | Path to input CSV file |
| `-o` / `--outfile` | Path to output file |
| `--seed` | Random seed for reproducibility |

## Python API example

```python
import numpy as np
from influ.F2 import F2
from influ.DMD import DMD
from influ.Kalman import Kalman

# counts and totcounts: shape (ND, Ntraj, T)
# ... load or simulate your data here ...

# F2 method
A_F2 = F2(counts, totcounts).infer()

# Least-squares DMD
A_LS = DMD(counts, totcounts).infer()

# EM / Kalman filter
A_EM = Kalman(counts + 1, totcounts).infer()
```
### Expected run time for the demo

Running the demo in `test/basic_example.ipynb` end-to-end — simulating the
example dataset (5 demes, 200 lineages, 5 time points) and running all four
inference methods (F2, DMD-LS, DMD-TLS, Kalman-EM) — takes **a few seconds
(< 10 s)** on a normal desktop computer. Runtime is dominated by the
Kalman-EM step (~3–4 s); F2 and DMD each finish in well under a second.

## Contributing

If you would like to contribute to the development of this package, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Make your changes and commit them with descriptive messages.
4. Push your changes to your fork.
5. Create a pull request to the main repository.

## Authors

Giulio Isacchini, Takashi Okada and Oskar Hallatschek

## License

This project is licensed under the GNU General Public License. See the `LICENSE` file for more details.
