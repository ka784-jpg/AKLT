# Physics and numerical methods

## What is being simulated?

Each physical site is a spin-1 degree of freedom with three basis states.
The AKLT state is a useful starting point for studying one-dimensional
symmetry-protected topological order. The paper asks how competing local
projections change string order, dimer order, and entanglement.

For two spin-1 sites, total spin is `J=0,1,2`. The AKLT update retains the
`J=0` and `J=1` subspaces and removes `J=2`:

```math
\begin{aligned}
P_{\mathrm{AKLT}} &= I-P^{J=2} = P^{J=0}+P^{J=1},\\
|\psi'\rangle &= \frac{P_{\mathrm{AKLT}}|\psi\rangle}{\left\|P_{\mathrm{AKLT}}|\psi\rangle\right\|}.
\end{aligned}
```

If `B = S_i . S_j`, the same projector is `I - B/2 - B^2/6`.
It is Hermitian, idempotent, and rank four. These are executable invariants in
the test suite, not just assumptions in the documentation.

The control randomness chooses **which projector to apply**. Outcomes are
forced and normalized, not sampled according to their Born probabilities.
This distinction matters: the code does not include the probability cost of
postselection or an experimental feedback implementation.

## Three competing-projector models

| Model | Competes with the NN AKLT projector | Diagnostic |
| --- | --- | --- |
| SPT to trivial | On-site projection onto `S^z=0` | String order and entanglement |
| SPT to dimer | Total-spin-zero projection on selected odd bonds | String order, dimer order, and entanglement |
| NN to NNN AKLT | AKLT projection on `(i,i+2)` rather than `(i,i+1)` | Two string observables, entanglement, and relaxation |

The NNN endpoint consists of two interwoven Haldane chains. The finite-size
NN-NNN data are consistent with an intermediate extended critical regime;
the repository does not establish its asymptotic field theory.

## Observables and conventions

The NN string observable has one spin operator at each endpoint:

```math
\mathcal{O}_{\mathrm{nn}}^{\mathrm{str}}(k)=\left\langle S_i^z\exp\!\left(i\pi\sum_{j=i+1}^{i+k-1}S_j^z\right)S_{i+k}^z\right\rangle.
```

The NNN observable has **two adjacent endpoint spins on each side**:

```math
\mathcal{O}_{\mathrm{nnn}}^{\mathrm{str}}(k)=\left\langle S_i^z S_{i+1}^z\exp\!\left(i\pi\sum_{j=i+2}^{i+k-2}S_j^z\right)S_{i+k-1}^z S_{i+k}^z\right\rangle.
```

These are different operators, not a change of spacing in the same endpoint
formula. The paper plots the magnitude of the averaged signed observable,
`abs(mean(O))`, not `mean(abs(O))`. The reference endpoint magnitudes are `4/9`
for NN AKLT order and `(4/9)^2` for NNN AKLT order.

Other conventions to preserve:

- Campaign inputs often use `p_nn`; paper plots usually use the competing
  probability `p_nnn`, `p_D`, or `p_TT`, equal to `1-p_nn`.
- Stationary runners count sweeps. Dynamics runners can count single gate
  slots. One round contains `L` slots.
- Half-chain entropy is obtained from Schmidt values. Check the logarithm
  base: the Renyi worker explicitly exports both bits and nats.
- A short seeded trajectory is not a stationary ensemble, and spatial
  samples from one trajectory are not independent trajectories.

## Two numerical representations

**Exact statevectors.** A flat `ComplexF64` vector stores `3^L` amplitudes.
The NN-NNN engines apply local 9-by-9 projectors directly using index arithmetic
and reusable scratch buffers. They do not construct a dense `3^L`-by-`3^L`
operator for each update. Dense embedding is used only in the independent
small-system tests. Entropy calculations require a bipartition and SVD.

**Matrix product states.** The TT/dimer and selected diagnostics use ITensors,
ITensorMPS, and the included QuantumCircuitsMPS2 snapshot. Bond dimensions and
truncation cutoffs control the approximation and resource cost. Increasing
system size is not automatically cheap: entanglement can drive bond growth.

These are preserved research campaign implementations, with differing launcher
signatures and observable formats. The new [quickstart](../examples/README.md)
provides one checked interface without claiming that all campaigns form a
uniform general-purpose library.

For the derivations, finite-size analyses, and full uncertainty estimates, see
the [paper](https://arxiv.org/abs/2609.26941). For source-to-figure evidence,
see [FIGURE_PROVENANCE.md](../FIGURE_PROVENANCE.md).
