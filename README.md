# Measurement-Only Dynamical Phase Transitions in Spin-1 Chains

Julia simulation sources associated with the AKLT manuscript (arXiv:2609.26941).
This collection contains the underlying MPS and exact-statevector simulations,
not the Python figure-generation notebooks or the numerical output archives.

The two original root files, `AKLT_nnn_nn` and `NN-NNN Forced`, are retained
unchanged as historical implementations. Start with `simulations/` for the
campaign sources collected for the current paper.

## What is included

| Directory | Simulation |
| --- | --- |
| `simulations/mps/trivial/` | NN AKLT versus on-site trivial projection; stationary observables and gate-time dynamics |
| `simulations/mps/dimer/` | NN AKLT versus dimer projection; string order, dimer order, sector-resolved entropy, Binder samples, and dynamics |
| `simulations/mps/bond_dimension/` | Bond dimensions for the trivial, dimer, and NN-NNN models |
| `simulations/mps/fidelity/` | Preparation fidelity in the three endpoint limits; fixed-product and random-MPS initializations are separate runners |
| `simulations/statevector/static/` | NN-NNN stationary observables, including entropy |
| `simulations/statevector/order/` | NN and NNN string-order sampling without entropy SVDs |
| `simulations/statevector/dynamics/` | Gate-resolved NN-NNN relaxation used for dynamical scaling |
| `simulations/statevector/relaxation/` | Later saturation campaigns, including multiple entropy cuts |
| `simulations/statevector/renyi/` | Multiple Renyi indices from a shared Schmidt spectrum |

See [FIGURE_PROVENANCE.md](FIGURE_PROVENANCE.md) for the figure-to-code mapping,
campaign evidence, and remaining historical-provenance limitations.

## Conventions

- The physical sites are spin-1 qutrits. These simulations use forced,
  normalized projections, not Born-rule sampling of the projector outcomes.
- Most runners take **`p_nn`**, the probability of choosing the NN AKLT
  projector. Thus `p_nnn = 1 - p_nn`, `p_D = 1 - p_nn`, and
  `p_TT = 1 - p_nn` in the respective competing-projector models.
- Stationary runners count complete sweeps. Gate-time runners count individual
  update slots. Check the runner before comparing time axes; one round has `L`
  slots.
- Preserve the signed trajectory observables. The paper's absolute-value SOP
  convention is applied to the ensemble estimate, not separately to every
  trajectory before averaging.
- Entropy log bases differ between outputs. The Renyi worker explicitly writes
  both `S_bits` and `S_nats`.

## Setup

Use Julia 1.11 or later. From this repository:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. --threads=1 tests/statevector_smoke.jl
```

The local `QuantumCircuitsMPS2` dependency snapshot is included under `vendor/`
with its original BSD 3-Clause license and attribution. It includes its own
circuit-display support because that is part of the package's include graph;
no paper plotting pipeline is included. This dependency snapshot is not a claim
that every historical cluster run used byte-identical package files.

No script here submits cluster jobs. Full production runs can be expensive:
statevector memory grows as `3^L`, and MPS bond dimensions can grow rapidly.
Start with the small examples in [RUNNING.md](RUNNING.md).
Local smoke-test results and their limits are recorded in
[VALIDATION.md](VALIDATION.md).

## Preservation and scope

The simulation algorithms have not been refactored. Four old template launch
lines were replaced by comments, and two bond-dimension templates no longer
activate a cluster-specific environment. Original file hashes are recorded in
[SOURCE_MANIFEST.json](SOURCE_MANIFEST.json). Other copied source files are
unchanged. Added tests and documentation are separate from those sources.

This is a source collection, not a complete bit-for-bit reproduction bundle:
legacy datasets, production seed ledgers, analysis/fit choices, and some exact
historical dependency versions are not included. No new license has been
assigned to the research scripts; the dependency license applies only to the
vendored package.
