# Reproducibility: what is covered

## 1. Runnable reference example

The small NN-NNN example has a committed configuration, runtime/dependency
lockfile, reference output, and regression tests. It invokes a preserved
simulation engine and records its source hash with the results. A fresh clone
needs only Julia and the small example environment.

The reference runtime is Julia 1.12.6. `examples/Manifest.toml` locks the
lightweight environment; the root `Manifest.toml` locks the larger MPS/test
environment. These are **current validation environments**, not asserted
historical environments for all paper runs.

```sh
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
julia --project=examples --threads=1 examples/quickstart.jl
julia --project=examples --threads=1 tests/quickstart_smoke.jl
```

The example records a single trajectory from a uniform product state. The
reference output is a regression target, not paper data or a substitute for
finite-size/ensemble convergence. It is compared numerically, not by a
cross-platform bitwise-equality claim.

## 2. Physics and behavior checks

The tests exercise independent dense embeddings of the local projector,
projector rank/idempotence/Hermiticity, normalization, seeded repeatability,
finite output, Renyi limits, bits-to-nats conversion, and tiny TT/dimer MPS
trajectories. The example tests add input rejection and no-overwrite behavior.

The [CI workflow](../.github/workflows/tests.yml) runs statevector/example checks
on Linux and Windows and MPS checks on Linux. CI status is shown live in the
README badge. Local observations are in [VALIDATION.md](../VALIDATION.md).

## 3. Source and figure integrity

`SOURCE_MANIFEST.json` records 75 collected files, original and packaged hashes,
and the six portability-only edits made during collection. It is deliberately
not rewritten when new documentation, examples, or tests are added.
`docs/figures/manifest.json` records the four selected PDFs and rendered PNGs.
The Python-standard-library integrity test checks both collections.

```sh
python tests/repository_integrity.py
```

The original `AKLT_nnn_nn` and `NN-NNN Forced` files remain unchanged. The
vendored dependency retains its original attribution and license. None of
these records establishes the exact dependency revision used by every
historical HPC job.

## 4. What full paper reproduction still requires

The [figure-to-code map](../FIGURE_PROVENANCE.md) separates exact campaign links
from producer-family identification. Recreating the paper's figures additionally
requires the contributing production datasets or complete production runs,
seed/campaign bookkeeping, dataset-selection rules, fitting and averaging code,
and sufficiently matched historical environments. Those are not all included.

Known caveats include mixed historical campaign contributions and a fidelity
plot filename that says random product while the identified data-producing
runner initializes a fixed uniform product state. The mapping records this
discrepancy openly; this repository update does not resolve it or modify the
manuscript.

The quickstart and smoke tests do not certify critical exponents, asymptotic
phase boundaries, every archived parameter combination, GPU execution,
experimental success probabilities, or performance at production scale.
