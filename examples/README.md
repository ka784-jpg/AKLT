# Simulation examples

## Model gallery

Each model has a paper reference, a small generated plot, saved numerical
outputs, and commands to rerun or replot it. The paper figures and the example
plots are explicitly distinguished.

| Model | Paper reference | Runnable example |
| --- | --- | --- |
| Trivial / zero | Figure 2 | [String order and entropy](trivial/README.md) |
| Dimer | Figure 3 | [String order, dimer order, and entropy](dimer/README.md) |
| NN-NNN | Figures 4, 6, 11 | [NN and NNN string order](nnn/README.md) |

The full [results guide](../docs/RESULTS.md) explains the panels and maps them
to the underlying simulation code. Plotting uses Python 3.11+ with
`python -m pip install -r examples/requirements-plot.txt`; the simulations
themselves use Julia. Plotting the committed CSVs needs no Julia installation.

## Small, reproducible NN-NNN quickstart

This interface calls the preserved order-only statevector engine. It does not
replace the campaign launchers or alter their algorithms. It adds named inputs,
bounded demo sizes, checked outputs, and a readable CSV schema.

```sh
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
julia --project=examples examples/quickstart.jl --config examples/quickstart.toml --output results/demo
```

Run from the repository root. Julia 1.12.6 is the reference runtime; the
lightweight environment has its own committed `Manifest.toml`.

## Inputs

| TOML field | Default | Contract |
| --- | --- | --- |
| `L` | `10` | `10` or `12`; physical spin-1 sites, periodic boundary conditions |
| `p_nnn` | `0.46` | Finite value in `[0,1]`; the engine receives `p_nn = 1-p_nnn` |
| `seed` | `1234` | Integer in `[0,2147483647]`; `MersenneTwister` gate-choice seed |
| `sweeps` | `24` | Final sweep, at most 100; one sweep contains `L` local update slots |
| `sample_start` | `20` | First recorded sweep, between 1 and `sweeps` |
| `sample_every` | `1` | Positive integer; the sampling sequence must reach `sweeps` exactly |

Unknown or missing keys, non-finite probabilities, unsupported sizes, and
inconsistent windows fail before the simulation starts. The limits are an
intentional guard against accidentally starting a production-sized dense run.
For larger systems use [the campaign interfaces](../RUNNING.md).

The initial state is the engine's uniform product state. The control sequence
chooses a nearest- or next-nearest-neighbor projector independently at each
slot and renormalizes after applying it. This example does not implement a
Born-sampled outcome or a feedback protocol.

## Outputs

`samples.csv` has one row per recorded sweep and spatial origin:

| Column | Meaning |
| --- | --- |
| `sweep` | Complete sweeps elapsed |
| `origin` | Starting site, `1` through `L/2` |
| `seed` | Gate-choice seed of this single trajectory |
| `nn`, `nnn` | Signed NN and NNN string observables at separation `L/2` |
| `nn_m1`, `nnn_m1` | The same observables at separation `L/2-1` |

`summary.csv` contains `sweep`, `mean_nn`, `mean_nnn`, `abs_mean_nn`, and
`abs_mean_nnn`. Means are over the recorded origins of **one trajectory**.
The absolute value is taken *after* that mean. These are not trajectory-ensemble
estimates or uncertainties. The paper averages over many trajectories as well.

`raw/` preserves the original four comma-delimited matrices, including the
metadata row with seed labels and an intentionally unused `NaN` corner.
The wrapper checks all observable entries for finiteness, their physical
`[-1,1]` bounds, dimensions, seed labels, and recorded times before publishing
the completed output directory. It does not change legitimate raw metadata.

`metadata.toml` records the configuration, initialization, probability and time
conventions, Julia version, and SHA256 of the included engine.

## Repeating a result

[`expected/summary.csv`](expected/summary.csv) was generated from the default
configuration using Julia 1.12.6, one Julia thread, and one BLAS thread.
The engine enforces the BLAS setting. The test checks an independent rerun
against this fixture with `rtol=1e-9`, `atol=1e-11`, and checks two runs in
the same process for identical files. Cross-platform bitwise identity is not
promised; Julia/runtime changes can alter random streams or floating-point
details.

```sh
julia --project=examples --threads=1 tests/quickstart_smoke.jl
```

The dense-reference projector tests in `tests/statevector_smoke.jl` complement
this regression fixture; a fixture alone would not establish physical correctness.

## Failure behavior

- Existing output directory or file: nonzero exit, no overwrite. Use a new path.
- Invalid input or failed output validation: nonzero exit, no completed result
  directory. The wrapper cleans its own temporary staging directory.
- Missing Julia dependencies: instantiate the lightweight environment first.
- First-run delay: Julia compilation is included in wall-clock time.

The legacy campaign engine can print a warning instead of throwing on a write
failure. The wrapper therefore verifies that all four files actually exist and
are valid before reporting success. This checked behavior belongs to the new
example, not to every archived launcher.
