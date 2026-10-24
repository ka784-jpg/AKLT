# Local validation

## Model galleries and results-page expansion: 2026-10-24

- Ran the preserved TT and dimer MPS engines at `L=4` for 16 sweeps each,
  plus the NN-NNN exact-statevector engine at `L=10` for all 24 sweeps.
  Each gallery saves the actual single-trajectory CSV and source-hash metadata.
- Rendered and visually inspected all three example plots using Python and
  Matplotlib 3.11.2. The plotting tests passed, including wrong-model,
  invalid-data, and no-overwrite cases (3 test methods).
- The MPS example-wrapper contract suite passed 28 assertions with explicit
  synthetic runner output; the separate real runs above exercise the engines.
- Copied and visually checked Figures 2 and 3 from the manuscript, without
  content changes. Integrity checks now cover 75 archived source hashes and
  12 paper-asset hashes, plus links in nested example pages.
- No archived simulation source, vendored algorithm, or manuscript was edited.
  The CI workflow also runs the new real examples and plotting checks.

The small traces are illustrative and finite-size; they are not paper ensemble
data or new estimates of phase boundaries. The earlier validation below is
retained as a dated record rather than relabeled as a new run.

Checked on Windows with Julia 1.12.6 on 2026-10-23.
The newly resolved environment used ITensors 0.9.32, ITensorMPS 0.3.45,
DelimitedFiles 1.9.1, and the included QuantumCircuitsMPS2 0.1.0 snapshot.
These are validation versions, not asserted historical production versions.

| Check | Result |
| --- | --- |
| Environment instantiation | Passed |
| Julia parsing of collected simulation and dependency sources | 65 files passed |
| Dense-reference projector checks | 36 assertions passed |
| Repeated-seed order-only trajectories | 9 assertions passed |
| Renyi entropy conventions | 14 assertions passed |
| Relaxation output and Renyi trajectory smoke checks | 6 assertions passed |
| Small TT and dimer MPS trajectories | 12 assertions passed |
| Collected-source hashes | 75 files checked; 69 unchanged copies and 6 documented portability edits |
| Original two repository files | Unchanged |

## Public example and documentation checks

Also checked on 2026-10-23 using the separate, locked `examples/` environment:

| Check | Result |
| --- | --- |
| README quickstart command | Completed; 25 signed spatial samples and 5 summary rows |
| Repeated output, reference fixture, and preservation checks | 18 assertions passed |
| Invalid configuration and CLI-option checks | 35 assertions passed |
| Statevector tests in lightweight environment | All 65 numerical assertions passed |
| Figure assets | Four copied PDFs matched source hashes; all four PNG previews visually checked |
| Published integrity | 75 collected-source hashes and 8 figure-asset hashes checked |
| Local Markdown file targets | Checked by `tests/repository_integrity.py` |

The root and example manifests now pin the validated dependency resolutions.
The new quickstart wraps the archived engine; no simulation or vendored
algorithm was modified for this documentation/example update. Together with
the 12 MPS checks above, the numerical suites contain 130 passing assertions.

The [GitHub Actions workflow](.github/workflows/tests.yml) repeats the lightweight
checks on Linux and Windows and the MPS checks on Linux. Its live status is
available from the README badge; local results alone are not evidence that a
hosted run completed.

```sh
julia --project=examples --startup-file=no --threads=1 examples/quickstart.jl
julia --project=examples --startup-file=no --threads=1 tests/quickstart_smoke.jl
julia --project=examples --startup-file=no --threads=1 tests/statevector_smoke.jl
python tests/repository_integrity.py
```

## Original collection checks

Commands:

```sh
julia --project=. --startup-file=no --threads=1 -e 'using Pkg; Pkg.instantiate(; allow_autoprecomp=false)'
julia --startup-file=no --threads=1 tests/parse_sources.jl
julia --project=. --startup-file=no --threads=1 tests/statevector_smoke.jl
julia --project=. --startup-file=no --threads=1 --compiled-modules=no --compile=min -O0 tests/mps_smoke.jl
```

The first MPS attempt was stopped during lengthy cold dependency precompilation;
the final command above bypassed cache compilation and passed. No production
cluster jobs or large ensembles were launched. GPU paths, every archived
runner/parameter combination, final fitted critical exponents, and complete
bit-for-bit paper reproduction have not been validated here.

Short-distance NNN channels are intentionally left as NaN by the legacy
order-only implementation when their endpoints do not meet its distance
condition. The finite-output regression uses L=10, where both NNN separations
are defined; it does not change this behavior in the collected source.
