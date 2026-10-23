# Contributing

For a bug report, include the repository commit, Julia version, operating
system, exact command, input configuration, and the first error message.
Attach a small reproducer or a few output rows instead of a production archive.
Do not post credentials, private cluster paths, or unpublished collaborator data.

## Before proposing a change

```sh
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
julia --startup-file=no tests/parse_sources.jl
julia --project=examples tests/statevector_smoke.jl
julia --project=examples tests/quickstart_smoke.jl
python tests/repository_integrity.py
```

For MPS changes, also instantiate the root environment and run
`tests/mps_smoke.jl` as documented in the README. Keep tests small enough to
run without a cluster.

## Scientific and archival changes

- Preserve the distinction between control randomness and Born-sampled outcomes.
- State the probability convention, time unit, boundary condition, initial
  state, and entropy logarithm base when adding a runner.
- Test numerical changes against a small independent reference where possible.
- Keep historical campaign copies stable. Put a revised algorithm in a clearly
  identified version, or document the change and update provenance deliberately.
- Do not regenerate reference CSVs merely to make a failing test pass. Explain
  the numerical reason and retain independent invariant tests.
- Keep plotting/analysis additions separate from the simulation engines.

New contributions follow the root MIT license unless explicitly agreed
otherwise. Do not remove dependency attribution or apply the code license to
paper figures; see [NOTICE.md](NOTICE.md).
