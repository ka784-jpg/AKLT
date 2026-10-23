# Running small simulations

For the shortest checked entry point, start with the
[README quickstart](README.md#quickstart) and its [named-input guide](examples/README.md).
The commands below expose the preserved campaign interfaces and use the full
root environment. Julia 1.12.6 and the committed root manifest are the validated
runtime/dependency combination.

Run from the repository root after `Pkg.instantiate()`. Examples are small
functional checks, not the production-size ensembles used for paper fits.
Run different MPS source files in separate Julia processes because their
shared function names and operator definitions can conflict in one session.

## NN-NNN stationary string order

```sh
mkdir results
julia --project=. --threads=1 simulations/statevector/order/run_order_worker.jl 10 0.54 0.46 1234 results/order_demo
```

Arguments: `L p_nn p_nnn seed output_prefix`. The worker runs one trajectory,
samples sweeps `2L..2L+4`, and writes signed NN/NNN string-order samples.
Use `L>=10` if both NNN separations are required: the legacy implementation
leaves unavailable short-distance NNN entries as `NaN` for smaller systems.

## NN-NNN relaxation

```sh
julia --project=. --threads=1 simulations/statevector/relaxation/run_entropy.jl simulations/statevector/relaxation/AKLTStatevectorNNNZ.jl 10 0.54 40 10 40 10 1234 5678 results entropy_demo 2
julia --project=. --threads=1 simulations/statevector/relaxation/run_order.jl simulations/statevector/relaxation/AKLTStatevectorNNNZ.jl 10 0.54 40 10 40 10 1234 results order_time_demo
```

Here `NSTEPS`, `TI`, `TF`, and `INTERVAL` count gate slots. Sampling every `L`
slots records integer rounds. The entropy runner's last argument is the number
of distinct cuts. It requires one Julia and one BLAS thread, as in the archived
campaign.

## Trivial and dimer MPS models

```sh
julia --project=. --threads=1 -e 'include("simulations/mps/trivial/run_zero_static.jl"); runy(6,0.5,1e-10,18,1,true,3,12,18,1,1234,"results/tt_demo")'
julia --project=. --threads=1 simulations/mps/dimer/run_dimer_sector_cli.jl 6 0.66 1e-10 20 1 true 3 18 20 1 1234 results/dimer_demo odd all half 0
```

The dimer example has `p_D=0.34`, because its input is `p_nn=0.66`. The dimer
CLI prints its supported arguments on an invalid argument count. The separate
Binder runner supports a `dimer_only` mode for recording trajectory samples
without entropy computation.

The historical TT and dimer dynamics templates now define `runy` without an
automatic placeholder invocation. Their typed signatures document the
gate-time arguments, observable modes, and entropy-cut choices.

## Bond dimension

`dimer_bond.jl` and `nnn_bond.jl` are function definitions rather than standalone
launchers. The NNN function has the signature

```julia
runy(L, cutoff, bc, p_nn, n_steps, iterations, self_av_space_it,
     ti, tf, interval, master_seed, output_prefix)
```

For the Fig. 10 NNN subset, use `cutoff=1e-10`, `bc=:periodic`,
`iterations=3`, `ti=4L`, and `tf=n_steps=6L`. These are not the parameters of
the later gap-fill launch script. See the provenance notes before combining
different production protocols.

## Fidelity and Renyi

The fidelity runner takes `TASKS_TSV ROW_INDEX OUT_CSV CAMPAIGN`.
`tasks_endpoint.tsv` is the archived production parameter table, not a cheap
smoke test. The fixed-product runner and random-MPS runner are distinct.

The Renyi runner takes `--task-file`, `--task-index`, `--worker-index`,
`--workers-per-task`, and `--tmp-out`. `tasks_renyi.csv` is the archived
production parameter table. It does not submit jobs by itself. Each sampled
Schmidt spectrum is reused for all requested Renyi indices.

## Validation

```sh
julia --project=. --threads=1 tests/statevector_smoke.jl
julia --project=. --threads=1 tests/mps_smoke.jl
julia --project=. --threads=1 tests/parse_sources.jl
```

These checks do not reproduce the full paper's large ensembles or certify
the fit parameters. Outputs from manual examples belong in ignored `results/`.
