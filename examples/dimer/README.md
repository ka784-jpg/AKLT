# NN AKLT to dimer model

## Paper reference: Figure 3

![Paper Figure 3: SPT-dimer transition.](../../docs/figures/fig03_dimer.png)

From Aziz, Pan, and Pixley, [arXiv:2609.26941](https://arxiv.org/abs/2609.26941).
Panels show string/dimer order, the centered Binder statistic, and entropy
scaling. [Panel guide and original PDF](../../docs/RESULTS.md#spt-to-dimerized-state-figure-3).
The paper figure is CC BY 4.0, not part of the MIT code license.

## Generated example: one small trajectory

![Single-trajectory dimer-model example, not paper data.](trace.png)

This runs the preserved sector-aware MPS engine at `L=4`, `p_D=0.34`,
16 full sweeps, master seed `1234`, cutoff `1e-10`, and periodic boundaries.
It starts from the uniform product state, with the odd dimer sector and
`halfcut_start_mode="half"`. The engine receives `p_nn=0.66`.
The competing branch projects selected odd bonds into total spin zero;
on the other bonds it is identity. One sweep contains `L` update slots.

The left panel shows the magnitude of the spatially averaged NN string order
and the magnitude of the alternating bond observable `D = sum_i (-1)^i
<S_i . S_(i+1)> / L`. The right panel is half-chain entropy in bits. The signed
`mean_nn` and `dimer_order` are retained in the CSV, alongside their magnitudes.
NN averaging uses `L/2` origins at separation `L/2`. The plotted entropy depends
on the fixed cut convention, especially for a dimerized state.

This is one four-site trajectory, not a Binder calculation, an uncertainty
estimate, or a reproduction of the paper's critical exponents.

## Run and plot

From the repository root, using Julia 1.12.6 and Python 3.11+:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate(; allow_autoprecomp=false)'
julia --project=. --threads=1 --compiled-modules=no --compile=min -O0 examples/dimer/run.jl
python -m pip install -r examples/requirements-plot.txt
python examples/plot_trace.py dimer results/dimer_example results/dimer_example/trace.png
```

The MPS compiler flags avoid a lengthy cold precompile, but the first run can
still take several minutes. [Configuration](config.toml) permits `L=4` or `6`
and at most 40 sweeps. A rerun can use `examples/dimer/run.jl results/dimer_again`.
Existing output directories and plot files are never overwritten.

Replot the saved data without Julia:

```sh
python examples/plot_trace.py dimer examples/dimer/expected results/dimer_saved.png
```

[Reference CSV](expected/summary.csv) | [Run metadata](expected/metadata.toml) |
[Simulation source](../../simulations/mps/dimer/run_dimer_sector_cli.jl).
Fresh runs retain the raw matrices, their trajectory seeds, and the engine hash.
