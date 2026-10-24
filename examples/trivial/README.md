# NN AKLT to zero / trivial model

## Paper reference: Figure 2

![Paper Figure 2: SPT-trivial transition.](../../docs/figures/fig02_trivial.png)

From Aziz, Pan, and Pixley, [arXiv:2609.26941](https://arxiv.org/abs/2609.26941).
Panels show string order and its collapse, entropy versus size, and entropy
variance. [Panel guide and original PDF](../../docs/RESULTS.md#spt-to-trivial-product-state-figure-2).
The paper figure is CC BY 4.0, not part of the MIT code license.

## Generated example: one small trajectory

![Single-trajectory trivial-model example, not paper data.](trace.png)

This is an actual run of the preserved MPS engine: `L=4`, `p_TT=0.50`,
16 full sweeps, master seed `1234`, cutoff `1e-10`, periodic boundaries,
and a uniform product initial state. A sweep contains `L` update slots.
`p_nn=0.50` is passed to the engine. Four sites are chosen to keep the example
cheap, not to resolve the transition.

The left panel takes the absolute value **after** averaging signed NN string
order at separation `L/2` over `L/2` origins. The right panel shows half-chain
von Neumann entropy in bits. These are single-trajectory values, not ensemble
averages or entropy variances. The zero operation in this legacy engine is
normalized `exp(50 * |0><0|)`, a finite-strength projector approximation.

## Run and plot

From the repository root, using Julia 1.12.6 and Python 3.11+:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate(; allow_autoprecomp=false)'
julia --project=. --threads=1 --compiled-modules=no --compile=min -O0 examples/trivial/run.jl
python -m pip install -r examples/requirements-plot.txt
python examples/plot_trace.py trivial results/trivial_example results/trivial_example/trace.png
```

The MPS compiler flags avoid a lengthy cold precompile; they are not performance
settings for production. The first run can still take several minutes. The
[configuration](config.toml) permits `L=4` or `6` and at most 40 sweeps.
Reruns require a fresh directory: `examples/trivial/run.jl results/trivial_again`.
Neither the simulation nor plotting command overwrites an existing output.

To replot the saved data without rerunning Julia:

```sh
python examples/plot_trace.py trivial examples/trivial/expected results/trivial_saved.png
```

[Reference CSV](expected/summary.csv) | [Run metadata](expected/metadata.toml) |
[Simulation source](../../simulations/mps/trivial/run_zero_static.jl).
Fresh runs also retain the raw engine output. `mean_nn` is signed,
`abs_mean_nn` is its magnitude, and `entropy_bits` uses base-2 logarithms.
