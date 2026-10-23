# Figure-to-simulation map

Figure numbers refer to the 14-figure manuscript checked during collection.
The mapping was traced from `main.tex` through the active
`RealPaperFiguresQCMPS_overleaf_only.ipynb` input paths and available campaign
records. The notebook was used as evidence, not included as simulation code.

| Figure | Numerical content | Relevant simulation sources |
| --- | --- | --- |
| 1 | Protocol schematic | No numerical simulation |
| 2 | SPT-trivial scaling | `mps/trivial/run_zero_static.jl` |
| 3 | SPT-dimer scaling | `mps/dimer/run_dimer_static_legacy.jl`, `run_dimer_sector_cli.jl`, `run_dimer_circuitshift_entropy.jl`, `run_dimer_binder_only_cli.jl` |
| 4 | NN-NNN order and collapse | `statevector/order/AKLTStatevectorOrderOnly.jl` and `run_order_worker.jl` |
| 5 | NN-NNN algebraic SOP decay | Same order-sampling family as Fig. 4 |
| 6 | NN-NNN entropy and variance | `statevector/static/AKLTStatevector.jl` |
| 7 | Dynamical-exponent collapses | `mps/trivial/run_zero_dynamics.jl`, `mps/dimer/run_dimer_dynamics.jl`, `statevector/dynamics/AKLTStatevectorNNNZ.jl` |
| 8 | Endpoint preparation fidelity | `mps/fidelity/run_fidelity_mps.jl`; see initialization caveat below |
| 9 | SOP collapse landscapes | Same order-sampling family as Fig. 4; landscape construction is analysis |
| 10 | Bond-dimension scaling | `mps/bond_dimension/run_zero_bond_batch.jl`, `dimer_bond.jl`, `nnn_bond.jl`, plus compatible legacy trajectories |
| 11 | Entropy relaxation | `statevector/relaxation/AKLTStatevectorNNNZ.jl` and `run_entropy.jl`, plus earlier saturation campaigns |
| 12 | NNN SOP relaxation | `statevector/relaxation/AKLTStatevectorNNNZ.jl` and `run_order.jl`, plus earlier saturation campaigns |
| 13 | Selected-probability SOP decay | Same order-sampling family as Fig. 4 |
| 14 | Renyi-index dependence | `statevector/renyi/AKLTStatevector.jl` and `run_sv_nnn_alpha_worker.jl` |

All paths in this table are relative to `simulations/`.

## Strong campaign links

- The string-order master records `legacy_sv_nnn_sat`,
  `svnnn_order1300_20260809_a5549b3c01d3`, and
  `svnnn_order_fill10k_20260810_a568bec405f5`. The included order-only module and
  worker are copied from the last campaign's staging snapshot. The worker
  samples sweeps `2L` through `2L+4` and uses `L/2` spatial samples.
- The dimer Binder master records a legacy equal-trajectory cache and
  `dimer_binder_only_10k_20260831`. The included Binder runner comes from that
  campaign's archived metadata, not a plot-fitting script.
- The saturation registry includes multiple older campaigns and the later
  `nnn_everything_roundonly_adaptive_amarel_20260826_7cfc4763` and
  `nnn_everything_roundonly_cont15_amarel_20260827_4deb023e` campaigns. The
  included multi-cut module and entropy/order runners come from the continuation
  snapshot. The notebook consumes their combined time-series aggregate rather
  than just one campaign.
- The Renyi worker comes from
  `sv_nnn_alpha_renyi_extended_amarel_stage_20260807_0453f23f`: `L=8,10,12,14`,
  `p_nnn=0.30,0.32,...,0.64`, sweeps `2L` through `2L+4`, and indices
  `0,0.7,1,2,3,4,5,inf`. The `n=0` rank diagnostic uses two tolerances and is
  distinct from the positive-index fits plotted in the paper.

## Important limitations

The TT, legacy dimer, stationary-entropy, and older dynamics sources are
identified producer families. The currently available aggregate files do not
establish an exact code hash for every contributing historical run. Do not
interpret this collection as proving that every plotted point was generated
by precisely the included revision.

The fidelity plotting cell reads the original `fidlim_20260718_203553_a7056d84`
`downloaded_outputs` directory. All 21 files in that directory identify
`fidlim_mps_deep_reference_v1`, matching `run_fidelity_mps.jl`. That runner
initializes a fixed uniform product state. The plot filename says
`fidelity_random_product_limits_t_over_L_1x3.pdf`, which alone is not evidence
of random initialization. `run_fidelity_random_mps.jl` is a separate available
extension, not silently substituted as the producer of those original files.
Resolving this naming/provenance discrepancy requires a separate manuscript or
data review; neither is changed by this repository update.

For Fig. 10, the active notebook selects compatible TT data at `3L..3L+4`,
dimer data at `3L..3L+2`, and NNN data with **three trajectories per original
file and the window `4L..6L`**. The later gap-fill launch script's different
NNN window is not used here as a universal reproduction command. The NNN
simulation function accepts explicit window arguments.

The vendored MPS dependency is the available local `Ounwy` snapshot, retained
with its license. Matching it to every historic cluster installation remains
unverified. Original source hashes and source labels are in
`SOURCE_MANIFEST.json`; local absolute paths and private operational files are
not published.
