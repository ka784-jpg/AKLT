module QuantumCircuitsMPS2

using ITensors
using ITensorMPS
using Random
using LinearAlgebra

# Core
include("Core/basis.jl")
include("Core/rng.jl")

# State
include("State/State.jl")
include("State/initialization.jl")

# Gates
include("Gates/Gates.jl")

# Geometry
include("Geometry/Geometry.jl")

# Core apply! (after State, Gates, Geometry)
include("Core/apply.jl")

# Observables
include("Observables/Observables.jl")

# API
include("API/imperative.jl")
include("API/functional.jl")
include("API/context.jl")
include("API/probabilistic.jl")

# Simulation styles (circuit-level APIs)
include("API/simulation_styles/style_imperative.jl")
include("API/simulation_styles/style_callback.jl")
include("API/simulation_styles/style_iterator.jl")

# Circuit (lazy mode API)
include("Circuit/Circuit.jl")

# Plotting
include("Plotting/Plotting.jl")

# === PUBLIC API EXPORTS ===
# State
export SimulationState, initialize!, ProductState, RandomMPS
# RNG
export RNGRegistry, get_rng  # NOTE: rand is extended, not exported
# Gates
export AbstractGate, PauliX, PauliY, PauliZ, Projection, HaarRandom, Measurement, Reset, CZ
export total_spin_projector, verify_spin_projectors
export SpinSectorProjection, SpinSectorMeasurement
# Geometry
export AbstractGeometry, SingleSite, AdjacentPair, NextNearestNeighbor, Bricklayer, AllSites
export StaircaseLeft, StaircaseRight
export Pointer, move!
# Observables
export AbstractObservable, DomainWall, BornProbability, EntanglementEntropy, StringOrder
export BondDimension, SpatiallyAveragedStringOrder, DimerOrder
export track!, record!, list_observables
export window_checker, SpatiallyAveragedStringOrder, record_step!
# API
export apply!, simulate, with_state, current_state, apply_with_prob!
export run_circuit!, simulate_circuits, CircuitSimulation
export record_every, record_at_circuits, record_always
export get_state, get_observables, circuits_run
# Circuit (lazy mode API)
export Circuit, expand_circuit, simulate!, ExpandedOp
export RecordingContext, every_n_gates, every_n_steps
# ASCII Plotting
export print_circuit
# Visualization (provided by Luxor extension)
function plot_circuit end
export plot_circuit

# === INTERNAL EXPORTS (for CT.jl parity/debugging) ===
# These are exported for testing/verification but not public API
export advance!, get_sites, current_position  # Geometry internals
export compute_site_staircase_right, compute_site_staircase_left, compute_pair_staircase  # Pure geometry computation
export apply_op_internal!, apply_post!        # Apply internals  
export born_probability                       # Observable internals
export compute_basis_mapping, physical_to_ram, ram_to_physical # Basis internals

# === Geometry Helpers === #
export wrap_site, sanitize_pair, build_nn_pairs, build_manual_pairs

end # module
