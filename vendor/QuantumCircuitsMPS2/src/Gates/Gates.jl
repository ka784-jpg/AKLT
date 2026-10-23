using ITensors
using LinearAlgebra

"""
Abstract base type for all quantum gates.
"""
abstract type AbstractGate end

"""
    support(gate::AbstractGate) -> Int

Return the number of sites this gate acts on (1 for single-qubit, 2 for two-qubit).
"""
function support end

"""
    build_operator(gate, site_or_sites, local_dim; kwargs...) -> ITensor

Build the operator tensor for a gate acting on given site(s).
"""
function build_operator end

# Include gate implementations
include("single_qubit.jl")
include("two_qubit.jl")
include("composite.jl")
include("spin_projectors.jl")
include("spin_measurement.jl")
