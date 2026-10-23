using Random

"""
    RNGRegistry

Container for named RNG streams for reproducible randomness.

Streams:
- :ctrl - decisions about whether to apply control operations (p_ctrl)
- :proj - decisions about whether to apply projections (p_proj)  
- :haar - Haar random unitary generation
- :born - Born rule measurement outcomes
- :state_init - random initial state generation (optional)
"""
struct RNGRegistry
    streams::Dict{Symbol, AbstractRNG}
end

"""
    RNGRegistry(; ctrl, proj, haar, born, state_init=0)

Create RNG registry with seeds for each stream.
First 4 arguments (ctrl, proj, haar, born) are REQUIRED.
"""
function RNGRegistry(;
    ctrl::Int,
    proj::Int,
    haar::Int,
    born::Int,
    state_init::Int = 0
)
    streams = Dict{Symbol, AbstractRNG}(
        :ctrl => MersenneTwister(ctrl),
        :proj => MersenneTwister(proj),
        :haar => MersenneTwister(haar),
        :born => MersenneTwister(born),
        :state_init => MersenneTwister(state_init)
    )
    return RNGRegistry(streams)
end

"""
    RNGRegistry(::Val{:ct_compat}; circuit, measurement)

Create CT.jl-compatible RNG registry where :ctrl, :proj, :haar share the SAME RNG.
This matches CT.jl's interleaved consumption pattern for verification.

Used ONLY for Task 8 CT.jl verification with p_proj=0.
"""
function RNGRegistry(::Val{:ct_compat}; circuit::Int, measurement::Int)
    # Shared RNG for circuit operations (matches CT.jl's rng_C)
    shared_circuit_rng = MersenneTwister(circuit)
    
    streams = Dict{Symbol, AbstractRNG}(
        :ctrl => shared_circuit_rng,      # ALIAS - same RNG object
        :proj => shared_circuit_rng,      # ALIAS
        :haar => shared_circuit_rng,      # ALIAS
        :born => MersenneTwister(measurement),
        :state_init => MersenneTwister(0)
    )
    return RNGRegistry(streams)
end

"""
    get_rng(registry::RNGRegistry, stream::Symbol) -> AbstractRNG

Get the raw RNG object for a stream. Low-level API.
"""
function get_rng(registry::RNGRegistry, stream::Symbol)
    haskey(registry.streams, stream) || throw(ArgumentError("Unknown RNG stream: $stream"))
    return registry.streams[stream]
end

"""
    rand(registry::RNGRegistry, stream::Symbol) -> Float64

Draw a random Float64 from the specified stream. Mid-level API.
"""
function Base.rand(registry::RNGRegistry, stream::Symbol)
    return rand(get_rng(registry, stream))
end

"""
    randn(registry::RNGRegistry, stream::Symbol) -> Float64

Draw a random normal Float64 from the specified stream.
"""
function Base.randn(registry::RNGRegistry, stream::Symbol)
    return randn(get_rng(registry, stream))
end

"""
    randn(registry::RNGRegistry, stream::Symbol, dims...) -> Array{Float64}

Draw random normal array from the specified stream. For Haar random matrices.
"""
function Base.randn(registry::RNGRegistry, stream::Symbol, dims...)
    return randn(get_rng(registry, stream), dims...)
end

# === SimulationState Extensions ===

# Note: SimulationState is defined in State/State.jl, which is included after this file.
# We use Any for the state argument to avoid circular dependency during inclusion.

"""
    rand(state, stream::Symbol) -> Float64

Draw a random number from the specified RNG stream.
Convenience wrapper that hides rng_registry internals.

Example:
    if rand(state, :ctrl) < p_ctrl
        # control branch
    end
"""
function Base.rand(state::Any, stream::Symbol)
    # Check if it's a SimulationState without requiring the type at compile time
    if nameof(typeof(state)) == :SimulationState
        isnothing(state.rng_registry) && error("SimulationState has no RNG registry attached.")
        return rand(state.rng_registry, stream)
    end
    # Fallback for other types if necessary, or just throw error
    throw(MethodError(rand, (state, stream)))
end
