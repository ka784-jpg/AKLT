# === CircuitBuilder (Internal, Do-Block API) ===
# Not exported - users interact via Circuit(f::Function; kwargs...) do-block

"""
    CircuitBuilder

Internal mutable structure for recording circuit operations during do-block construction.

Users never see this type directly - they interact via:

```julia
circuit = Circuit(L=4, bc=:periodic) do c
    apply!(c, Reset(), SingleSite(1))
    apply_with_prob!(c; rng=:ctrl, outcomes=[
        (probability=0.5, gate=PauliX(), geometry=SingleSite(1))
    ])
end
```

The builder records operations as NamedTuples which are then passed to the Circuit constructor.

# Fields
- `L::Int`: Number of physical sites
- `bc::Symbol`: Boundary conditions (`:periodic` or `:open`)
- `operations::Vector{NamedTuple}`: Accumulated operation records
- `params::Dict{Symbol,Any}`: User-defined parameters passed from outer constructor
"""
mutable struct CircuitBuilder
    L::Int
    bc::Symbol
    operations::Vector{NamedTuple}
    params::Dict{Symbol,Any}
end

CircuitBuilder(L::Int, bc::Symbol, params::Dict{Symbol,Any}=Dict{Symbol,Any}()) = CircuitBuilder(L, bc, NamedTuple[], params)

"""
    apply!(builder::CircuitBuilder, gate, geometry)

Record a deterministic gate operation in the circuit builder.

Stores operation as: `(type=:deterministic, gate=gate, geometry=geometry)`

# Example
```julia
Circuit(L=4, bc=:periodic) do c
    apply!(c, Hadamard(), SingleSite(1))
    apply!(c, CNOT(), StaircaseRight(1))
end
```
"""
function apply!(builder::CircuitBuilder, gate, geometry)
    push!(builder.operations, (type=:deterministic, gate=gate, geometry=geometry))
    return nothing
end

"""
    apply_with_prob!(builder::CircuitBuilder; rng::Symbol=:ctrl, outcomes)

Record a stochastic operation in the circuit builder.

Stores operation as: `(type=:stochastic, rng=rng, outcomes=collect(outcomes))`

# Arguments
- `rng::Symbol`: RNG source identifier (must be `:ctrl` in Phase 1)
- `outcomes`: Vector of NamedTuples with keys `(:probability, :gate, :geometry)`

# Validations
- Throws if `rng != :ctrl` (Phase 1 constraint)
- Throws if `outcomes` is empty
- Throws if probabilities sum to > 1.0

# Example
```julia
Circuit(L=4, bc=:periodic) do c
    apply_with_prob!(c; rng=:ctrl, outcomes=[
        (probability=0.3, gate=PauliX(), geometry=SingleSite(1)),
        (probability=0.2, gate=PauliZ(), geometry=SingleSite(1))
    ])
end
```
"""
function apply_with_prob!(
    builder::CircuitBuilder;
    rng::Symbol = :ctrl,
    outcomes::Vector{<:NamedTuple{(:probability, :gate, :geometry)}}
)
    # Phase 1 constraint: only :ctrl RNG supported
    if rng != :ctrl
        throw(ArgumentError("Only rng=:ctrl is supported in Phase 1 (got: $rng)"))
    end
    
    # Must provide at least one outcome
    if isempty(outcomes)
        throw(ArgumentError("outcomes cannot be empty"))
    end
    
    # Validate probabilities sum to ≤ 1.0
    probs = [o.probability for o in outcomes]
    total_prob = sum(probs)
    if total_prob > 1.0 + 1e-10
        throw(ArgumentError("Probabilities sum to $total_prob (must be ≤ 1)"))
    end
    
    # Record stochastic operation
    push!(builder.operations, (type=:stochastic, rng=rng, outcomes=collect(outcomes)))
    return nothing
end

"""
    Circuit(f::Function; L::Int, bc::Symbol, n_steps::Int=1)

Construct a Circuit using do-block syntax with a CircuitBuilder.

The function `f` receives a `CircuitBuilder` instance and can call:
- `apply!(builder, gate, geometry)` - for deterministic operations
- `apply_with_prob!(builder; rng, outcomes)` - for stochastic operations

# Arguments
- `f::Function`: Builder function that receives CircuitBuilder
- `L::Int`: Number of physical sites
- `bc::Symbol`: Boundary conditions (`:periodic` or `:open`)
- `n_steps::Int`: Number of circuit timesteps (default: 1)
- `kwargs...`: Additional keyword arguments stored in circuit.params Dict

# Example
```julia
circuit = Circuit(L=10, bc=:periodic) do c
    apply!(c, Hadamard(), SingleSite(1))
    apply!(c, CNOT(), StaircaseRight(1))
    apply_with_prob!(c; rng=:ctrl, outcomes=[
        (probability=0.5, gate=PauliX(), geometry=SingleSite(2))
    ])
end
```
"""
function Circuit(f::Function; L::Int, bc::Symbol, n_steps::Int=1, kwargs...)
    params = Dict{Symbol,Any}(kwargs)
    builder = CircuitBuilder(L, bc, NamedTuple[], params)
    f(builder)
    return Circuit(L=L, bc=bc, operations=builder.operations, n_steps=n_steps, params=params)
end
