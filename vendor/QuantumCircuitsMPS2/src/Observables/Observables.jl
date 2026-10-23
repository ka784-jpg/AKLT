using ITensors
using ITensorMPS

"""
Abstract base type for observable specifications.
"""
abstract type AbstractObservable end

# Include implementations
include("born.jl")
include("domain_wall.jl")
include("entanglement.jl")
include("string_order.jl")
include("bond_dimension.jl")
include("SelfAveraging.jl")
include("dimer_order.jl") 


# === Observable Tracking API ===

"""
    track!(state::SimulationState, spec::Pair{Symbol, AbstractObservable})

Register an observable to be tracked. Values are stored in state.observables.

Example: track!(state, :dw1 => DomainWall(order=1))
"""
function track!(state, spec::Pair{Symbol, <:AbstractObservable})
    name, obs = spec
    state.observable_specs[name] = obs
    state.observables[name] = Float64[]
    return nothing
end

"""
    record!(state::SimulationState; i1::Union{Int,Nothing}=nothing)

Compute all tracked observables and append values to state.observables.

For DomainWall observables:
- If i1_fn was provided at construction, it will be called automatically
- Otherwise, i1 parameter must be provided explicitly
"""
function record!(state; i1::Union{Int,Nothing}=nothing)
    for (name, obs) in state.observable_specs
        if obs isa DomainWall
            if obs.i1_fn !== nothing
                # i1_fn is set - call observable without i1, it will use i1_fn
                value = obs(state)
            elseif i1 !== nothing
                # Explicit i1 passed - use it
                value = obs(state, i1)
            else
                throw(ArgumentError(
                    "DomainWall '$name' requires either i1_fn at registration or i1 at record! call"
                ))
            end
        else
            value = obs(state)
        end
        push!(state.observables[name], value)
    end
    return nothing
end

"""
    list_observables() -> Vector{String}

Return a list of available observable type names.

Returns the names of all observable types that can be used with the tracking API.

Example:
```julia
obs_types = list_observables()
# Returns: ["DomainWall", "BornProbability"]
```
"""
function list_observables()::Vector{String}
    return ["DomainWall", "BornProbability", "EntanglementEntropy", "StringOrder", "BondDimension","Dimer"]
end
