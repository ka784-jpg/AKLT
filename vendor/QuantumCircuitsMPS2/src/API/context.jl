# Context API - thread-local state management

const CURRENT_STATE = Ref{Union{SimulationState, Nothing}}(nothing)

"""
    with_state(fn::Function, state::SimulationState)

Execute `fn` with `state` as the implicit current state.
Allows calling `apply!(gate, geo)` without passing the state explicitly.
"""
function with_state(fn::Function, state::SimulationState)
    old = CURRENT_STATE[]
    CURRENT_STATE[] = state
    try
        fn()
    finally
        CURRENT_STATE[] = old
    end
end

"""
    current_state()::SimulationState

Retrieve the current implicit simulation state.
Throws an error if no state is set.
"""
function current_state()::SimulationState
    s = CURRENT_STATE[]
    s === nothing && error("No current state. Use with_state(state) do ... end")
    return s
end

# Implicit apply! overloads (call explicit version from Core/apply.jl)
"""
    apply!(gate::AbstractGate, geo::AbstractGeometry)
    apply!(gate::AbstractGate, sites::Vector{Int})
    apply!(gate::AbstractGate, site::Int)

Apply a gate to the current implicit state.
"""
apply!(gate::AbstractGate, geo::AbstractGeometry) = apply!(current_state(), gate, geo)
apply!(gate::AbstractGate, sites::Vector{Int}) = apply!(current_state(), gate, sites)
apply!(gate::AbstractGate, site::Int) = apply!(current_state(), gate, [site])
