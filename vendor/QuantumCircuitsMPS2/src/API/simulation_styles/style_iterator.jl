#=
Style 3: Iterator Pattern
=========================

Philosophy: Lazy evaluation, user controls loop with iterator utilities

Pros:
- Clean separation of setup vs. execution
- Lazy evaluation - compute only what you need
- User controls loop but with less boilerplate
- Composable with Julia's Iterators (take, drop, enumerate)
- Natural for "run until condition" patterns

Cons:
- Iterator semantics might be unfamiliar
- Yields same mutable object (not copies!)
- Debugging can be trickier

When to Use:
Choose this for exploratory simulation where you want to inspect
state between circuits, or for "run until convergence" patterns.

See also: examples/ct_model_simulation_styles.jl for side-by-side comparison
=#

# This file is meant to be included in the QuantumCircuitsMPS module context

"""
    CircuitSimulation

A lazy simulation iterator. Each iteration runs one circuit (L steps).

**WARNING**: Each iteration yields the SAME mutable state object.
If you need snapshots, copy the state yourself.

# Usage
```julia
sim = CircuitSimulation(L=10, bc=:periodic, ...)

# Run 20 circuits, record every 2
record!(sim.state; i1=1)  # Initial
for (n, state) in enumerate(Iterators.take(sim, 20))
    if n % 2 == 0
        record!(state; i1=get_i1())
    end
end

results = get_observables(sim)
```
"""
mutable struct CircuitSimulation
    L::Int
    bc::Symbol
    circuit_step!::Function
    reset_geometry!::Union{Function,Nothing}
    state::SimulationState
    circuit_count::Int
end

"""
    CircuitSimulation(; L, bc, init, circuit_step!, observables, rng, reset_geometry!=nothing)

Create a lazy circuit simulation iterator.

# Arguments
- `circuit_step!::Function`: (state) -> Nothing
- `observables::Vector{Pair{Symbol,<:AbstractObservable}}`
- `reset_geometry!::Function`: Optional () -> Nothing, called at START of each circuit

# Example
```julia
left = StaircaseLeft(L)

sim = CircuitSimulation(
    L = 10,
    bc = :periodic,
    init = ProductState(x0 = 1//2^10),
    circuit_step! = state -> apply_with_prob!(state; ...),
    observables = [:DW1 => DomainWall(order=1)],
    rng = rng,
    reset_geometry! = () -> (left._position = L)
)
```
"""
function CircuitSimulation(;
    L::Int,
    bc::Symbol,
    init::AbstractInitialState,
    circuit_step!::Function,
    observables::Vector,
    rng::RNGRegistry,
    reset_geometry!::Union{Function,Nothing} = nothing
)
    state = SimulationState(L=L, bc=bc, rng=rng)
    initialize!(state, init)
    for (name, obs) in observables
        track!(state, name => obs)
    end
    return CircuitSimulation(L, bc, circuit_step!, reset_geometry!, state, 0)
end

# Julia iteration protocol

function Base.iterate(sim::CircuitSimulation)
    sim.reset_geometry! !== nothing && sim.reset_geometry!()
    for step in 1:sim.L
        sim.circuit_step!(sim.state)
    end
    sim.circuit_count = 1
    return (sim.state, sim.circuit_count)
end

function Base.iterate(sim::CircuitSimulation, prev_circuit::Int)
    sim.reset_geometry! !== nothing && sim.reset_geometry!()
    for step in 1:sim.L
        sim.circuit_step!(sim.state)
    end
    sim.circuit_count = prev_circuit + 1
    return (sim.state, sim.circuit_count)
end

Base.IteratorSize(::Type{CircuitSimulation}) = Base.IsInfinite()
Base.eltype(::Type{CircuitSimulation}) = SimulationState

# Convenience methods

"""Get current simulation state."""
get_state(sim::CircuitSimulation) = sim.state

"""Get recorded observables dictionary."""
get_observables(sim::CircuitSimulation) = sim.state.observables

"""Get number of circuits run so far."""
circuits_run(sim::CircuitSimulation) = sim.circuit_count

"""
    run!(sim::CircuitSimulation, n_circuits::Int)

Run n circuits without yielding. Useful for burn-in.
"""
function run!(sim::CircuitSimulation, n_circuits::Int)
    for _ in 1:n_circuits
        sim.reset_geometry! !== nothing && sim.reset_geometry!()
        for step in 1:sim.L
            sim.circuit_step!(sim.state)
        end
        sim.circuit_count += 1
    end
    return sim
end
