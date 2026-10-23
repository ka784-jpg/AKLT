#=
Style 1: Imperative Loop
========================

Philosophy: Maximum flexibility - user controls EVERYTHING

Pros:
- Maximum flexibility - user controls the entire loop
- Explicit control flow - easy to understand and debug
- No hidden magic - what you write is what happens
- Easy to add conditional logic (early stopping, adaptive recording)

Cons:
- More boilerplate code
- User must remember to call record!() themselves
- No structure enforced

When to Use:
Choose this for complex simulations with irregular recording patterns
or when you need maximum control over every aspect.

See also: examples/ct_model_simulation_styles.jl for side-by-side comparison
=#

# This file is meant to be included in the QuantumCircuitsMPS module context
# where SimulationState, record!, etc. are already defined.
# It should NOT be loaded standalone.

"""
    run_circuit!(state, circuit_step!, L)

Execute one complete circuit (L applications of circuit_step!).

A "circuit" is one full sweep across the system: L steps.
This matches physicist intuition: "run 20 circuits" rather than "run 200 steps".

# Arguments
- `state::SimulationState`: State to evolve
- `circuit_step!::Function`: (state) -> Nothing, applies gates for one step
- `L::Int`: System size (number of steps per circuit)

# Note on circuit_step! signature
This function expects `circuit_step!(state)` (1-arg), not `circuit!(state, t)` (2-arg).
If you have a 2-arg function, wrap it: `circuit_step!(s) = my_circuit!(s, 0)`

# Example
```julia
state = SimulationState(L=10, bc=:periodic, rng=rng)
initialize!(state, ProductState(x0 = 1//2^L))
track!(state, :DW1 => DomainWall(order=1))

left = StaircaseLeft(L)
circuit_step!(s) = apply_with_prob!(s; rng=:ctrl, outcomes=[...])

record!(state; i1=1)  # Initial recording
for circuit in 1:n_circuits
    run_circuit!(state, circuit_step!, L)
    if circuit % 2 == 0
        record!(state; i1=(current_position(left) % L) + 1)
    end
end
```
"""
function run_circuit!(state, circuit_step!::Function, L::Int)
    for step in 1:L
        circuit_step!(state)
    end
    return nothing
end

"""
    run_circuit!(state, circuit_step!, L, reset_geometry!)

Execute one circuit with geometry reset at the START.

# Arguments
- `reset_geometry!::Function`: () -> Nothing, resets geometry state before circuit

# What reset_geometry! should do
Staircases have internal `_position` fields that accumulate across steps.
Your reset function should set these to known starting positions:

```julia
left = StaircaseLeft(L)   # _position starts at L
right = StaircaseRight(1) # _position starts at 1

reset_geometry!() = begin
    left._position = L
    right._position = 1
end
```
"""
function run_circuit!(state, circuit_step!::Function, L::Int, reset_geometry!::Function)
    reset_geometry!()
    for step in 1:L
        circuit_step!(state)
    end
    return nothing
end
