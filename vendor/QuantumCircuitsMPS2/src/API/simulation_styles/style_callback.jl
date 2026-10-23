#=
Style 2: Callback-based simulate_circuits()
==========================================

Philosophy: Structure provided, flexibility via callbacks

Pros:
- Less boilerplate than imperative
- Structure provided - harder to forget steps
- "circuits" parameter matches physicist thinking
- on_circuit! callback gives flexibility without full loop control

Cons:
- Callback pattern can be less intuitive for some
- Slightly hidden control flow
- Harder to do early stopping

When to Use:
Choose this for standard simulations with regular recording patterns
when you want structure but still need per-circuit flexibility.

See also: examples/ct_model_simulation_styles.jl for side-by-side comparison
=#

# This file is meant to be included in the QuantumCircuitsMPS module context

"""
    simulate_circuits(; L, bc, init, circuit_step!, circuits, observables, rng, ...)

Run a simulation measured in circuits (not raw steps).

One circuit = L applications of circuit_step!. This matches physicist intuition:
"run 20 circuits" rather than "run 200 steps" for L=10.

# Arguments
- `circuit_step!::Function`: (state) -> Nothing, applies gates for one step
- `circuits::Int`: Number of circuits to run (total steps = circuits * L)
- `observables::Vector{Pair{Symbol,<:AbstractObservable}}`: e.g., [:DW1 => DomainWall(order=1)]
- `on_circuit!::Function`: Optional (state, circuit_num, get_i1) -> Nothing
- `reset_geometry!::Function`: Optional () -> Nothing, called at START of each circuit
- `record_initial::Bool`: Whether to record at t=0 (default: true)
- `i1_fn::Function`: Optional () -> Int for DomainWall i1

# Example
```julia
left = StaircaseLeft(L)
right = StaircaseRight(1)

circuit_step!(state) = apply_with_prob!(state; ...)

results = simulate_circuits(
    L = L,
    bc = :periodic,
    init = ProductState(x0 = 1//2^L),
    circuit_step! = circuit_step!,
    circuits = 2 * L,
    observables = [:DW1 => DomainWall(order=1)],
    rng = rng,
    on_circuit! = record_every(2),
    reset_geometry! = () -> (left._position = L; right._position = 1),
    i1_fn = () -> (current_position(left) % L) + 1
)
```
"""
function simulate_circuits(;
    L::Int,
    bc::Symbol,
    init::AbstractInitialState,
    circuit_step!::Function,
    circuits::Int,
    observables::Vector,
    rng::RNGRegistry,
    on_circuit!::Union{Function,Nothing} = nothing,
    reset_geometry!::Union{Function,Nothing} = nothing,
    record_initial::Bool = true,
    i1_fn::Union{Function,Nothing} = nothing
)
    # 1. Create and initialize state
    state = SimulationState(L=L, bc=bc, rng=rng)
    initialize!(state, init)
    
    # 2. Register observables
    for (name, obs) in observables
        track!(state, name => obs)
    end
    
    # 3. Helper to get i1
    get_i1 = i1_fn !== nothing ? i1_fn : () -> 1
    
    # 4. Initial recording (t=0)
    if record_initial
        record!(state; i1=get_i1())
    end
    
    # 5. Main loop - by CIRCUITS
    for circuit_num in 1:circuits
        # Reset geometry at start of circuit if provided
        reset_geometry! !== nothing && reset_geometry!()
        
        # Run one circuit = L steps
        for step in 1:L
            circuit_step!(state)
        end
        
        # Call user's on_circuit! callback
        on_circuit! !== nothing && on_circuit!(state, circuit_num, get_i1)
    end
    
    return state.observables
end

# Convenience callbacks

"""
    record_every(n::Int)

Create callback that records every n circuits.
Example: `record_every(2)` records after circuits 2, 4, 6, ...
"""
record_every(n::Int) = (state, circuit_num, get_i1) -> begin
    if circuit_num % n == 0
        record!(state; i1=get_i1())
    end
end

"""
    record_at_circuits(circuit_nums::Vector{Int})

Create callback that records at specific circuit numbers.
Does NOT include circuit 0 (use record_initial=true).

Example: `record_at_circuits([10, 50, 100])`
"""
record_at_circuits(circuit_nums::Vector{Int}) = begin
    circuit_set = Set(circuit_nums)
    (state, circuit_num, get_i1) -> begin
        if circuit_num in circuit_set
            record!(state; i1=get_i1())
        end
    end
end

"""
    record_always()

Create callback that records after every circuit.
"""
record_always() = (state, circuit_num, get_i1) -> record!(state; i1=get_i1())
