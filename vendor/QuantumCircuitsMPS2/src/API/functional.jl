"""
    simulate(; L, bc, init, circuit!, steps, observables, rng, record_at=:every, record_fn=nothing, i1_fn=nothing)

Functional API for running a full simulation.

Arguments:
- L: System size
- bc: Boundary conditions (:open or :periodic)
- init: Initial state specification (AbstractInitialState)
- circuit!: Function (state, t) -> Nothing that applies gates for step t
- steps: Number of simulation steps
- observables: Vector of Pair{Symbol, AbstractObservable} to track
- rng: RNGRegistry for reproducibility
- record_at: When to record observables (:every, :final, or :custom)
- record_fn: Custom recording function (state, t) -> Nothing if record_at=:custom
- i1_fn: Function (state, t) -> Int to determine i1 for DomainWall recording

Returns:
- Dict{Symbol, Vector}: The recorded observables
"""
function simulate(;
    L::Int,
    bc::Symbol,
    init::AbstractInitialState,
    circuit!::Function,           # f(state, t) -> Nothing
    steps::Int,
    observables::Vector,  # Vector{Pair{Symbol, AbstractObservable}} but Julia's type system needs this
    rng::RNGRegistry,
    record_at::Symbol = :every,   # :every | :final | :custom
    record_fn::Union{Function,Nothing} = nothing,
    i1_fn::Union{Function,Nothing} = nothing  # for DomainWall
)
    # 1. Create and initialize state
    state = SimulationState(L=L, bc=bc, rng=rng)
    initialize!(state, init)
    
    # 2. Register observables
    for (name, obs) in observables
        track!(state, name => obs)
    end
    
    # 3. Initial recording (t=0) if :every
    if record_at == :every
        i1 = i1_fn !== nothing ? i1_fn(state, 0) : 1
        record!(state; i1=i1)
    end
    
    # 4. Main simulation loop
    for t in 1:steps
        circuit!(state, t)
        
        if record_at == :every
            i1 = i1_fn !== nothing ? i1_fn(state, t) : 1
            record!(state; i1=i1)
        elseif record_at == :custom && record_fn !== nothing
            record_fn(state, t)
        end
    end
    
    # 5. Final recording if :final
    if record_at == :final
        i1 = i1_fn !== nothing ? i1_fn(state, steps) : 1
        record!(state; i1=i1)
    end
    
    # 6. Return observables dict
    return state.observables
end
