# === Circuit Execution Engine ===
# Executes Circuit objects on SimulationState with stochastic branch resolution

"""
    simulate!(circuit::Circuit, state::SimulationState; n_circuits::Int=1, record_when::Union{Symbol,Function}=:every_step)

Execute a circuit on a simulation state, applying gates and recording observables.

This function runs the circuit `n_circuits` times, with each run executing all `circuit.n_steps` timesteps.
For each timestep, all operations in `circuit.operations` are processed in order.

# Arguments
- `circuit::Circuit`: The circuit to execute (symbolic operations)
- `state::SimulationState`: The state to modify in-place
- `n_circuits::Int`: Number of times to execute the full circuit (default: 1)
- `record_when::Union{Symbol,Function}`: Controls when observables are recorded (default: `:every_step`)

# Recording Options
The `record_when` parameter accepts:
- `:every_step` (default): Record once per circuit, after the last gate of the last timestep
- `:every_gate`: Record after every gate execution
- `:final_only`: Record only after the final circuit completes
- Custom function `(ctx::RecordingContext) -> Bool`: Record when function returns true

# RecordingContext Fields
Custom recording functions receive a `RecordingContext` with:
- `step_idx::Int`: Current circuit execution index (1 to n_circuits)  
- `gate_idx::Int`: Cumulative gate count across all circuits (never resets)
- `gate_type::Any`: The gate being applied
- `is_step_boundary::Bool`: True when at the last gate of the current timestep

# Examples
```julia
# Basic execution with 5 runs
circuit = Circuit(L=4, bc=:periodic, n_steps=10) do c
    apply_with_prob!(c; rng=:ctrl, outcomes=[
        (probability=0.5, gate=Reset(), geometry=StaircaseRight(1)),
        (probability=0.5, gate=HaarRandom(), geometry=StaircaseLeft(4))
    ])
end

rng = RNGRegistry(ctrl=42, proj=43, haar=44, born=45)
state = SimulationState(L=4, bc=:periodic, rng=rng)
initialize!(state, ProductState(binary_int=1))
track!(state, :dw => DomainWall(order=1, i1_fn=() -> 1))

# Record after every circuit (default)
simulate!(circuit, state; n_circuits=5)

# Record after every gate
simulate!(circuit, state; n_circuits=5, record_when=:every_gate)

# Record only at the end
simulate!(circuit, state; n_circuits=5, record_when=:final_only)

# Custom: record every 10 gates
simulate!(circuit, state; n_circuits=5, record_when=every_n_gates(10))

# Custom: record every 2 steps
simulate!(circuit, state; n_circuits=10, record_when=every_n_steps(2))

# Custom: record at specific gate
simulate!(circuit, state; n_circuits=5, record_when=ctx -> ctx.gate_idx == 100)
```

# Migration from Old API
Breaking change: The old `record_initial` and `record_every` parameters have been removed.

Old code (no longer works):
```julia
simulate!(circuit, state; n_circuits=100, record_initial=true, record_every=10)
```

New equivalent:
```julia
record!(state)  # Record initial state if desired
simulate!(circuit, state; n_circuits=100, record_when=every_n_steps(10))
```

# RNG Alignment
RNG consumption MUST match `expand_circuit` exactly:
- For each stochastic operation: consume exactly ONE `rand()` call
- Selection logic: cumulative probability with STRICT `<` comparison
- Same seed in state.rng_registry[op.rng] → same branch selection

# Validation
- Throws `ArgumentError` if `n_circuits < 1`
- Throws `ArgumentError` for unknown symbol presets

# See Also
- `expand_circuit`: Visualize which gates will be applied
- `record!`: Manual observable recording
- `RecordingContext`: Context passed to custom recording functions
- `every_n_gates`, `every_n_steps`: Helper functions for common patterns
"""
function simulate!(circuit::Circuit, state::SimulationState;
                   n_circuits::Int=1,
                   record_when::Union{Symbol,Function}=:every_step)
    # Validation
    n_circuits >= 1 || throw(ArgumentError("n_circuits must be >= 1, got $n_circuits"))
    if record_when isa Symbol && record_when ∉ (:every_step, :every_gate, :final_only)
        throw(ArgumentError("Unknown record_when symbol: $record_when. Valid options: :every_step, :every_gate, :final_only"))
    end
    
    # Gate index tracks cumulative gates executed across ALL circuits
    gate_idx = 0
    
    # Execute n_circuits repetitions
    for circuit_idx in 1:n_circuits
        should_record_this_step = false  # Flag for this circuit
        
        # Execute all n_steps of this circuit
        for step in 1:circuit.n_steps
            for (op_idx, op) in enumerate(circuit.operations)
                gate_executed = false
                current_gate = nothing
                
                if op.type == :deterministic
                    if is_compound_geometry(op.geometry)
                        # Compound geometry: iterate over elements
                        elements = get_compound_elements(op.geometry, circuit.L, circuit.bc)
                        for sites in elements
                            execute_gate!(state, op.gate, sites)
                            gate_idx += 1
                            is_step_boundary = (step == circuit.n_steps) && (op_idx == length(circuit.operations)) && (sites == elements[end])
                            ctx = RecordingContext(circuit_idx, gate_idx, op.gate, is_step_boundary)
                            
                            # Evaluate recording
                            set_flag, record_now = _evaluate_recording(record_when, ctx, circuit_idx, n_circuits)
                            should_record_this_step |= set_flag
                            record_now && record!(state)
                        end
                        gate_executed = false  # Already handled above
                        current_gate = nothing
                    else
                        # Simple geometry: existing path
                        sites = compute_sites_dispatch(op.geometry, op.gate, step, circuit.L, circuit.bc)
                        execute_gate!(state, op.gate, sites)
                        gate_executed = true
                        current_gate = op.gate
                    end
                    
                elseif op.type == :stochastic
                    actual_rng = get_rng(state.rng_registry, op.rng)
                    
                    # Check if ANY outcome has compound geometry
                    has_compound = any(is_compound_geometry(o.geometry) for o in op.outcomes)
                    
                    if has_compound
                        # Compound stochastic: draw RNG ONCE to select outcome, then apply to ALL elements of selected geometry
                        r = rand(actual_rng)  # Single draw to select which outcome
                        cumulative = 0.0
                        selected_outcome = nothing
                        for outcome in op.outcomes
                            cumulative += outcome.probability
                            if r < cumulative
                                selected_outcome = outcome
                                break
                            end
                        end
                        
                        if selected_outcome !== nothing
                            # Get elements from the SELECTED outcome's geometry
                            elements = get_compound_elements(selected_outcome.geometry, circuit.L, circuit.bc)
                            
                            for sites in elements
                                execute_gate!(state, selected_outcome.gate, sites)
                                gate_idx += 1
                                is_step_boundary = (step == circuit.n_steps) && (op_idx == length(circuit.operations)) && (sites == elements[end])
                                ctx = RecordingContext(circuit_idx, gate_idx, selected_outcome.gate, is_step_boundary)
                                
                                set_flag, record_now = _evaluate_recording(record_when, ctx, circuit_idx, n_circuits)
                                should_record_this_step |= set_flag
                                record_now && record!(state)
                            end
                        end
                        # If no outcome selected: "do nothing" - no gates applied
                        
                        # For :every_step mode with compound geometry, always check step boundary
                        is_step_boundary = (step == circuit.n_steps) && (op_idx == length(circuit.operations))
                        if is_step_boundary && _should_record_at_step_boundary(record_when, circuit_idx, n_circuits)
                            should_record_this_step = true
                        end
                        
                        gate_executed = false  # Already handled
                        current_gate = nothing
                    else
                        # Simple stochastic: existing single-draw path
                        r = rand(actual_rng)
                        cumulative = 0.0
                        for outcome in op.outcomes
                            cumulative += outcome.probability
                            if r < cumulative
                                sites = compute_sites_dispatch(outcome.geometry, outcome.gate, step, circuit.L, circuit.bc)
                                execute_gate!(state, outcome.gate, sites)
                                gate_executed = true
                                current_gate = outcome.gate
                                break
                            end
                        end
                    end
                end
                
                # Only process recording logic if a gate was actually executed
                if gate_executed
                    gate_idx += 1
                    is_step_boundary = (step == circuit.n_steps) && (op_idx == length(circuit.operations))
                    ctx = RecordingContext(circuit_idx, gate_idx, current_gate, is_step_boundary)
                    
                    # Evaluate recording criteria
                    set_flag, record_now = _evaluate_recording(record_when, ctx, circuit_idx, n_circuits)
                    should_record_this_step |= set_flag
                    record_now && record!(state)
                end
            end
        end
        
        # Record after this circuit completes (flag set by :every_step, :final_only, or custom function)
        if should_record_this_step
            record!(state)
        end
    end
    
    return nothing
end

"""
    execute_gate!(state::SimulationState, gate::AbstractGate, sites::Vector{Int})

Apply a gate to specific sites, handling special cases like Reset.

# Special Cases
- **Reset**: Must use `SingleSite` wrapper because Reset's `build_operator` throws for Vector{Int}.
  This triggers the specialized `_apply_dispatch!(state, ::Reset, ::SingleSite)` method.
- **All other gates**: Use `apply!(state, gate, sites)` directly.

# Arguments
- `state`: The simulation state to modify
- `gate`: The gate to apply
- `sites`: Physical site indices for this operation

# Internal Implementation Detail
This function is part of the circuit execution engine and should not be called directly
by users. Use `simulate!` or the imperative `apply!` API instead.
"""
function execute_gate!(state::SimulationState, gate::AbstractGate, sites::Vector{Int})
    if gate isa Reset
        # Reset requires SingleSite wrapper to trigger correct dispatch
        # (Reset's build_operator(gate, ::Vector{Int}) throws error)
        site = sites[1]  # Reset is always single-site
        apply!(state, gate, SingleSite(site))
    elseif gate isa Measurement
        # Measurement requires SingleSite wrapper (like Reset)
        site = sites[1]  # Measurement is always single-site
        apply!(state, gate, SingleSite(site))
    else
        # Normal gates use sites vector directly
        apply!(state, gate, sites)
    end
end

# Note: compute_sites_dispatch is defined in expand.jl (shared helper)
