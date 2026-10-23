"""
    RecordingContext

Context information passed to recording predicate functions during circuit execution.

# Fields
- `step_idx::Int`: Current circuit execution index (1 to n_circuits)
- `gate_idx::Int`: Cumulative gate count across all steps (never resets)
- `gate_type::Any`: The gate being applied
- `is_step_boundary::Bool`: True when at the last gate of the current step

# Example
```julia
# Custom recording function
function my_recorder(ctx::RecordingContext)
    return ctx.gate_idx > 10 && ctx.is_step_boundary
end
```
"""
struct RecordingContext
    step_idx::Int
    gate_idx::Int
    gate_type::Any
    is_step_boundary::Bool
end

"""
    every_n_gates(n::Int)

Create a recording predicate that triggers every `n` gates.

# Arguments
- `n::Int`: Record every n gates (based on cumulative gate_idx)

# Returns
Function that takes a `RecordingContext` and returns `Bool`

# Example
```julia
simulate!(state, circuit; record_when=every_n_gates(5))
```
"""
function every_n_gates(n::Int)
    return ctx -> ctx.gate_idx % n == 0
end

"""
    every_n_steps(n::Int)

Create a recording predicate that triggers every `n` steps at step boundaries.

Records once per n steps, only when `is_step_boundary` is true (after all gates 
in the step have been executed).

# Arguments
- `n::Int`: Record every n steps (based on step_idx)

# Returns
Function that takes a `RecordingContext` and returns `Bool`

# Example
```julia
simulate!(state, circuit; record_when=every_n_steps(2))
```
"""
function every_n_steps(n::Int)
    return ctx -> ctx.step_idx % n == 0 && ctx.is_step_boundary
end

# === Recording Evaluation Helpers (used by simulate!) ===

"""
    _evaluate_recording(record_when, ctx, circuit_idx, n_circuits) -> (set_flag::Bool, record_now::Bool)

Evaluate recording criteria and return whether to set the recording flag and/or record immediately.

For compound geometries (called inside element loops), `:every_gate` triggers immediate recording.
For simple geometries, the caller handles the immediate recording after setting the flag.

Returns a tuple:
- `set_flag`: Whether to set `should_record_this_step = true`
- `record_now`: Whether to call `record!(state)` immediately
"""
function _evaluate_recording(record_when::Symbol, ctx::RecordingContext, circuit_idx::Int, n_circuits::Int)
    is_step_boundary = ctx.is_step_boundary
    
    if record_when == :every_step && is_step_boundary
        return (true, false)
    elseif record_when == :every_gate
        return (false, true)  # Record immediately for compound geometry case
    elseif record_when == :final_only && is_step_boundary && circuit_idx == n_circuits
        return (true, false)
    else
        return (false, false)
    end
end

function _evaluate_recording(record_when::Function, ctx::RecordingContext, circuit_idx::Int, n_circuits::Int)
    if record_when(ctx)
        return (true, false)
    else
        return (false, false)
    end
end

"""
    _should_record_at_step_boundary(record_when, circuit_idx, n_circuits) -> Bool

Check if recording should occur at a step boundary (simplified check without gate context).
Used after compound geometry loops where step boundary may need to be handled.
"""
function _should_record_at_step_boundary(record_when::Symbol, circuit_idx::Int, n_circuits::Int)
    record_when == :every_step || (record_when == :final_only && circuit_idx == n_circuits)
end

_should_record_at_step_boundary(record_when::Function, circuit_idx::Int, n_circuits::Int) = false
