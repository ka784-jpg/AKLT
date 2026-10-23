# === Composite Gates ===

"""
    Measurement(axis::Symbol)

Pure projective measurement in the specified basis.

- `Measurement(:Z)` - Z-basis measurement: projects to |0⟩ or |1⟩ based on Born probability

This is a FUNDAMENTAL operation. Unlike `Reset`, measurement leaves the qubit
in the measured state (|0⟩ or |1⟩) without resetting it.

# Physics
1. Compute Born probability P(0|ψ) for the qubit
2. Sample outcome ∈ {0, 1} according to Born rule
3. Apply projection operator to collapse wavefunction
4. Normalize the state

# Example
```julia
apply!(state, Measurement(:Z), SingleSite(1))  # Measure qubit 1 in Z-basis
apply!(state, Measurement(:Z), AllSites())     # Measure all qubits
```
"""
struct Measurement <: AbstractGate
    axis::Symbol
    
    function Measurement(axis::Symbol)
        axis == :Z || throw(ArgumentError("Only :Z axis supported currently. Got: $axis"))
        new(axis)
    end
end

support(::Measurement) = 1

# Measurement requires Born sampling - cannot be a simple operator
function build_operator(gate::Measurement, site::Index, local_dim::Int; kwargs...)
    error("Measurement gate cannot be built as a single operator. Use apply!(state, Measurement(:Z), geo) instead.")
end

"""
    Reset

Reset gate: projects to |0⟩ or |1⟩ based on Born probability, then flips to |0⟩ if needed.
Equivalent to measure + conditional X.
"""
struct Reset <: AbstractGate end
support(::Reset) = 1

# Note: Reset doesn't have a simple build_operator because it requires
# measurement + conditional logic. It's handled specially in apply!.
# We provide a stub that throws to catch misuse.
function build_operator(gate::Reset, site::Index, local_dim::Int; kwargs...)
    error("Reset gate cannot be built as a single operator. Use apply!(state, Reset(), geo) instead.")
end
