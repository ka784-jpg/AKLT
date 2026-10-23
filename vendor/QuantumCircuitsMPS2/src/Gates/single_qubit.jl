# === Custom S=1 Zero Projector ===

struct FZeroGate <: QuantumCircuitsMPS2.AbstractGate
    tau::Float64
end
QuantumCircuitsMPS2.support(::FZeroGate) = 1
function QuantumCircuitsMPS2.build_operator(gate::FZeroGate, site::Index, local_dim::Int; kwargs...)
    projzero = [0.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 0.0]
    mat = exp(projzero * gate.tau)
    return itensor(mat, site', site)
end 


"""
    ZeroProjector(tau::Float64)
 
Projector onto the S=1 zero state, evolved in imaginary time by tau.
Computes exp(|0⟩⟨0| * tau).
"""
struct ZeroProjector <: AbstractGate
    tau::Float64
end
support(::ZeroProjector) = 1

"""
    build_operator(gate::ZeroProjector, site::Index, local_dim::Int) -> ITensor

Build the Zero Projector operator tensor.
"""
function build_operator(gate::ZeroProjector, site::Index, local_dim::Int; kwargs...)
    projzero = [0.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 0.0]
    mat = exp(projzero * gate.tau)
    return itensor(mat, site', site)
end

# === Single-Qubit Gates ===

"""Pauli X gate (NOT gate, bit flip)."""
struct PauliX <: AbstractGate end
support(::PauliX) = 1

"""Pauli Y gate."""
struct PauliY <: AbstractGate end
support(::PauliY) = 1

"""Pauli Z gate (phase flip)."""
struct PauliZ <: AbstractGate end
support(::PauliZ) = 1

"""
    TGate

Single-qubit T gate (π/8 phase gate).
Applies a phase of e^(iπ/4) to the |1⟩ computational basis state.
"""
struct TGate <: AbstractGate end
support(::TGate) = 1

# === Single-Qubit Rotation Gates ===

"""
    Rx(theta::Float64)
Rotation around the X-axis by angle theta.
"""
struct Rx <: AbstractGate
    theta::Float64
end
support(::Rx) = 1

"""
    Ry(theta::Float64)
Rotation around the Y-axis by angle theta.
"""
struct Ry <: AbstractGate
    theta::Float64
end
support(::Ry) = 1

"""
    Rz(theta::Float64)
Rotation around the Z-axis by angle theta.
"""
struct Rz <: AbstractGate
    theta::Float64
end
support(::Rz) = 1

"""
    Projection(outcome::Int)

Projector onto computational basis state |outcome⟩.
outcome=0 projects onto |0⟩, outcome=1 projects onto |1⟩.
"""
struct Projection <: AbstractGate
    outcome::Int
    
    function Projection(outcome::Int)
        outcome in (0, 1) || throw(ArgumentError("Projection outcome must be 0 or 1, got $outcome"))
        new(outcome)
    end
end
support(::Projection) = 1

# === build_operator implementations ===

"""
    build_operator(gate::PauliX, site::Index, local_dim::Int) -> ITensor

Build Pauli X operator tensor.
"""
function build_operator(gate::PauliX, site::Index, local_dim::Int; kwargs...)
    # Use ITensors' built-in op function
    return op("X", site)
end

"""
    build_operator(gate::PauliY, site::Index, local_dim::Int) -> ITensor

Build Pauli Y operator tensor.
"""
function build_operator(gate::PauliY, site::Index, local_dim::Int; kwargs...)
    return op("Y", site)
end

"""
    build_operator(gate::PauliZ, site::Index, local_dim::Int) -> ITensor

Build Pauli Z operator tensor.
"""
function build_operator(gate::PauliZ, site::Index, local_dim::Int; kwargs...)
    return op("Z", site)
end

"""
    build_operator(gate::Projection, site::Index, local_dim::Int) -> ITensor

Build projection operator |outcome⟩⟨outcome|.
"""
function build_operator(gate::Projection, site::Index, local_dim::Int; kwargs...)
    if gate.outcome == 0
        return op("Proj0", site)  # |0⟩⟨0|
    else
        return op("Proj1", site)  # |1⟩⟨1|
    end
end

"""
    build_operator(gate::TGate, site::Index, local_dim::Int) -> ITensor

Build the T gate operator tensor using ITensors' built-in "T" operator.
"""
function build_operator(gate::TGate, site::Index, local_dim::Int; kwargs...)
    return op("T", site)
end

# === build_operator implementations ===

function build_operator(gate::Rx, site::Index, local_dim::Int; kwargs...)
    c = cos(gate.theta / 2.0)
    s = sin(gate.theta / 2.0)
    
    mat = [c + 0.0im      0.0 - s*im;
           0.0 - s*im     c + 0.0im]
           
    return itensor(mat, site', site)
end

function build_operator(gate::Ry, site::Index, local_dim::Int; kwargs...)
    c = cos(gate.theta / 2.0)
    s = sin(gate.theta / 2.0)
    
    mat = [c + 0.0im     -s + 0.0im;
           s + 0.0im      c + 0.0im]
           
    return itensor(mat, site', site)
end

function build_operator(gate::Rz, site::Index, local_dim::Int; kwargs...)
    phase = exp(-im * gate.theta / 2.0)
    
    mat = [phase          0.0 + 0.0im;
           0.0 + 0.0im    conj(phase)]
           
    return itensor(mat, site', site)
end
