# === Two-Qubit Gates ===

"""
    HaarRandom

Two-qubit Haar random unitary gate.
Requires RNG from state for reproducibility.
"""
struct HaarRandom <: AbstractGate end
support(::HaarRandom) = 2

"""
    CZ

Controlled-Z gate. Symmetric under qubit exchange.
|00⟩→|00⟩, |01⟩→|01⟩, |10⟩→|10⟩, |11⟩→-|11⟩
"""
struct CZ <: AbstractGate end
support(::CZ) = 2

"""
    build_operator(gate::HaarRandom, sites::Vector{Index}, local_dim::Int; rng::RNGRegistry) -> ITensor

Build Haar random unitary operator.
Uses exact CT.jl algorithm for reproducibility (from CT.jl U() function lines 585-592).
"""
function build_operator(gate::HaarRandom, sites::Vector{<:Index}, local_dim::Int; rng)
    length(sites) == 2 || throw(ArgumentError("HaarRandom requires exactly 2 sites"))
    
    # Get the Haar RNG stream
    haar_rng = get_rng(rng, :haar)
    
    # CT.jl U(n, rng) algorithm - EXACT reproduction
    n = local_dim^2  # 4 for qubits
    
    # Generate complex Gaussian matrix: real + imag parts separately
    z = randn(haar_rng, n, n) + randn(haar_rng, n, n) * im
    
    # QR decomposition
    Q, R = qr(z)
    Q = Matrix(Q)  # Convert from QRCompactWY to Matrix
    
    # Phase correction: multiply by diagonal of R/|R|
    r_diag = diag(R)
    Lambda = Diagonal(r_diag ./ abs.(r_diag))
    U_matrix = Q * Lambda
    
    # Build ITensor from 4x4 matrix
    U_4 = reshape(U_matrix, 2, 2, 2, 2)
    s1, s2 = sites[1], sites[2]
    op_tensor = ITensor(U_4, s1, s2, s1', s2')
    
    return op_tensor
end

"""
    build_operator(gate::CZ, sites::Vector{Index}, local_dim::Int) -> ITensor

Build CZ gate operator.
"""
function build_operator(gate::CZ, sites::Vector{<:Index}, local_dim::Int; rng=nothing)
    length(sites) == 2 || throw(ArgumentError("CZ requires exactly 2 sites"))
    
    s1, s2 = sites[1], sites[2]
    
    # CZ matrix: diagonal with -1 at |11⟩
    # |00⟩→|00⟩, |01⟩→|01⟩, |10⟩→|10⟩, |11⟩→-|11⟩
    op_tensor = ITensor(ComplexF64, s1', s2', dag(s1), dag(s2))
    
    for i1 in 1:local_dim, i2 in 1:local_dim
        for j1 in 1:local_dim, j2 in 1:local_dim
            if i1 == j1 && i2 == j2  # diagonal
                if i1 == local_dim && i2 == local_dim  # |11⟩ state
                    op_tensor[s1' => i1, s2' => i2, s1 => j1, s2 => j2] = -1.0 + 0.0im
                else
                    op_tensor[s1' => i1, s2' => i2, s1 => j1, s2 => j2] = 1.0 + 0.0im
                end
            end
        end
    end
    
    return op_tensor
end
