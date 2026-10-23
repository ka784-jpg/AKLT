# === Spin Sector Projectors for S=1 Chains ===
#
# Projectors onto total spin sectors for two spin-1 particles.
# Used for AKLT forced measurement simulations.
#
# Basis ordering: |1,1⟩, |1,0⟩, |1,-1⟩, |0,1⟩, |0,0⟩, |0,-1⟩, |-1,1⟩, |-1,0⟩, |-1,-1⟩
# (i.e., m₁ ∈ {1,0,-1}, m₂ ∈ {1,0,-1}, lexicographic order)

using LinearAlgebra

"""
    spin1_operators()

Return the spin-1 operators Sz, S+, S- as 3×3 matrices.
Basis ordering: |+1⟩, |0⟩, |-1⟩ (descending m).
"""
function spin1_operators()
    # Sz = diag(1, 0, -1)
    Sz = [1.0 0.0 0.0; 
          0.0 0.0 0.0; 
          0.0 0.0 -1.0]
    
    # S+ raises m by 1: S+|m⟩ = √(s(s+1) - m(m+1)) |m+1⟩
    # For s=1: S+|-1⟩ = √2|0⟩, S+|0⟩ = √2|+1⟩, S+|+1⟩ = 0
    Sp = [0.0 sqrt(2.0) 0.0;
          0.0 0.0 sqrt(2.0);
          0.0 0.0 0.0]
    
    Sm = Sp'  # S- = (S+)†
    
    return Sz, Sp, Sm
end

"""
    s1_dot_s2()

Compute the S₁·S₂ operator for two spin-1 particles.
Returns a 9×9 matrix in the tensor product basis.

S₁·S₂ = Sz₁⊗Sz₂ + (1/2)(S+₁⊗S-₂ + S-₁⊗S+₂)
"""
function s1_dot_s2()
    Sz, Sp, Sm = spin1_operators()
    
    # S₁·S₂ = Sz⊗Sz + (1/2)(S+⊗S- + S-⊗S+)
    S1dotS2 = kron(Sz, Sz) + 0.5 * (kron(Sp, Sm) + kron(Sm, Sp))
    
    return S1dotS2
end

"""
    total_spin_projector(S::Int; d::Int=3) -> Matrix{Float64}

Construct the projector onto total spin sector S for two spin-1 particles.

For two spin-1 particles (d=3 each), the tensor product decomposes as:
1 ⊗ 1 = 0 ⊕ 1 ⊕ 2

Returns a d²×d² (9×9) projector matrix.

# Arguments
- `S`: Total spin sector (0, 1, or 2)
- `d`: Local dimension (default 3 for spin-1)

# Returns
- `P_S`: Projector matrix onto sector S

# Examples
```julia
P0 = total_spin_projector(0)  # Singlet projector (dim=1)
P1 = total_spin_projector(1)  # Triplet projector (dim=3)
P2 = total_spin_projector(2)  # Quintet projector (dim=5)

# Verify completeness
@assert P0 + P1 + P2 ≈ I(9)

# Verify idempotence
@assert P0 * P0 ≈ P0
```

# Physics
The projector formulas are derived from Clebsch-Gordan decomposition:
- P₂ = (1/6)(S₁·S₂)² + (1/2)(S₁·S₂) + (1/3)I
- P₁ = -(1/2)(S₁·S₂)² - (1/2)(S₁·S₂) + I
- P₀ = (1/3)(S₁·S₂)² - (1/3)I
"""
function total_spin_projector(S::Int; d::Int=3)
    d == 3 || throw(ArgumentError("Only d=3 (spin-1) is currently supported"))
    S in (0, 1, 2) || throw(ArgumentError("S must be 0, 1, or 2 for spin-1 ⊗ spin-1"))
    
    S1S2 = s1_dot_s2()
    S1S2_sq = S1S2 * S1S2
    I9 = Matrix{Float64}(I, 9, 9)
    
    if S == 2
        # P₂ = (1/6)(S₁·S₂)² + (1/2)(S₁·S₂) + (1/3)I
        P = (1/6) * S1S2_sq + (1/2) * S1S2 + (1/3) * I9
    elseif S == 1
        # P₁ = -(1/2)(S₁·S₂)² - (1/2)(S₁·S₂) + I
        P = -(1/2) * S1S2_sq - (1/2) * S1S2 + I9
    else  # S == 0
        # P₀ = (1/3)(S₁·S₂)² - (1/3)I
        P = (1/3) * S1S2_sq - (1/3) * I9
    end
    
    return P
end

"""
    verify_spin_projectors(; tol::Float64=1e-10)

Verify that the spin projectors satisfy required properties.
Returns true if all checks pass, throws error otherwise.

Checks:
1. Completeness: P₀ + P₁ + P₂ = I
2. Idempotence: Pₛ² = Pₛ for all S
3. Orthogonality: Pᵢ·Pⱼ = 0 for i ≠ j
4. Correct dimensions: tr(P₀)=1, tr(P₁)=3, tr(P₂)=5
"""
function verify_spin_projectors(; tol::Float64=1e-10)
    P0 = total_spin_projector(0)
    P1 = total_spin_projector(1)
    P2 = total_spin_projector(2)
    I9 = Matrix{Float64}(I, 9, 9)
    
    # Completeness
    @assert norm(P0 + P1 + P2 - I9) < tol "Completeness failed: P₀ + P₁ + P₂ ≠ I"
    
    # Idempotence
    @assert norm(P0 * P0 - P0) < tol "Idempotence failed for P₀"
    @assert norm(P1 * P1 - P1) < tol "Idempotence failed for P₁"
    @assert norm(P2 * P2 - P2) < tol "Idempotence failed for P₂"
    
    # Orthogonality
    @assert norm(P0 * P1) < tol "Orthogonality failed: P₀·P₁ ≠ 0"
    @assert norm(P0 * P2) < tol "Orthogonality failed: P₀·P₂ ≠ 0"
    @assert norm(P1 * P2) < tol "Orthogonality failed: P₁·P₂ ≠ 0"
    
    # Correct dimensions (trace = dimension of sector)
    @assert abs(tr(P0) - 1) < tol "Trace failed: tr(P₀) ≠ 1"
    @assert abs(tr(P1) - 3) < tol "Trace failed: tr(P₁) ≠ 3"
    @assert abs(tr(P2) - 5) < tol "Trace failed: tr(P₂) ≠ 5"
    
    return true
end
