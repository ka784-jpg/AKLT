# === Two-Site Spin Sector Operations for S=1 Chains ===
#
# This module provides two distinct operations for AKLT forced measurement:
# 1. SpinSectorProjection: Coherent projection preserving superposition
# 2. SpinSectorMeasurement: Born rule measurement with outcome collapse

"""
    SpinSectorProjection(projector::Matrix{Float64})

Coherent projection onto specified spin sectors (no measurement/collapse).

Applies projector operator P to two adjacent spin-1 sites, then renormalizes:
    |ψ⟩ → P|ψ⟩ / ||P|ψ⟩||

# Example
```julia
# Project onto S=0 and S=1 sectors (remove S=2)
P01 = total_spin_projector(0) + total_spin_projector(1)
gate = SpinSectorProjection(P01)
```

# Physics
This is a coherent operation that preserves quantum superposition.
For AKLT: Repeated application of P₀+P₁ should converge to ground state.
"""
struct SpinSectorProjection <: AbstractGate
    projector::Matrix{Float64}
    
    function SpinSectorProjection(projector::Matrix{Float64})
        # Validate projector is 9×9 (two spin-1 particles)
        size(projector) == (9, 9) || throw(ArgumentError(
            "SpinSectorProjection requires 9×9 projector for two spin-1 sites"
        ))
        return new(projector)
    end
end

support(::SpinSectorProjection) = 2

"""
    SpinSectorMeasurement(sectors::Vector{Int}=)

True Born measurement of total spin sector for two adjacent spin-1 sites.

Performs projective measurement that collapses the state to a definite spin sector.
Outcome probabilities follow Born rule: P(S) = ⟨ψ|Pₛ|ψ⟩

# Arguments
- `sectors`: Which sectors to measure (default: [0, 1, 2] for all sectors)

# Example
```julia
# Measure all three sectors
gate = SpinSectorMeasurement([0, 1, 2])

# Measure only S=0 or S=1 (post-select)
gate = SpinSectorMeasurement([0, 1])
```

# Physics
This is the research question: Does forced measurement to S∈{0,1} produce
different physics than coherent projection? Unknown behavior to explore.

# Returns
After application, the measurement outcome S can be retrieved from state history.
"""
struct SpinSectorMeasurement <: AbstractGate
    sectors::Vector{Int}
    
    function SpinSectorMeasurement(sectors::Vector{Int}=[0, 1, 2])
        # Validate sectors are valid for spin-1 ⊗ spin-1
        all(s -> s in (0, 1, 2), sectors) || throw(ArgumentError(
            "sectors must be subset of {0, 1, 2} for two spin-1 sites"
        ))
        !isempty(sectors) || throw(ArgumentError(
            "sectors must be non-empty"
        ))
        return new(sectors)
    end
end

support(::SpinSectorMeasurement) = 2

# === build_operator implementations ===

"""
    build_operator(gate::SpinSectorProjection, sites::Vector{Index}, local_dim::Int; kwargs...) -> ITensor

Build projector operator for two spin-1 sites.
Returns ITensor representation of the projector matrix.
"""
function build_operator(gate::SpinSectorProjection, sites::Vector{<:Index}, local_dim::Int; kwargs...)
    length(sites) == 2 || throw(ArgumentError("SpinSectorProjection requires exactly 2 sites"))
    local_dim == 3 || throw(ArgumentError("SpinSectorProjection requires local_dim=3 (spin-1)"))
    
    # Convert 9×9 matrix to ITensor
    # sites = [site_i, site_j] for two adjacent spins
    site_i, site_j = sites
    
    # Reshape 9×9 matrix to (3,3,3,3) tensor
    # 
    # The 9×9 projector matrix has basis ordering (m1, m2) where:
    # - m1 ∈ {+1, 0, -1} indexes the first spin (slow index in row/col)
    # - m2 ∈ {+1, 0, -1} indexes the second spin (fast index in row/col)
    # So matrix row/col = (m1-1)*3 + m2 (1-indexed with m2 cycling fast)
    #
    # Julia reshape(M, 3,3,3,3) produces tensor T where:
    # - T[i1, i2, i3, i4] with i1 fastest, i4 slowest
    # - For M[row,col]: row → (i1=m2_out, i2=m1_out), col → (i3=m2_in, i4=m1_in)
    #
    # Physical meaning: m1 = site_i, m2 = site_j
    # So reshaped tensor has indices: (site_j, site_i, site_j, site_i)
    #
    # ITensor construction must match: ITensor(data, j, i, j', i')
    proj_tensor = reshape(gate.projector, local_dim, local_dim, local_dim, local_dim)
    
    # Create ITensor with correct index ordering to match matrix basis
    # The reshaped tensor has (m2=site_j fast, m1=site_i slow) for both row and col
    op_tensor = ITensor(proj_tensor, site_j, site_i, site_j', site_i')
    
    return op_tensor
end

"""
    compute_two_site_born_probability(mps::MPS, projector::Matrix{Float64}, ram_sites::Vector{Int}, local_dim::Int) -> Float64

Compute the Born probability ⟨ψ|P|ψ⟩ for a two-site projector.

Computes ⟨ψ|P|ψ⟩ by:
1. Orthogonalizing MPS to the target region
2. Contracting the two-site block into a local tensor
3. Computing ⟨T|P|T⟩ using proper index contraction

# Arguments
- `mps`: Current MPS state  
- `projector`: 9×9 projector matrix (d²×d² where d=local_dim)
- `ram_sites`: RAM indices of the two sites [i, j]
- `local_dim`: Local Hilbert space dimension (3 for spin-1)

# Returns
- Probability p = ⟨ψ|P|ψ⟩ (real, non-negative)
"""
function compute_two_site_born_probability(mps::MPS, projector::Matrix{Float64}, ram_sites::Vector{Int}, local_dim::Int)
    # Get sorted RAM indices (must be adjacent for SpinSectorMeasurement)
    sorted_sites = sort(ram_sites)
    i_ram, j_ram = sorted_sites[1], sorted_sites[2]
    
    # Sanity check: sites must be adjacent
    if j_ram != i_ram + 1
        error("SpinSectorMeasurement requires adjacent sites, got RAM indices $i_ram and $j_ram")
    end
    
    # Get site indices from MPS
    site_i = siteind(mps, i_ram)
    site_j = siteind(mps, j_ram)
    
    # Build projector ITensor with correct index structure
    # Projector is 9×9, reshape to (3,3,3,3) tensor
    # Basis ordering: m2 (site_j) is fast, m1 (site_i) is slow
    proj_tensor = reshape(projector, local_dim, local_dim, local_dim, local_dim)
    P_op = ITensor(proj_tensor, site_j, site_i, site_j', site_i')
    
    # Make copy and orthogonalize to the bond between i and j
    psi = copy(mps)
    orthogonalize!(psi, i_ram)
    
    # Contract the two-site block: T = A_i * A_j
    # When orthogonalized to i_ram, we have ⟨ψ|P|ψ⟩ = ⟨T|P|T⟩ (local expectation)
    T = psi[i_ram] * psi[j_ram]
    
    # Compute ⟨T|P|T⟩ = Tr[T† P T]
    # = (T†)_{j',i'} P_{j',i',j,i} T_{j,i}
    # = sum over all indices of conj(T) * P * T
    
    # Method: compute P|T⟩ first, then contract with conj(T)
    # P_op has indices (j, i, j', i') acting as P_{out, in} = P_{(j,i), (j',i')}
    # We want (P T)_{j', i'} = P_{j',i',j,i} T_{j,i}
    # Then ⟨T|P|T⟩ = conj(T_{j',i'}) (PT)_{j',i'}
    
    # Apply projector: P|T⟩ → has indices (j', i') after contraction over (j, i)
    P_T = T * P_op  # Contracts over site_j and site_i → leaves site_j' and site_i'
    
    # Now compute overlap ⟨T|P|T⟩ = (dag(T) * P_T)
    # dag(T) has same indices but conjugated values
    # To contract with P_T (which has primed indices), we need dag(T) with primed indices
    T_dag_primed = prime(dag(T), "Site")  # Now has indices (site_j', site_i')
    
    # Contract T_dag_primed with P_T
    # Both have (site_j', site_i') plus possibly link indices
    overlap_tensor = T_dag_primed * P_T
    
    # The result should be a scalar (all site indices contracted)
    # Link indices might remain if not at edges - but for orthogonalized MPS at i_ram,
    # the two-site block should give a scalar when contracted properly
    result = scalar(overlap_tensor)
    
    # Return real part, ensuring non-negative (numerical precision)
    return max(real(result), 0.0)
end

"""
    build_operator(gate::SpinSectorMeasurement, sites::Vector{Index}, local_dim::Int; rng, mps, ram_sites) -> ITensor

Build measurement operator for two spin-1 sites with proper Born sampling.

Implements the Born rule measurement:
1. Compute Born probabilities p(S) = ⟨ψ|P_S|ψ⟩ for each allowed sector S
2. Normalize probabilities over allowed sectors
3. Sample outcome S with probability p(S) using the provided RNG
4. Return the projector P_S onto the sampled sector

# Arguments
- `gate`: SpinSectorMeasurement specifying allowed sectors
- `sites`: ITensor site indices for the two sites
- `local_dim`: Local Hilbert space dimension (must be 3 for spin-1)
- `rng`: RNG registry for reproducible sampling (uses :born stream)
- `mps`: Current MPS state for computing Born probabilities
- `ram_sites`: RAM indices of the two sites

# Returns
- ITensor projector onto the randomly sampled spin sector
"""
function build_operator(gate::SpinSectorMeasurement, sites::Vector{<:Index}, local_dim::Int; rng, mps, ram_sites)
    length(sites) == 2 || throw(ArgumentError("SpinSectorMeasurement requires exactly 2 sites"))
    local_dim == 3 || throw(ArgumentError("SpinSectorMeasurement requires local_dim=3 (spin-1)"))
    
    # === Step 1: Compute Born probabilities for each allowed sector ===
    probs = Float64[]
    projectors = Matrix{Float64}[]
    
    for S in gate.sectors
        P_S = total_spin_projector(S)
        push!(projectors, P_S)
        
        # Compute ⟨ψ|P_S|ψ⟩ via MPS contraction
        prob = compute_two_site_born_probability(mps, P_S, ram_sites, local_dim)
        push!(probs, prob)
    end
    
    # === Step 2: Normalize probabilities over allowed sectors ===
    total_prob = sum(probs)
    if total_prob < 1e-14
        error("SpinSectorMeasurement: State has zero overlap with all allowed sectors $(gate.sectors). " *
              "Probabilities: $probs. The state may already be in an orthogonal sector.")
    end
    probs ./= total_prob
    
    # === Step 3: Sample from the probability distribution ===
    # Get the born RNG stream for reproducibility
    born_rng = get_rng(rng, :born)
    r = rand(born_rng)
    
    cumprob = 0.0
    chosen_idx = length(probs)  # Default to last if rounding issues
    for (idx, p) in enumerate(probs)
        cumprob += p
        if r < cumprob
            chosen_idx = idx
            break
        end
    end
    
    # === Step 4: Return the chosen projector as ITensor ===
    P_chosen = projectors[chosen_idx]
    site_i, site_j = sites
    
    # Reshape 9×9 matrix to (3,3,3,3) tensor
    # Same basis ordering as SpinSectorProjection: m2 (site_j) is fast, m1 (site_i) is slow
    proj_tensor = reshape(P_chosen, local_dim, local_dim, local_dim, local_dim)
    
    # Create ITensor with correct index ordering to match matrix basis
    op_tensor = ITensor(proj_tensor, site_j, site_i, site_j', site_i')
    
    return op_tensor
end
