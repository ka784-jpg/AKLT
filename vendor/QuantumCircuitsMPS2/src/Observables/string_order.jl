# === String Order Parameter Observable for AKLT Chains ===
#
# Computes the string order parameter:
#   O_string(i,j) = ⟨Sz[i] * exp(iπ Σ_{k=i+1}^{j-1} Sz[k]) * Sz[j]⟩
#
# For AKLT ground state: |O_string| ≈ 4/9 ≈ 0.444

using ITensors
using ITensorMPS

"""
    StringOrder(i::Int, j::Int; order::Int=1)

String order parameter observable for spin-1 chains.

# Formulas

**order=1** (nearest-neighbor AKLT):
```
O¹(i,j) = ⟨Sz[i] * exp(iπ Σ_{k=i+1}^{j-1} Sz[k]) * Sz[j]⟩
```
Expected: |O¹| ≈ 4/9 ≈ 0.444 for NN AKLT ground state

**order=2** (next-nearest-neighbor AKLT):
```
O²(n,m) = ⟨Sz[n]·Sz[n+1] * exp(iπ Σ_{k=n+2}^{m-2} Sz[k]) * Sz[m-1]·Sz[m]⟩
```
Expected: |O²| ≈ (4/9)² ≈ 0.198 for NNN AKLT ground state

# Arguments
- `i`: First site index (physical indexing)
- `j`: Second site index (physical indexing, must be j > i)
- `order`: 1 for NN formula (default), 2 for NNN formula with paired endpoints

# Notes
- order=2 requires j >= i+4 (for non-overlapping endpoint pairs)
- NNN AKLT creates two decoupled chains; paired endpoints project onto both

# Example
```julia
s = SimulationState(L=8, bc=:periodic, site_type="S=1")
initialize!(s, ProductState(binary_int=0))
# ... apply AKLT protocol ...
so1 = compute(StringOrder(1, 5), s)           # order=1 (default)
so2 = compute(StringOrder(1, 7, order=2), s)  # order=2
```

# References
- AKLT (1987): Rigorous results on valence-bond ground states
- String order distinguishes Haldane phase from trivial phases
"""
struct StringOrder <: AbstractObservable
    i::Int
    j::Int
    order::Int
    
    function StringOrder(i::Int, j::Int; order::Int=1)
        i > 0 || throw(ArgumentError("i must be positive, got $i"))
        j > i || throw(ArgumentError("j must be > i, got j=$j, i=$i"))
        order in (1, 2) || throw(ArgumentError("order must be 1 or 2, got $order"))
        if order == 2
            j >= i + 4 || throw(ArgumentError("order=2 requires j >= i+4 for non-overlapping endpoint pairs, got i=$i, j=$j"))
        end
        new(i, j, order)
    end
end

"""
    (obs::StringOrder)(state::SimulationState) -> Float64

Compute string order parameter via MPS contraction.
"""
function (obs::StringOrder)(state::SimulationState)
    i_phys = obs.i
    j_phys = obs.j
    L = state.L
    
    # Validate sites are in bounds
    if i_phys > L || j_phys > L
        throw(ArgumentError(
            "StringOrder sites ($i_phys, $j_phys) exceed system size L=$L"
        ))
    end
    
    # Work on a copy of the MPS
    psi_copy = copy(state.mps)
    
    if obs.order == 1
        # Order 1: ⟨Sz[i] * exp(iπΣ) * Sz[j]⟩
        # Convert physical sites to RAM indices
        i_ram = state.phy_ram[i_phys]
        j_ram = state.phy_ram[j_phys]
        
        # Get site indices
        site_i = state.sites[i_ram]
        site_j = state.sites[j_ram]
        
        # Apply Sz at site i
        Sz_i = op("Sz", site_i)
        psi_copy[i_ram] = psi_copy[i_ram] * Sz_i
        
        # Apply exp(iπ Sz) to all sites between i and j
        for k_phys in (i_phys+1):(j_phys-1)
            k_ram = state.phy_ram[k_phys]
            site_k = state.sites[k_ram]
            expSz_k = op("expSz", site_k)
            psi_copy[k_ram] = psi_copy[k_ram] * expSz_k
        end
        
        # Apply Sz at site j
        Sz_j = op("Sz", site_j)
        psi_copy[j_ram] = psi_copy[j_ram] * Sz_j
        
    elseif obs.order == 2
        # Order 2: ⟨Sz[n]·Sz[n+1] * exp(iπΣ_{n+2:m-2}) * Sz[m-1]·Sz[m]⟩
        n_phys, m_phys = i_phys, j_phys
        
        # Left endpoint pair: Sz[n] · Sz[n+1]
        for site_phys in (n_phys, n_phys+1)
            site_ram = state.phy_ram[site_phys]
            Sz = op("Sz", state.sites[site_ram])
            psi_copy[site_ram] = psi_copy[site_ram] * Sz
        end
        
        # String: exp(iπ Sz) for sites n+2 to m-2
        for k_phys in (n_phys+2):(m_phys-2)
            k_ram = state.phy_ram[k_phys]
            expSz_k = op("expSz", state.sites[k_ram])
            psi_copy[k_ram] = psi_copy[k_ram] * expSz_k
        end
        
        # Right endpoint pair: Sz[m-1] · Sz[m]
        for site_phys in (m_phys-1, m_phys)
            site_ram = state.phy_ram[site_phys]
            Sz = op("Sz", state.sites[site_ram])
            psi_copy[site_ram] = psi_copy[site_ram] * Sz
        end
    end
    
    # Compute expectation value: ⟨ψ|O|ψ⟩
    # Remove prime marks from site indices added by operator application
    noprime!(psi_copy)
    result = real(inner(state.mps, psi_copy))
    
    return result
end

# === Define custom ITensor operator for exp(iπ Sz) ===

"""
Define exp(iπ Sz) operator for S=1 sites.

For spin-1: Sz = diag(+1, 0, -1) in basis |+1⟩, |0⟩, |-1⟩
exp(iπ Sz) = diag(exp(iπ), exp(0), exp(-iπ))
           = diag(-1, 1, -1)

This is a diagonal operator in the Sz basis.
"""
function ITensors.op(::OpName"expSz", ::SiteType"S=1")
    return [
        -1.0  0.0  0.0;   # |Up⟩ (m=+1): exp(iπ·1) = -1
         0.0  1.0  0.0;   # |Z0⟩ (m=0):  exp(iπ·0) = +1
         0.0  0.0 -1.0    # |Dn⟩ (m=-1): exp(iπ·(-1)) = -1
    ]
end
