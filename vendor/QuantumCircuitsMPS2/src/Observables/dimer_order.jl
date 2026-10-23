using ITensors
using ITensorMPS

"""
    DimerOrder(; periodic::Bool=false)

Dimer order parameter observable for spin-1 chains.
Computes terms sequentially to minimize memory usage.

# Arguments
- `periodic`: If `true`, includes the bond between site L and site 1. 
              (Note: For the alternating sign (-1)^i to be consistent in PBC, L must be even).

# Formula
O_dimer = (1 / N_bonds) * Σ_{i=1}^{N_bonds} (-1)^i ⟨S_i ⋅ S_{i+1}⟩
"""
struct DimerOrder <: AbstractObservable
    periodic::Bool

    # Inner constructor for keyword argument support
    function DimerOrder(; periodic::Bool=false)
        return new(periodic)
    end
end

function (obs::DimerOrder)(state::SimulationState)
    L = state.L
    dimer_sum = 0.0
    
    # Determine number of bonds based on boundary conditions
    num_bonds = obs.periodic ? L : L - 1

    # Loop over bonds
    for i_phys in 1:num_bonds
        
        # 1. Identify Physical Indices
        #    If periodic and i=L, the neighbor j is 1. Otherwise j = i+1.
        j_phys = (i_phys == L) ? 1 : i_phys + 1

        # 2. Resolve to RAM Indices
        i_ram = state.phy_ram[i_phys]
        j_ram = state.phy_ram[j_phys]
        
        site_i = state.sites[i_ram]
        site_j = state.sites[j_ram]

        # --- Term 1: <Sz_i * Sz_{j}> ---
        psi_copy = copy(state.mps)
        
        Sz_i = op("Sz", site_i)
        psi_copy[i_ram] *= Sz_i
        
        Sz_j = op("Sz", site_j)
        psi_copy[j_ram] *= Sz_j
        
        noprime!(psi_copy)
        val_zz = real(inner(state.mps, psi_copy))
        
        # DISCARD: psi_copy is overwritten below


        # --- Term 2: <S+_i * S-_{j}> ---
        psi_copy = copy(state.mps)
        
        Sp_i = op("S+", site_i)
        psi_copy[i_ram] *= Sp_i
        
        Sm_j = op("S-", site_j)
        psi_copy[j_ram] *= Sm_j
        
        noprime!(psi_copy)
        val_pm = real(inner(state.mps, psi_copy))


        # --- Term 3: <S-_i * S+_{j}> ---
        psi_copy = copy(state.mps)
        
        Sm_i = op("S-", site_i)
        psi_copy[i_ram] *= Sm_i
        
        Sp_j = op("S+", site_j)
        psi_copy[j_ram] *= Sp_j
        
        noprime!(psi_copy)
        val_mp = real(inner(state.mps, psi_copy))


        # --- Combine ---
        # S⋅S = SzSz + 0.5(S+S- + S-S+)
        bond_val = val_zz + 0.5 * (val_pm + val_mp)

        # Add to sum with alternating sign (-1)^i
        dimer_sum += (-1)^i_phys * bond_val
    end

    # Normalize by the actual number of bonds summed
    return dimer_sum / num_bonds
end
