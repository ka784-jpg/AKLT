"""
Compute physical-to-RAM and RAM-to-physical site mappings.

BC options:
- :open          -> identity mapping (1:L -> 1:L)
- :periodic      -> CT.jl folded mapping  ram_phy = [1, L, 2, L-1, 3, L-2, ...]  (requires even L)
- :periodic_nnn  -> "NNN-friendly" mapping using your forward_perm/position_map

Returns: (phy_ram, ram_phy) where:
- phy_ram[physical_site] = ram_index
- ram_phy[ram_index] = physical_site
"""

# =========================
# :periodic_nnn mapping (your mapping)
# =========================
function forward_perm(n::Int)
    perm = Int[]
    mid = n ÷ 2

    left = mid
    right = mid + 1

    while left ≥ 1 || right ≤ n
        if left ≥ 1
            push!(perm, left)
            left -= 1
        end
        if left ≥ 1
            push!(perm, left)
            left -= 1
        end
        if right ≤ n
            push!(perm, right)
            right += 1
        end
    end

    return perm
end

# Inverse permutation (physical -> RAM if ram_phy == forward_perm)
function position_map(n::Int)
    return invperm(forward_perm(n))
end

function compute_basis_mapping(L::Int, bc::Symbol)
    bc in (:open, :periodic, :periodic_nnn) ||
        throw(ArgumentError("bc must be :open, :periodic, or :periodic_nnn, got $bc"))

    if bc == :open
        # OBC: identity
        return collect(1:L), collect(1:L)

    elseif bc == :periodic
        # PBC: CT.jl folded mapping (interleave from both ends)
        iseven(L) || throw(ArgumentError("PBC folded basis requires even L, got L=$L"))

        # ram_phy = [1, L, 2, L-1, 3, L-2, ...]
        ram_phy = Int[]
        sizehint!(ram_phy, L)
        for (a, b) in zip(1:L÷2, reverse((L÷2+1):L))
            push!(ram_phy, a)
            push!(ram_phy, b)
        end

        phy_ram = zeros(Int, L)
        for (ram_idx, phy_site) in enumerate(ram_phy)
            phy_ram[phy_site] = ram_idx
        end

        return phy_ram, ram_phy

    else
        # :periodic_nnn -> your mapping
        # Use your permutation as RAM order:
        # ram_phy[ram_index] = physical_site
        ram_phy = forward_perm(L)

        # and its inverse as physical -> RAM
        phy_ram = invperm(ram_phy)  # == position_map(L)

        return phy_ram, ram_phy
    end
end

# Convenience accessors
physical_to_ram(state, phy_site::Int) = state.phy_ram[phy_site]
ram_to_physical(state, ram_site::Int) = state.ram_phy[ram_site]
