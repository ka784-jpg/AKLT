using ITensors
using ITensorMPS

"""
    BondDimension()

Observable that computes the maximum bond dimension of the current MPS state.

Max Bond Dimension:
χ_max = max(dim(linkind(ψ, b)) for b in 1:L-1)

Notes:
- Returns 0 if L < 2 (no links).
- Checks all bonds 1 through L-1 to find the global maximum.
"""
struct BondDimension <: AbstractObservable end

"""
    (obs::BondDimension)(state::SimulationState) -> Int

Compute the maximum link dimension of the MPS.
"""
function (obs::BondDimension)(state::SimulationState)::Int
    psi = state.mps
    L = length(psi)

    # Edge case: Chain too short to have links
    if L < 2
        return 0
    end

    max_dim = 0

    # Iterate through all bonds (1 to L-1) to find the max dimension
    for b in 1:(L - 1)
        l_ind = linkind(psi, b)  # link index between site b and b+1

        # linkind returns `nothing` if the link is trivial/missing
        if !isnothing(l_ind)
            d = dim(l_ind)
            if d > max_dim
                max_dim = d
            end
        end
    end

    return max_dim
end
