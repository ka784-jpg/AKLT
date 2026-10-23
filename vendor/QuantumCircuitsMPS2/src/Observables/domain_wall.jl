"""
    DomainWall(; order::Int, i1_fn::Union{Function, Nothing}=nothing)

Domain wall observable for CT model.
Only supports xj=Set([0]) case (first "1" in bit string).

The domain wall computes:
  DW = Σ_j (L-j+1)^order * P(first "1" at position j starting from i1)

where position j is measured cyclically starting from i1.

Parameters:
- order: The order of the domain wall (≥ 1)
- i1_fn: Optional function that returns the sampling site i1 when called
         If provided, record! can be called without i1 parameter
"""
struct DomainWall <: AbstractObservable
    order::Int
    i1_fn::Union{Function, Nothing}
    
    function DomainWall(; order::Int, i1_fn::Union{Function, Nothing}=nothing)
        order >= 1 || throw(ArgumentError("DomainWall order must be >= 1"))
        new(order, i1_fn)
    end
end

# Callable struct interface
function (dw::DomainWall)(state, i1::Union{Int, Nothing}=nothing)
    # Determine which i1 to use
    actual_i1 = if dw.i1_fn !== nothing
        dw.i1_fn()  # Call the captured function
    elseif i1 !== nothing
        i1  # Use explicit parameter
    else
        throw(ArgumentError(
            "DomainWall requires either i1_fn at construction or i1 at call time"
        ))
    end
    return domain_wall(state, actual_i1, dw.order)
end

"""
    domain_wall(state::SimulationState, i1::Int, order::Int) -> Float64

Compute domain wall observable at sampling site i1 with given order.
Ports CT.jl's dw_FM algorithm for xj=Set([0]) case.

The domain wall measures where the first "1" appears in the bit string
when scanning cyclically from position i1.
"""
function domain_wall(state, i1::Int, order::Int)
    L = state.L
    
    # Physical site list starting from i1, wrapping around
    # phy_list[j] = the j-th physical site in scanning order
    # CT.jl line 595: phy_list = [mod(i1+j-2, L)+1 for j in 1:L]
    phy_list = [mod(i1 + j - 2, L) + 1 for j in 1:L]
    
    dw_value = 0.0
    
    for j in 1:L
        # Weight for finding first "1" at position j
        # CT.jl line 598: (L-j+1)^order
        weight = Float64((L - j + 1)^order)
        
        # Probability of:
        # - Sites 1..j-1 being "0" (all zeros before position j)
        # - Site j being "1" (the first "1")
        
        # Get the physical sites in scanning order up to position j
        sites_zero = phy_list[1:j-1]  # Should be "0"
        site_one = phy_list[j]         # Should be "1"
        
        # Build the probability using projector products
        # P = ⟨ψ| (∏_{k<j} P0_k) P1_j |ψ⟩
        
        prob = compute_projector_product_expectation(state, sites_zero, site_one)
        dw_value += weight * prob
    end
    
    return dw_value
end

"""
    compute_projector_product_expectation(state, sites_zero::Vector{Int}, site_one::Int) -> Float64

Compute ⟨ψ| (∏_k P0_k) P1 |ψ⟩ where P0 projects to |0⟩ and P1 projects to |1⟩.
sites_zero: physical sites that should be "0"
site_one: physical site that should be "1"

This uses MPO construction for the projector product.
"""
function compute_projector_product_expectation(state, sites_zero::Vector{Int}, site_one::Int)
    L = state.L
    
    # Build MPO for the projector product
    # Each site gets either P0, P1, or I (identity)
    
    # Determine which operator goes at each RAM site
    ops_at_ram = fill("Id", L)  # Default to identity
    
    for phy_site in sites_zero
        ram_idx = state.phy_ram[phy_site]
        ops_at_ram[ram_idx] = "Proj0"
    end
    
    ram_idx_one = state.phy_ram[site_one]
    ops_at_ram[ram_idx_one] = "Proj1"
    
    # Build single-site operator tensors and contract
    # For efficiency, we use the fact that projectors are diagonal
    # and compute the expectation directly
    
    # Alternative: Build MPO and use inner()
    # This is cleaner and works for general cases
    
    mpo_tensors = ITensor[]
    for (ram_idx, op_name) in enumerate(ops_at_ram)
        site_idx = state.sites[ram_idx]
        push!(mpo_tensors, op(op_name, site_idx))
    end
    
    # Create MPO from tensors
    proj_mpo = MPO(mpo_tensors)
    
    # Compute ⟨ψ|O|ψ⟩
    return real(inner(state.mps', proj_mpo, state.mps))
end
