# === Gate Application Engine ===
# Core apply! function implementing CT.jl-style MPS contraction

using ITensors
using ITensorMPS

"""
    apply!(state::SimulationState, gate::AbstractGate, geo::AbstractGeometry)

Apply a gate to the state at sites specified by geometry.
Modifies state.mps in-place.

Normalization dispatch (Contract 3.5):
- Unitaries (HaarRandom, CZ, PauliX/Y/Z): NO normalize after apply
- Projections: YES normalize after apply
"""
function apply!(state::SimulationState, gate::AbstractGate, geo::AbstractGeometry)
    # Dispatch to appropriate handler based on geometry type
    _apply_dispatch!(state, gate, geo)
end

"""
    apply!(state::SimulationState, gate::AbstractGate, sites::Vector{Int})

Apply a gate to specific physical sites. Direct site specification.
"""
function apply!(state::SimulationState, gate::AbstractGate, sites::Vector{Int})
    _apply_single!(state, gate, sites)
end

# === Dispatch handlers for different geometry types ===

function _apply_dispatch!(state::SimulationState, gate::AbstractGate, geo::SingleSite)
    sites = get_sites(geo, state)
    _apply_single!(state, gate, sites)
end

function _apply_dispatch!(state::SimulationState, gate::AbstractGate, geo::AdjacentPair)
    sites = get_sites(geo, state)
    _apply_single!(state, gate, sites)
end

function _apply_dispatch!(state::SimulationState, gate::AbstractGate, geo::AbstractStaircase)
    sites = get_sites(geo, state)
    _apply_single!(state, gate, sites)
    # Advance staircase AFTER application
    advance!(geo, state.L, state.bc)
end

function _apply_dispatch!(state::SimulationState, gate::AbstractGate, geo::Bricklayer)
    pairs = get_pairs(geo, state)
    for (p1, p2) in pairs
        _apply_single!(state, gate, [p1, p2])
    end
end

function _apply_dispatch!(state::SimulationState, gate::AbstractGate, geo::AllSites)
    all_sites = get_all_sites(geo, state)
    for site in all_sites
        _apply_single!(state, gate, [site])
    end
end

# Pointer does NOT auto-advance - user controls movement via move!()
function _apply_dispatch!(state::SimulationState, gate::AbstractGate, geo::Pointer)
    sites = get_sites(geo, state)
    _apply_single!(state, gate, sites)
    # NO advance! - user explicitly calls move!()
end

# === Internal helper for Born-sampled projection ===

"""
    _measure_single_site!(state::SimulationState, site::Int) -> Int

Perform Born-sampled projective measurement on a single site.
Returns the measurement outcome (0 or 1).

This is the FUNDAMENTAL measurement operation:
1. Compute Born probability P(0|ψ)
2. Sample outcome using :born RNG stream
3. Apply Projection operator
4. Return outcome (for conditional logic in Reset)
"""
function _measure_single_site!(state::SimulationState, site::Int)
    p_0 = born_probability(state, site, 0)
    born_rng = get_rng(state.rng_registry, :born)
    outcome = rand(born_rng) < p_0 ? 0 : 1
    _apply_single!(state, Projection(outcome), [site])
    return outcome
end

# === Measurement gate dispatch (FUNDAMENTAL - pure projection) ===

function _apply_dispatch!(state::SimulationState, gate::Measurement, geo::SingleSite)
    site = get_sites(geo, state)[1]
    _measure_single_site!(state, site)
    return nothing
end

function _apply_dispatch!(state::SimulationState, gate::Measurement, geo::AllSites)
    all_sites = get_all_sites(geo, state)
    for site in all_sites
        _measure_single_site!(state, site)  # Independent per-site sampling
    end
    return nothing
end

function _apply_dispatch!(state::SimulationState, gate::Measurement, geo::AbstractStaircase)
    site = geo._position
    _measure_single_site!(state, site)
    advance!(geo, state.L, state.bc)
    return nothing
end

function _apply_dispatch!(state::SimulationState, gate::Measurement, geo::Pointer)
    site = geo._position
    _measure_single_site!(state, site)
    # NO advance! - user explicitly calls move!()
    return nothing
end

# === Reset gate dispatch (DERIVED - measurement + conditional X) ===

function _apply_dispatch!(state::SimulationState, gate::Reset, geo::SingleSite)
    site = get_sites(geo, state)[1]
    outcome = _measure_single_site!(state, site)
    if outcome == 1
        _apply_single!(state, PauliX(), [site])
    end
    return nothing
end

function _apply_dispatch!(state::SimulationState, gate::Reset, geo::AllSites)
    all_sites = get_all_sites(geo, state)
    for site in all_sites
        outcome = _measure_single_site!(state, site)
        if outcome == 1
            _apply_single!(state, PauliX(), [site])
        end
    end
    return nothing
end

function _apply_dispatch!(state::SimulationState, gate::Reset, geo::AbstractStaircase)
    site = geo._position
    outcome = _measure_single_site!(state, site)
    if outcome == 1
        _apply_single!(state, PauliX(), [site])
    end
    advance!(geo, state.L, state.bc)
    return nothing
end

function _apply_dispatch!(state::SimulationState, gate::Reset, geo::Pointer)
    site = geo._position
    outcome = _measure_single_site!(state, site)
    if outcome == 1
        _apply_single!(state, PauliX(), [site])
    end
    # NO advance!
    return nothing
end

# === Core application logic ===

"""
    _apply_single!(state::SimulationState, gate::AbstractGate, phy_sites::Vector{Int})

Apply gate to specific physical sites. Internal workhorse.

Steps:
1. Validate support matches site count
2. Convert physical sites to RAM indices
3. Build operator with physical site indices
4. Apply operator to MPS
5. Normalize if gate is Projection
"""
function _apply_single!(state::SimulationState, gate::AbstractGate, phy_sites::Vector{Int})
    # Contract 2.1: Support validation
    if support(gate) != length(phy_sites)
        throw(ArgumentError("Gate support $(support(gate)) does not match sites $(length(phy_sites))"))
    end
    
    # Convert physical sites to RAM indices
    ram_sites = [state.phy_ram[ps] for ps in phy_sites]
    
    # Build operator with state.sites indices (in physical pair order)
    op = _build_gate_operator(state, gate, phy_sites, ram_sites)
    
    # Apply operator using CT.jl algorithm
    apply_op_internal!(state.mps, op, state.sites, state.cutoff, state.maxdim)
    
    # Contract 3.5: Normalization dispatch
    if gate isa Projection || gate isa SpinSectorProjection || gate isa SpinSectorMeasurement
        normalize!(state.mps)
    end
    # Unitaries (HaarRandom, CZ, PauliX/Y/Z): NO normalize
end

"""
    _build_gate_operator(state, gate, phy_sites, ram_sites) -> ITensor

Build the operator tensor for the gate.
"""
function _build_gate_operator(state::SimulationState, gate::AbstractGate, phy_sites::Vector{Int}, ram_sites::Vector{Int})
    if length(ram_sites) == 1
        # Single-site gate
        site_idx = state.sites[ram_sites[1]]
        return build_operator(gate, site_idx, state.local_dim)
    else
        # Multi-site gate: use indices in RAM order
        site_indices = [state.sites[rs] for rs in ram_sites]
        return build_operator(gate, site_indices, state.local_dim; 
                              rng=state.rng_registry, mps=state.mps, ram_sites=ram_sites)
    end
end

"""
    apply_op_internal!(mps::MPS, op::ITensor, sites::Vector{Index}, cutoff::Float64, maxdim::Int)

Apply operator to MPS following CT.jl algorithm (lines 147-172).

Contract 3.6: Index matching via Index comparison, NOT tag parsing.
"""
function apply_op_internal!(mps::MPS, op::ITensor, sites::Vector{Index}, cutoff::Float64, maxdim::Int)
    # Get RAM site indices from operator indices (Contract 3.6)
    i_list = get_op_ram_sites(op, sites)
    sort!(i_list)
    
    # Orthogonalize MPS to first site
    orthogonalize!(mps, i_list[1])
    
    # Contract MPS tensors in range
    mps_ij = mps[i_list[1]]
    for idx in i_list[1]+1:i_list[end]
        mps_ij *= mps[idx]
    end
    
    # Apply operator
    mps_ij *= op
    noprime!(mps_ij)
    
    if length(i_list) == 1
        # Single-site: direct assignment
        mps[i_list[1]] = mps_ij
    else
        # Multi-site: SVD chain reconstruction
        lefttags = (i_list[1] == 1) ? nothing : tags(linkind(mps, i_list[1] - 1))
        
        for idx in i_list[1]:i_list[end]-1
            if idx == 1
                inds1 = [siteind(mps, 1)]
            else
                inds1 = [findindex(mps[idx-1], lefttags), findindex(mps[idx], "Site")]
            end
            
            lefttags = tags(linkind(mps, idx))
            U, S, V = svd(mps_ij, inds1; cutoff=cutoff, lefttags=lefttags, maxdim=maxdim)
            mps[idx] = U
            mps_ij = S * V
        end
        
        mps[i_list[end]] = mps_ij
    end
    
    return nothing
end

"""
    get_op_ram_sites(op::ITensor, sites::Vector{Index}) -> Vector{Int}

Get RAM site indices from operator indices using Index comparison (Contract 3.6).
Does NOT parse tags.
"""
function get_op_ram_sites(op::ITensor, sites::Vector{Index})
    op_inds = inds(op)
    ram_sites = Int[]
    
    for op_idx in op_inds
        # Only process unprimed indices (inputs)
        if plev(op_idx) != 0
            continue
        end
        
        # Find matching site by Index comparison
        found = false
        for (ram_idx, site_idx) in enumerate(sites)
            if noprime(op_idx) == noprime(site_idx)
                push!(ram_sites, ram_idx)
                found = true
                break
            end
        end
        
        if !found
            error("Operator index $op_idx not found in state sites")
        end
    end
    
    return ram_sites
end
