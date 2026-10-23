using ITensors
using ITensorMPS

# Forward declaration for RNGRegistry (defined in Task 2)
# For now, use Union{Nothing, Any} to avoid dependency
const RNGRegistryType = Any

"""
    SimulationState

Main simulation state container holding MPS and metadata.

Fields:
- mps: The MPS tensor network (Nothing until initialize! called)
- sites: ITensor site indices
- phy_ram: physical site -> RAM index mapping
- ram_phy: RAM index -> physical site mapping
- L: system size
- bc: boundary condition (:open or :periodic)
- site_type: site index type ("Qubit", "S=1", "Qudit")
- local_dim: local Hilbert space dimension (default 2 for qubits)
- cutoff: SVD truncation cutoff
- maxdim: maximum bond dimension
- rng_registry: RNG streams for reproducibility
- observables: tracked observable values
- observable_specs: observable specifications

Supported site_type values:
- "Qubit": spin-1/2 (local_dim=2, default)
- "S=1": spin-1 (local_dim=3)
- "Qudit": arbitrary dimension (requires local_dim parameter)
"""
mutable struct SimulationState
    mps::Union{MPS, Nothing}
    sites::Vector{Index}
    phy_ram::Vector{Int}
    ram_phy::Vector{Int}
    L::Int
    bc::Symbol
    site_type::String
    local_dim::Int
    cutoff::Float64
    maxdim::Int
    rng_registry::Union{RNGRegistryType, Nothing}
    observables::Dict{Symbol, Vector}
    observable_specs::Dict{Symbol, Any}
end

"""
    SimulationState(; L, bc, site_type="Qubit", local_dim=2, cutoff=1e-10, maxdim=100, rng=nothing)

Create a new simulation state. MPS is created later via initialize!().

Parameters:
- L: system size
- bc: boundary condition (:open or :periodic)
- site_type: site index type ("Qubit", "S=1", "Qudit")
- local_dim: local Hilbert space dimension (default 2)
- cutoff: SVD truncation cutoff
- maxdim: maximum bond dimension
- rng: RNGRegistry for reproducible randomness

For "Qudit" site type, local_dim specifies the dimension (e.g., local_dim=4 for d=4).
"""
function SimulationState(;
    L::Int,
    bc::Symbol,
    site_type::String = "Qubit",
    local_dim::Int = 2,
    cutoff::Float64 = 1e-10,
    maxdim::Int = 100,
    rng = nothing  # RNGRegistry, attached later or passed here
)
    # Validate bc
    bc in (:open, :periodic, :periodic_nnn) ||
        throw(ArgumentError("bc must be :open or :periodic or :periodic_nnn, got $bc"))

    # Auto-detect local_dim from site_type if not explicitly set
    if site_type == "S=1" && local_dim == 2  # default not overridden
        local_dim = 3
    end

    # Compute basis mapping (OBC works now, PBC throws until Task 4)
    phy_ram, ram_phy = compute_basis_mapping(L, bc)

    # Create site indices in RAM order
    if site_type == "Qudit"
        sites = siteinds("Qudit", L; dim=local_dim)
    else
        sites = siteinds(site_type, L)
    end

    # Return state with MPS=nothing (deferred to initialize!)
    return SimulationState(
        nothing,  # mps - set by initialize!
        sites,
        phy_ram,
        ram_phy,
        L,
        bc,
        site_type,
        local_dim,
        cutoff,
        maxdim,
        rng,
        Dict{Symbol, Vector}(),  # observables
        Dict{Symbol, Any}()      # observable_specs
    )
end
