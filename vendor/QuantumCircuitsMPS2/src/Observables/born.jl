"""
    BornProbability(site::Int, outcome::Int)

Observable for Born rule probability P(measurement outcome | state) at a physical site.
"""
struct BornProbability <: AbstractObservable
    site::Int      # Physical site index
    outcome::Int   # 0 or 1
    
    function BornProbability(site::Int, outcome::Int)
        outcome in (0, 1) || throw(ArgumentError("outcome must be 0 or 1"))
        new(site, outcome)
    end
end

# Callable struct interface
function (bp::BornProbability)(state)
    return born_probability(state, bp.site, bp.outcome)
end

"""
    born_probability(state::SimulationState, physical_site::Int, outcome::Int) -> Float64

Compute Born probability P(outcome | state) at a physical site.
Converts physical site to RAM index for MPS access.
"""
function born_probability(state, physical_site::Int, outcome::Int)
    # Convert physical site to RAM index
    ram_idx = state.phy_ram[physical_site]
    
    # Use ITensorMPS expect() with projector operator
    # Proj0 = |0⟩⟨0|, Proj1 = |1⟩⟨1|
    proj_op = outcome == 0 ? "Proj0" : "Proj1"
    
    # expect() returns Vector for all sites, index by RAM position
    all_probs = expect(state.mps, proj_op)
    return real(all_probs[ram_idx])
end
