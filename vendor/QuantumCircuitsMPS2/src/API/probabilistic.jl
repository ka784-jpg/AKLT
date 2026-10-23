# Probabilistic Branching API
# ==========================
# 
# This is the finalized probabilistic API using fully named parameters (Style C).
# Selected for: self-documenting code, no position memorization, idiomatic Julia.
#
# For historical context on alternative styles that were considered, see:
# - examples/ct_model_styles.jl (comparison of 4 styles)

# Part of the Probabilistic API (Contract 4.4)

"""
    apply_with_prob!(state; rng=:ctrl, outcomes)

Execute one action from outcomes with fully named parameters.
Probabilities may sum to ≤ 1 (implicit "do nothing" branch for remainder).
Throws error if sum > 1.

Each outcome is a NamedTuple with fields:
- probability: Float64 (required)
- gate: AbstractGate (required)
- geometry: AbstractGeometry (required)

Example:
    apply_with_prob!(state;
        rng = :ctrl,
        outcomes = [
            (probability=p_ctrl, gate=Reset(), geometry=left),
            (probability=1-p_ctrl, gate=HaarRandom(), geometry=right)
        ]
    )

Example (sum < 1):
    apply_with_prob!(state;
        outcomes = [
            (probability=0.3, gate=PauliX(), geometry=site),
            (probability=0.4, gate=PauliY(), geometry=site)
            # 0.3 probability of "do nothing"
        ]
    )
"""
function apply_with_prob!(
    state::SimulationState;
    rng::Symbol = :ctrl,
    outcomes::Vector{<:NamedTuple{(:probability, :gate, :geometry)}}
)
    probs = [o.probability for o in outcomes]
    total_prob = sum(probs)
    
    # Error if probabilities sum to more than 1
    if total_prob > 1.0 + 1e-10
        error("Probabilities sum to $total_prob (must be ≤ 1)")
    end
    
    # CRITICAL: Draw BEFORE checking (Contract 4.4)
    actual_rng = get_rng(state.rng_registry, rng)
    r = rand(actual_rng)
    
    # Check each outcome
    cumulative = 0.0
    for outcome in outcomes
        cumulative += outcome.probability
        if r < cumulative
            apply!(state, outcome.gate, outcome.geometry)
            return nothing
        end
    end
    
    # If we get here: "do nothing" branch selected (r >= sum(probs))
    return nothing
end
