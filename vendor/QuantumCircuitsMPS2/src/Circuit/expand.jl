# === Circuit Expansion (Symbolic → Concrete) ===
# Converts Circuit's symbolic operations to concrete site lists with deterministic RNG sampling

using Random

"""
    ExpandedOp

Represents a concrete gate operation at specific sites for a specific timestep.

# Fields
- `step::Int`: Circuit timestep (1-indexed)
- `gate::AbstractGate`: The gate to apply
- `sites::Vector{Int}`: Physical sites for this operation
- `label::String`: Short label for visualization (e.g., "Rst", "Haar", "CZ")

# Usage
Produced by `expand_circuit` for visualization and manual execution.
"""
struct ExpandedOp
    step::Int
    gate::AbstractGate
    sites::Vector{Int}
    label::String
end

"""
    gate_label(gate::AbstractGate) -> String

Return a short visualization label for a gate type.

# Labels
- Reset → "Rst"
- HaarRandom → "Haar"
- Projection → "Prj"
- PauliX → "X"
- PauliY → "Y"
- PauliZ → "Z"
- CZ → "CZ"
- Other → Type name as string

# Examples
```julia
gate_label(Reset())       # Returns "Rst"
gate_label(HaarRandom())  # Returns "Haar"
gate_label(CZ())          # Returns "CZ"
```
"""
gate_label(::Reset) = "Rst"
gate_label(::HaarRandom) = "Haar"
gate_label(::Projection) = "Prj"
gate_label(::Measurement) = "Meas"
gate_label(::PauliX) = "X"
gate_label(::PauliY) = "Y"
gate_label(::PauliZ) = "Z"
gate_label(::CZ) = "CZ"
gate_label(::SpinSectorProjection) = "P(S≠2)"
gate_label(g::AbstractGate) = string(typeof(g))  # Fallback

"""
    validate_geometry(geo::AbstractGeometry)

Validate that a geometry type is supported for circuit expansion.

Phase 1 supports:
- `StaircaseRight`
- `StaircaseLeft`
- `SingleSite`
- `AdjacentPair`

# Throws
`ArgumentError` if geometry is not supported.
"""
function validate_geometry(geo::AbstractGeometry)
    if geo isa StaircaseRight
        # supported
    elseif geo isa StaircaseLeft
        # supported
    elseif geo isa SingleSite
        # supported
    elseif geo isa AdjacentPair
        # supported
    elseif geo isa NextNearestNeighbor
        # supported (custom geometry: (i, i+2) with PBC wrap)
    elseif geo isa Bricklayer
        # supported
    elseif geo isa AllSites
        # supported
    else
        throw(ArgumentError(
            "Phase 1 does not support geometry type $(typeof(geo)). " *
            "Supported: StaircaseRight, StaircaseLeft, SingleSite, AdjacentPair, " *
            "NextNearestNeighbor, Bricklayer, AllSites"
        ))
    end
end


"""
    select_branch(rng::AbstractRNG, outcomes) -> Union{NamedTuple, Nothing}

Select a stochastic outcome using cumulative probability matching.

CRITICAL: This MUST match the RNG consumption pattern in `src/API/probabilistic.jl:56-68`.

# Algorithm
1. Draw `r = rand(rng)` ONCE (before checking)
2. Accumulate probabilities cumulatively
3. Return first outcome where `r < cumulative` (STRICT <, not <=)
4. If no outcome selected: return `nothing` (do-nothing branch)

# Arguments
- `rng`: Random number generator
- `outcomes`: Vector of NamedTuples with fields `(probability, gate, geometry)`

# Returns
- Selected outcome NamedTuple if `r` falls in any outcome's range
- `nothing` if "do nothing" branch is selected (r >= sum(probabilities))

# Examples
```julia
rng = MersenneTwister(42)
outcomes = [(probability=0.5, gate=Reset(), geometry=StaircaseRight(1)),
            (probability=0.3, gate=PauliX(), geometry=SingleSite(2))]

# If rand() = 0.4 → selects first outcome (0.4 < 0.5)
# If rand() = 0.7 → selects second outcome (0.7 < 0.8)
# If rand() = 0.9 → returns nothing (do-nothing branch)
```
"""
function select_branch(rng::AbstractRNG, outcomes)
    # CRITICAL: Draw BEFORE checking (matches apply_with_prob!)
    r = rand(rng)
    
    cumulative = 0.0
    for outcome in outcomes
        cumulative += outcome.probability
        if r < cumulative  # STRICT <, not <=
            return outcome
        end
    end
    
    # If we get here: "do nothing" branch selected
    return nothing
end

"""
    compute_sites_dispatch(geo::AbstractGeometry, gate::AbstractGate, step::Int, L::Int, bc::Symbol) -> Vector{Int}

Dispatch compute_sites with appropriate arguments based on geometry type.

For StaircaseRight/StaircaseLeft: requires gate parameter to determine support.
For SingleSite/AdjacentPair: gate parameter not needed.
"""
function compute_sites_dispatch(geo::AbstractGeometry, gate::AbstractGate, step::Int, L::Int, bc::Symbol)
    if geo isa StaircaseRight || geo isa StaircaseLeft
        return compute_sites(geo, step, L, bc, gate)
    else
        return compute_sites(geo, step, L, bc)
    end
end

"""
    expand_circuit(circuit::Circuit; seed::Int=0) -> Vector{Vector{ExpandedOp}}

Expand a symbolic circuit to concrete gate operations with deterministic RNG sampling.

Converts Circuit's lazy operations into explicit gate applications at specific sites
for each timestep. Stochastic branches are resolved using a seeded RNG.

# Arguments
- `circuit::Circuit`: Symbolic circuit to expand
- `seed::Int`: Random seed for stochastic branch selection (default: 0)

# Returns
- `Vector{Vector{ExpandedOp}}`: Outer vector has length `n_steps`, inner vectors contain
  operations for that timestep. Inner vectors may be empty if "do nothing" is selected
  for all stochastic operations.

# RNG Alignment
The `seed` parameter creates a `MersenneTwister` that consumes randomness in this order:
1. For each step (1 to `circuit.n_steps`)
2. For each operation in `circuit.operations`
3. If `op.type == :stochastic`: consume ONE `rand()` to select branch

This matches `simulate!` behavior when the RNG registry stream for the operation's
`rng` field is seeded with the same value.

# Determinism
Same seed → same expansion. This enables:
- Reproducible visualizations
- Verification that simulation matches expansion
- Debugging stochastic circuits

# Examples
```julia
circuit = Circuit(L=4, bc=:periodic, n_steps=4) do c
    apply_with_prob!(c; rng=:ctrl, outcomes=[
        (probability=0.5, gate=Reset(), geometry=StaircaseRight(1)),
        (probability=0.5, gate=HaarRandom(), geometry=StaircaseLeft(4))
    ])
end

ops = expand_circuit(circuit; seed=42)
length(ops) == 4  # Always n_steps outer vectors

# Determinism check
ops2 = expand_circuit(circuit; seed=42)
all(length(ops[i]) == length(ops2[i]) for i in 1:4)  # true
```

# See Also
- `ExpandedOp`: Concrete operation representation
- `simulate!`: Execute circuit (uses same RNG pattern)
"""
function expand_circuit(circuit::Circuit; seed::Int=0)
    # Validate all geometries upfront
    for op in circuit.operations
        if op.type == :deterministic
            validate_geometry(op.geometry)
        elseif op.type == :stochastic
            for outcome in op.outcomes
                validate_geometry(outcome.geometry)
            end
        end
    end
    
    # Create seeded RNG for stochastic branch selection
    rng = MersenneTwister(seed)
    
    # Result: Vector{Vector{ExpandedOp}} with length n_steps
    result = Vector{Vector{ExpandedOp}}()
    
    # Expand each timestep
    for step in 1:circuit.n_steps
        step_ops = ExpandedOp[]
        
        # Process each operation in the circuit
        for op in circuit.operations
            if op.type == :deterministic
    if is_compound_geometry(op.geometry)
        # Expand compound geometries into individual elements
        elements = get_compound_elements(op.geometry, circuit.L, circuit.bc)
                    for sites in elements
                        push!(step_ops, ExpandedOp(
                            step,
                            op.gate,
                            sites,
                            gate_label(op.gate)
                        ))
                    end
                else
                    # Simple geometry: single ExpandedOp
                    sites = compute_sites_dispatch(op.geometry, op.gate, step, circuit.L, circuit.bc)
                    push!(step_ops, ExpandedOp(
                        step,
                        op.gate,
                        sites,
                        gate_label(op.gate)
                    ))
                end
                
            elseif op.type == :stochastic
                # Check if ANY outcome has compound geometry
                has_compound = any(is_compound_geometry(o.geometry) for o in op.outcomes)
                
                if has_compound
                    # Compound geometry → single RNG draw, then expand selected geometry
                    selected = select_branch(rng, op.outcomes)
                    
                    if selected !== nothing
                        # Get elements from the SELECTED outcome's geometry
                        elements = get_compound_elements(selected.geometry, circuit.L, circuit.bc)
                        
                        for sites in elements
                            push!(step_ops, ExpandedOp(
                                step,
                                selected.gate,
                                sites,
                                gate_label(selected.gate)
                            ))
                        end
                    end
                    # If nothing selected: "do nothing" - no entries added
                else
                    # Simple stochastic: single RNG draw
                    selected = select_branch(rng, op.outcomes)
                    
                    if selected !== nothing
                        # Branch was selected - compute sites and add operation
                        sites = compute_sites_dispatch(selected.geometry, selected.gate, step, circuit.L, circuit.bc)
                        push!(step_ops, ExpandedOp(
                            step,
                            selected.gate,
                            sites,
                            gate_label(selected.gate)
                        ))
                    end
                    # If nothing selected: "do nothing", no entry added
                end
            end
        end
        
        push!(result, step_ops)
    end
    
    return result
end
