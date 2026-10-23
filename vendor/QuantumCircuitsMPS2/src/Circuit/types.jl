# === Circuit Type ===
# Lazy/symbolic representation of quantum circuit operations

"""
    Circuit(; L::Int, bc::Symbol, operations=NamedTuple[], n_steps::Int=1)

Lazy representation of a quantum circuit as a sequence of symbolic operations.

A `Circuit` does NOT execute immediately - it stores operations symbolically and
expands them to concrete gate applications only when passed to `simulate!`.

# Fields
- `L::Int`: Number of physical sites in the system
- `bc::Symbol`: Boundary conditions (`:periodic` or `:open` or ':periodic_nnn')
- `operations::Vector{NamedTuple}`: Internal symbolic operation list
- `n_steps::Int`: Number of circuit timesteps (default: 1)
- `params::Dict{Symbol,Any}`: User-defined parameters (default: empty Dict)

# Operation Representation
Operations are stored as NamedTuples with different formats:

**Deterministic gates:**
```julia
(type = :deterministic, gate = gate, geometry = geometry)
```

**Stochastic outcomes (apply_with_prob!):**
```julia
(type = :stochastic, rng = :ctrl, outcomes = [(probability=p, gate=g, geometry=geo), ...])
```

# Construction
Users construct circuits via the do-block API (see `CircuitBuilder`):

```julia
circuit = Circuit(L=10, bc=:periodic) do c
    apply!(c, Hadamard(), SingleSite(1))
    apply!(c, CNOT(), StaircaseRight(1), steps=5)
end
```

# Execution
Pass the circuit to `simulate!` to execute:

```julia
state = SimulationState(...)
simulate!(state, circuit)  # Expands and executes all operations
```

# See Also
- `expand_circuit`: Expands symbolic operations to concrete site lists
- `simulate!`: Executes a circuit on a simulation state
"""
Base.@kwdef struct Circuit
    L::Int
    bc::Symbol
    operations::Vector{NamedTuple} = NamedTuple[]
    n_steps::Int = 1
    params::Dict{Symbol,Any} = Dict{Symbol,Any}()
end
