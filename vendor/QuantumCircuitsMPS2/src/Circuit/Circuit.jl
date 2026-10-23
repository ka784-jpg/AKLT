"""
    Circuit Module

Lazy/symbolic circuit construction and execution for quantum circuits.

This module provides a high-level API for building quantum circuits symbolically
before execution. Circuits can be visualized and inspected without running
simulations, enabling workflow patterns like:

1. **Build**: Construct circuit using do-block syntax with `apply!` and `apply_with_prob!`
2. **Visualize**: Inspect structure with `print_circuit` or `plot_circuit`
3. **Execute**: Run with `simulate!` on a `SimulationState`

# Core Types
- [`Circuit`](@ref): Symbolic circuit representation with lazy operations
- [`ExpandedOp`](@ref): Concrete operation at specific sites and timestep

# Core Functions
- [`expand_circuit`](@ref): Expand symbolic circuit to concrete operations
- [`simulate!`](@ref): Execute circuit on simulation state

# Construction Pattern
```julia
circuit = Circuit(L=4, bc=:periodic, n_steps=10) do c
    # Deterministic operations
    apply!(c, Reset(), StaircaseRight(1))
    
    # Stochastic branching
    apply_with_prob!(c; rng=:ctrl, outcomes=[
        (probability=0.3, gate=HaarRandom(), geometry=StaircaseLeft(4)),
        (probability=0.5, gate=Reset(), geometry=SingleSite(1))
        # Implicit 0.2 probability of "do nothing"
    ])
end
```

# See Also
- [`SimulationState`](@ref): Execution target for circuits
- [`print_circuit`](@ref): ASCII visualization
- [`plot_circuit`](@ref): SVG visualization (requires Luxor)
"""

# Circuit types and internal operation representation
include("types.jl")

# CircuitBuilder and do-block API
include("builder.jl")

# Circuit expansion (symbolic → concrete)
include("expand.jl")

# Recording context and presets (must come before execute.jl)
include("recording.jl")

# Circuit executor (depends on recording.jl for RecordingContext)
include("execute.jl")
