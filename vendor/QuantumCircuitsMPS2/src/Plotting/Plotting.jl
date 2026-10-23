"""
    Plotting Module

Circuit visualization for quantum circuits.

This module provides visualization tools for inspecting circuit structure
before or after simulation. Visualizations are deterministic (same seed →
same diagram) and help debug stochastic circuits.

# Visualization Functions
- [`print_circuit`](@ref): ASCII/Unicode terminal visualization
- [`plot_circuit`](@ref): SVG export (requires `using Luxor`)

# ASCII Visualization
Renders circuits as qubit wires with gate boxes using either Unicode
box-drawing characters (default) or ASCII fallback for compatibility.

```julia
circuit = Circuit(L=4, bc=:periodic, n_steps=5) do c
    apply!(c, Reset(), StaircaseRight(1))
end

print_circuit(circuit; seed=42)  # Terminal output
print_circuit(circuit; seed=42, unicode=false)  # ASCII mode
```

# SVG Visualization
Exports high-quality vector graphics using Luxor.jl (optional dependency).
Files can be embedded in documentation, presentations, or papers.

```julia
using Luxor  # Load extension
plot_circuit(circuit, "diagram.svg"; seed=42)
```

# Deterministic Rendering
Both visualization methods use the same `seed` parameter for stochastic
branch resolution, ensuring reproducible diagrams that match simulation
behavior when using the same RNG seeds.

# See Also
- [`Circuit`](@ref): Build circuits for visualization
- [`expand_circuit`](@ref): Get concrete operations (used internally)
- [`simulate!`](@ref): Execute circuits after visualizing
"""

# ASCII circuit visualization
include("ascii.jl")
