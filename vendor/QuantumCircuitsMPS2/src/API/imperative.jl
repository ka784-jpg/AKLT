# Imperative API style - direct state mutation
# The apply!() function is defined in Core/apply.jl
# This file exists only to document the imperative API pattern

# Example usage:
#   state = SimulationState(L=10, bc=:periodic, rng=RNGRegistry(...))
#   initialize!(state, ProductState(binary_int=0))
#   apply!(state, HaarRandom(), Bricklayer(:odd))
#   apply!(state, Projection(0), SingleSite(1))
