"""
    EntanglementEntropy(; cut::Int, renyi_index::Int=1, threshold::Float64=1e-16, base::Real=2)

Entanglement entropy observable.

Computes the entanglement entropy across a specified cut in the MPS.

# Arguments
- `cut::Int`: Physical site where the cut is made (must satisfy 1 <= cut < L)
- `renyi_index::Int=1`: Rényi index for entropy (must be >= 1)
  - `renyi_index=1`: von Neumann entropy S₁ = -Σᵢ λᵢ log(λᵢ) (default)
  - `renyi_index=2`: Rényi-2 entropy S₂ = log(Σᵢ λᵢ²) / (1-2)
  - `renyi_index=n`: Rényi-n entropy Sₙ = log(Σᵢ λᵢⁿ) / (1-n)
- `threshold::Float64=1e-16`: Minimum threshold for singular values (default: 1e-16)
- `base::Real=2`: Base of logarithm for entropy computation (default: 2 for bits)

!!! note "Hartley entropy (renyi_index=0) is NOT supported"
    Hartley entropy (renyi_index=0) measures log₂(Schmidt rank), but is not available via 
    this interface because:
    - MPS compression retains singular values above a cutoff threshold (~1e-10)
    - Numerically, "zero" singular values are never truly zero in floating-point arithmetic
    - This makes `log(rank)` give `log(maxdim)` instead of `log(true_rank)`
    - Result is threshold-dependent and unreliable
    
    **Alternative**: Access MPS singular values directly via `orthogonalize!` + `svd`,
    then apply your own threshold to determine the Schmidt rank.

# Implementation
The entropy is computed by:
1. Converting the physical cut position to RAM ordering
2. Orthogonalizing the MPS at the cut site
3. Performing SVD to obtain Schmidt values
4. Computing the entropy from the Schmidt spectrum

# Example
```julia
ee = EntanglementEntropy(; cut=2, renyi_index=1)
entropy = ee(state)
```
"""
struct EntanglementEntropy <: AbstractObservable
    cut::Int
    renyi_index::Int
    threshold::Float64
    base::Float64
    
    function EntanglementEntropy(; cut::Int, renyi_index::Int=1, threshold::Float64=1e-16, base::Real=2)
        cut >= 1 || throw(ArgumentError("EntanglementEntropy cut must be >= 1"))
        renyi_index >= 1 || throw(ArgumentError("EntanglementEntropy renyi_index must be >= 1"))
        threshold > 0 || throw(ArgumentError("EntanglementEntropy threshold must be > 0"))
        base > 0 || throw(ArgumentError("EntanglementEntropy base must be > 0"))
        new(cut, renyi_index, threshold, Float64(base))
    end
end

# Callable struct interface
function (ee::EntanglementEntropy)(state)
    # Validate cut is in valid range
    1 <= ee.cut < state.L || throw(ArgumentError("cut must satisfy 1 <= cut < L"))
    
    # Determine RAM cut position based on boundary conditions
    # For periodic BC with folded MPS [1,L,2,L-1,...], RAM cut at position k
    # partitions into two contiguous arcs. The cut parameter directly specifies
    # the RAM bond, giving a half-chain cut when cut=L÷2.
    # For open BC, ram_phy is identity so this also works correctly.
    ram_cut = ee.cut
    
    # Compute entropy using internal helper
    return _von_neumann_entropy(state.mps, ram_cut; n=ee.renyi_index, threshold=ee.threshold, base=ee.base)
end

"""
    _von_neumann_entropy(mps::MPS, i::Int; n::Int=1, threshold::Float64=1e-16, base::Float64=2.0) -> Float64

Compute entanglement entropy at bond i of an MPS.

Arguments:
- mps: The MPS state
- i: The bond index (site index) where entropy is computed
- n: Rényi index (1=von Neumann, n=Rényi-n)
- threshold: Minimum threshold for singular values to avoid log(0)
- base: Base of logarithm for entropy computation (default: 2.0 for bits)

Returns:
- Entanglement entropy value

The function:
1. Orthogonalizes the MPS to site i
2. Performs SVD on the tensor to extract Schmidt values
3. Computes probabilities from Schmidt values (squared)
4. Returns entropy based on Rényi index:
   - n=1: von Neumann entropy S₁ = -Σ p log_b(p)
   - n=0: Hartley entropy S₀ = log_b(rank)
   - n≠1: Rényi entropy Sₙ = log_b(Σ pⁿ) / (1-n)
"""
function _von_neumann_entropy(
    mps::MPS,
    i::Int;
    n::Int=1,
    threshold::Float64=1e-16,
    base::Float64=2.0,
)
    # Orthogonalize MPS to site i
    mps_ = orthogonalize(mps, i)
    
    # Perform SVD on the link between site i and i+1
    # Extract singular values from the bond
    _, S = svd(mps_[i], (linkind(mps_, i),))
    
    # Get singular values and compute probabilities (squared for normalization)
    # Apply threshold to avoid numerical issues with log(0)
    singular_vals = diag(S)
    p = max.(singular_vals, threshold) .^ 2
    
    # Define log with specified base: log_b(x) = log(x) / log(b)
    log_fn = x -> log(x) / log(base)
    
    # Compute entropy based on Rényi index
    if n == 1
        # von Neumann entropy: S₁ = -Σ p log_b(p)
        return -sum(p .* log_fn.(p))
    elseif n == 0
        # Hartley entropy: S₀ = log_b(rank)
        return log_fn(length(p))
    else
        # Rényi entropy: Sₙ = log_b(Σ pⁿ) / (1-n)
        return log_fn(sum(p .^ n)) / (1 - n)
    end
end
