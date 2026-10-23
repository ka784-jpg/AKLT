"""
Wrap a 1-indexed site i onto the range 1:L using periodic
(1-based modular) arithmetic.

Examples:
- wrap_site(L+1, L) == 1
- wrap_site(0, L) == L
"""
wrap_site(i::Int, L::Int) = ((i - 1) % L) + 1


"""
Sanitize and validate a pair of site indices (i, j)
according to boundary condition bc.

If bc is :periodic or :periodic_nnn:
    - indices are wrapped into 1:L
    - always returns ok = true

If bc is :open:
    - indices are not wrapped
    - ok is true only if both are in 1:L and i != j

Returns (ii, jj, ok).
"""
function sanitize_pair(i::Int, j::Int, L::Int, bc::Symbol)
    if bc in (:periodic, :periodic_nnn)
        return wrap_site(i, L), wrap_site(j, L), true
    elseif bc == :open
        ok = (1 <= i <= L) && (1 <= j <= L) && (i != j)
        return i, j, ok
    else
        error("bc must be :open, :periodic, or :periodic_nnn, got $bc")
    end
end


"""
Build a list of nearest-neighbor pairs (i, i+1).

- Repeats the sweep n_steps times.
- Uses sweep_sites to determine which starting sites i to use.
- Does not apply boundary conditions.
"""
function build_nn_pairs(L::Int; n_steps::Int=1, sweep_sites=1:L)
    pairs = Vector{Tuple{Int,Int}}()
    sizehint!(pairs, n_steps * length(sweep_sites))
    for _ in 1:n_steps
        for i in sweep_sites
            push!(pairs, (i, i + 1))
        end
    end
    return pairs
end


"""
Build a stochastic list of NN or NNN pairs.

For each site i in sweep_sites and each timestep:
    - with probability p_nn, add (i, i+1)
    - otherwise add (i, i+2)

Does not apply boundary conditions or wrapping.
Returns raw index pairs.
"""
function build_manual_pairs(ctrl_rng, L::Int, bc::Symbol, p_nn::Float64;
                            n_steps::Int=1, sweep_sites=1:L)
    pairs = Vector{Tuple{Int,Int}}()
    sizehint!(pairs, n_steps * length(sweep_sites))
    for t in 1:n_steps
        for i in sweep_sites
            if rand(ctrl_rng) < p_nn
                push!(pairs, (i, i + 1))
            else
                push!(pairs, (i, i + 2))
            end
        end
    end
    return pairs
end
