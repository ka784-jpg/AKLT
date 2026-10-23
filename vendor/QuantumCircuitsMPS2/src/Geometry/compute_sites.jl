# === Pure Site Computation Functions ===
# Pure functional versions of geometry site computation without mutation.
# These enable symbolic circuit expansion where step replaces mutable _position.

"""
    compute_site_staircase_right(start::Int, step::Int, L::Int, bc::Symbol) -> Int

Compute the position of a StaircaseRight geometry after `step` iterations.

# Arguments
- `start`: Starting position (1-indexed)
- `step`: Step number (1 = initial position, 2 = after first advance, etc.)
- `L`: System size (number of sites)
- `bc`: Boundary condition (`:periodic`, `:periodic_nnn`, or `:open`)
"""
function compute_site_staircase_right(start::Int, step::Int, L::Int, bc::Symbol)
    # Validation
    step < 1 && throw(ArgumentError("step must be >= 1, got $step"))
    L < 2 && throw(ArgumentError("L must be >= 2 for staircase geometry, got $L"))
    bc ∉ [:periodic, :periodic_nnn, :open] &&
        throw(ArgumentError("bc must be :periodic, :periodic_nnn, or :open, got $bc"))
    (start < 1 || start > L) && throw(ArgumentError("start must be in range 1:$L, got $start"))

    advances = step - 1

    if is_periodic_bc(bc)
        # PBC: cycles over 1:L
        pos = start
        for _ in 1:advances
            pos = (pos % L) + 1
        end
    else
        # OBC: cycles over 1:(L-1)
        max_pos = L - 1
        pos = start
        for _ in 1:advances
            pos = (pos % max_pos) + 1
        end
    end

    return pos
end

"""
    compute_site_staircase_left(start::Int, step::Int, L::Int, bc::Symbol) -> Int

Compute the position of a StaircaseLeft geometry after `step` iterations.

# Arguments
- `start`: Starting position (1-indexed)
- `step`: Step number (1 = initial position, 2 = after first advance, etc.)
- `L`: System size (number of sites)
- `bc`: Boundary condition (`:periodic`, `:periodic_nnn`, or `:open`)
"""
function compute_site_staircase_left(start::Int, step::Int, L::Int, bc::Symbol)
    # Validation
    step < 1 && throw(ArgumentError("step must be >= 1, got $step"))
    L < 2 && throw(ArgumentError("L must be >= 2 for staircase geometry, got $L"))
    bc ∉ [:periodic, :periodic_nnn, :open] &&
        throw(ArgumentError("bc must be :periodic, :periodic_nnn, or :open, got $bc"))
    (start < 1 || start > L) && throw(ArgumentError("start must be in range 1:$L, got $start"))

    advances = step - 1

    if is_periodic_bc(bc)
        # PBC: cycles over 1:L (backwards)
        pos = start
        for _ in 1:advances
            pos = pos == 1 ? L : pos - 1
        end
    else
        # OBC: cycles over 1:(L-1) (backwards)
        max_pos = L - 1
        pos = start
        for _ in 1:advances
            pos = pos == 1 ? max_pos : pos - 1
        end
    end

    return pos
end

"""
    compute_pair_staircase(pos::Int, L::Int, bc::Symbol) -> Vector{Int}

Convert a single staircase position to an adjacent pair [pos, pos+1].

# Arguments
- `pos`: Position (1-indexed)
- `L`: System size (number of sites)
- `bc`: Boundary condition (`:periodic`, `:periodic_nnn`, or `:open`)
"""
function compute_pair_staircase(pos::Int, L::Int, bc::Symbol)
    # Validation
    L < 2 && throw(ArgumentError("L must be >= 2 for staircase geometry, got $L"))
    bc ∉ [:periodic, :periodic_nnn, :open] &&
        throw(ArgumentError("bc must be :periodic, :periodic_nnn, or :open, got $bc"))
    (pos < 1 || pos > L) && throw(ArgumentError("pos must be in range 1:$L, got $pos"))

    # OBC-specific validation: cannot form pair at position L
    if bc == :open && pos == L
        throw(ArgumentError("Cannot form adjacent pair at position $pos with open boundary conditions (would require site $(L+1))"))
    end

    # Compute second site with PBC wrapping
    second = (pos == L && is_periodic_bc(bc)) ? 1 : pos + 1

    return [pos, second]
end

"""
    compute_sites(geo::SingleSite, step::Int, L::Int, bc::Symbol) -> Vector{Int}

Compute sites for SingleSite geometry (always returns [geo.site]).
"""
function compute_sites(geo::SingleSite, step::Int, L::Int, bc::Symbol)
    return [geo.site]
end

"""
    compute_sites(geo::AdjacentPair, step::Int, L::Int, bc::Symbol) -> Vector{Int}

Compute sites for AdjacentPair geometry.
"""
function compute_sites(geo::AdjacentPair, step::Int, L::Int, bc::Symbol)
    second = (geo.first == L && is_periodic_bc(bc)) ? 1 : geo.first + 1
    return [geo.first, second]
end

"""
    compute_sites(geo::NextNearestNeighbor, step::Int, L::Int, bc::Symbol) -> Vector{Int}

Compute sites for NextNearestNeighbor geometry (i, i+2).
"""
function compute_sites(geo::NextNearestNeighbor, step::Int, L::Int, bc::Symbol)
    s1 = geo.first

    # Logic for Next-Nearest Neighbor (stride 2)
    if is_periodic_bc(bc)
        if s1 == L - 1
            s2 = 1   # Wrap (L-1) + 2 -> L+1 -> 1
        elseif s1 == L
            s2 = 2   # Wrap L + 2     -> L+2 -> 2
        else
            s2 = s1 + 2
        end
    else
        # Open boundary conditions (just add 2)
        s2 = s1 + 2
    end

    return [s1, s2]
end

"""
    compute_sites(geo::StaircaseRight, step::Int, L::Int, bc::Symbol, gate::AbstractGate) -> Vector{Int}

Compute sites for StaircaseRight geometry based on gate support.
"""
function compute_sites(geo::StaircaseRight, step::Int, L::Int, bc::Symbol, gate::AbstractGate)
    pos = compute_site_staircase_right(geo._position, step, L, bc)
    if support(gate) == 1
        return [pos]
    else  # support(gate) == 2
        return compute_pair_staircase(pos, L, bc)
    end
end

"""
    compute_sites(geo::StaircaseLeft, step::Int, L::Int, bc::Symbol, gate::AbstractGate) -> Vector{Int}

Compute sites for StaircaseLeft geometry based on gate support.
"""
function compute_sites(geo::StaircaseLeft, step::Int, L::Int, bc::Symbol, gate::AbstractGate)
    pos = compute_site_staircase_left(geo._position, step, L, bc)
    if support(gate) == 1
        return [pos]
    else  # support(gate) == 2
        return compute_pair_staircase(pos, L, bc)
    end
end
