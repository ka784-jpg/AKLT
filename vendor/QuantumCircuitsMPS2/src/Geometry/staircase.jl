# === Staircase Geometry Types ===
# Geometries with internal state (mutable pointer)
"""
    AbstractStaircase <: AbstractGeometry

Base type for staircase geometries with internal pointer.
"""
abstract type AbstractStaircase <: AbstractGeometry end

"""
    StaircaseRight(start_position::Int; range::Int=1)

Staircase that moves right: applies at (pos, pos+range), then advances pos.

# Arguments
- `start_position`: Initial position
- `range`: Distance between sites (default 1 for nearest neighbors)

# Examples
```julia
StaircaseRight(1)           # NN: (1,2), (2,3), ...
StaircaseRight(1; range=2)  # NNN: (1,3), (2,4), ...
```
"""
mutable struct StaircaseRight <: AbstractStaircase
    _position::Int  # internal, use current_position() to read
    range::Int      # distance between sites
    
    function StaircaseRight(start::Int; range::Int=1)
        range >= 1 || throw(ArgumentError("range must be >= 1"))
        new(start, range)
    end
end

"""
    StaircaseLeft(start_position::Int; range::Int=1)

Staircase that moves left: applies at (pos, pos+range), then decrements pos.

# Arguments
- `start_position`: Initial position
- `range`: Distance between sites (default 1 for nearest neighbors)

# Examples
```julia
StaircaseLeft(1)           # NN: (1,2), (2,3), ...
StaircaseLeft(1; range=2)  # NNN: (1,3), (2,4), ...
```
"""
mutable struct StaircaseLeft <: AbstractStaircase
    _position::Int  # internal, use current_position() to read
    range::Int      # distance between sites
    
    function StaircaseLeft(start::Int; range::Int=1)
        range >= 1 || throw(ArgumentError("range must be >= 1"))
        new(start, range)
    end
end

"""
    current_position(geo::AbstractStaircase) -> Int

Get the current position of the staircase (READ-ONLY accessor).
"""
current_position(geo::AbstractStaircase) = geo._position

"""
    get_sites(geo::AbstractStaircase, state) -> Vector{Int}

Get current pair of physical sites for the staircase.
Returns [pos, pos+range] with proper boundary condition handling.
"""
function get_sites(geo::AbstractStaircase, state)
    pos = geo._position
    L = state.L
    range = geo.range
    
    # Compute second site with wrapping for PBC
    if is_periodic_bc(state.bc)
        second = mod1(pos + range, L)
    else
        second = pos + range
        # Validate second site is within bounds for OBC
        if second > L
            throw(ArgumentError(
                "Staircase at position $pos with range=$range exceeds system size L=$L (OBC)"
            ))
        end
    end
    
    return [pos, second]
end

"""
    advance!(geo::StaircaseRight, L::Int, bc::Symbol)

Advance staircase right by one position. Internal use by apply!.
- StaircaseRight: pos += 1, wraps L → 1 (PBC) or L-1 → 1 (OBC)
"""
function advance!(geo::StaircaseRight, L::Int, bc::Symbol)
    if is_periodic_bc(state.bc)
        # PBC: position cycles 1 → 2 → ... → L → 1
        geo._position = (geo._position % L) + 1
    else
        # OBC: position cycles 1 → 2 → ... → L-1 → 1 (can't apply at L since no L+1)
        max_pos = L - 1
        geo._position = (geo._position % max_pos) + 1
    end
end

"""
    advance!(geo::StaircaseLeft, L::Int, bc::Symbol)

Advance staircase left by one position. Internal use by apply!.
- StaircaseLeft: pos -= 1, wraps 1 → L (PBC) or 1 → L-1 (OBC)
"""
function advance!(geo::StaircaseLeft, L::Int, bc::Symbol)
    if is_periodic_bc(state.bc)
        # PBC: position cycles L → L-1 → ... → 1 → L
        geo._position = geo._position == 1 ? L : geo._position - 1
    else
        # OBC: position cycles L-1 → L-2 → ... → 1 → L-1
        max_pos = L - 1
        geo._position = geo._position == 1 ? max_pos : geo._position - 1
    end
end
