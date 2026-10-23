# === Pointer Geometry Type ===
# Bidirectional pointer for algorithms that need to move in either direction

"""
    Pointer(start_position::Int)

A bidirectional pointer that can move left or right.
Unlike StaircaseLeft/StaircaseRight which are unidirectional,
Pointer supports explicit direction control via move!().

For two-qubit gates, the pair is (position, position+1) with PBC wrap.
"""
mutable struct Pointer <: AbstractGeometry
    _position::Int
    Pointer(start::Int) = new(start)
end

"""
    current_position(p::Pointer) -> Int

Get the current position (read-only).
"""
current_position(p::Pointer) = p._position

"""
    get_sites(p::Pointer, state) -> Vector{Int}

Get the current pair of sites [pos, pos+1] for two-qubit gates.
Handles PBC wrap when pos == L.
"""
function get_sites(p::Pointer, state)
    pos = p._position
    L = state.L
    second = (pos == L && (state.bc == :periodic || state.bc == :periodic_nnn)) ? 1 : pos + 1
    return [pos, second]
end

"""
    move!(p::Pointer, direction::Symbol, L::Int, bc::Symbol=:periodic)

Move the pointer in the specified direction.
- direction = :left → position decreases (wraps L → 1 for PBC)
- direction = :right → position increases (wraps 1 → L for PBC)

This is the PUBLIC API for pointer movement, unlike advance!() which is internal.
"""
function move!(p::Pointer, direction::Symbol, L::Int, bc::Symbol=:periodic)
    if direction == :left
        if bc == :periodic || bc == :periodic_nnn
            p._position = p._position == 1 ? L : p._position - 1
        else
            p._position = p._position == 1 ? L - 1 : p._position - 1
        end
    elseif direction == :right
        if bc == :periodic || bc == :periodic_nnn
            p._position = (p._position % L) + 1
        else
            max_pos = L - 1
            p._position = (p._position % max_pos) + 1
        end
    else
        throw(ArgumentError("direction must be :left or :right, got $direction"))
    end
    return nothing
end

# Pointer does NOT auto-advance when used with apply!
# Movement is explicit via move!()
