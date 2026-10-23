# === Static Geometry Types ===
# Geometries where sites are known at construction time

"""
    SingleSite(site::Int)

Geometry specifying a single physical site.
Used for single-qubit gates like PauliX, Projection.
"""
struct SingleSite <: AbstractGeometry
    site::Int
end

get_sites(geo::SingleSite, state) = [geo.site]

"""
    AdjacentPair(first::Int)

Geometry specifying an adjacent pair of physical sites: (first, first+1).
For PBC, wraps: (L, 1) when first=L.
"""
struct AdjacentPair <: AbstractGeometry
    first::Int
end

function get_sites(geo::AdjacentPair, state)
    L = state.L
    second = (geo.first == L && is_periodic_bc(state.bc)) ? 1 : geo.first + 1
    return [geo.first, second]
end

"""
    NextNearestNeighbor(first::Int)

Geometry specifying a next-nearest-neighbor pair of physical sites: (first, first+2).
For PBC, wraps: (L-1, 1) and (L, 2).
"""
struct NextNearestNeighbor <: AbstractGeometry
    first::Int
end

function get_sites(geo::NextNearestNeighbor, state)
    L = state.L
    # If periodic-like and we are at L-1 or L, we wrap.
    # L-1 wraps to 1. L wraps to 2.
    # Otherwise, we just add 2.
    second = (is_periodic_bc(state.bc) && geo.first >= L - 1) ? (geo.first + 2 - L) : geo.first + 2
    return [geo.first, second]
end

"""
    Bricklayer(parity::Symbol)

Geometry for bricklayer gate application pattern.

Nearest-neighbor (NN) modes:
- `:odd` parity → pairs (1,2), (3,4), (5,6), ...
- `:even` parity → pairs (2,3), (4,5), ... plus (L,1) for PBC
- `:nn` parity → ALL NN pairs (combines :odd + :even)

Next-nearest-neighbor (NNN) modes (4 sublayers covering all 12 NNN pairs for L=12):
- `:nnn_odd_1` parity → pairs (1,3), (5,7), (9,11), ... (stride 4, offset 1)
- `:nnn_odd_2` parity → pairs (3,5), (7,9), (11,1), ... (stride 4, offset 3, PBC wrap)
- `:nnn_even_1` parity → pairs (2,4), (6,8), (10,12), ... (stride 4, offset 2)
- `:nnn_even_2` parity → pairs (4,6), (8,10), (12,2), ... (stride 4, offset 4, PBC wrap)
- `:nnn` parity → ALL NNN pairs (combines all 4 sublayers)

apply! loops internally over all pairs.
"""
struct Bricklayer <: AbstractGeometry
    parity::Symbol

    function Bricklayer(parity::Symbol)
        parity in (:odd, :even, :nn, :nnn, :nnn_odd_1, :nnn_odd_2, :nnn_even_1, :nnn_even_2) ||
            throw(ArgumentError("Bricklayer parity must be :odd, :even, :nn, :nnn, :nnn_odd_1, :nnn_odd_2, :nnn_even_1, or :nnn_even_2, got $parity"))
        new(parity)
    end
end

"""
    get_pairs(geo::Bricklayer, state) -> Vector{Tuple{Int,Int}}

Get all pairs for bricklayer pattern. Returns pairs of physical sites.
"""
function get_pairs(geo::Bricklayer, state)
    L = state.L
    bc = state.bc
    pairs = Tuple{Int,Int}[]

    if geo.parity == :odd
        # Odd pairs: (1,2), (3,4), (5,6), ...
        for i in 1:2:L-1
            push!(pairs, (i, i+1))
        end
    elseif geo.parity == :even
        # Even pairs: (2,3), (4,5), ...
        for i in 2:2:L-1
            push!(pairs, (i, i+1))
        end
        # For PBC, also include (L, 1)
        if is_periodic_bc(bc)
            push!(pairs, (L, 1))
        end
    elseif geo.parity == :nn
        # All NN pairs: combines :odd and :even
        # For L=12 periodic: 12 pairs covering all bonds
        for i in 1:2:L-1  # Odd pairs: (1,2), (3,4), ...
            push!(pairs, (i, i+1))
        end
        for i in 2:2:L-1  # Even pairs: (2,3), (4,5), ...
            push!(pairs, (i, i+1))
        end
        if is_periodic_bc(bc)
            push!(pairs, (L, 1))  # Wrap: (12,1) for L=12
        end
    elseif geo.parity == :nnn
        # All NNN pairs: combines 4 sublayers
        # For L=12 periodic: 12 pairs covering all NNN bonds
        # Sublayer 1: (1,3), (5,7), (9,11)
        for i in 1:4:L-2
            push!(pairs, (i, i+2))
        end
        # Sublayer 2: (3,5), (7,9)
        for i in 3:4:L-2
            push!(pairs, (i, i+2))
        end
        if is_periodic_bc(bc) && L >= 4
            push!(pairs, (L-1, 1))  # (11,1) for L=12
        end
        # Sublayer 3: (2,4), (6,8), (10,12)
        for i in 2:4:L-2
            push!(pairs, (i, i+2))
        end
        # Sublayer 4: (4,6), (8,10)
        for i in 4:4:L-2
            push!(pairs, (i, i+2))
        end
        if is_periodic_bc(bc) && L >= 4
            push!(pairs, (L, 2))  # (12,2) for L=12
        end
    elseif geo.parity == :nnn_odd_1
        # NNN odd sublayer 1: (1,3), (5,7), (9,11), ... (stride 4, offset 1)
        for i in 1:4:L-2
            push!(pairs, (i, i+2))
        end
    elseif geo.parity == :nnn_odd_2
        # NNN odd sublayer 2: (3,5), (7,9), (11,1), ... (stride 4, offset 3)
        for i in 3:4:L-2
            push!(pairs, (i, i+2))
        end
        if is_periodic_bc(bc) && L >= 4
            push!(pairs, (L-1, 1))  # Wrap: (11,1) for L=12
        end
    elseif geo.parity == :nnn_even_1
        # NNN even sublayer 1: (2,4), (6,8), (10,12), ... (stride 4, offset 2)
        for i in 2:4:L-2
            push!(pairs, (i, i+2))
        end
    elseif geo.parity == :nnn_even_2
        # NNN even sublayer 2: (4,6), (8,10), (12,2), ... (stride 4, offset 4)
        for i in 4:4:L-2
            push!(pairs, (i, i+2))
        end
        if is_periodic_bc(bc) && L >= 4
            push!(pairs, (L, 2))  # Wrap: (12,2) for L=12
        end
    end

    return pairs
end

"""
    AllSites

Geometry for applying single-site gates to all sites.
apply! loops internally over all L sites.
"""
struct AllSites <: AbstractGeometry end

"""
    get_all_sites(geo::AllSites, state) -> Vector{Int}

Get all physical sites (1:L).
"""
get_all_sites(geo::AllSites, state) = collect(1:state.L)
