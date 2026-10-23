# === Compound Geometry Helpers ===
"""
Checks for either periodic boundary condition
"""
is_periodic_bc(bc::Symbol) = (bc == :periodic || bc == :periodic_nnn)

"""
    is_compound_geometry(geo::AbstractGeometry) -> Bool

Check if geometry requires element-by-element iteration.

Compound geometries (Bricklayer, AllSites) need to be expanded into
multiple individual gate applications, one for each element/site pair.
"""
is_compound_geometry(::Bricklayer) = true
is_compound_geometry(::AllSites) = true
is_compound_geometry(::AbstractGeometry) = false

"""
    get_compound_elements(geo::AbstractGeometry, L::Int, bc::Symbol) -> Vector{Vector{Int}}

Get elements for compound geometry iteration.

Returns a vector of site vectors, where each inner vector represents the sites
for one gate application.

# Arguments
- `geo`: The geometry object (Bricklayer or AllSites)
- `L`: System size (number of sites)
- `bc`: Boundary condition (:open or :periodic)

# Returns
- `Vector{Vector{Int}}`: Each inner vector is the sites for one gate application

# Examples
```julia
# Bricklayer with odd parity on L=4 system
geo = Bricklayer(:odd)
get_compound_elements(geo, 4, :open)  # [[1, 2], [3, 4]]

# AllSites on L=3 system
geo = AllSites()
get_compound_elements(geo, 3, :open)  # [[1], [2], [3]]
```
"""
function get_compound_elements(geo::Bricklayer, L::Int, bc::Symbol)
    pairs = Tuple{Int,Int}[]
    if geo.parity == :odd
        # NN odd pairs: (1,2), (3,4), (5,6), ...
        for i in 1:2:L-1
            push!(pairs, (i, i+1))
        end
    elseif geo.parity == :even
        # NN even pairs: (2,3), (4,5), ...
        for i in 2:2:L-1
            push!(pairs, (i, i+1))
        end
        # For PBC, also include (L, 1)
        if bc == :periodic
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
    return [[p1, p2] for (p1, p2) in pairs]
end

function get_compound_elements(geo::AllSites, L::Int, bc::Symbol)
    return [[site] for site in 1:L]
end
