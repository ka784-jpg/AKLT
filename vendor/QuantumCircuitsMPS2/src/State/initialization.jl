using ITensors
using ITensorMPS

"""
Abstract type for initial state specifications.
"""
abstract type AbstractInitialState end

"""
    ProductState(; binary_int=nothing, binary_decimal=nothing, bitstring=nothing)

Product state initialization for qubits/qudits. Exactly one parameter must be specified.

# Arguments (mutually exclusive)
- `binary_int::Integer`: Integer representation (e.g., 0, 1, 5)
- `binary_decimal::AbstractFloat`: Binary decimal notation (e.g., 0.101 = "101")
- `bitstring::AbstractString`: Explicit bitstring (e.g., "101")

# Site ordering
MSB at site 1, LSB at site L (CT.jl convention):
- `binary_int=1` with L=4 → "0001" → site 4 has "1"
- `bitstring="1000"` → site 1 has "1"

# Examples
```julia
# All-zero state (4 qubits)
ProductState(binary_int=0)

# Site 4 has "1", others "0"
ProductState(binary_int=1)

# Binary decimal: "101" (sites 1,3 have "1")
ProductState(binary_decimal=0.101)

# Explicit bitstring
ProductState(bitstring="1010")
```
"""
struct ProductState <: AbstractInitialState
    binary_int::Union{Nothing, BigInt}
    binary_decimal::Union{Nothing, Float64}
    bitstring::Union{Nothing, String}
    
    function ProductState(; 
        binary_int::Union{Nothing, Integer}=nothing,
        binary_decimal::Union{Nothing, AbstractFloat}=nothing,
        bitstring::Union{Nothing, AbstractString}=nothing
    )
        # Validate exactly one parameter is specified
        params_specified = sum([
            binary_int !== nothing,
            binary_decimal !== nothing,
            bitstring !== nothing
        ])
        
        if params_specified == 0
            throw(ArgumentError(
                "ProductState requires exactly one of: binary_int, binary_decimal, or bitstring"
            ))
        elseif params_specified > 1
            throw(ArgumentError(
                "ProductState accepts only one parameter, got multiple"
            ))
        end
        
        # Convert to appropriate types
        bi = binary_int === nothing ? nothing : BigInt(binary_int)
        bd = binary_decimal === nothing ? nothing : Float64(binary_decimal)
        bs = bitstring === nothing ? nothing : String(bitstring)
        
        # Validate bitstring contains only 0/1
        if bs !== nothing && !all(c in ('0', '1') for c in bs)
            throw(ArgumentError("bitstring must contain only '0' and '1' characters"))
        end
        
        return new(bi, bd, bs)
    end
end

"""
    RandomMPS(; bond_dim::Int)

Random MPS with specified bond dimension.
Requires RNGRegistry with :state_init stream.
"""
struct RandomMPS <: AbstractInitialState
    bond_dim::Int
    
    function RandomMPS(; bond_dim::Int = 1)
        return new(bond_dim)
    end
end

"""
    initialize!(state::SimulationState, init::ProductState)

Initialize state with a product state based on specified initialization method.
Supports binary_int, binary_decimal, or bitstring.
Uses CT.jl MSB ordering: site 1 = MSB, site L = LSB.
"""
function initialize!(state::SimulationState, init::ProductState)
    L = state.L
    
    # Convert init specification to bit pattern string
    bit_pattern_str::String = if init.binary_int !== nothing
        # Convert integer to binary string, padded to L digits
        lpad(string(init.binary_int, base=2), L, "0")
    elseif init.binary_decimal !== nothing
        # Parse binary decimal: 0.101 → "101"
        decimal_str = string(init.binary_decimal)
        if !startswith(decimal_str, "0.")
            throw(ArgumentError("binary_decimal must be in format 0.xxx (e.g., 0.101)"))
        end
        bitstr = decimal_str[3:end]  # Skip "0."
        # Validate only 0/1
        if !all(c in ('0', '1') for c in bitstr)
            throw(ArgumentError("binary_decimal digits must be 0 or 1"))
        end
        # Pad or truncate to L
        if length(bitstr) < L
            rpad(bitstr, L, "0")
        elseif length(bitstr) > L
            bitstr[1:L]
        else
            bitstr
        end
    elseif init.bitstring !== nothing
        # Use bitstring directly, pad or truncate to L
        bitstr = init.bitstring
        if length(bitstr) < L
            rpad(bitstr, L, "0")
        elseif length(bitstr) > L
            bitstr[1:L]
        else
            bitstr
        end
    else
        throw(ArgumentError("ProductState has no initialization method specified"))
    end
    
    # bit_pattern_str[i] is the bit value at PHYSICAL site i (MSB at site 1)
    vec_int_pos = [string(c) for c in bit_pattern_str]
    
    # Map to state names based on site_type
    site_type = state.site_type
    state_names_physical = if site_type == "Qubit"
        # "0" → "0", "1" → "1"
        vec_int_pos
    elseif site_type == "S=1"
        # For S=1: "0" → "Up" (m=+1), "1" → "Dn" (m=-1)
        # ITensor uses "Up"/"Z0"/"Dn" for m = +1, 0, -1
        # Binary encoding: 0 = spin up, 1 = spin down
        [b == "0" ? "Up" : "Dn" for b in vec_int_pos]
    elseif site_type == "Qudit"
        # Generic qudit: "0" → "1", "1" → "2", etc. (1-indexed states)
        # For binary qudits, "0" → "1", "1" → "2"
        [string(parse(Int, b) + 1) for b in vec_int_pos]
    else
        throw(ArgumentError("Unknown site_type: $site_type"))
    end
    
    # Reorder to RAM order using ram_phy
    # ram_state_names[ram_idx] = state name for RAM site ram_idx
    ram_state_names = [state_names_physical[state.ram_phy[i]] for i in 1:L]
    
    # Create MPS from state names
    state.mps = MPS(state.sites, ram_state_names)
    
    return nothing
end

"""
    initialize!(state::SimulationState, init::RandomMPS)

Initialize state with a random MPS.
Requires RNGRegistry with :state_init stream attached to state.
"""
function initialize!(state::SimulationState, init::RandomMPS)
    if state.rng_registry === nothing
        throw(ArgumentError(
            "RandomMPS requires RNGRegistry with :state_init stream. " *
            "Attach RNG before calling initialize! via: " *
            "state = SimulationState(..., rng=RNGRegistry(...))"
        ))
    end
    
    # Use ITensorMPS randomMPS with specified bond dimension
    # Note: ITensorMPS 0.3+ uses Random.default_rng() internally
    # For reproducibility with our RNG, we'd need to seed it
    # For now, just use the specified bond_dim
    state.mps = randomMPS(state.sites; linkdims=init.bond_dim)
    
    return nothing
end
