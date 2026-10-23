module AKLTStatevector

# Refactored AKLT spin-1 statevector code.
# Key difference from original: flat Vector{ComplexF64} of length 3^L instead of
# a rank-L tensor, and all gate/observable operations work in-place using direct
# index arithmetic (no permutedims, no temporary arrays during the hot loop).
# Pattern follows DURandomCircuit.jl: pre-allocate small scratch buffers once,
# pass them into every hot function.

using Random, LinearAlgebra, DelimitedFiles, Dates
BLAS.set_num_threads(1)  # hard-limit BLAS to 1 thread regardless of node core count

const CUDA_IS_LOADED = Ref(false)
const CUDA_LOAD_ERROR = Ref{Any}(nothing)
try
    @eval using CUDA
    CUDA_IS_LOADED[] = true
catch e
    CUDA_LOAD_ERROR[] = e
end

# ── Spin-1 operators (module-level constants, computed once) ──────────────────
const Sz_sv       = ComplexF64[1 0 0; 0 0 0; 0 0 -1]
const Sp_sv       = ComplexF64[0 sqrt(2) 0; 0 0 sqrt(2); 0 0 0]
const Sm_sv       = Matrix(Sp_sv')
const S1S2        = kron(Sz_sv, Sz_sv) + 0.5*(kron(Sp_sv, Sm_sv) + kron(Sm_sv, Sp_sv))
const S1S2_sq     = S1S2 * S1S2
const I9          = Matrix{ComplexF64}(I, 9, 9)
const P0_sv       = (1/3)*S1S2_sq - (1/3)*I9
const P1_sv       = -(1/2)*S1S2_sq - (1/2)*S1S2 + I9
const PNOT2       = P0_sv + P1_sv
const PNOT2_22    = PNOT2[2, 2]
const PNOT2_24    = PNOT2[2, 4]
const PNOT2_33    = PNOT2[3, 3]
const PNOT2_35    = PNOT2[3, 5]
const PNOT2_37    = PNOT2[3, 7]
const PNOT2_42    = PNOT2[4, 2]
const PNOT2_44    = PNOT2[4, 4]
const PNOT2_53    = PNOT2[5, 3]
const PNOT2_55    = PNOT2[5, 5]
const PNOT2_57    = PNOT2[5, 7]
const PNOT2_66    = PNOT2[6, 6]
const PNOT2_68    = PNOT2[6, 8]
const PNOT2_73    = PNOT2[7, 3]
const PNOT2_75    = PNOT2[7, 5]
const PNOT2_77    = PNOT2[7, 7]
const PNOT2_86    = PNOT2[8, 6]
const PNOT2_88    = PNOT2[8, 8]
const Sz_1site    = ComplexF64[1 0 0; 0 0 0; 0 0 -1]
const expSz_1site = ComplexF64[-1 0 0; 0 1 0; 0 0 -1]

# ── State initialization ──────────────────────────────────────────────────────
"""Flat spin-1 state vector, uniform superposition, length 3^L."""
init_psi(L::Int) = fill(ComplexF64(3.0^(-L/2)), 3^L)

# ── In-place two-site gate ────────────────────────────────────────────────────
# Spin-1 flat-vector index convention (column-major, Julia default):
#   idx(m_1,...,m_L) = 1 + sum_k  m_k * 3^(k-1),   m_k in {0,1,2}
#
# For sites i < j the 3^(L-2) "background" configurations split into:
#   nR in 0:3^(i-1)-1      (sites 1..i-1,   stride 1 to 3^(i-2))
#   nm in 0:3^(j-i-1)-1    (sites i+1..j-1, stride 3^i to 3^(j-2))
#   nL in 0:3^(L-j)-1      (sites j+1..L,   stride 3^j to 3^(L-1))
#
# base_0indexed = nR  +  nm * 3^i  +  nL * 3^j
# element at (alpha_i, alpha_j) = psi[1 + base + alpha_i*3^(i-1) + alpha_j*3^(j-1)]
#
# Buffer packing:  buf[1 + alpha_i + 3*alpha_j]   (alpha_i varies fastest)
# PNOT2 is invariant under site-swap so buffer ordering does not affect physics.

"""
Apply 9x9 operator P in-place on sites i < j of flat spin-1 vector psi (length 3^L).
buf_in, buf_out: pre-allocated length-9 scratch vectors (reused across calls).
"""
function apply_two_site_op!(psi::Vector{ComplexF64}, P::Matrix{ComplexF64},
                              i::Int, j::Int, L::Int,
                              buf_in::Vector{ComplexF64}, buf_out::Vector{ComplexF64})
    # PNOT2 is symmetric under site swap, so we can always enforce i < j
    i, j = i < j ? (i, j) : (j, i)
    si     = 3^(i-1)
    sj     = 3^(j-1)
    nR_max = si            # = 3^(i-1)
    nm_max = 3^(j-i-1)
    nL_max = 3^(L-j)
    step_m = 3^i           # stride between middle-block groups
    step_L = 3^j           # stride between left-block groups

    @inbounds for nL in 0:(nL_max-1)
        base_L = nL * step_L
        for nm in 0:(nm_max-1)
            base_m = base_L + nm * step_m
            for nR in 0:(nR_max-1)
                base = base_m + nR   # 0-indexed

                # Pack: buf_in[1 + alpha_i + 3*alpha_j]
                for αj in 0:2
                    bj = base + αj * sj
                    for αi in 0:2
                        buf_in[1 + αi + 3*αj] = psi[1 + bj + αi * si]
                    end
                end

                # Apply operator (no allocation: in-place BLAS gemv)
                mul!(buf_out, P, buf_in)

                # Unpack back
                for αj in 0:2
                    bj = base + αj * sj
                    for αi in 0:2
                        psi[1 + bj + αi * si] = buf_out[1 + αi + 3*αj]
                    end
                end
            end
        end
    end
end

@inline function apply_pnot2_block!(buf_out::Vector{ComplexF64}, buf_in::Vector{ComplexF64})
    @inbounds begin
        x2 = buf_in[2]; x3 = buf_in[3]; x4 = buf_in[4]; x5 = buf_in[5]
        x6 = buf_in[6]; x7 = buf_in[7]; x8 = buf_in[8]

        buf_out[1] = 0
        buf_out[2] = PNOT2_22*x2 + PNOT2_24*x4
        buf_out[3] = PNOT2_33*x3 + PNOT2_35*x5 + PNOT2_37*x7
        buf_out[4] = PNOT2_42*x2 + PNOT2_44*x4
        buf_out[5] = PNOT2_53*x3 + PNOT2_55*x5 + PNOT2_57*x7
        buf_out[6] = PNOT2_66*x6 + PNOT2_68*x8
        buf_out[7] = PNOT2_73*x3 + PNOT2_75*x5 + PNOT2_77*x7
        buf_out[8] = PNOT2_86*x6 + PNOT2_88*x8
        buf_out[9] = 0
    end
    return buf_out
end

function apply_pnot2_op!(psi::Vector{ComplexF64}, i::Int, j::Int, L::Int,
                         buf_in::Vector{ComplexF64}, buf_out::Vector{ComplexF64})
    i, j = i < j ? (i, j) : (j, i)
    si     = 3^(i-1)
    sj     = 3^(j-1)
    nR_max = si
    nm_max = 3^(j-i-1)
    nL_max = 3^(L-j)
    step_m = 3^i
    step_L = 3^j

    @inbounds for nL in 0:(nL_max-1)
        base_L = nL * step_L
        for nm in 0:(nm_max-1)
            base_m = base_L + nm * step_m
            for nR in 0:(nR_max-1)
                base = base_m + nR

                for αj in 0:2
                    bj = base + αj * sj
                    for αi in 0:2
                        buf_in[1 + αi + 3*αj] = psi[1 + bj + αi * si]
                    end
                end

                apply_pnot2_block!(buf_out, buf_in)

                for αj in 0:2
                    bj = base + αj * sj
                    for αi in 0:2
                        psi[1 + bj + αi * si] = buf_out[1 + αi + 3*αj]
                    end
                end
            end
        end
    end
end

"""Apply PNOT2 on sites i,j in-place, then renormalise psi."""
function project_normalize!(psi::Vector{ComplexF64}, i::Int, j::Int, L::Int,
                              buf9_in::Vector{ComplexF64}, buf9_out::Vector{ComplexF64})
    apply_pnot2_op!(psi, i, j, L, buf9_in, buf9_out)
    normalize!(psi)   # BLAS nrm2 + rmul! -- zero allocation
end

# ── In-place one-site operator ────────────────────────────────────────────────
"""
Apply 3x3 operator A in-place on site k of flat spin-1 vector psi.
buf_in, buf_out: pre-allocated length-3 scratch vectors.
"""
function apply_one_site!(psi::Vector{ComplexF64}, A::Matrix{ComplexF64},
                          k::Int, L::Int,
                          buf_in::Vector{ComplexF64}, buf_out::Vector{ComplexF64})
    sk     = 3^(k-1)
    step_k = 3 * sk      # = 3^k
    nR_max = sk
    nL_max = 3^(L-k)

    @inbounds for nL in 0:(nL_max-1)
        base_L = nL * step_k
        for nR in 0:(nR_max-1)
            base = base_L + nR

            buf_in[1] = psi[1 + base]
            buf_in[2] = psi[1 + base + sk]
            buf_in[3] = psi[1 + base + 2*sk]

            mul!(buf_out, A, buf_in)

            psi[1 + base]        = buf_out[1]
            psi[1 + base + sk]   = buf_out[2]
            psi[1 + base + 2*sk] = buf_out[3]
        end
    end
end

# ── Entanglement entropy ──────────────────────────────────────────────────────
# Still needs an SVD, but can reuse caller-provided work for the full-state matrix copy.
@inline function is_identity_perm(p::Vector{Int})
    @inbounds for i in eachindex(p)
        p[i] == i || return false
    end
    return true
end

function entanglement_entropy(psi::Vector{ComplexF64}, psi_work::Vector{ComplexF64},
                               L::Int, cut::Int, ram_phy::Vector{Int};
                               renyi::Int=1, logbase::Float64=2.0, threshold::Float64=0.0)
    dims = ntuple(_->3, L)
    if is_identity_perm(ram_phy)
        copyto!(psi_work, psi)
    else
        permutedims!(reshape(psi_work, dims), reshape(psi, dims), ram_phy)
    end

    M = reshape(psi_work, 3^cut, 3^(L-cut))
    s = svdvals!(M)

    invlogbase = inv(log(logbase))
    if renyi == 1
        entropy = 0.0
        @inbounds for σ in s
            p = σ^2
            if p > threshold
                entropy -= p * log(p) * invlogbase
            end
        end
        return entropy
    else
        acc = 0.0
        @inbounds for σ in s
            p = σ^2
            if p > threshold
                acc += p^renyi
            end
        end
        return log(acc) * invlogbase / (1-renyi)
    end
end

function entanglement_entropy(psi::Vector{ComplexF64}, L::Int, cut::Int, ram_phy::Vector{Int};
                               renyi::Int=1, logbase::Float64=2.0, threshold::Float64=0.0)
    psi_work = similar(psi)
    return entanglement_entropy(psi, psi_work, L, cut, ram_phy;
                                renyi=renyi, logbase=logbase, threshold=threshold)
end

function entropy_from_probabilities(vals; renyi::Int=1, logbase::Float64=2.0,
                                    threshold::Float64=0.0)
    invlogbase = inv(log(logbase))
    if renyi == 1
        entropy = 0.0
        @inbounds for p0 in vals
            p = real(p0)
            if p > threshold
                entropy -= p * log(p) * invlogbase
            end
        end
        return entropy
    else
        acc = 0.0
        @inbounds for p0 in vals
            p = real(p0)
            if p > threshold
                acc += p^renyi
            end
        end
        return log(acc) * invlogbase / (1-renyi)
    end
end

function entropy_from_singular_values(vals; renyi::Int=1, logbase::Float64=2.0,
                                      threshold::Float64=0.0)
    invlogbase = inv(log(logbase))
    if renyi == 1
        entropy = 0.0
        @inbounds for σ in vals
            p = real(σ)^2
            if p > threshold
                entropy -= p * log(p) * invlogbase
            end
        end
        return entropy
    else
        acc = 0.0
        @inbounds for σ in vals
            p = real(σ)^2
            if p > threshold
                acc += p^renyi
            end
        end
        return log(acc) * invlogbase / (1-renyi)
    end
end

struct EntropyGpuWorkspace
    state_work::Any
    rho_work::Any
end

struct EntropyRegionSpec
    cut::Int
    ram_phy::Vector{Int}
    rowdim::Int
    coldim::Int
    is_identity::Bool
    use_reduced_density::Bool
    row_offsets::Vector{Int}
    col_offsets::Vector{Int}
end

struct InformationRegionSpecs
    tmi_A::EntropyRegionSpec
    tmi_B::EntropyRegionSpec
    tmi_C::EntropyRegionSpec
    tmi_AB::EntropyRegionSpec
    tmi_AC::EntropyRegionSpec
    tmi_BC::EntropyRegionSpec
    tmi_ABC::EntropyRegionSpec
end

@inline normalize_site(i::Int, L::Int) = ((i-1) % L) + 1

function unique_sites(L::Int, regions::Vector{Int}...)
    seen = falses(L)
    out = Int[]
    for region in regions
        for site in region
            s = normalize_site(site, L)
            if !seen[s]
                seen[s] = true
                push!(out, s)
            end
        end
    end
    return out
end

function complement_sites(region::Vector{Int}, L::Int)
    in_region = falses(L)
    for site in region
        in_region[normalize_site(site, L)] = true
    end
    return [site for site in 1:L if !in_region[site]]
end

function circular_region(start_site::Int, len::Int, L::Int)
    len <= 0 && return Int[]
    return [normalize_site(start_site + k, L) for k in 0:(len-1)]
end

function basis_offsets(sites::Vector{Int})
    offsets = Vector{Int}(undef, 3^length(sites))
    strides = 3 .^ (sites .- 1)
    @inbounds for idx in 0:(length(offsets)-1)
        x = idx
        offset = 0
        for q in eachindex(sites)
            digit = x % 3
            x = x ÷ 3
            offset += digit * strides[q]
        end
        offsets[idx+1] = offset
    end
    return offsets
end

function entropy_region_spec(L::Int, region::Vector{Int})
    reg = unique_sites(L, region)
    if isempty(reg) || length(reg) == L
        return EntropyRegionSpec(0, collect(1:L), 1, 3^L, true, false, [0], [0])
    end

    if length(reg) > L ÷ 2
        reg = complement_sites(reg, L)
    end

    cut = length(reg)
    comp = complement_sites(reg, L)
    ram_phy = vcat(reg, comp)
    return EntropyRegionSpec(cut, ram_phy, 3^cut, 3^(L-cut),
                             is_identity_perm(ram_phy), 2 * cut < L,
                             basis_offsets(reg), basis_offsets(comp))
end

function entropy_region_spec_from_permutation(L::Int, cut::Int, ram_phy::Vector{Int})
    @assert length(ram_phy) == L
    @assert 0 <= cut <= L
    if cut == 0 || cut == L
        return EntropyRegionSpec(0, copy(ram_phy), 1, 3^L, true, false, [0], [0])
    end

    row_sites = collect(@view ram_phy[1:cut])
    col_sites = collect(@view ram_phy[(cut+1):end])
    return EntropyRegionSpec(cut, copy(ram_phy), 3^cut, 3^(L-cut),
                             is_identity_perm(ram_phy), 2 * cut < L,
                             basis_offsets(row_sites), basis_offsets(col_sites))
end

function build_information_region_specs(L::Int, shift::Int)
    L % 4 == 0 || throw(ArgumentError("TMI/BMI regions require L divisible by 4, got L=$L"))
    quarter_L = L ÷ 4

    A = circular_region(1 + shift,             quarter_L, L)
    B = circular_region(1 + quarter_L + shift, quarter_L, L)
    C = circular_region(1 + 2quarter_L + shift, quarter_L, L)

    return InformationRegionSpecs(
        entropy_region_spec(L, A),
        entropy_region_spec(L, B),
        entropy_region_spec(L, C),
        entropy_region_spec(L, unique_sites(L, A, B)),
        entropy_region_spec(L, unique_sites(L, A, C)),
        entropy_region_spec(L, unique_sites(L, B, C)),
        entropy_region_spec(L, unique_sites(L, A, B, C)),
    )
end

@inline function entropy_from_spec(psi::Vector{ComplexF64}, psi_work::Vector{ComplexF64},
                                   L::Int, spec::EntropyRegionSpec)
    spec.cut == 0 && return 0.0
    if spec.use_reduced_density
        return entanglement_entropy_reduced_density(psi, psi_work, L, spec)
    end
    return entanglement_entropy_from_spec(psi, psi_work, L, spec)
end

@inline function entropy_from_spec(psi::Vector{ComplexF64}, psi_work::Vector{ComplexF64},
                                   rho_work::Matrix{ComplexF64}, L::Int,
                                   spec::EntropyRegionSpec)
    spec.cut == 0 && return 0.0
    if spec.use_reduced_density
        return entanglement_entropy_reduced_density(psi, psi_work, rho_work, L, spec)
    end
    return entanglement_entropy_from_spec(psi, psi_work, L, spec)
end

function copy_bipartition_matrix!(psi_work::Vector{ComplexF64}, psi::Vector{ComplexF64},
                                  spec::EntropyRegionSpec)
    @assert length(spec.row_offsets) == spec.rowdim
    @assert length(spec.col_offsets) == spec.coldim
    @inbounds for col in 1:spec.coldim
        base = spec.col_offsets[col]
        outbase = (col - 1) * spec.rowdim
        for row in 1:spec.rowdim
            psi_work[outbase + row] = psi[1 + spec.row_offsets[row] + base]
        end
    end
    return psi_work
end

function entanglement_entropy_from_spec(psi::Vector{ComplexF64},
                                        psi_work::Vector{ComplexF64},
                                        L::Int, spec::EntropyRegionSpec;
                                        renyi::Int=1, logbase::Float64=2.0,
                                        threshold::Float64=0.0)
    if spec.is_identity
        copyto!(psi_work, psi)
    else
        copy_bipartition_matrix!(psi_work, psi, spec)
    end

    M = reshape(psi_work, spec.rowdim, spec.coldim)
    s = svdvals!(M)

    invlogbase = inv(log(logbase))
    if renyi == 1
        entropy = 0.0
        @inbounds for σ in s
            p = σ^2
            if p > threshold
                entropy -= p * log(p) * invlogbase
            end
        end
        return entropy
    else
        acc = 0.0
        @inbounds for σ in s
            p = σ^2
            if p > threshold
                acc += p^renyi
            end
        end
        return log(acc) * invlogbase / (1-renyi)
    end
end

function entanglement_entropy_reduced_density(psi::Vector{ComplexF64},
                                              psi_work::Vector{ComplexF64},
                                              L::Int, spec::EntropyRegionSpec;
                                              renyi::Int=1, logbase::Float64=2.0,
                                              threshold::Float64=0.0)
    rho_work = Matrix{ComplexF64}(undef, spec.rowdim, spec.rowdim)
    return entanglement_entropy_reduced_density(psi, psi_work, rho_work, L, spec;
                                                renyi=renyi, logbase=logbase,
                                                threshold=threshold)
end

function entanglement_entropy_reduced_density(psi::Vector{ComplexF64},
                                              psi_work::Vector{ComplexF64},
                                              rho_work::Matrix{ComplexF64},
                                              L::Int, spec::EntropyRegionSpec;
                                              renyi::Int=1, logbase::Float64=2.0,
                                              threshold::Float64=0.0)
    @assert size(rho_work, 1) == spec.rowdim && size(rho_work, 2) == spec.rowdim
    if spec.is_identity
        copyto!(psi_work, psi)
    else
        copy_bipartition_matrix!(psi_work, psi, spec)
    end

    M = reshape(psi_work, spec.rowdim, spec.coldim)
    mul!(rho_work, M, adjoint(M))
    vals = eigvals!(Hermitian(rho_work))
    return entropy_from_probabilities(vals; renyi=renyi, logbase=logbase, threshold=threshold)
end

function ensure_cuda_functional!()
    CUDA_IS_LOADED[] ||
        throw(ArgumentError("CUDA.jl is not loaded in this Julia environment: $(CUDA_LOAD_ERROR[])"))
    Base.invokelatest(CUDA.functional) ||
        throw(ArgumentError("CUDA.jl loaded, but CUDA.functional() is false on this node"))
    return nothing
end

function normalize_entropy_backend(mode::Symbol)
    mode in (:cpu, :CPU) && return :cpu
    mode in (:gpu, :cuda, :CUDA) && return :gpu
    mode in (:auto, :cuda_auto, :gpu_auto) && return :auto
    throw(ArgumentError("entropy_backend must be :cpu, :gpu, or :auto, got $mode"))
end

function maybe_init_entropy_gpu_workspace(mode::Symbol, L::Int, needs_entropy::Bool,
                                          needs_reduced_density::Bool)
    needs_entropy || return nothing
    mode == :cpu && return nothing

    try
        ensure_cuda_functional!()
    catch e
        if mode == :auto
            println("=== GPU entropy backend unavailable; falling back to CPU: ", e)
            return nothing
        end
        rethrow(e)
    end

    state_work = CUDA.CuArray{ComplexF64}(undef, 3^L)
    rho_dim = needs_reduced_density ? 3^(L ÷ 4) : 1
    rho_work = CUDA.CuArray{ComplexF64}(undef, rho_dim, rho_dim)
    return EntropyGpuWorkspace(state_work, rho_work)
end

function copy_spec_matrix_to_gpu!(gpu_work::EntropyGpuWorkspace,
                                  psi::Vector{ComplexF64},
                                  psi_work::Vector{ComplexF64},
                                  spec::EntropyRegionSpec)
    if spec.is_identity
        copyto!(psi_work, psi)
    else
        copy_bipartition_matrix!(psi_work, psi, spec)
    end
    copyto!(gpu_work.state_work, psi_work)
    return reshape(gpu_work.state_work, spec.rowdim, spec.coldim)
end

function entanglement_entropy_from_spec_gpu(psi::Vector{ComplexF64},
                                            psi_work::Vector{ComplexF64},
                                            gpu_work::EntropyGpuWorkspace,
                                            L::Int, spec::EntropyRegionSpec;
                                            renyi::Int=1, logbase::Float64=2.0,
                                            threshold::Float64=0.0)
    M = copy_spec_matrix_to_gpu!(gpu_work, psi, psi_work, spec)
    s = svdvals!(M; alg=CUDA.CUSOLVER.QRAlgorithm())
    return entropy_from_singular_values(Array(s); renyi=renyi, logbase=logbase,
                                        threshold=threshold)
end

function entanglement_entropy_reduced_density_gpu(psi::Vector{ComplexF64},
                                                  psi_work::Vector{ComplexF64},
                                                  gpu_work::EntropyGpuWorkspace,
                                                  L::Int, spec::EntropyRegionSpec;
                                                  renyi::Int=1, logbase::Float64=2.0,
                                                  threshold::Float64=0.0)
    @assert size(gpu_work.rho_work, 1) >= spec.rowdim &&
            size(gpu_work.rho_work, 2) >= spec.rowdim
    M = copy_spec_matrix_to_gpu!(gpu_work, psi, psi_work, spec)
    rho = @view gpu_work.rho_work[1:spec.rowdim, 1:spec.rowdim]
    mul!(rho, M, adjoint(M))
    # CUDA.jl does not provide a Hermitian eigvals! path for this CuArray on
    # Amarel's Julia/CUDA stack, and the CPU fallback tries scalar GPU indexing.
    # The reduced matrix is small for the TMI quarter regions, so copy it back
    # after the GPU GEMM and use the ordinary CPU eigensolver.
    vals = eigvals!(Hermitian(Array(rho)))
    return entropy_from_probabilities(Array(vals); renyi=renyi, logbase=logbase,
                                      threshold=threshold)
end

@inline function entropy_from_spec_backend(psi::Vector{ComplexF64},
                                           psi_work::Vector{ComplexF64},
                                           rho_work::Matrix{ComplexF64},
                                           gpu_work,
                                           L::Int, spec::EntropyRegionSpec)
    spec.cut == 0 && return 0.0
    if gpu_work === nothing
        return entropy_from_spec(psi, psi_work, rho_work, L, spec)
    elseif spec.use_reduced_density
        return entanglement_entropy_reduced_density_gpu(psi, psi_work, gpu_work, L, spec)
    else
        return entanglement_entropy_from_spec_gpu(psi, psi_work, gpu_work, L, spec)
    end
end

function mutual_information_observables(psi::Vector{ComplexF64}, psi_work::Vector{ComplexF64},
                                        rho_work::Matrix{ComplexF64},
                                        L::Int, specs::InformationRegionSpecs)
    SA   = entropy_from_spec(psi, psi_work, rho_work, L, specs.tmi_A)
    SB   = entropy_from_spec(psi, psi_work, rho_work, L, specs.tmi_B)
    SC   = entropy_from_spec(psi, psi_work, rho_work, L, specs.tmi_C)
    SAB  = entropy_from_spec(psi, psi_work, rho_work, L, specs.tmi_AB)
    SAC  = entropy_from_spec(psi, psi_work, rho_work, L, specs.tmi_AC)
    SBC  = entropy_from_spec(psi, psi_work, rho_work, L, specs.tmi_BC)
    SABC = entropy_from_spec(psi, psi_work, rho_work, L, specs.tmi_ABC)
    tmi = SA + SB + SC - SAB - SAC - SBC + SABC
    bmi = SA + SC - SAC
    return tmi, bmi
end

function mutual_information_observables(psi::Vector{ComplexF64}, psi_work::Vector{ComplexF64},
                                        rho_work::Matrix{ComplexF64},
                                        gpu_work::EntropyGpuWorkspace,
                                        L::Int, specs::InformationRegionSpecs)
    SA   = entropy_from_spec_backend(psi, psi_work, rho_work, gpu_work, L, specs.tmi_A)
    SB   = entropy_from_spec_backend(psi, psi_work, rho_work, gpu_work, L, specs.tmi_B)
    SC   = entropy_from_spec_backend(psi, psi_work, rho_work, gpu_work, L, specs.tmi_C)
    SAB  = entropy_from_spec_backend(psi, psi_work, rho_work, gpu_work, L, specs.tmi_AB)
    SAC  = entropy_from_spec_backend(psi, psi_work, rho_work, gpu_work, L, specs.tmi_AC)
    SBC  = entropy_from_spec_backend(psi, psi_work, rho_work, gpu_work, L, specs.tmi_BC)
    SABC = entropy_from_spec_backend(psi, psi_work, rho_work, gpu_work, L, specs.tmi_ABC)
    tmi = SA + SB + SC - SAB - SAC - SBC + SABC
    bmi = SA + SC - SAC
    return tmi, bmi
end

# ── String order parameter ────────────────────────────────────────────────────
@inline function permuted_site_index(n::Int, ram_phy::Vector{Int}, strides)
    dest = 0
    @inbounds for q in eachindex(ram_phy)
        dest += ((n ÷ strides[ram_phy[q]]) % 3) * strides[q]
    end
    return dest
end

function permute_sites_inplace!(psi::Vector{ComplexF64}, L::Int, ram_phy::Vector{Int})
    is_identity_perm(ram_phy) && return psi

    strides = ntuple(k -> 3^(k-1), L)
    @inbounds for start in 0:(length(psi)-1)
        j = permuted_site_index(start, ram_phy, strides)
        while j > start
            j = permuted_site_index(j, ram_phy, strides)
        end
        j == start || continue

        temp = psi[start+1]
        cur = start
        while true
            nxt = permuted_site_index(cur, ram_phy, strides)
            if nxt == start
                psi[start+1] = temp
                break
            end
            psi[nxt+1], temp = temp, psi[nxt+1]
            cur = nxt
        end
    end

    return psi
end

function entanglement_entropy_destructive!(psi::Vector{ComplexF64},
                                           L::Int, cut::Int, ram_phy::Vector{Int};
                                           renyi::Int=1, logbase::Float64=2.0,
                                           threshold::Float64=0.0)
    permute_sites_inplace!(psi, L, ram_phy)
    M = reshape(psi, 3^cut, 3^(L-cut))
    s = svdvals!(M)

    invlogbase = inv(log(logbase))
    if renyi == 1
        entropy = 0.0
        @inbounds for σ in s
            p = σ^2
            if p > threshold
                entropy -= p * log(p) * invlogbase
            end
        end
        return entropy
    else
        acc = 0.0
        @inbounds for σ in s
            p = σ^2
            if p > threshold
                acc += p^renyi
            end
        end
        return log(acc) * invlogbase / (1-renyi)
    end
end

"""
<psi | Sz_i (expSz_{i+1}...expSz_{j-1}) Sz_j | psi>  (order=1)
<psi | Sz_i Sz_{i+1} (expSz...) Sz_{j-1} Sz_j | psi>  (order=2)

The operators are diagonal, so this is computed directly from |psi[idx]|^2.
No full-size scratch vector is needed for string order.
"""
@inline spin1_digit(n::Int, stride::Int) = (n ÷ stride) % 3
@inline spin1_sz_value(d::Int) = d == 0 ? 1.0 : (d == 1 ? 0.0 : -1.0)
@inline spin1_exp_value(d::Int) = d == 1 ? 1.0 : -1.0

const SO_KIND_NN = 1
const SO_KIND_NNN = 2
const SO_KIND_NNM1 = 3
const SO_KIND_NNNM1 = 4

struct StringOrderSpec
    kind::Int
    self_index::Int
    sz_sites::Vector{Int}
    exp_sites::Vector{Int}
end

function string_order_spec(i::Int, j::Int, order::Int, kind::Int, self_index::Int)
    @assert j > i "string_order requires j > i (got i=$i, j=$j)"
    @assert order == 1 || order == 2 "string_order order must be 1 or 2"

    if order == 1
        return StringOrderSpec(kind, self_index, [i, j], collect((i+1):(j-1)))
    end
    return StringOrderSpec(kind, self_index, [i, i+1, j-1, j], collect((i+2):(j-2)))
end

function build_string_order_specs(L::Int, half_L::Int, self_av_space_it::Int)
    specs = StringOrderSpec[]
    for k in 0:(self_av_space_it-1)
        i_site    = mod1(1 + k,          L)
        j_site    = mod1(half_L + 1 + k, L)
        j_site_m1 = mod1(half_L + k,     L)
        self_index = k + 1

        push!(specs, string_order_spec(i_site, j_site, 1, SO_KIND_NN, self_index))
        push!(specs, string_order_spec(i_site, j_site_m1, 1, SO_KIND_NNM1, self_index))
        if j_site - i_site >= 4
            push!(specs, string_order_spec(i_site, j_site, 2, SO_KIND_NNN, self_index))
        end
        if j_site_m1 - i_site >= 4
            push!(specs, string_order_spec(i_site, j_site_m1, 2, SO_KIND_NNNM1, self_index))
        end
    end
    return specs
end

function string_order_observables!(values::Vector{Float64}, psi::Vector{ComplexF64},
                                   specs::Vector{StringOrderSpec}, digits::Vector{Int})
    @assert length(values) == length(specs)
    fill!(values, 0.0)

    @inbounds for n in 0:(length(psi)-1)
        x = n
        for site in eachindex(digits)
            digits[site] = x % 3
            x = div(x, 3)
        end

        prob = abs2(psi[n+1])
        for idx in eachindex(specs)
            spec = specs[idx]
            coeff = 1.0
            for site in spec.sz_sites
                d = digits[site]
                if d == 1
                    coeff = 0.0
                    break
                elseif d == 2
                    coeff = -coeff
                end
            end
            coeff == 0.0 && continue

            for site in spec.exp_sites
                digits[site] != 1 && (coeff = -coeff)
            end
            values[idx] += prob * coeff
        end
    end

    return values
end

function string_order(psi::Vector{ComplexF64}, i::Int, j::Int, order::Int, L::Int)
    @assert j > i "string_order requires j > i (got i=$i, j=$j)"
    @assert order == 1 || order == 2 "string_order order must be 1 or 2"

    strides = ntuple(k -> 3^(k-1), L)
    acc = 0.0

    @inbounds for n in 0:(length(psi)-1)
        coeff = spin1_sz_value(spin1_digit(n, strides[i]))

        if order == 1
            coeff *= spin1_sz_value(spin1_digit(n, strides[j]))
            coeff == 0.0 && continue
            for k in (i+1):(j-1)
                coeff *= spin1_exp_value(spin1_digit(n, strides[k]))
            end
        else
            coeff *= spin1_sz_value(spin1_digit(n, strides[i+1]))
            coeff *= spin1_sz_value(spin1_digit(n, strides[j-1]))
            coeff *= spin1_sz_value(spin1_digit(n, strides[j]))
            coeff == 0.0 && continue
            for k in (i+2):(j-2)
                coeff *= spin1_exp_value(spin1_digit(n, strides[k]))
            end
        end

        acc += abs2(psi[n+1]) * coeff
    end

    return acc
end

function string_order(psi::Vector{ComplexF64}, psi_work::Vector{ComplexF64},
                       i::Int, j::Int, order::Int, L::Int,
                       buf3_in::Vector{ComplexF64}, buf3_out::Vector{ComplexF64})
    return string_order(psi, i, j, order, L)
end

# ── BC helpers ────────────────────────────────────────────────────────────────
wrap_site(i::Int, L::Int) = ((i-1) % L) + 1

function sanitize_pair(i::Int, j::Int, L::Int, bc::Symbol)
    if bc == :periodic
        return wrap_site(i, L), wrap_site(j, L), true
    elseif bc == :open
        ok = (1 <= i <= L) && (1 <= j <= L) && i != j
        return i, j, ok
    else
        error("bc must be :open or :periodic, got $bc")
    end
end

function compute_basis_mapping(L::Int, bc::Symbol)
    bc == :open && return collect(1:L), collect(1:L)
    iseven(L) || throw(ArgumentError("PBC folded basis requires even L"))
    ram_phy = Int[]
    for (a, b) in zip(1:L÷2, reverse((L÷2+1):L))
        push!(ram_phy, a); push!(ram_phy, b)
    end
    return invperm(ram_phy), ram_phy
end

function normalize_observable_mode(mode::Symbol)
    mode in (:all, :both) && return :all
    mode in (:standard, :standard_only, :old_only) && return :standard_only
    mode in (:information, :information_only, :info_only, :tmi_bmi_only) && return :info_only
    throw(ArgumentError("observable_mode must be :all, :standard_only, or :info_only, got $mode"))
end

# ── Main simulation ───────────────────────────────────────────────────────────
function runy(
    L::Int,
    cutoffin::Float64,
    bc::Symbol,
    p_nn::Float64,
    n_steps::Int,
    iterations::Int,
    self_av_space_it::Int,
    ti::Int,
    tf::Int,
    interval::Int,
    master_seed::Int,
    filename::AbstractString,
    observable_mode::Symbol=:all,
    entropy_backend::Symbol=:cpu,
)
    t0 = now()
    obs_mode = normalize_observable_mode(observable_mode)
    entropy_mode = normalize_entropy_backend(entropy_backend)
    println("\n=== AKLTStatevector.runy (in-place) | L=$L | bc=$bc | p_nn=$p_nn | observables=$obs_mode | entropy_backend=$entropy_mode")
    println("=== master_seed=$master_seed | iterations=$iterations")

    @assert iseven(L)
    @assert 1 <= ti <= tf <= n_steps
    @assert (tf - ti) % interval == 0

    master_rng = MersenneTwister(master_seed)
    rec_times  = collect(ti:interval:tf)
    nmeas      = length(rec_times)
    nrows      = nmeas + 1
    final_entropy_only = nmeas == 1 && rec_times[1] == n_steps
    compute_standard = obs_mode != :info_only
    compute_information = obs_mode != :standard_only && L % 4 == 0
    if obs_mode != :standard_only && L % 4 != 0
        obs_mode == :info_only &&
            throw(ArgumentError("observable_mode=:info_only requires L divisible by 4, got L=$L"))
        println("=== TMI/BMI skipped because L=$L is not divisible by 4")
    end

    empty_mat     = Matrix{Float64}(undef, 0, 0)
    S_mat         = compute_standard ? fill(NaN, nrows, 1 + iterations) : empty_mat
    SO_NN_mat     = compute_standard ? fill(NaN, nrows, 1 + iterations * self_av_space_it) : empty_mat
    SO_NNN_mat    = compute_standard ? fill(NaN, nrows, 1 + iterations * self_av_space_it) : empty_mat
    SO_NNm1_mat   = compute_standard ? fill(NaN, nrows, 1 + iterations * self_av_space_it) : empty_mat
    SO_NNNm1_mat  = compute_standard ? fill(NaN, nrows, 1 + iterations * self_av_space_it) : empty_mat
    TMI_mat        = compute_information ? fill(NaN, nrows, 1 + iterations * self_av_space_it) : empty_mat
    BMI_mat        = compute_information ? fill(NaN, nrows, 1 + iterations * self_av_space_it) : empty_mat

    if compute_standard
        for mat in (S_mat, SO_NN_mat, SO_NNN_mat, SO_NNm1_mat, SO_NNNm1_mat)
            mat[2:end, 1] .= rec_times
        end
    end
    if compute_information
        TMI_mat[2:end, 1] .= rec_times
        BMI_mat[2:end, 1] .= rec_times
    end

    _phy_ram, ram_phy = compute_basis_mapping(L, bc)
    half_L = L ÷ 2
    standard_entropy_spec = entropy_region_spec_from_permutation(L, half_L, ram_phy)
    use_fused_string_order = compute_standard && L >= 16
    string_order_specs = use_fused_string_order ? build_string_order_specs(L, half_L, self_av_space_it) :
                         StringOrderSpec[]
    info_specs = compute_information ?
                 [build_information_region_specs(L, k) for k in 0:(self_av_space_it-1)] :
                 InformationRegionSpecs[]

    # ── Pre-allocate all scratch buffers once ──────────────────────────────
    buf9_in  = zeros(ComplexF64, 9)
    buf9_out = zeros(ComplexF64, 9)
    state_len = 3^L
    init_amp = ComplexF64(3.0^(-L/2))
    psi      = Vector{ComplexF64}(undef, state_len)
    entropy_work = compute_information || !final_entropy_only ? zeros(ComplexF64, state_len) : nothing
    rho_dim = compute_information ? 3^(L ÷ 4) : 0
    rho_work = Matrix{ComplexF64}(undef, rho_dim, rho_dim)
    entropy_gpu_work = maybe_init_entropy_gpu_workspace(entropy_mode, L,
                                                        compute_information || compute_standard,
                                                        compute_information)
    entropy_gpu_work !== nothing && println("=== CUDA entropy backend active")
    string_order_vals = use_fused_string_order ? zeros(Float64, length(string_order_specs)) : Float64[]
    string_order_digits = use_fused_string_order ? Vector{Int}(undef, L) : Int[]

    for it in 1:iterations
        seed_ctrl_it = Int(rand(master_rng, UInt32))
        compute_standard && (S_mat[1, 1 + it] = seed_ctrl_it)
        for ks in 1:self_av_space_it
            col = 1 + (it-1)*self_av_space_it + ks
            if compute_standard
                SO_NN_mat[1,col] = SO_NNN_mat[1,col] =
                SO_NNm1_mat[1,col] = SO_NNNm1_mat[1,col] = seed_ctrl_it
            end
            if compute_information
                TMI_mat[1,col] = BMI_mat[1,col] = seed_ctrl_it
            end
        end

        ctrl_rng = MersenneTwister(seed_ctrl_it)
        fill!(psi, init_amp)

        meas_row = 0
        for t in 1:n_steps
            # Generate and apply each gate immediately to avoid storing n_steps sweeps.
            for i_raw in 1:L
                u     = rand(ctrl_rng)
                j_raw = i_raw + (u < p_nn ? 1 : 2)
                i2, j2, ok = sanitize_pair(i_raw, j_raw, L, bc)
                ok && project_normalize!(psi, i2, j2, L, buf9_in, buf9_out)
            end

            if ti <= t <= tf && (t - ti) % interval == 0
                meas_row += 1
                row = meas_row + 1

                if compute_standard
                    if use_fused_string_order
                        string_order_observables!(string_order_vals, psi, string_order_specs,
                                                  string_order_digits)
                        for idx in eachindex(string_order_specs)
                            spec = string_order_specs[idx]
                            col = 1 + (it-1)*self_av_space_it + spec.self_index
                            if spec.kind == SO_KIND_NN
                                SO_NN_mat[row, col] = string_order_vals[idx]
                            elseif spec.kind == SO_KIND_NNN
                                SO_NNN_mat[row, col] = string_order_vals[idx]
                            elseif spec.kind == SO_KIND_NNM1
                                SO_NNm1_mat[row, col] = string_order_vals[idx]
                            else
                                SO_NNNm1_mat[row, col] = string_order_vals[idx]
                            end
                        end
                    else
                        for k in 0:(self_av_space_it-1)
                            i_site    = mod1(1 + k,          L)
                            j_site    = mod1(half_L + 1 + k, L)
                            j_site_m1 = mod1(half_L + k,     L)
                            col = 1 + (it-1)*self_av_space_it + (k+1)

                            SO_NN_mat[row, col]   = string_order(psi, i_site, j_site,    1, L)
                            SO_NNm1_mat[row, col] = string_order(psi, i_site, j_site_m1, 1, L)
                            if j_site - i_site >= 4
                                SO_NNN_mat[row, col] = string_order(psi, i_site, j_site, 2, L)
                            end
                            if j_site_m1 - i_site >= 4
                                SO_NNNm1_mat[row, col] = string_order(psi, i_site, j_site_m1, 2, L)
                            end
                        end
                    end
                end

                if compute_information
                    for k in 0:(self_av_space_it-1)
                        col = 1 + (it-1)*self_av_space_it + (k+1)
                        specs = info_specs[k+1]
                        if entropy_gpu_work === nothing
                            TMI_mat[row, col], BMI_mat[row, col] =
                                mutual_information_observables(psi, entropy_work::Vector{ComplexF64},
                                                               rho_work, L, specs)
                        else
                            TMI_mat[row, col], BMI_mat[row, col] =
                                mutual_information_observables(psi, entropy_work::Vector{ComplexF64},
                                                               rho_work, entropy_gpu_work, L, specs)
                        end
                    end
                end

                if compute_standard
                    if final_entropy_only && t == n_steps
                        S_mat[row, 1+it] = entanglement_entropy_destructive!(psi, L, half_L, ram_phy)
                    else
                        S_mat[row, 1+it] =
                            entropy_from_spec_backend(psi, entropy_work::Vector{ComplexF64},
                                                      rho_work, entropy_gpu_work, L,
                                                      standard_entropy_spec)
                    end
                end
            end
        end
        @assert meas_row == nmeas
    end

    println("Run complete in ", now() - t0)

    header = string("_pn=", p_nn, "_cutoff=", cutoffin,
                    "_it=", iterations, "_ti=", ti,
                    "_tf=", n_steps, "_PBC=", bc == :periodic)
    try
        if compute_standard
            writedlm(string(filename, header, "_S_SV_nnn"),        S_mat,        ", ")
            writedlm(string(filename, header, "_SO_NN_SV_nnn"),    SO_NN_mat,    ", ")
            writedlm(string(filename, header, "_SO_NNN_SV_nnn"),   SO_NNN_mat,   ", ")
            writedlm(string(filename, header, "_SO_NNm1_SV_nnn"),  SO_NNm1_mat,  ", ")
            writedlm(string(filename, header, "_SO_NNNm1_SV_nnn"), SO_NNNm1_mat, ", ")
        end
        if compute_information
            writedlm(string(filename, header, "_TMI_SV_nnn"),  TMI_mat,      ", ")
            writedlm(string(filename, header, "_BMI_SV_nnn"),  BMI_mat,      ", ")
        end
    catch e
        println("Warning: could not write output files: ", e)
    end
end

end # module AKLTStatevector
