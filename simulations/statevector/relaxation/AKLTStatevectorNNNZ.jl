module AKLTStatevectorNNNZ

# Refactored AKLT spin-1 statevector code.
# Key difference from original: flat Vector{ComplexF64} of length 3^L instead of
# a rank-L tensor, and all gate/observable operations work in-place using direct
# index arithmetic (no permutedims, no temporary arrays during the hot loop).
# Pattern follows DURandomCircuit.jl: pre-allocate small scratch buffers once,
# pass them into every hot function.

using Random, LinearAlgebra, DelimitedFiles, Dates
BLAS.set_num_threads(1)  # hard-limit BLAS to 1 thread regardless of node core count

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

"""Apply PNOT2 on sites i,j in-place, then renormalise psi."""
function project_normalize!(psi::Vector{ComplexF64}, i::Int, j::Int, L::Int,
                              buf9_in::Vector{ComplexF64}, buf9_out::Vector{ComplexF64})
    apply_two_site_op!(psi, PNOT2, i, j, L, buf9_in, buf9_out)
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
                               renyi::Int=1, logbase::Float64=2.0, threshold::Float64=1e-16)
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
            p = max(σ, threshold)^2
            entropy -= p * log(p) * invlogbase
        end
        return entropy
    else
        acc = 0.0
        @inbounds for σ in s
            p = max(σ, threshold)^2
            acc += p^renyi
        end
        return log(acc) * invlogbase / (1-renyi)
    end
end

function entanglement_entropy(psi::Vector{ComplexF64}, L::Int, cut::Int, ram_phy::Vector{Int};
                               renyi::Int=1, logbase::Float64=2.0, threshold::Float64=1e-16)
    psi_work = similar(psi)
    return entanglement_entropy(psi, psi_work, L, cut, ram_phy;
                                renyi=renyi, logbase=logbase, threshold=threshold)
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
                                           threshold::Float64=1e-16)
    permute_sites_inplace!(psi, L, ram_phy)
    M = reshape(psi, 3^cut, 3^(L-cut))
    s = svdvals!(M)

    invlogbase = inv(log(logbase))
    if renyi == 1
        entropy = 0.0
        @inbounds for σ in s
            p = max(σ, threshold)^2
            entropy -= p * log(p) * invlogbase
        end
        return entropy
    else
        acc = 0.0
        @inbounds for σ in s
            p = max(σ, threshold)^2
            acc += p^renyi
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
function string_order_periodic_distance(psi::Vector{ComplexF64},
                                        start::Int,
                                        dist::Int,
                                        order::Int,
                                        L::Int)
    @assert 1 <= start <= L
    @assert 1 <= dist <= L - 1
    @assert order == 1 || order == 2 "string_order order must be 1 or 2"
    order == 2 && @assert dist >= 3 "order=2 needs at least four distinct sites"

    strides = ntuple(k -> 3^(k-1), L)
    site(offset::Int) = mod1(start + offset, L)
    acc = 0.0

    @inbounds for n in 0:(length(psi)-1)
        coeff = spin1_sz_value(spin1_digit(n, strides[start]))

        if order == 1
            coeff *= spin1_sz_value(spin1_digit(n, strides[site(dist)]))
            coeff == 0.0 && continue
            for offset in 1:(dist-1)
                coeff *= spin1_exp_value(spin1_digit(n, strides[site(offset)]))
            end
        else
            coeff *= spin1_sz_value(spin1_digit(n, strides[site(1)]))
            coeff *= spin1_sz_value(spin1_digit(n, strides[site(dist-1)]))
            coeff *= spin1_sz_value(spin1_digit(n, strides[site(dist)]))
            coeff == 0.0 && continue
            for offset in 2:(dist-2)
                coeff *= spin1_exp_value(spin1_digit(n, strides[site(offset)]))
            end
        end

        acc += abs2(psi[n+1]) * coeff
    end

    return acc
end

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

# ── Main simulation ───────────────────────────────────────────────────────────
function folded_contiguous_region(start::Int, len::Int, L::Int)
    region = [wrap_site(start + k, L) for k in 0:(len-1)]
    center = div(len + 1, 2)
    out = Int[region[center]]
    for radius in 1:len
        left = center - radius
        right = center + radius
        left >= 1 && push!(out, region[left])
        right <= len && push!(out, region[right])
    end
    return out
end

function nnn_z_halfcut_folded_ram_phy(L::Int, halfcut_start::Int)
    @assert iseven(L)
    half_L = div(L, 2)
    a = folded_contiguous_region(halfcut_start, half_L, L)
    b = folded_contiguous_region(halfcut_start + half_L, half_L, L)
    ram_phy = vcat(a, b)
    length(unique(ram_phy)) == L || error("invalid folded halfcut permutation")
    return ram_phy
end

function normalize_nnn_z_halfcut_start_mode(mode::AbstractString, L::Int)
    s = lowercase(strip(mode))
    s in ("random", "rand") && return :random, 0, "random"
    s in ("sequential", "seq", "cycle") && return :sequential, 0, "sequential"

    for prefix in ("start=", "start:", "fixed=", "fixed:")
        if startswith(s, prefix)
            raw = parse(Int, s[(lastindex(prefix)+1):end])
            fixed_start = wrap_site(raw, L)
            return :fixed, fixed_start, string("fixed", fixed_start)
        end
    end

    raw = try
        parse(Int, s)
    catch
        throw(ArgumentError("halfcut_start_mode must be random, sequential, or an integer site; got $mode"))
    end
    fixed_start = wrap_site(raw, L)
    return :fixed, fixed_start, string("fixed", fixed_start)
end

derive_nnn_z_halfcut_seed(master_seed::Int, halfcut_start_seed::Int) =
    halfcut_start_seed == 0 ? master_seed + 314159 : halfcut_start_seed

function choose_nnn_z_halfcut_start(mode::Symbol,
                                    fixed_start::Int,
                                    it::Int,
                                    L::Int,
                                    rng::AbstractRNG)
    mode == :random && return rand(rng, 1:L)
    mode == :sequential && return wrap_site(it, L)
    mode == :fixed && return fixed_start
    error("unknown halfcut mode $mode")
end

function choose_nnn_z_halfcut_starts(mode::Symbol,
                                     fixed_start::Int,
                                     it::Int,
                                     L::Int,
                                     rng::AbstractRNG,
                                     count::Int)
    unique_count = div(L, 2)
    1 <= count <= unique_count || throw(ArgumentError(
        "halfcut_count must be between 1 and L/2=$unique_count, got $count"
    ))
    if mode == :random
        # Starts separated by L/2 define complementary copies of the same cut.
        return randperm(rng, unique_count)[1:count]
    elseif mode == :sequential
        return [mod1(it + offset, unique_count) for offset in 0:(count-1)]
    elseif mode == :fixed
        count == 1 || throw(ArgumentError("fixed halfcut mode supports halfcut_count=1 only"))
        return [mod1(fixed_start, unique_count)]
    end
    error("unknown halfcut mode $mode")
end

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
    halfcut_start_mode::AbstractString="random",
    halfcut_start_seed::Int=0,
    halfcut_count::Int=1,
)
    t0 = now()
    mode = Symbol(observable_mode)
    valid_modes = (:all, :standard_only, :entropy_only, :order_only, :order_parameters_only, :no_entanglement)
    mode in valid_modes || throw(ArgumentError("observable_mode must be one of $valid_modes, got $observable_mode"))
    compute_entropy = !(mode in (:order_only, :order_parameters_only, :no_entanglement))
    compute_order = mode != :entropy_only
    println("\n=== AKLTStatevectorNNNZ.runy (gate timestep) | L=$L | bc=$bc | p_nn=$p_nn | observables=$mode")
    println("=== master_seed=$master_seed | iterations=$iterations")

    @assert iseven(L)
    bc == :periodic || throw(ArgumentError("AKLTStatevectorNNNZ currently expects periodic boundary conditions"))
    @assert 1 <= ti <= tf <= n_steps
    @assert (tf - ti) % interval == 0
    @assert 1 <= self_av_space_it <= L

    master_rng = MersenneTwister(master_seed)
    hcut_mode, hcut_fixed_start, hcut_mode_label = normalize_nnn_z_halfcut_start_mode(halfcut_start_mode, L)
    hcut_seed_eff = derive_nnn_z_halfcut_seed(master_seed, halfcut_start_seed)
    hcut_rng = MersenneTwister(hcut_seed_eff)
    rec_times  = collect(ti:interval:tf)
    nmeas      = length(rec_times)
    nrows      = nmeas + 1
    final_entropy_only = nmeas == 1 && rec_times[1] == n_steps

    entropy_columns = iterations * halfcut_count
    S_mat         = compute_entropy ? fill(NaN, nrows, 1 + entropy_columns) : Matrix{Float64}(undef, 0, 0)
    HcutStart_mat = compute_entropy ? fill(NaN, 2, 1 + entropy_columns) : Matrix{Float64}(undef, 0, 0)
    SO_NN_mat    = compute_order ? fill(NaN, nrows, 1 + iterations * self_av_space_it) : Matrix{Float64}(undef, 0, 0)
    SO_NNN_mat   = compute_order ? fill(NaN, nrows, 1 + iterations * self_av_space_it) : Matrix{Float64}(undef, 0, 0)
    SO_NNm1_mat  = compute_order ? fill(NaN, nrows, 1 + iterations * self_av_space_it) : Matrix{Float64}(undef, 0, 0)
    SO_NNNm1_mat = compute_order ? fill(NaN, nrows, 1 + iterations * self_av_space_it) : Matrix{Float64}(undef, 0, 0)

    if compute_entropy
        S_mat[2:end, 1] .= rec_times
        HcutStart_mat[1, 1] = -1
        HcutStart_mat[2, 1] = -2
    end
    if compute_order
        for mat in (SO_NN_mat, SO_NNN_mat, SO_NNm1_mat, SO_NNNm1_mat)
            mat[2:end, 1] .= rec_times
        end
    end

    half_L = div(L, 2)
    hcut_start_header = hcut_mode == :fixed ? string(hcut_fixed_start) : "mixed"

    # ── Pre-allocate all scratch buffers once ──────────────────────────────
    buf9_in  = zeros(ComplexF64, 9)
    buf9_out = zeros(ComplexF64, 9)
    state_len = 3^L
    init_amp = ComplexF64(3.0^(-L/2))
    psi      = Vector{ComplexF64}(undef, state_len)
    entropy_work = (compute_entropy && !final_entropy_only) ? zeros(ComplexF64, state_len) : nothing

    for it in 1:iterations
        # Production workers run one trajectory, so retain the manifest's
        # globally unique controller seed instead of reducing it to UInt32.
        seed_ctrl_it = iterations == 1 ? master_seed : Int(rand(master_rng, UInt32))
        hcut_starts = compute_entropy ? choose_nnn_z_halfcut_starts(
            hcut_mode, hcut_fixed_start, it, L, hcut_rng, halfcut_count
        ) : Int[]
        ram_phys = compute_entropy ? [nnn_z_halfcut_folded_ram_phy(L, start) for start in hcut_starts] : Vector{Vector{Int}}()
        if compute_entropy
            for cut_index in 1:halfcut_count
                col = 1 + (it - 1) * halfcut_count + cut_index
                S_mat[1, col] = seed_ctrl_it
                HcutStart_mat[1, col] = seed_ctrl_it
                HcutStart_mat[2, col] = hcut_starts[cut_index]
            end
        end
        if compute_order
            for ks in 1:self_av_space_it
                col = 1 + (it-1)*self_av_space_it + ks
                SO_NN_mat[1,col] = SO_NNN_mat[1,col] =
                SO_NNm1_mat[1,col] = SO_NNNm1_mat[1,col] = seed_ctrl_it
            end
        end

        ctrl_rng = MersenneTwister(seed_ctrl_it)
        fill!(psi, init_amp)

        meas_row = 0
        for t in 1:n_steps
            # One z-timestep is one microscopic projector/gate attempt.
            i_raw = mod1(t, L)
            u     = rand(ctrl_rng)
            j_raw = i_raw + (u < p_nn ? 1 : 2)
            i2, j2, ok = sanitize_pair(i_raw, j_raw, L, bc)
            ok && project_normalize!(psi, i2, j2, L, buf9_in, buf9_out)

            if ti <= t <= tf && (t - ti) % interval == 0
                meas_row += 1
                row = meas_row + 1

                if compute_order
                    for k in 0:(self_av_space_it-1)
                        i_site = mod1(1 + k, L)
                        col = 1 + (it-1)*self_av_space_it + (k+1)

                        SO_NN_mat[row, col]   = string_order_periodic_distance(psi, i_site, half_L,     1, L)
                        SO_NNm1_mat[row, col] = string_order_periodic_distance(psi, i_site, half_L - 1, 1, L)
                        if half_L >= 4
                            SO_NNN_mat[row, col]   = string_order_periodic_distance(psi, i_site, half_L,     2, L)
                        end
                        if half_L - 1 >= 4
                            SO_NNNm1_mat[row, col] = string_order_periodic_distance(psi, i_site, half_L - 1, 2, L)
                        end
                    end
                end

                if compute_entropy
                    for cut_index in 1:halfcut_count
                        col = 1 + (it - 1) * halfcut_count + cut_index
                        S_mat[row, col] = entanglement_entropy(
                            psi, entropy_work::Vector{ComplexF64}, L, half_L, ram_phys[cut_index]
                        )
                    end
                end
            end
        end
        @assert meas_row == nmeas
    end

    println("Run complete in ", now() - t0)

    header = string("_pn=", p_nn, "_cutoff=", cutoffin,
                    "_it=", iterations, "_ti=", ti,
                    "_tf=", tf, "_nsteps=", n_steps,
                    "_tstep=gate",
                    "_hcutmethod=folded_basisrot",
                    "_hcutmode=", hcut_mode_label,
                    "_hcutstart=", hcut_start_header,
                    "_hcutseed=", hcut_seed_eff,
                    "_hcutcount=", halfcut_count,
                    "_hcutunique=true",
                    "_observables=", mode,
                    "_sa=", self_av_space_it,
                    "_PBC=", bc == :periodic)
    try
        if compute_entropy
            writedlm(string(filename, header, "_S_SV_nnn_z"), S_mat, ", ")
            writedlm(string(filename, header, "_HcutStart_SV_nnn_z"), HcutStart_mat, ", ")
        end
        if compute_order
            writedlm(string(filename, header, "_SO_NN_SV_nnn_z"),    SO_NN_mat,    ", ")
            writedlm(string(filename, header, "_SO_NNN_SV_nnn_z"),   SO_NNN_mat,   ", ")
            writedlm(string(filename, header, "_SO_NNm1_SV_nnn_z"),  SO_NNm1_mat,  ", ")
            writedlm(string(filename, header, "_SO_NNNm1_SV_nnn_z"), SO_NNNm1_mat, ", ")
        end
    catch e
        println("Warning: could not write output files: ", e)
    end
end

end # module AKLTStatevectorNNNZ
