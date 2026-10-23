using Random
using Dates
using DelimitedFiles
using QuantumCircuitsMPS2
using ITensors
using ITensorMPS
using LinearAlgebra

BLAS.set_num_threads(1)

ITensors.state(::StateName"+Z1", ::SiteType"S=1") = [1, 1, 1] ./ sqrt(3)

const DIMER_RUNNER_VERSION = "v9_dimer_sector_circuitshift_order_fullspavg"

function normalize_observable_mode(mode::AbstractString)
    m = lowercase(strip(mode))
    m in ("all", "full", "standard") && return :all
    m in ("entropy_only", "entanglement_only", "s_only", "s") && return :entropy_only
    m in ("order_only", "orders_only", "op_only", "so_only") && return :order_only
    throw(ArgumentError("observable_mode must be all, entropy_only, or order_only, got $mode"))
end

function normalize_halfcut_start_mode(mode::AbstractString)
    m = lowercase(strip(mode))
    m in ("half", "halfcut", "fixed_half") && return :half
    m in ("random", "rand") && return :random
    k = try
        parse(Int, m)
    catch
        throw(ArgumentError("halfcut_start_mode must be half, random, or a positive integer halfcut start, got $mode"))
    end
    k >= 1 || throw(ArgumentError("halfcut start must be >= 1, got $mode"))
    return k
end

derive_halfcut_start_seed(master_seed::Int) = master_seed + 1_000_003

function choose_halfcut_start(L::Int, half_L::Int, cut_mode, cut_rng)
    cut_mode === :half && return 1
    cut_mode === :random && return rand(cut_rng, 1:half_L)
    start = cut_mode::Int
    1 <= start <= half_L || throw(ArgumentError("fixed halfcut start must satisfy 1 <= start <= L/2, got start=$start for L=$L"))
    return start
end

@inline physical_to_mps_site(site::Int, halfcut_start::Int, L::Int) =
    mod1(site - halfcut_start + 1, L)

function halfcut_rotated_ram_phy(L::Int, halfcut_start::Int)
    return [mod1(halfcut_start + offset, L) for offset in 0:(L - 1)]
end

function apply_halfcut_basis_rotation!(sim::SimulationState, halfcut_start::Int)
    sim.ram_phy = halfcut_rotated_ram_phy(sim.L, halfcut_start)
    sim.phy_ram = invperm(sim.ram_phy)
    return nothing
end

function default_folded_halfcut_start(L::Int)
    half_L = div(L, 2)
    return mod1(1 - div(half_L, 2), L)
end

function circuit_shift_for_halfcut_start(L::Int, halfcut_start::Int)
    return mod(default_folded_halfcut_start(L) - halfcut_start, L)
end

function entropy_from_density_eigs(vals; renyi_index::Real=1, base::Real=2)
    probs = [max(real(p), 0.0) for p in vals]
    tr = sum(probs)
    tr > 0 || return 0.0
    probs ./= tr
    probs = [p for p in probs if p > 1e-14]
    isempty(probs) && return 0.0
    if renyi_index == 1
        return -sum(p -> p * log(p) / log(base), probs)
    end
    return log(sum(p -> p^renyi_index, probs)) / (1 - renyi_index) / log(base)
end

function shifted_halfcut_entropy(psi::MPS, start::Int, L::Int; renyi_index::Real=1, base::Real=2)
    half_L = div(L, 2)
    stop = start + half_L - 1
    @assert 1 <= start <= half_L "start must be in 1:L/2 for unique shifted halfcuts."
    @assert stop <= L "shifted halfcut wraps; use the complementary unique start in 1:L/2."

    psi_c = orthogonalize(psi, start)
    rho_boundary = ITensor(1.0)
    for n in start:stop
        A = psi_c[n]
        rho_boundary *= A * dag(prime(A, "Link"))
    end

    boundary = Index[]
    start > 1 && push!(boundary, linkind(psi_c, start - 1))
    stop < L && push!(boundary, linkind(psi_c, stop))
    isempty(boundary) && return 0.0

    boundary_p = prime.(boundary)
    if length(boundary) == 1
        rho = Matrix(Array(rho_boundary, boundary[1], boundary_p[1]))
    else
        row_combiner = combiner(boundary...; tags="halfcut_row")
        col_combiner = combiner(boundary_p...; tags="halfcut_col")
        rho_combined = rho_boundary * row_combiner * col_combiner
        rho = Matrix(Array(rho_combined, combinedind(row_combiner), combinedind(col_combiner)))
    end

    vals = eigvals(Hermitian(rho))
    return entropy_from_density_eigs(vals; renyi_index=renyi_index, base=base)
end

# =========================
# DEFINE PROJECTORS (MATCHING YOUR SNIPPET)
# =========================
P0 = total_spin_projector(0)
P1 = total_spin_projector(1)
P_not_2 = P0 + P1

proj_gate          = SpinSectorProjection(P_not_2)  # NN AKLT projector
proj_gate_dimer    = SpinSectorProjection(P0)       # dimer projector

I9                 = Matrix{Float64}(I, 9, 9)
proj_gate_identity = SpinSectorProjection(I9)       # identity projector (replaces "Nothing")

    
function runy(el::Int,
              pnn::Float64,
              cutoffin::Float64,
              timesteps::Int,          # = number of sweeps
              iterations::Int,         # = number of runs
              periodic::Bool,
              self_av_space_it::Int,   # Active for SO_NN spatial averaging
              ti::Int,                 # timestep start (1..timesteps)
              tf::Int,                 # timestep end
              interval::Int,           # timestep spacing
              master_seed::Int,
              filename::String,
              dimer_sector::String="odd",
              observable_mode::String="all",
              halfcut_start_mode::String="half",
              halfcut_start_seed::Int=0
              )

    # ----------------------------------
    # MASTER RNG (controls seed generation only)
    # ----------------------------------
    master_rng = MersenneTwister(master_seed)

    # -------------------------
    # checks
    # -------------------------
    L = el
    @assert iseven(L) "L must be even."
    @assert 0.0 ≤ pnn ≤ 1.0 "pnn must be in [0,1]."
    @assert timesteps ≥ 1 "timesteps must be >= 1."
    @assert iterations ≥ 1 "iterations must be >= 1."
    @assert interval ≥ 1 "interval must be >= 1."
    @assert 1 ≤ ti ≤ tf ≤ timesteps "Need 1 ≤ ti ≤ tf ≤ timesteps."
    @assert (tf - ti) % interval == 0 "Need (tf - ti) divisible by interval."
    @assert self_av_space_it ≥ 1 "self_av_space_it must be >= 1."
    @assert dimer_sector in ("odd", "even") "dimer_sector must be \"odd\" or \"even\"."

    # -------------------------
    # map periodic::Bool -> bc::Symbol
    # -------------------------
    bc::Symbol = periodic ? :periodic : :open
    is_periodic = periodic

    # -------------------------
    # measurement grid (TIMESTEPS)
    # -------------------------
    times = collect(ti:interval:tf)
    nmeas = length(times)

    # ----------------------------------
    # Output matrices
    # Reserve ONE header row:
    #   row 1: seed_ctrl_it  (per iteration column)
    # Rows 2..end are the actual timeseries with time in col 1.
    # ----------------------------------
    nrows = nmeas + 1

    S_mat           = fill(NaN, nrows, 1 + iterations)
    Scut_mat        = fill(NaN, nrows, 1 + iterations)
    D_mat           = fill(NaN, nrows, 1 + iterations)
    bond_mat        = fill(NaN, nrows, 1 + iterations)

    # SO_NN gets self_av_space_it columns per iteration
    SO_NN_mat       = fill(NaN, nrows, 1 + (iterations * self_av_space_it))
    SO_NN_m1_mat    = fill(NaN, nrows, 1 + (iterations * self_av_space_it))

    S_mat[2:end, 1]        .= times
    Scut_mat[2:end, 1]     .= times
    D_mat[2:end, 1]        .= times
    bond_mat[2:end, 1]     .= times
    SO_NN_mat[2:end, 1]    .= times
    SO_NN_m1_mat[2:end, 1] .= times

    # -------------------------
    # geometry per sweep (timestep)
    # -------------------------
    pairs_per_sweep = periodic ? L : (L - 1)
    Ldiv2 = div(L, 2)
    is_dimer_sector_bond(i_raw::Int) = dimer_sector == "odd" ? isodd(i_raw) : iseven(i_raw)
    obs_mode = normalize_observable_mode(observable_mode)
    cut_mode = normalize_halfcut_start_mode(halfcut_start_mode)
    compute_entropy_observables = obs_mode != :order_only
    compute_order_observables = obs_mode != :entropy_only
    cut_seed = halfcut_start_seed == 0 ? derive_halfcut_start_seed(master_seed) : halfcut_start_seed
    cut_rng = MersenneTwister(cut_seed)
    halfcut_start_by_iter = Vector{Int}(undef, iterations)
    circuit_shift_by_iter = Vector{Int}(undef, iterations)

    # store compiled plan per iteration if you want it back
    compiled_per_iter = Vector{Vector{NamedTuple{(:t,:i,:j,:gate_name),Tuple{Int,Int,Int,String}}}}(undef, iterations)

    # =========================
    # DEFINE PROJECTORS (From Gold Standard)
    # =========================
    P0 = total_spin_projector(0)
    P1 = total_spin_projector(1)
    P_not_2 = P0 + P1

    proj_gate          = SpinSectorProjection(P_not_2)  # NN AKLT projector
    proj_gate_dimer    = SpinSectorProjection(P0)       # dimer projector
    I9                 = Matrix{Float64}(I, 9, 9)
    proj_gate_identity = SpinSectorProjection(I9)       # Valid identity projector!

    t_global_start = now()

    for it in 1:iterations
        # ----------------------------------
        # Generate fresh random seeds INSIDE loop (from master_rng)
        # ----------------------------------
        seed_ctrl_it  = Int(rand(master_rng, UInt32))
        if obs_mode == :order_only
            halfcut_start_it = 1
            circuit_shift_it = 0
        else
            halfcut_start_it = choose_halfcut_start(L, Ldiv2, cut_mode, cut_rng)
            circuit_shift_it = circuit_shift_for_halfcut_start(L, halfcut_start_it)
        end
        halfcut_start_by_iter[it] = halfcut_start_it
        circuit_shift_by_iter[it] = circuit_shift_it

        # Put seed into the top row for this iteration column
        S_mat[1, 1 + it]    = seed_ctrl_it
        Scut_mat[1, 1 + it] = seed_ctrl_it
        Scut_mat[2:end, 1 + it] .= halfcut_start_it
        D_mat[1, 1 + it]    = seed_ctrl_it
        bond_mat[1, 1 + it] = seed_ctrl_it

        # Write the seed into EVERY column corresponding to this iteration for SO_NN and SO_NN_m1
        for k_shift in 1:self_av_space_it
            col_idx = 1 + (it - 1) * self_av_space_it + k_shift
            SO_NN_mat[1, col_idx]    = seed_ctrl_it
            SO_NN_m1_mat[1, col_idx] = seed_ctrl_it
        end

        # RNG registry (only :ctrl matters for compilation)
        rng_reg = RNGRegistry(ctrl=seed_ctrl_it, proj=1, haar=1, born=1)

        # SimulationState
        sim = SimulationState(L=L, bc=bc, site_type="S=1",
                              cutoff=cutoffin, maxdim=Int(1e10), rng=rng_reg)

        # Change "+Z1" to your X-basis string (e.g. "XUp" or "+X1") to match paper and avoid state annihilation!
        sim.mps = MPS(ComplexF64, sim.sites, ["+Z1" for _ in 1:L])

        # Keep QuantumCircuitsMPS2's folded PBC RAM order intact for fast gates.
        # Translate the whole physical circuit instead, so the fixed folded
        # cut=L/2 measures the target physical block beginning at halfcut_start_it.
        obs_entropy = compute_entropy_observables ? EntanglementEntropy(cut=Ldiv2, renyi_index=1, base=2) : nothing
        obs_dimer   = compute_order_observables ? DimerOrder(periodic=is_periodic) : nothing

        # ctrl RNG (seed_ctrl only)
        ctrl_rng = get_rng(sim.rng_registry, :ctrl)

        # =========================
        # PRECOMPILE CIRCUIT INSIDE runy (grouped by timestep)
        # =========================
        compiled_by_t = Vector{Vector{NamedTuple{(:i,:j,:gate,:gate_name),Tuple{Int,Int,Any,String}}}}(undef, timesteps)
        for t in 1:timesteps
            compiled_by_t[t] = NamedTuple{(:i,:j,:gate,:gate_name),Tuple{Int,Int,Any,String}}[]
            sizehint!(compiled_by_t[t], pairs_per_sweep)
        end

        for t in 1:timesteps
            # sweep i=1..L-1 always
            for i_raw in 1:(L-1)
                j_raw = i_raw + 1
                i_shift = mod1(i_raw + circuit_shift_it, L)
                j_shift = mod1(j_raw + circuit_shift_it, L)
                i, j, ok = sanitize_pair(i_shift, j_shift, L, bc)
                ok || continue

                u = rand(ctrl_rng)
                is_sector_bond = is_dimer_sector_bond(i_raw)

                if is_sector_bond
                    if u < pnn
                        gate = proj_gate
                        gate_name = "NN-AKLT"
                    else
                        gate = proj_gate_dimer
                        gate_name = "DIMER"
                    end
                else
                    if u < pnn
                        gate = proj_gate
                        gate_name = "NN-AKLT"
                    else
                        gate = proj_gate_identity
                        gate_name = "IDENTITY"
                    end
                end

                push!(compiled_by_t[t], (i=i, j=j, gate=gate, gate_name=gate_name))
            end

            # periodic wrap bond (L,1) if requested
            if periodic
                i_raw, j_raw = L, 1
                i_shift = mod1(i_raw + circuit_shift_it, L)
                j_shift = mod1(j_raw + circuit_shift_it, L)
                i, j, ok = sanitize_pair(i_shift, j_shift, L, bc)
                if ok
                    u = rand(ctrl_rng)
                    is_sector_bond = is_dimer_sector_bond(i_raw)

                    if is_sector_bond
                        if u < pnn
                            gate = proj_gate
                            gate_name = "NN-AKLT"
                        else
                            gate = proj_gate_dimer
                            gate_name = "DIMER"
                        end
                    else
                        if u < pnn
                            gate = proj_gate
                            gate_name = "NN-AKLT"
                        else
                            gate = proj_gate_identity
                            gate_name = "IDENTITY"
                        end
                    end

                    push!(compiled_by_t[t], (i=i, j=j, gate=gate, gate_name=gate_name))
                end
            end
        end

        # flatten for returning/debugging
        compiled_flat = NamedTuple{(:t,:i,:j,:gate_name),Tuple{Int,Int,Int,String}}[]
        for t in 1:timesteps
            for g in compiled_by_t[t]
                push!(compiled_flat, (t=t, i=g.i, j=g.j, gate_name=g.gate_name))
            end
        end
        compiled_per_iter[it] = compiled_flat

        # =========================
        # RUN: execute compiled circuit
        # =========================
        meas_row = 0
        for t in 1:timesteps
            for g in compiled_by_t[t]
                apply!(sim, g.gate, [g.i, g.j])
            end

            if (t ≥ ti) && (t ≤ tf) && ((t - ti) % interval == 0)
                meas_row += 1
                row = 1 + meas_row  # shift by +1 due to the single seed header

                if compute_entropy_observables
                    S = obs_entropy(sim)
                    bdim = maximum(linkdims(sim.mps))

                    S_mat[row, 1 + it]     = S
                    bond_mat[row, 1 + it]  = bdim
                end

                if compute_order_observables
                    d  = obs_dimer(sim)
                    D_mat[row, 1 + it] = d

                    # --- Exact True Translational Spatial Averaging ---
                    for k in 1:self_av_space_it
                        i_site = mod1(k, L)

                        # separation = L/2
                        j_site = mod1(Ldiv2 + k, L)
                        val = real(StringOrder(i_site, j_site; order=1)(sim))

                        # separation = L/2 - 1
                        j_site_m1 = mod1(Ldiv2 - 1 + k, L)
                        val_m1 = real(StringOrder(i_site, j_site_m1; order=1)(sim))

                        # Column Index logic
                        col_idx = 1 + (it - 1) * self_av_space_it + k
                        SO_NN_mat[row, col_idx]    = val
                        SO_NN_m1_mat[row, col_idx] = val_m1
                    end

                    # --- Console Output matching Gold Standard ---
                    kmax = min(2, self_av_space_it)
                    base_col = 1 + (it - 1) * self_av_space_it

                    nn1    = SO_NN_mat[row, base_col + 1]
                    nn2    = (kmax >= 2) ? SO_NN_mat[row, base_col + 2] : NaN
                    nn1_m1 = SO_NN_m1_mat[row, base_col + 1]
                    nn2_m1 = (kmax >= 2) ? SO_NN_m1_mat[row, base_col + 2] : NaN
                end

#                if kmax >= 2
#                    println("it=$it t=$t: S_half=$S | bond_dim=$bdim | SO_NN[k=1]=$nn1 | SO_NN[k=2]=$nn2 | SO_NN_m1[k=1]=$nn1_m1 | SO_NN_m1[k=2]=$nn2_m1 | Dimer=$d")
#                else
#                    println("it=$it t=$t: S_half=$S | bond_dim=$bdim | SO_NN[k=1]=$nn1 | SO_NN_m1[k=1]=$nn1_m1 | Dimer=$d")
#                end
            end
        end

        @assert meas_row == nmeas "Internal error: meas_row=$meas_row but expected nmeas=$nmeas."
    end

    t_global_end = now()

    # =========================
    # WRITE OUTPUTS
    # =========================
    # Construct a descriptive header from the simulation parameters
    hcut_token = iterations == 1 ? string(halfcut_start_by_iter[1]) : "mixed"
    hcut_method = "circuitshift"
    cshift_token = iterations == 1 ? string(circuit_shift_by_iter[1]) : "mixed"
    header = string("_pn=", pnn,
                "_cutoff=", cutoffin,
                "_it=", iterations,
                "_ti=", ti,
                "_tf=", timesteps,
                "_sa=", self_av_space_it,
                "_obs=", obs_mode,
                "_soavg=periodic_fullpath",
                "_hcutmethod=", hcut_method,
                "_hcutmode=", cut_mode,
                "_hcutstart=", hcut_token,
                "_cshift=", cshift_token,
                "_hlen=", Ldiv2,
                "_dsec_", dimer_sector,
                "_PBC=", periodic)

    if compute_entropy_observables
        writedlm(string(filename, header, "_S_QCMPS_dimer"),         S_mat,        ", ")
        writedlm(string(filename, header, "_HcutStart_QCMPS_dimer"), Scut_mat,     ", ")
        writedlm(string(filename, header, "_bond_QCMPS_dimer"),      bond_mat,     ", ")
    end
    if compute_order_observables
        writedlm(string(filename, header, "_SO_NN_QCMPS_dimer"),     SO_NN_mat,    ", ")
        writedlm(string(filename, header, "_SO_NNm1_QCMPS_dimer"),   SO_NN_m1_mat, ", ")
        writedlm(string(filename, header, "_Dimer_QCMPS_dimer"),     D_mat,        ", ")
    end

    println(DIMER_RUNNER_VERSION)
    return (S_mat=S_mat,
            Scut_mat=Scut_mat,
            bond_mat=bond_mat,
            SO_NN_mat=SO_NN_mat,
            SO_NN_m1_mat=SO_NN_m1_mat,
            Dimer_mat=D_mat,
            compiled_per_iter=compiled_per_iter,
            bc=bc,
            dimer_sector=dimer_sector,
            periodic=is_periodic,
            observable_mode=obs_mode,
            halfcut_start_mode=cut_mode,
            halfcut_start_by_iter=halfcut_start_by_iter,
            circuit_shift_by_iter=circuit_shift_by_iter)
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 13 || length(ARGS) > 16
        error("Usage: julia run_dimer_sector_cli.jl L pnn cutoff timesteps iterations periodic self_av_space_it ti tf interval master_seed output_prefix dimer_sector [observable_mode=all|entropy_only|order_only] [halfcut_start_mode=half|random|K] [halfcut_start_seed=0]")
    end

    L = parse(Int, ARGS[1])
    pnn = parse(Float64, ARGS[2])
    cutoffin = parse(Float64, ARGS[3])
    timesteps = parse(Int, ARGS[4])
    iterations = parse(Int, ARGS[5])
    periodic = parse(Bool, lowercase(ARGS[6]))
    self_av_space_it = parse(Int, ARGS[7])
    ti = parse(Int, ARGS[8])
    tf = parse(Int, ARGS[9])
    interval = parse(Int, ARGS[10])
    master_seed = parse(Int, ARGS[11])
    filename = ARGS[12]
    dimer_sector = ARGS[13]
    observable_mode = length(ARGS) >= 14 ? ARGS[14] : "all"
    halfcut_start_mode = length(ARGS) >= 15 ? ARGS[15] : "half"
    halfcut_start_seed = length(ARGS) >= 16 ? parse(Int, ARGS[16]) : 0

    runy(L, pnn, cutoffin, timesteps, iterations, periodic, self_av_space_it,
         ti, tf, interval, master_seed, filename, dimer_sector,
         observable_mode, halfcut_start_mode, halfcut_start_seed)
end
