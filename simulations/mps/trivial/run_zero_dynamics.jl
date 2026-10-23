using Random
using Dates
using DelimitedFiles
using QuantumCircuitsMPS2
using ITensors
using ITensorMPS
using LinearAlgebra # Needed for Identity (I) and exp

# ==========================================
# CUSTOM 1-SITE GATE FOR QMPS ENGINE
# ==========================================
struct FZeroGate <: QuantumCircuitsMPS2.AbstractGate
    tau::Float64
end
QuantumCircuitsMPS2.support(::FZeroGate) = 1
function QuantumCircuitsMPS2.build_operator(gate::FZeroGate, site::Index, local_dim::Int; kwargs...)
    projzero = [0.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 0.0]
    mat = exp(projzero * gate.tau)
    return itensor(mat, site', site)
end 

ITensors.state(::StateName"+Z1", ::SiteType"S=1") = [1.0, 1.0, 1.0]

function normalize_zero_z_observable_mode(mode::AbstractString)
    m = lowercase(strip(String(mode)))
    m in ("all", "full") && return :all
    m in ("entropy_only", "entanglement_only", "s_only", "s") && return :entropy_only
    throw(ArgumentError("observable_mode must be all or entropy_only, got $mode"))
end

function normalize_zero_z_halfcut_start_mode(mode::AbstractString)
    m = lowercase(strip(String(mode)))
    m in ("half", "halfcut", "fixed_half") && return :half
    m in ("random", "rand") && return :random
    k = try
        parse(Int, m)
    catch
        throw(ArgumentError("halfcut_start_mode must be half, random, or an integer start, got $mode"))
    end
    k >= 1 || throw(ArgumentError("halfcut start must be >= 1, got $mode"))
    return k
end

derive_zero_z_halfcut_seed(master_seed::Int) = master_seed + 314159

function choose_zero_z_halfcut_start(L::Int, half_L::Int, mode, rng)
    mode === :half && return 1
    mode === :random && return rand(rng, 1:half_L)
    start = mode::Int
    1 <= start <= half_L || throw(ArgumentError("fixed halfcut start must satisfy 1 <= start <= L/2, got start=$start for L=$L"))
    return start
end

function zero_z_halfcut_folded_ram_phy(L::Int, halfcut_start::Int)
    half_L = div(L, 2)
    center = mod1(halfcut_start + div(half_L, 2), L)
    ram_phy = Int[]
    sizehint!(ram_phy, L)
    seen = falses(L)

    for offset in 0:L
        left = mod1(center - offset, L)
        if !seen[left]
            push!(ram_phy, left)
            seen[left] = true
        end
        length(ram_phy) == L && break

        right = mod1(center + offset, L)
        if !seen[right]
            push!(ram_phy, right)
            seen[right] = true
        end
        length(ram_phy) == L && break
    end

    @assert length(ram_phy) == L
    @assert Set(ram_phy[1:half_L]) == Set(halfcut_start:(halfcut_start + half_L - 1))
    return ram_phy
end

function apply_zero_z_halfcut_basis_rotation!(sim::SimulationState, halfcut_start::Int)
    sim.ram_phy = zero_z_halfcut_folded_ram_phy(sim.L, halfcut_start)
    sim.phy_ram = invperm(sim.ram_phy)
    return nothing
end # zero_z_folded_basisrot_halfcut_v1

function zero_z_entropy_from_density_eigs(vals; renyi_index::Real=1, base::Real=2)
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

function zero_z_shifted_halfcut_entropy(psi::MPS, start::Int, L::Int; renyi_index::Real=1, base::Real=2)
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
    return zero_z_entropy_from_density_eigs(vals; renyi_index=renyi_index, base=base)
end # zero_z_shifted_halfcut_v1

function zero_z_periodic_string_order_1(state::SimulationState, i_phys::Int, j_phys::Int)
    L = state.L
    1 <= i_phys <= L || throw(ArgumentError("i_phys=$i_phys outside 1:$L"))
    1 <= j_phys <= L || throw(ArgumentError("j_phys=$j_phys outside 1:$L"))
    i_phys != j_phys || throw(ArgumentError("string endpoints must be distinct"))

    psi_copy = copy(state.mps)

    i_ram = state.phy_ram[i_phys]
    j_ram = state.phy_ram[j_phys]
    psi_copy[i_ram] = psi_copy[i_ram] * op("Sz", state.sites[i_ram])

    k_phys = mod1(i_phys + 1, L)
    while k_phys != j_phys
        k_ram = state.phy_ram[k_phys]
        psi_copy[k_ram] = psi_copy[k_ram] * op("expSz", state.sites[k_ram])
        k_phys = mod1(k_phys + 1, L)
    end

    psi_copy[j_ram] = psi_copy[j_ram] * op("Sz", state.sites[j_ram])
    noprime!(psi_copy)
    return real(inner(state.mps, psi_copy))
end # zero_z_periodic_forward_string_order_v1


# ==========================================
# MAIN FUNCTION (Zero Code - z dynamics; timestep = one local gate/projector)
# ==========================================
function runy(el::Int, pnn::Float64, cutoffin::Float64, timesteps::Int, iterations::Int, periodic::Bool, self_av_space_it::Int, ti::Int, tf::Int,
interval::Int, master_seed::Int, filename::String, observable_mode::AbstractString="all", halfcut_start_mode::AbstractString="random", halfcut_start_seed::Int=0)

    # ----------------------------------
    # MASTER RNG
    # ----------------------------------
    master_rng = MersenneTwister(master_seed)
    obs_mode = normalize_zero_z_observable_mode(observable_mode) # zero_z_observable_mode_v1
    halfcut_mode = normalize_zero_z_halfcut_start_mode(halfcut_start_mode)
    hcut_seed = halfcut_start_seed == 0 ? derive_zero_z_halfcut_seed(master_seed) : halfcut_start_seed
    halfcut_rng = MersenneTwister(hcut_seed) # zero_z_random_entropy_cut_v1
    compute_order_observables = obs_mode == :all

    # -------------------------
    # checks
    # -------------------------
    L = el
    @assert iseven(L) "L must be even."
    @assert 0.0 <= pnn <= 1.0 "pnn must be in [0,1]."
    @assert timesteps >= 1 "timesteps must be >= 1."
    @assert iterations >= 1 "iterations must be >= 1."
    @assert interval >= 1 "interval must be >= 1."
    @assert 1 <= ti <= tf <= timesteps "Need 1 <= ti <= tf <= timesteps."
    @assert (tf - ti) % interval == 0 "Need (tf - ti) divisible by interval."
    @assert self_av_space_it >= 1 "self_av_space_it must be >= 1."

    @assert self_av_space_it <= L "full spatial averaging expects self_av_space_it <= L; got $self_av_space_it for L=$L"

    bc::Symbol = periodic ? :periodic : :open
    is_periodic = periodic

    # -------------------------
    # measurement grid
    # -------------------------
    times = collect(ti:interval:tf)
    nmeas = length(times)

    nrows = nmeas + 1

    S_mat       = fill(NaN, nrows, 1 + iterations)
    bond_mat    = fill(NaN, nrows, 1 + iterations)
    Scut_mat    = fill(NaN, nrows, 1 + iterations)
    SO_NN_mat   = fill(NaN, nrows, 1 + (iterations * self_av_space_it))
    SO_NNm1_mat = fill(NaN, nrows, 1 + (iterations * self_av_space_it)) # NEW: Initialize L/2-1 matrix

    S_mat[2:end, 1]       .= times
    bond_mat[2:end, 1]    .= times
    Scut_mat[2:end, 1]    .= times
    SO_NN_mat[2:end, 1]   .= times
    SO_NNm1_mat[2:end, 1] .= times # NEW: Set time column

    Ldiv2 = div(L, 2)
    halfcut_start_by_iter = fill(1, iterations)
    # Z template: each timestep stores exactly one local gate/projector
    compiled_per_iter = Vector{Vector{NamedTuple{(:t,:sites,:gate_name),Tuple{Int,Vector{Int},String}}}}(undef, iterations)

    # =========================
    # DEFINE PROJECTORS
    # =========================
    P0 = total_spin_projector(0)
    P1 = total_spin_projector(1)
    P_not_2 = P0 + P1
    proj_gate = SpinSectorProjection(P_not_2)  # NN AKLT projector (2-site)

    # NATIVE 1-SITE GATE
    tau = 50.0
    proj_gate_zero = FZeroGate(tau)

    t_global_start = now()

    for it in 1:iterations
        # ----------------------------------
        # Seed Generation & Matrices
        # ----------------------------------
        seed_ctrl_it  = Int(rand(master_rng, UInt32))
        halfcut_start_it = choose_zero_z_halfcut_start(L, Ldiv2, halfcut_mode, halfcut_rng)
        halfcut_start_by_iter[it] = halfcut_start_it

        S_mat[1, 1 + it]    = seed_ctrl_it
        bond_mat[1, 1 + it] = seed_ctrl_it
        Scut_mat[1, 1 + it] = seed_ctrl_it
        Scut_mat[2:end, 1 + it] .= halfcut_start_it

        if compute_order_observables
            for k_shift in 1:self_av_space_it
                col_idx = 1 + (it - 1) * self_av_space_it + k_shift
                SO_NN_mat[1, col_idx] = seed_ctrl_it
                SO_NNm1_mat[1, col_idx] = seed_ctrl_it # NEW: Store seed in header row
            end
        end

        rng_reg = RNGRegistry(ctrl=seed_ctrl_it, proj=1, haar=1, born=1)

        sim = SimulationState(L=L, bc=bc, site_type="S=1",
                              cutoff=cutoffin, maxdim=Int(1e10), rng=rng_reg)
        apply_zero_z_halfcut_basis_rotation!(sim, halfcut_start_it)

        # !!! IMPORTANT FIX !!!
        # Change "+Z1" to your X-basis string (e.g. "XUp" or "+X1") to match paper and avoid state annihilation!
        sim.mps = MPS(ComplexF64, sim.sites, ["+Z1" for _ in 1:L]) 

        # The folded trajectory-specific RAM order keeps the selected physical
        # block [halfcut_start_it, halfcut_start_it + L/2 - 1] at cut=L/2.
        obs_entropy = EntanglementEntropy(cut=Ldiv2, renyi_index=1, base=2)
        ctrl_rng = get_rng(sim.rng_registry, :ctrl)

        # =========================
        # PRECOMPILE CIRCUIT (Z TEMPLATE: ONE LOCAL GATE/PROJECTOR PER TIMESTEP)
        # =========================
        # In this z-dynamics template, t is a microscopic local update: exactly
        # one projector/gate at site i = mod1(t, L). A full lattice sweep is L
        # such timesteps. Observables below are recorded after that single update.
        compiled_by_t = Vector{Vector{NamedTuple{(:sites,:gate,:gate_name),Tuple{Vector{Int},Any,String}}}}(undef, timesteps)

        for t in 1:timesteps
            compiled_by_t[t] = NamedTuple{(:sites,:gate,:gate_name),Tuple{Vector{Int},Any,String}}[]

            i = mod1(t, L)
            u = rand(ctrl_rng)

            if u < pnn
                if !(periodic == false && i == L)
                    i_next = (i % L) + 1
                    push!(compiled_by_t[t], (sites=[i, i_next], gate=proj_gate, gate_name="NN-AKLT"))
                else
                    # Do nothing branch on boundary (Open BC wrap)
                    push!(compiled_by_t[t], (sites=[i], gate=proj_gate, gate_name="IDENTITY"))
                end
            else
                # Pass only ONE site for the native 1-site gate
                push!(compiled_by_t[t], (sites=[i], gate=proj_gate_zero, gate_name="ZERO"))
            end
        end

        # Flatten for returning/debugging AND PRINTING
        compiled_flat = NamedTuple{(:t,:sites,:gate_name),Tuple{Int,Vector{Int},String}}[]

        println("\n--- Compiled Circuit Blueprint (Iteration $it; z gate-timestep convention) ---")
        for t in 1:timesteps
            for g in compiled_by_t[t]
                push!(compiled_flat, (t=t, sites=g.sites, gate_name=g.gate_name))
                # Print each microscopic timestep to the console
                println("  t=$(lpad(t, 2)) | Gate: $(rpad(g.gate_name, 8)) | Sites: $(g.sites)")
            end
        end
        println("--------------------------------------------------\n")

        compiled_per_iter[it] = compiled_flat

        # =========================
        # RUN: execute compiled circuit (Z TEMPLATE: timestep = one local gate/projector)
        # =========================
        meas_row = 0
        for t in 1:timesteps
            
            # Apply the single local gate/projector for this microscopic timestep
            for g in compiled_by_t[t]
                if g.gate_name != "IDENTITY"
                    apply!(sim, g.gate, g.sites)
                    normalize!(sim.mps) # Keep state physically normalized
                end
            end

            # Measurements happen after each microscopic gate/projector timestep
            if (t >= ti) && (t <= tf) && ((t - ti) % interval == 0)
                meas_row += 1
                row = 1 + meas_row

                S  = obs_entropy(sim)
                bdim = maximum(linkdims(sim.mps))

                S_mat[row, 1 + it]     = S
                bond_mat[row, 1 + it]  = bdim

                if compute_order_observables
                    # --- Periodic translational spatial averaging.
                    # The wrapped physical forward path makes k=1:L a true
                    # full-ring average instead of stopping at the j > i seam.
                    for k in 1:self_av_space_it
                        i_site = mod1(k, L)
                        j_site = mod1(Ldiv2 + k, L)
                        j_site_m1 = mod1(Ldiv2 - 1 + k, L) # NEW: Calculate j_site at separation L/2 - 1

                        val = zero_z_periodic_string_order_1(sim, i_site, j_site)
                        val_m1 = zero_z_periodic_string_order_1(sim, i_site, j_site_m1)

                        col_idx = 1 + (it - 1) * self_av_space_it + k
                        SO_NN_mat[row, col_idx] = val
                        SO_NNm1_mat[row, col_idx] = val_m1 # NEW: Store into matrix
                    end
                end
            end
        end

        @assert meas_row == nmeas "Internal error: meas_row=$meas_row but expected nmeas=$nmeas."
        println("Iteration $it calculation complete.")
    end

    t_global_end = now()

    # =========================
    # WRITE OUTPUTS
    # =========================
    hcut_token = iterations == 1 ? string(halfcut_start_by_iter[1]) : "mixed"
    obs_token = obs_mode == :entropy_only ? "_obs=entropy_only" : ""
    header = string("_tstep=gate",
                    obs_token,
                    "_hcutmethod=folded_basisrot",
                    "_soavg=periodic_fullpath",
                    "_hcutmode=", halfcut_mode,
                    "_hcutstart=", hcut_token,
                    "_hlen=", Ldiv2,
                    "_pn=", pnn,
                    "_cutoff=", cutoffin,
                    "_it=", iterations,
                    "_tf=", timesteps,
                    "_sa=", self_av_space_it,
                    "_PBC=", periodic)

    writedlm(string(filename, header, "_S_QCMPS_zero"),     S_mat,     ", ")
    writedlm(string(filename, header, "_HcutStart_QCMPS_zero"),  Scut_mat,  ", ")
    writedlm(string(filename, header, "_zero_bond_QCMPS_zero"),  bond_mat,  ", ")
    if compute_order_observables
        writedlm(string(filename, header, "_SO_NN_QCMPS_zero"), SO_NN_mat, ", ")
        writedlm(string(filename, header, "_SO_NNm1_QCMPS_zero"), SO_NNm1_mat, ", ") # NEW: Write output to file
    end

    println("z-gate-timestep template - full periodic spatial string-order averaging")
end

# Call runy(...) with the desired parameters after including this file.
