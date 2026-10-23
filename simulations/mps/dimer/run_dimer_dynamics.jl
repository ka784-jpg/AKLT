using Random
using Dates
using DelimitedFiles
using QuantumCircuitsMPS2
using ITensors
using ITensorMPS
using LinearAlgebra

BLAS.set_num_threads(1)

ITensors.state(::StateName"+Z1", ::SiteType"S=1") = [1.0, 1.0, 1.0] ./ sqrt(3)

const DIMER_Z_TEMPLATE_VERSION = "v2_dimer_z_gate_timestep_circuitshift_halfcut_fullspavg"

function normalize_dimer_z_observable_mode(mode::AbstractString)
    m = lowercase(strip(String(mode)))
    m in ("all", "full", "standard") && return :all
    m in ("entropy_only", "entanglement_only", "s_only", "s") && return :entropy_only
    throw(ArgumentError("observable_mode must be all or entropy_only, got $mode"))
end

function normalize_dimer_z_halfcut_start_mode(mode::AbstractString)
    m = lowercase(strip(String(mode)))
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

derive_dimer_z_halfcut_start_seed(master_seed::Int) = master_seed + 1_000_003

function choose_dimer_z_halfcut_start(L::Int, half_L::Int, cut_mode, cut_rng)
    cut_mode === :half && return 1
    cut_mode === :random && return rand(cut_rng, 1:half_L)
    start = cut_mode::Int
    1 <= start <= half_L || throw(ArgumentError("fixed halfcut start must satisfy 1 <= start <= L/2, got start=$start for L=$L"))
    return start
end

function default_folded_halfcut_start(L::Int)
    half_L = div(L, 2)
    return mod1(1 - div(half_L, 2), L)
end

function circuit_shift_for_halfcut_start(L::Int, halfcut_start::Int)
    return mod(default_folded_halfcut_start(L) - halfcut_start, L)
end

function dimer_z_periodic_string_order_1(state::SimulationState, i_phys::Int, j_phys::Int)
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
end

function runy(el::Int,
              pnn::Float64,
              cutoffin::Float64,
              timesteps::Int,
              iterations::Int,
              periodic::Bool,
              self_av_space_it::Int,
              ti::Int,
              tf::Int,
              interval::Int,
              master_seed::Int,
              filename::String,
              dimer_sector::AbstractString="odd",
              observable_mode::AbstractString="entropy_only",
              halfcut_start_mode::AbstractString="random",
              halfcut_start_seed::Int=0)

    master_rng = MersenneTwister(master_seed)
    obs_mode = normalize_dimer_z_observable_mode(observable_mode)
    cut_mode = normalize_dimer_z_halfcut_start_mode(halfcut_start_mode)
    compute_order_observables = obs_mode == :all

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

    dsec = lowercase(strip(String(dimer_sector)))
    @assert dsec in ("odd", "even") "dimer_sector must be \"odd\" or \"even\"."

    bc::Symbol = periodic ? :periodic : :open
    is_periodic = periodic
    pairs_per_sweep = periodic ? L : (L - 1)
    Ldiv2 = div(L, 2)

    times = collect(ti:interval:tf)
    nmeas = length(times)
    nrows = nmeas + 1

    S_mat        = fill(NaN, nrows, 1 + iterations)
    Scut_mat     = fill(NaN, nrows, 1 + iterations)
    bond_mat     = fill(NaN, nrows, 1 + iterations)
    D_mat        = fill(NaN, nrows, 1 + iterations)
    SO_NN_mat    = fill(NaN, nrows, 1 + (iterations * self_av_space_it))
    SO_NNm1_mat  = fill(NaN, nrows, 1 + (iterations * self_av_space_it))

    S_mat[2:end, 1]       .= times
    Scut_mat[2:end, 1]    .= times
    bond_mat[2:end, 1]    .= times
    D_mat[2:end, 1]       .= times
    SO_NN_mat[2:end, 1]   .= times
    SO_NNm1_mat[2:end, 1] .= times

    P0 = total_spin_projector(0)
    P1 = total_spin_projector(1)
    P_not_2 = P0 + P1

    proj_gate = SpinSectorProjection(P_not_2)
    proj_gate_dimer = SpinSectorProjection(P0)
    proj_gate_identity = SpinSectorProjection(Matrix{Float64}(I, 9, 9))

    cut_seed = halfcut_start_seed == 0 ? derive_dimer_z_halfcut_start_seed(master_seed) : halfcut_start_seed
    cut_rng = MersenneTwister(cut_seed)
    halfcut_start_by_iter = Vector{Int}(undef, iterations)
    circuit_shift_by_iter = Vector{Int}(undef, iterations)
    compiled_per_iter = Vector{Vector{NamedTuple{(:t,:i_raw,:i,:j,:gate_name),Tuple{Int,Int,Int,Int,String}}}}(undef, iterations)

    is_dimer_sector_bond(i_raw::Int) = dsec == "odd" ? isodd(i_raw) : iseven(i_raw)

    for it in 1:iterations
        seed_ctrl_it = Int(rand(master_rng, UInt32))
        halfcut_start_it = choose_dimer_z_halfcut_start(L, Ldiv2, cut_mode, cut_rng)
        circuit_shift_it = circuit_shift_for_halfcut_start(L, halfcut_start_it)
        halfcut_start_by_iter[it] = halfcut_start_it
        circuit_shift_by_iter[it] = circuit_shift_it

        S_mat[1, 1 + it] = seed_ctrl_it
        Scut_mat[1, 1 + it] = seed_ctrl_it
        Scut_mat[2:end, 1 + it] .= halfcut_start_it
        bond_mat[1, 1 + it] = seed_ctrl_it
        D_mat[1, 1 + it] = seed_ctrl_it

        if compute_order_observables
            for k_shift in 1:self_av_space_it
                col_idx = 1 + (it - 1) * self_av_space_it + k_shift
                SO_NN_mat[1, col_idx] = seed_ctrl_it
                SO_NNm1_mat[1, col_idx] = seed_ctrl_it
            end
        end

        rng_reg = RNGRegistry(ctrl=seed_ctrl_it, proj=1, haar=1, born=1)
        sim = SimulationState(L=L, bc=bc, site_type="S=1",
                              cutoff=cutoffin, maxdim=Int(1e10), rng=rng_reg)

        sim.mps = MPS(ComplexF64, sim.sites, ["+Z1" for _ in 1:L])

        obs_entropy = EntanglementEntropy(cut=Ldiv2, renyi_index=1, base=2)
        obs_dimer = compute_order_observables ? DimerOrder(periodic=is_periodic) : nothing
        ctrl_rng = get_rng(sim.rng_registry, :ctrl)

        compiled_by_t = Vector{Vector{NamedTuple{(:i_raw,:i,:j,:gate,:gate_name),Tuple{Int,Int,Int,Any,String}}}}(undef, timesteps)
        compiled_flat = NamedTuple{(:t,:i_raw,:i,:j,:gate_name),Tuple{Int,Int,Int,Int,String}}[]
        for t in 1:timesteps
            compiled_by_t[t] = NamedTuple{(:i_raw,:i,:j,:gate,:gate_name),Tuple{Int,Int,Int,Any,String}}[]

            i_raw = mod1(t, pairs_per_sweep)
            j_raw = i_raw == L ? 1 : i_raw + 1
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
                push!(compiled_by_t[t], (i_raw=i_raw, i=i, j=j, gate=gate, gate_name=gate_name))
                push!(compiled_flat, (t=t, i_raw=i_raw, i=i, j=j, gate_name=gate_name))
            end
        end
        compiled_per_iter[it] = compiled_flat

        meas_row = 0
        for t in 1:timesteps
            for g in compiled_by_t[t]
                g.gate_name == "IDENTITY" && continue
                apply!(sim, g.gate, [g.i, g.j])
            end

            if (t >= ti) && (t <= tf) && ((t - ti) % interval == 0)
                meas_row += 1
                row = 1 + meas_row

                S_mat[row, 1 + it] = obs_entropy(sim)
                bond_mat[row, 1 + it] = maximum(linkdims(sim.mps))

                if compute_order_observables
                    D_mat[row, 1 + it] = obs_dimer(sim)
                    for k in 1:self_av_space_it
                        i_site = mod1(k, L)
                        j_site = mod1(Ldiv2 + k, L)
                        j_site_m1 = mod1(Ldiv2 - 1 + k, L)
                        col_idx = 1 + (it - 1) * self_av_space_it + k
                        SO_NN_mat[row, col_idx] = dimer_z_periodic_string_order_1(sim, i_site, j_site)
                        SO_NNm1_mat[row, col_idx] = dimer_z_periodic_string_order_1(sim, i_site, j_site_m1)
                    end
                end
            end
        end

        @assert meas_row == nmeas "Internal error: meas_row=$meas_row but expected nmeas=$nmeas."
        println("Iteration $it calculation complete.")
    end

    hcut_token = iterations == 1 ? string(halfcut_start_by_iter[1]) : "mixed"
    cshift_token = iterations == 1 ? string(circuit_shift_by_iter[1]) : "mixed"
    obs_token = obs_mode == :entropy_only ? "_obs=entropy_only" : "_obs=all"
    header = string("_tstep=gate",
                    "_model=dimer",
                    obs_token,
                    "_hcutmethod=circuitshift",
                    "_soavg=periodic_fullpath",
                    "_hcutmode=", cut_mode,
                    "_hcutstart=", hcut_token,
                    "_cshift=", cshift_token,
                    "_hlen=", Ldiv2,
                    "_pn=", pnn,
                    "_cutoff=", cutoffin,
                    "_it=", iterations,
                    "_ti=", ti,
                    "_tf=", timesteps,
                    "_sa=", self_av_space_it,
                    "_dsec_", dsec,
                    "_PBC=", periodic)

    writedlm(string(filename, header, "_S_QCMPS_dimer"), S_mat, ", ")
    writedlm(string(filename, header, "_HcutStart_QCMPS_dimer"), Scut_mat, ", ")
    writedlm(string(filename, header, "_bond_QCMPS_dimer"), bond_mat, ", ")
    if compute_order_observables
        writedlm(string(filename, header, "_SO_NN_QCMPS_dimer"), SO_NN_mat, ", ")
        writedlm(string(filename, header, "_SO_NNm1_QCMPS_dimer"), SO_NNm1_mat, ", ")
        writedlm(string(filename, header, "_Dimer_QCMPS_dimer"), D_mat, ", ")
    end

    println(DIMER_Z_TEMPLATE_VERSION)
    println("dimer z-gate-timestep template - timestep equals one local dimer/AKLT bond update")
    return (S_mat=S_mat,
            Scut_mat=Scut_mat,
            bond_mat=bond_mat,
            SO_NN_mat=SO_NN_mat,
            SO_NNm1_mat=SO_NNm1_mat,
            Dimer_mat=D_mat,
            compiled_per_iter=compiled_per_iter,
            dimer_sector=dsec,
            observable_mode=obs_mode,
            halfcut_start_mode=cut_mode,
            halfcut_start_by_iter=halfcut_start_by_iter,
            circuit_shift_by_iter=circuit_shift_by_iter)
end

# Call runy(...) with the desired parameters after including this file.
