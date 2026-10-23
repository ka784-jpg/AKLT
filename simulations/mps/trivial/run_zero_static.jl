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

# ==========================================
# MAIN FUNCTION (Zero Code - Full Sweep Timesteps)
# ==========================================
function runy(el::Int, pnn::Float64, cutoffin::Float64, timesteps::Int, iterations::Int, periodic::Bool, self_av_space_it::Int, ti::Int, tf::Int,
interval::Int, master_seed::Int, filename::String)

    # ----------------------------------
    # MASTER RNG
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
    SO_NN_mat   = fill(NaN, nrows, 1 + (iterations * self_av_space_it))
    SO_NNm1_mat = fill(NaN, nrows, 1 + (iterations * self_av_space_it)) # NEW: Initialize L/2-1 matrix

    S_mat[2:end, 1]       .= times
    bond_mat[2:end, 1]    .= times
    SO_NN_mat[2:end, 1]   .= times
    SO_NNm1_mat[2:end, 1] .= times # NEW: Set time column

    Ldiv2 = div(L, 2)
    # UPDATED: Vector of Vectors to store a sweep of gates per timestep
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

        S_mat[1, 1 + it]    = seed_ctrl_it
        bond_mat[1, 1 + it] = seed_ctrl_it

        for k_shift in 1:self_av_space_it
            col_idx = 1 + (it - 1) * self_av_space_it + k_shift
            SO_NN_mat[1, col_idx] = seed_ctrl_it
            SO_NNm1_mat[1, col_idx] = seed_ctrl_it # NEW: Store seed in header row
        end

        rng_reg = RNGRegistry(ctrl=seed_ctrl_it, proj=1, haar=1, born=1)

        sim = SimulationState(L=L, bc=bc, site_type="S=1",
                              cutoff=cutoffin, maxdim=Int(1e10), rng=rng_reg)

        # !!! IMPORTANT FIX !!!
        # Change "+Z1" to your X-basis string (e.g. "XUp" or "+X1") to match paper and avoid state annihilation!
        sim.mps = MPS(ComplexF64, sim.sites, ["+Z1" for _ in 1:L]) 

        obs_entropy = EntanglementEntropy(cut=Ldiv2, renyi_index=1, base=2)
        ctrl_rng = get_rng(sim.rng_registry, :ctrl)

        # =========================
        # PRECOMPILE CIRCUIT (UPDATED FOR FULL SWEEPS)
        # =========================
        compiled_by_t = Vector{Vector{NamedTuple{(:sites,:gate,:gate_name),Tuple{Vector{Int},Any,String}}}}(undef, timesteps)

        for t in 1:timesteps
            compiled_by_t[t] = NamedTuple{(:sites,:gate,:gate_name),Tuple{Vector{Int},Any,String}}[]
            
            # Inner loop: sweep across the entire lattice (1 to L)
            for i in 1:L
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
        end

        # Flatten for returning/debugging AND PRINTING
        compiled_flat = NamedTuple{(:t,:sites,:gate_name),Tuple{Int,Vector{Int},String}}[]

        println("Compiled Circuit Blueprint suppressed for production (Iteration $it)")
        for t in 1:timesteps
            for g in compiled_by_t[t]
                push!(compiled_flat, (t=t, sites=g.sites, gate_name=g.gate_name))
                # Print each step to the console
                # quiet production: per-gate circuit print suppressed
            end
        end
        println("Circuit blueprint suppression complete.\n")

        compiled_per_iter[it] = compiled_flat

        # =========================
        # RUN: execute compiled circuit (UPDATED FOR FULL SWEEPS)
        # =========================
        meas_row = 0
        for t in 1:timesteps
            
            # Apply all gates for this full sweep
            for g in compiled_by_t[t]
                if g.gate_name != "IDENTITY"
                    apply!(sim, g.gate, g.sites)
                    normalize!(sim.mps) # Keep state physically normalized
                end
            end

            # Measurements happen ONCE per full sweep (timestep)
            if (t ≥ ti) && (t ≤ tf) && ((t - ti) % interval == 0)
                meas_row += 1
                row = 1 + meas_row

                S  = obs_entropy(sim)
                bdim = maximum(linkdims(sim.mps))

                S_mat[row, 1 + it]     = S
                bond_mat[row, 1 + it]  = bdim

                # --- Exact True Translational Spatial Averaging ---
                for k in 1:self_av_space_it
                    i_site = mod1(k, L)
                    j_site = mod1(Ldiv2 + k, L)
                    j_site_m1 = mod1(Ldiv2 - 1 + k, L) # NEW: Calculate j_site at separation L/2 - 1

                    val = real(StringOrder(i_site, j_site; order=1)(sim))
                    val_m1 = real(StringOrder(i_site, j_site_m1; order=1)(sim)) # NEW: Calculate new String Order

                    col_idx = 1 + (it - 1) * self_av_space_it + k
                    SO_NN_mat[row, col_idx] = val
                    SO_NNm1_mat[row, col_idx] = val_m1 # NEW: Store into matrix
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
    header = string("_pn=", pnn,
                    "_cutoff=", cutoffin,
                    "_it=", iterations,
                    "_tf=", timesteps,
                    "_sa=", self_av_space_it,
                    "_PBC=", periodic)

    writedlm(string(filename, header, "_S_QCMPS_zero"),     S_mat,     ", ")
    writedlm(string(filename, header, "_zero_bond_QCMPS_zero"),  bond_mat,  ", ")
    writedlm(string(filename, header, "_SO_NN_QCMPS_zero"), SO_NN_mat, ", ")
    writedlm(string(filename, header, "_SO_NNm1_QCMPS_zero"), SO_NNm1_mat, ", ") # NEW: Write output to file

    println("v3 - with circuit printout")
end

# Call runy(...) with the desired parameters after including this file.
