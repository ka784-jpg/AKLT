# Activate the repository environment with julia --project=.
using Random
using Dates
using ITensors
using ITensorMPS
using DelimitedFiles
using LinearAlgebra

BLAS.set_num_threads(1)
using QuantumCircuitsMPS2

# =========================
# GLOBAL PHYSICS DEFINITIONS
# =========================

# 1. State Initialization
ITensors.state(::StateName"+Z1", ::SiteType"S=1") = [1, 1, 1] ./ sqrt(3)

# 2. Projection Gate (Removed 'const' so you can re-run this cell endlessly)
P0 = total_spin_projector(0)
P1 = total_spin_projector(1)
P_not_2 = P0 + P1
proj_gate = SpinSectorProjection(P_not_2)

# =========================
# MAIN SIMULATION FUNCTION
# ==========================================
function runy(
    L::Int,
    cutoffin::Float64,
    bc::Symbol,               # :open, :periodic, or :periodic_nnn
    p_nn::Float64,            # probability of choosing (i,i+1) vs (i,i+2)
    n_steps::Int,             # total number of sweeps (timesteps)
    iterations::Int,          # total number of runs
    self_av_space_it::Int,
    ti::Int,                  # measurement start sweep
    tf::Int,                  # measurement end sweep
    interval::Int,            # measurement interval
    master_seed::Int,         # Master seed for the ensemble
    filename::AbstractString
)
    t_global_start = now()

    # =========================
    # CHECKS
    # =========================
    @assert iseven(L) "L must be even."
    @assert 1 ≤ ti ≤ tf ≤ n_steps "Need 1 ≤ ti ≤ tf ≤ n_steps."
    @assert interval ≥ 1 "interval must be >= 1."
    @assert (tf - ti) % interval == 0 "Need (tf - ti) divisible by interval."
    @assert iterations ≥ 1 "iterations must be >= 1."

    # ------------------------------------------------------------
    # MASTER SEED & RNG
    # ------------------------------------------------------------
    println("\n========================================")
    println("MASTER SEED = $master_seed | ITERATIONS = $iterations")
    println("========================================")

    master_rng = MersenneTwister(master_seed)

    # ------------------------------------------------------------
    # MEASUREMENT GRID & STORAGE
    # ------------------------------------------------------------
    rec_times = collect(ti:interval:tf)
    nmeas = length(rec_times)
    nrows = nmeas + 1 # +1 for the seed header row

    # Allocate matrices with columns for each iteration
    bond_mat     = fill(NaN, nrows, 1 + iterations)

    # Write Time Columns
    bond_mat[2:end, 1] .= rec_times

    # ============================================================
    # MAIN ITERATION LOOP
    # ============================================================
    for it in 1:iterations
        # 1. Generate unique sub-seed for this iteration
        seed_ctrl_it = Int(rand(master_rng, UInt32))

        # 2. Write Seed Headers to the correct columns
        bond_mat[1, 1 + it] = seed_ctrl_it

        # 3. Build SimulationState for this iteration
        rng_reg   = RNGRegistry(ctrl=seed_ctrl_it, proj=1, haar=1, born=1)
        sim_state = SimulationState(L=L, bc=bc, site_type="S=1", cutoff=cutoffin,
                                    maxdim=Int(1e10), rng=rng_reg)

        # TI fixed product init (+Z1)
        sim_state.mps = MPS(ComplexF64, sim_state.sites, ["+Z1" for _ in 1:L])
        ctrl_rng = get_rng(sim_state.rng_registry, :ctrl)

        # 4. PRECOMPILE CIRCUIT
        compiled_gates = Vector{Vector{NamedTuple{(:i,:j),Tuple{Int,Int}}}}(undef, n_steps)
        for t in 1:n_steps
            sweep_gates = NamedTuple{(:i,:j),Tuple{Int,Int}}[]
            sizehint!(sweep_gates, L)

            for i_raw in 1:L
                u = rand(ctrl_rng)
                j_raw = i_raw + (u < p_nn ? 1 : 2)

                i, j, ok = sanitize_pair(i_raw, j_raw, L, bc)
                if ok
                    push!(sweep_gates, (i=i, j=j))
                end
            end
            compiled_gates[t] = sweep_gates
        end

        # 5. EXECUTE SWEEPS & RECORD
        meas_row = 0
        for t in 1:n_steps
            # Apply sweep of gates
            for g in compiled_gates[t]
                apply!(sim_state, proj_gate, [g.i, g.j])
            end

            # Measure on grid
            if t ≥ ti && t ≤ tf && ((t - ti) % interval == 0)
                meas_row += 1
                row = meas_row + 1

                # Fig. 9 bond-only measurement.
                bond_mat[row, 1 + it] = maximum(linkdims(sim_state.mps))
            end
        end
        @assert meas_row == nmeas "Internal error: meas_row=$meas_row but expected nmeas=$nmeas."

        # Free iteration-local objects before next iteration
        GC.gc(); GC.gc()
    end

    t_global_end = now()
    println("\n------------------------------------------------")
    println("Run complete in ", t_global_end - t_global_start)
    println("------------------------------------------------\n")

    # ------------------------------------------------------------
    # WRITE OUTPUTS
    # ------------------------------------------------------------
    header = string("_pn=", p_nn,
                "_cutoff=", cutoffin,
                "_it=", iterations,
                "_ti=", ti,
                "_tf=", n_steps,
                "_PBC=", bc == :periodic)

    try
        writedlm(string(filename, header, "_bond_QCMPS_nnn"),     bond_mat,     ", ")
    catch e
        println("Warning: Could not save files. Error:\n", e)
    end
end
