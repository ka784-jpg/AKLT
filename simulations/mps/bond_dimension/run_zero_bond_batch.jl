using Random
using DelimitedFiles
using QuantumCircuitsMPS2
using ITensors
using ITensorMPS
using LinearAlgebra

BLAS.set_num_threads(1)

struct FZeroBondGate <: QuantumCircuitsMPS2.AbstractGate
    tau::Float64
end

QuantumCircuitsMPS2.support(::FZeroBondGate) = 1

function QuantumCircuitsMPS2.build_operator(
    gate::FZeroBondGate,
    site::Index,
    local_dim::Int;
    kwargs...,
)
    projzero = [0.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 0.0]
    return itensor(exp(projzero * gate.tau), site', site)
end

ITensors.state(::StateName"+Z1", ::SiteType"S=1") = [1.0, 1.0, 1.0]

function run_zero_bond_batch(
    L::Int,
    pnn::Float64,
    iterations::Int,
    master_seed::Int,
    prefix::String,
)
    cutoff = 1.0e-10
    ti = 3 * L
    tf = ti + 4
    interval = 1

    @assert iseven(L)
    @assert 0.0 <= pnn <= 1.0
    @assert iterations >= 1

    times = collect(ti:interval:tf)
    bond_mat = fill(NaN, length(times) + 1, iterations + 1)
    bond_mat[2:end, 1] .= times

    p_not_2 = total_spin_projector(0) + total_spin_projector(1)
    nn_gate = SpinSectorProjection(p_not_2)
    zero_gate = FZeroBondGate(50.0)
    master_rng = MersenneTwister(master_seed)

    for iteration in 1:iterations
        trajectory_seed = Int(rand(master_rng, UInt32))
        bond_mat[1, 1 + iteration] = trajectory_seed
        rng_registry = RNGRegistry(ctrl=trajectory_seed, proj=1, haar=1, born=1)
        sim = SimulationState(
            L=L,
            bc=:periodic,
            site_type="S=1",
            cutoff=cutoff,
            maxdim=Int(1e10),
            rng=rng_registry,
        )
        sim.mps = MPS(ComplexF64, sim.sites, ["+Z1" for _ in 1:L])
        ctrl_rng = get_rng(sim.rng_registry, :ctrl)

        measurement_row = 0
        for t in 1:tf
            for site in 1:L
                if rand(ctrl_rng) < pnn
                    next_site = (site % L) + 1
                    apply!(sim, nn_gate, [site, next_site])
                else
                    apply!(sim, zero_gate, [site])
                    normalize!(sim.mps)
                end
            end

            if ti <= t <= tf && (t - ti) % interval == 0
                measurement_row += 1
                bond_mat[1 + measurement_row, 1 + iteration] = maximum(linkdims(sim.mps))
            end
        end
        @assert measurement_row == length(times)
    end

    ptt = 1.0 - pnn
    suffix = string(
        "_pn=", pnn,
        "_ptt=", ptt,
        "_cutoff=", cutoff,
        "_it=", iterations,
        "_ti=", ti,
        "_tf=", tf,
        "_sa=1_PBC=true_zero_bond_QCMPS_zero",
    )
    output_path = string(prefix, suffix)
    writedlm(output_path, bond_mat, ", ")
    println("output=", output_path)
end

length(ARGS) == 5 || error(
    "usage: run_zero_bond_batch.jl L P_NN ITERATIONS MASTER_SEED OUTPUT_PREFIX",
)

run_zero_bond_batch(
    parse(Int, ARGS[1]),
    parse(Float64, ARGS[2]),
    parse(Int, ARGS[3]),
    parse(Int, ARGS[4]),
    ARGS[5],
)
