#!/usr/bin/env julia

using Dates
using ITensors
using ITensorMPS
using LinearAlgebra
using QuantumCircuitsMPS2
using Random

BLAS.set_num_threads(1)

ITensors.state(::StateName"+Z1", ::SiteType"S=1") = ComplexF64[1, 1, 1] ./ sqrt(3)

const VERSION = "fidlim_random_mps_reference_v1"
const BC = :periodic

function parse_task_file(path::AbstractString, row_index::Int)
    lines = readlines(path)
    length(lines) >= 2 || error("Task file $path has no task rows")
    1 <= row_index <= length(lines) - 1 || error("row_index=$row_index outside 1:$(length(lines)-1)")
    header = split(lines[1], '\t')
    values = split(lines[row_index + 1], '\t')
    length(header) == length(values) || error("Malformed TSV row $row_index in $path")
    return Dict(header .=> values)
end

function nnn_folded_ram_phy(L::Int)
    iseven(L) || error("NNN folded ordering requires even L, got $L")
    return vcat(collect(1:2:L), collect(2:2:L))
end

function configure_layout!(sim::SimulationState, model::AbstractString)
    if model == "nnn_aklt"
        sim.ram_phy = nnn_folded_ram_phy(sim.L)
        sim.phy_ram = invperm(sim.ram_phy)
    end
    return nothing
end

function build_initial_mps(sim::SimulationState, task)
    init_kind = get(task, "init_kind", "fixed_plus_z1")
    if init_kind == "fixed_plus_z1"
        return MPS(ComplexF64, sim.sites, ["+Z1" for _ in 1:sim.L])
    elseif init_kind == "random_mps"
        linkdim = parse(Int, get(task, "init_linkdim", "4"))
        rng = MersenneTwister(parse(Int, task["seed"]))
        psi = random_mps(rng, ComplexF64, sim.sites; linkdims=linkdim)
        normalize!(psi)
        return psi
    end
    error("Unknown init_kind=$init_kind")
end

function init_sim(L::Int, cutoff::Float64, maxdim::Int, seed::Int, model::AbstractString, task)
    rng_reg = RNGRegistry(ctrl=seed, proj=1, haar=1, born=1)
    sim = SimulationState(L=L, bc=BC, site_type="S=1", cutoff=cutoff, maxdim=maxdim, rng=rng_reg)
    configure_layout!(sim, model)
    sim.mps = build_initial_mps(sim, task)
    return sim
end

function gate_for_timestep(model::AbstractString, L::Int, t::Int, proj_nn, proj_dimer)
    i_raw = mod1(t, L)

    if model == "dimer_odd"
        if isodd(i_raw)
            return (active=true, i=i_raw, j=mod1(i_raw + 1, L), gate=proj_dimer, gate_name="DIMER_P0")
        else
            return (active=false, i=i_raw, j=mod1(i_raw + 1, L), gate=proj_dimer, gate_name="IDENTITY_OFF_DIMER_SECTOR")
        end
    elseif model == "dimer_even"
        if iseven(i_raw)
            return (active=true, i=i_raw, j=mod1(i_raw + 1, L), gate=proj_dimer, gate_name="DIMER_P0")
        else
            return (active=false, i=i_raw, j=mod1(i_raw + 1, L), gate=proj_dimer, gate_name="IDENTITY_OFF_DIMER_SECTOR")
        end
    elseif model == "nn_aklt"
        return (active=true, i=i_raw, j=mod1(i_raw + 1, L), gate=proj_nn, gate_name="NN_AKLT_P0P1")
    elseif model == "nnn_aklt"
        return (active=true, i=i_raw, j=mod1(i_raw + 2, L), gate=proj_nn, gate_name="NNN_AKLT_P0P1")
    end

    error("Unknown model=$model")
end

function apply_limit_step!(sim::SimulationState, model::AbstractString, t::Int, proj_nn, proj_dimer)
    g = gate_for_timestep(model, sim.L, t, proj_nn, proj_dimer)
    if !g.active
        return g
    end

    i, j, ok = sanitize_pair(g.i, g.j, sim.L, BC)
    if ok
        apply!(sim, g.gate, [i, j])
        normalize!(sim.mps)
    end
    return g
end

function fidelity_to_reference(psi::MPS, ref::MPS)
    npsi = max(real(inner(psi, psi)), eps(Float64))
    nref = max(real(inner(ref, ref)), eps(Float64))
    return abs2(inner(psi, ref)) / (npsi * nref)
end

function push_measurement!(rows, campaign, row_index, task, sim, ref_mps, t, gate_name, active, elapsed_s)
    fid = fidelity_to_reference(sim.mps, ref_mps)
    L = parse(Int, task["L"])
    model = task["model"]
    push!(rows, (
        campaign=campaign,
        version=VERSION,
        host=get(ENV, "HOSTNAME", "unknown"),
        slurm_job_id=get(ENV, "SLURM_JOB_ID", "local"),
        slurm_array_task_id=get(ENV, "SLURM_ARRAY_TASK_ID", "0"),
        slot_id=get(ENV, "SLOT_ID", "0"),
        row_index=row_index,
        task_id=parse(Int, task["task_id"]),
        model=model,
        L=L,
        t=t,
        t_over_L=t / L,
        t_over_L2=t / L^2,
        p_dimer=startswith(model, "dimer") ? 1.0 : 0.0,
        p_nn_aklt=model == "nn_aklt" ? 1.0 : 0.0,
        p_nnn_aklt=model == "nnn_aklt" ? 1.0 : 0.0,
        fidelity=fid,
        bond_dim=maximum(linkdims(sim.mps)),
        gate_name=gate_name,
        active_gate=active,
        cutoff=parse(Float64, task["cutoff"]),
        maxdim=parse(Int, task["maxdim"]),
        timesteps=parse(Int, task["timesteps"]),
        interval=parse(Int, task["interval"]),
        reference_steps=parse(Int, task["reference_steps"]),
        init_kind=get(task, "init_kind", "fixed_plus_z1"),
        init_linkdim=parse(Int, get(task, "init_linkdim", "1")),
        init_sample=parse(Int, get(task, "sample_id", "1")),
        target_definition="deep_reference_from_same_initial_state",
        timestep_definition="single_projector_or_gate_slot",
        layout=model == "nnn_aklt" ? "folded_odd_even_ram_phy" : "physical_order",
        elapsed_s=elapsed_s,
    ))
    return nothing
end

function write_csv(path::AbstractString, rows)
    mkpath(dirname(path))
    tmp = path * ".tmp"
    open(tmp, "w") do io
        header = collect(keys(rows[1]))
        println(io, join(header, ","))
        for r in rows
            println(io, join((getfield(r, h) for h in header), ","))
        end
    end
    mv(tmp, path; force=true)
    return path
end

function main()
    length(ARGS) == 4 || error("Usage: julia run_fidelity_random_mps.jl TASKS_TSV ROW_INDEX OUT_CSV CAMPAIGN")
    task_path, row_index_s, out_csv, campaign = ARGS
    row_index = parse(Int, row_index_s)
    task = parse_task_file(task_path, row_index)

    model = task["model"]
    L = parse(Int, task["L"])
    timesteps = parse(Int, task["timesteps"])
    interval = parse(Int, task["interval"])
    reference_steps = parse(Int, task["reference_steps"])
    cutoff = parse(Float64, task["cutoff"])
    maxdim = parse(Int, task["maxdim"])
    seed = parse(Int, task["seed"])

    interval >= 1 || error("interval must be >= 1")
    reference_steps >= timesteps || error("reference_steps must be >= timesteps")

    P0 = total_spin_projector(0)
    P1 = total_spin_projector(1)
    proj_nn = SpinSectorProjection(P0 + P1)
    proj_dimer = SpinSectorProjection(P0)

    sim = init_sim(L, cutoff, maxdim, seed, model, task)
    psi0 = copy(sim.mps)

    for t in 1:reference_steps
        apply_limit_step!(sim, model, t, proj_nn, proj_dimer)
    end
    ref_mps = copy(sim.mps)

    sim.mps = copy(psi0)
    rows = NamedTuple[]
    t0 = time()
    push_measurement!(rows, campaign, row_index, task, sim, ref_mps, 0, "INITIAL", false, 0.0)

    for t in 1:timesteps
        g = apply_limit_step!(sim, model, t, proj_nn, proj_dimer)
        if t % interval == 0
            push_measurement!(rows, campaign, row_index, task, sim, ref_mps, t, g.gate_name, g.active, time() - t0)
        end
    end

    write_csv(out_csv, rows)
    println("Wrote ", out_csv, " rows=", length(rows), " model=", model, " L=", L,
            " init=", get(task, "init_kind", "fixed_plus_z1"),
            " sample=", get(task, "sample_id", "1"))
end

main()
