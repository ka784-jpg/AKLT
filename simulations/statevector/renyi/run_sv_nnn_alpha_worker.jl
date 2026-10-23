#!/usr/bin/env julia

using Dates
using LinearAlgebra
using Printf
using Random

BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "AKLTStatevector.jl"))
const AKLT = AKLTStatevector

struct RenyiSpec
    label::String
    value::Float64
end

function arg_value(args::Vector{String}, name::String; default=nothing)
    idx = findfirst(==(name), args)
    idx === nothing && return default
    idx == length(args) && error("Missing value for $name")
    return args[idx + 1]
end

function read_task_row(path::String, task_index::Int)
    lines = readlines(path)
    task_index >= 1 || error("task_index must be 1-based")
    task_index + 1 <= length(lines) || error("task_index $task_index exceeds task file length")
    header = split(chomp(lines[1]), ',')
    vals = split(chomp(lines[task_index + 1]), ',')
    length(header) == length(vals) || error("Malformed task row $task_index")
    return Dict(header .=> vals)
end

function parse_renyi_list(s::String)
    specs = RenyiSpec[]
    for raw in split(s, '|')
        token = strip(raw)
        isempty(token) && continue
        lowered = lowercase(token)
        if lowered in ("inf", "+inf", "infinity")
            push!(specs, RenyiSpec("inf", Inf))
        else
            value = parse(Float64, token)
            isfinite(value) && value >= 0 || error("Renyi index must be nonnegative or inf: $token")
            push!(specs, RenyiSpec(token, value))
        end
    end
    isempty(specs) && error("renyi_list is empty")
    return specs
end

function parse_positive_float_list(s::String, name::String)
    vals = Float64[]
    for raw in split(s, '|')
        token = strip(raw)
        isempty(token) && continue
        value = parse(Float64, token)
        isfinite(value) && value > 0 || error("$name entries must be finite and positive: $token")
        push!(vals, value)
    end
    isempty(vals) && error("$name is empty")
    return vals
end

function stable_seed(seed_base::UInt64, task_index::Int, global_traj::Int, worker_index::Int)
    x = xor(seed_base, UInt64(task_index) * 0x9e3779b97f4a7c15)
    x = xor(x, UInt64(global_traj) * 0xbf58476d1ce4e5b9)
    x = xor(x, UInt64(worker_index) * 0x94d049bb133111eb)
    x = xor(x, x >> 30)
    x *= 0xbf58476d1ce4e5b9
    x = xor(x, x >> 27)
    x *= 0x94d049bb133111eb
    x = xor(x, x >> 31)
    return Int(mod(x, UInt64(typemax(Int))))
end

function probabilities_from_singular_values!(svals::Vector{Float64})
    total = 0.0
    @inbounds for i in eachindex(svals)
        p = abs2(svals[i])
        svals[i] = p
        total += p
    end
    total > 0 || error("Schmidt spectrum has zero total probability")
    invtotal = inv(total)
    @inbounds for i in eachindex(svals)
        svals[i] *= invtotal
    end
    return svals
end

@inline function entropy_from_probabilities_nats(probs::Vector{Float64}, renyi::Float64;
                                                  rank_rel_tol::Float64=0.0)
    if renyi == 0.0
        pmax = maximum(probs)
        threshold = rank_rel_tol * pmax
        rank = count(p -> p > threshold, probs)
        rank > 0 || error("Numerical Schmidt rank is zero at relative tolerance $rank_rel_tol")
        return log(rank)
    elseif renyi == 1.0
        entropy = 0.0
        @inbounds for p in probs
            p > 0 && (entropy -= p * log(p))
        end
        return entropy
    elseif isinf(renyi)
        return -log(maximum(probs))
    else
        acc = 0.0
        @inbounds for p in probs
            p > 0 && (acc += p^renyi)
        end
        return log(acc) / (1.0 - renyi)
    end
end

function halfcut_region(L::Int, origin::Int)
    return [mod1(origin + k, L) for k in 0:(div(L, 2) - 1)]
end

function entropy_spec_for_origin(L::Int, origin::Int, ram_phy::Vector{Int})
    if origin == 1
        return AKLT.entropy_region_spec_from_permutation(L, div(L, 2), ram_phy)
    end
    return AKLT.entropy_region_spec(L, halfcut_region(L, origin))
end

function probabilities_from_spec!(psi::Vector{ComplexF64},
                                  psi_work::Vector{ComplexF64},
                                  L::Int,
                                  spec)
    dims = ntuple(_ -> 3, L)
    if AKLT.is_identity_perm(spec.ram_phy)
        copyto!(psi_work, psi)
    else
        permutedims!(reshape(psi_work, dims), reshape(psi, dims), spec.ram_phy)
    end
    M = reshape(psi_work, spec.rowdim, spec.coldim)
    return probabilities_from_singular_values!(svdvals!(M))
end

function measurement_origins(L::Int, cut_mode::String, self_av_cuts::Int, rng::AbstractRNG)
    if cut_mode == "standard"
        return [1]
    elseif cut_mode == "random_per_trajectory"
        return [rand(rng, 1:L)]
    elseif cut_mode == "first_k"
        1 <= self_av_cuts <= L || error("self_av_cuts must be in 1:L for first_k mode")
        return collect(1:self_av_cuts)
    elseif cut_mode == "all"
        return collect(1:L)
    else
        error("Unknown cut_mode=$cut_mode")
    end
end

function write_entropy_row(io,
                           task_index::Int,
                           worker_index::Int,
                           campaign::String,
                           L::Int,
                           p_nn::Float64,
                           p_nnn::Float64,
                           global_traj::Int,
                           seed::Int,
                           bc::Symbol,
                           ti::Int,
                           tf::Int,
                           interval::Int,
                           t::Int,
                           cut_mode::String,
                           cut_origin::Int,
                           renyi_label::String,
                           renyi_value::Float64,
                           rank_tol_label::String,
                           rank_rel_tol::Float64,
                           s_nats::Float64,
                           elapsed_sec::Float64)
    s_bits = s_nats / log(2.0)
    renyi_value_text = isinf(renyi_value) ? "inf" : @sprintf("%.15g", renyi_value)
    @printf(io,
            "%s,%d,%d,%d,%.8f,%.8f,%d,%d,%s,%d,%d,%d,%d,%s,%d,%s,%s,%s,%.15g,%.15g,%.15g,%.6f\n",
            campaign, task_index, worker_index, L, p_nn, p_nnn,
            global_traj, seed, String(bc), ti, tf, interval, t,
            cut_mode, cut_origin, renyi_label, renyi_value_text,
            rank_tol_label, rank_rel_tol, s_bits, s_nats, elapsed_sec)
end

function run_trajectory!(io,
                         task::AbstractDict,
                         task_index::Int,
                         worker_index::Int,
                         global_traj::Int,
                         seed::Int,
                         renyi_specs::Vector{RenyiSpec},
                         rank_rel_tols::Vector{Float64})
    L = parse(Int, task["L"])
    p_nn = parse(Float64, task["p_nn"])
    p_nnn = parse(Float64, task["p_nnn"])
    ti = parse(Int, task["ti"])
    tf = parse(Int, task["tf"])
    interval = parse(Int, task["interval"])
    bc = Symbol(String(task["bc"]))
    cut_mode = String(task["cut_mode"])
    self_av_cuts = parse(Int, task["self_av_cuts"])
    campaign = String(task["campaign"])

    ctrl_rng = MersenneTwister(seed)
    cut_rng = MersenneTwister(seed + 7919)
    rec_times = Set(ti:interval:tf)

    _phy_ram, ram_phy = AKLT.compute_basis_mapping(L, bc)
    cut_origins = measurement_origins(L, cut_mode, self_av_cuts, cut_rng)
    cut_specs = [entropy_spec_for_origin(L, origin, ram_phy) for origin in cut_origins]

    buf9_in = zeros(ComplexF64, 9)
    buf9_out = zeros(ComplexF64, 9)
    state_len = 3^L
    psi = fill(ComplexF64(3.0^(-L / 2)), state_len)
    entropy_work = similar(psi)

    started = time()
    for t in 1:tf
        for i_raw in 1:L
            u = rand(ctrl_rng)
            j_raw = i_raw + (u < p_nn ? 1 : 2)
            i2, j2, ok = AKLT.sanitize_pair(i_raw, j_raw, L, bc)
            ok && AKLT.project_normalize!(psi, i2, j2, L, buf9_in, buf9_out)
        end

        if t in rec_times
            for (cut_origin, spec) in zip(cut_origins, cut_specs)
                probs = probabilities_from_spec!(psi, entropy_work, L, spec)
                elapsed = time() - started
                for renyi_spec in renyi_specs
                    if renyi_spec.value == 0.0
                        for rank_rel_tol in rank_rel_tols
                            tol_label = @sprintf("relp_%.0e", rank_rel_tol)
                            s_nats = entropy_from_probabilities_nats(
                                probs, 0.0; rank_rel_tol=rank_rel_tol
                            )
                            write_entropy_row(
                                io, task_index, worker_index, campaign, L, p_nn, p_nnn,
                                global_traj, seed, bc, ti, tf, interval, t, cut_mode,
                                cut_origin, "0", 0.0, tol_label, rank_rel_tol,
                                s_nats, elapsed,
                            )
                        end
                    else
                        s_nats = entropy_from_probabilities_nats(probs, renyi_spec.value)
                        write_entropy_row(
                            io, task_index, worker_index, campaign, L, p_nn, p_nnn,
                            global_traj, seed, bc, ti, tf, interval, t, cut_mode,
                            cut_origin, renyi_spec.label, renyi_spec.value, "none", 0.0,
                            s_nats, elapsed,
                        )
                    end
                end
            end
        end
    end
end

function main()
    args = collect(ARGS)
    task_file = String(arg_value(args, "--task-file"))
    task_index = parse(Int, arg_value(args, "--task-index"))
    worker_index = parse(Int, arg_value(args, "--worker-index"))
    workers_per_task = parse(Int, arg_value(args, "--workers-per-task"))
    tmp_out = String(arg_value(args, "--tmp-out"))

    task = read_task_row(task_file, task_index)
    renyi_specs = parse_renyi_list(String(task["renyi_list"]))
    rank_rel_tols = parse_positive_float_list(String(task["rank_rel_tols"]), "rank_rel_tols")
    seed_base = parse(UInt64, task["seed_base"])
    traj_start = parse(Int, task["traj_start"])
    traj_stop = parse(Int, task["traj_stop"])

    mkpath(dirname(tmp_out))
    open(tmp_out, "w") do io
        println(io, "campaign,task_index,worker_index,L,p_nn,p_nnn,global_traj,seed,bc,ti,tf,interval,t,cut_mode,cut_origin,renyi_label,renyi_value,rank_tol_label,rank_rel_probability_tol,S_bits,S_nats,elapsed_sec")
        for (offset, global_traj) in enumerate(traj_start:traj_stop)
            if mod(offset - 1, workers_per_task) != worker_index - 1
                continue
            end
            seed = stable_seed(seed_base, task_index, global_traj, worker_index)
            run_trajectory!(
                io, task, task_index, worker_index, global_traj, seed,
                renyi_specs, rank_rel_tols,
            )
            flush(io)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
