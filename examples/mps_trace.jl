module AKLTMPSExample

using DelimitedFiles, SHA, Statistics, TOML

const KEYS = Set(["L", "p_competing", "seed", "sweeps", "cutoff"])

function read_config(path)
    c = TOML.parsefile(path)
    Set(keys(c)) == KEYS || throw(ArgumentError("Config requires L, p_competing, seed, sweeps, cutoff"))
    for key in ("L", "seed", "sweeps")
        c[key] isa Integer && !(c[key] isa Bool) || throw(ArgumentError("$key must be an integer"))
    end
    c["L"] in (4, 6) || throw(ArgumentError("Demo L must be 4 or 6"))
    1 <= c["sweeps"] <= 40 || throw(ArgumentError("Demo sweeps must be in 1:40"))
    0 <= c["seed"] <= typemax(Int32) || throw(ArgumentError("Invalid seed"))
    for key in ("p_competing", "cutoff")
        c[key] isa Real && !(c[key] isa Bool) && isfinite(c[key]) || throw(ArgumentError("$key must be finite"))
    end
    0 <= c["p_competing"] <= 1 || throw(ArgumentError("p_competing must be in [0,1]"))
    1e-12 <= c["cutoff"] <= 1e-4 || throw(ArgumentError("Demo cutoff must be in [1e-12,1e-4]"))
    return c
end

function read_channel(raw, suffix, c, columns)
    files = filter(p -> endswith(p, suffix), readdir(raw; join=true))
    length(files) == 1 || error("Expected one $suffix output")
    values = readdlm(only(files), ',', Float64)
    size(values) == (c["sweeps"] + 1, columns + 1) || error("Unexpected shape: $suffix")
    values[2:end, 1] == 1:c["sweeps"] || error("Unexpected sweep labels: $suffix")
    all(isfinite, values[1, 2:end]) || error("Invalid seed header: $suffix")
    all(isfinite, values[2:end, 2:end]) || error("Non-finite observable: $suffix")
    return values[2:end, 2:end], values[1, 2:end]
end

"""Wrap a preserved MPS runner; publish validated outputs without overwriting."""
function run_demo(runner, model, config_path, output, engine_path)
    model in ("trivial", "dimer") || throw(ArgumentError("Unknown MPS model"))
    c = read_config(config_path)
    destination = abspath(output)
    (ispath(destination) || islink(destination)) && throw(ArgumentError("Output exists: $destination"))
    mkpath(dirname(destination))
    stage = mktempdir(dirname(destination); prefix=".aklt-mps-example-")
    try
        raw = joinpath(stage, "raw")
        mkpath(raw)
        args = (c["L"], 1.0-c["p_competing"], Float64(c["cutoff"]), c["sweeps"],
            1, true, div(c["L"], 2), 1, c["sweeps"], 1, c["seed"], joinpath(raw, "trajectory"))
        if model == "dimer"
            runner(args..., "odd", "all", "half", 0)
        else
            runner(args...)
        end
        suffix = model == "trivial" ? "zero" : "dimer"
        nn, seeds = read_channel(raw, "_SO_NN_QCMPS_$suffix", c, div(c["L"], 2))
        entropy, entropy_seeds = read_channel(raw, "_S_QCMPS_$suffix", c, 1)
        all(==(only(entropy_seeds)), seeds) || error("Inconsistent trajectory seeds")
        all(x -> abs(x) <= 1 + 1e-10, nn) || error("String order outside physical bounds")
        all(x -> -1e-10 <= x <= div(c["L"], 2)*log2(3)+1e-10, entropy) || error("Entropy outside physical bounds")
        signed_nn = vec(mean(nn; dims=2))
        summary = hcat(1:c["sweeps"], signed_nn, abs.(signed_nn), vec(entropy))
        header = "sweep,mean_nn,abs_mean_nn,entropy_bits"
        if model == "dimer"
            dimer, dimer_seeds = read_channel(raw, "_Dimer_QCMPS_dimer", c, 1)
            dimer_seeds == entropy_seeds || error("Inconsistent dimer seed")
            all(x -> abs(x) <= 2 + 1e-10, dimer) || error("Dimer order outside physical bounds")
            summary = hcat(summary, vec(dimer), abs.(vec(dimer)))
            header *= ",dimer_order,abs_dimer_order"
        end
        open(joinpath(stage, "summary.csv"), "w") do io
            println(io, header)
            writedlm(io, summary, ',')
        end
        metadata = Dict(
            "model" => model, "parameters" => c, "trajectories" => 1,
            "trajectory_seed" => Int(only(entropy_seeds)), "p_nn" => 1.0-c["p_competing"],
            "julia_version" => string(VERSION), "boundary" => "periodic",
            "initial_state" => "uniform product state",
            "time_unit" => "one sweep = L local update slots",
            "summary_convention" => "NN: signed mean over L/2 origins, then absolute value; one trajectory",
            "entropy_unit" => "bits", "engine_sha256" => bytes2hex(open(sha256, engine_path)),
            "engine" => replace(relpath(engine_path, joinpath(@__DIR__, "..")), '\\' => '/'),
        )
        if model == "dimer"
            metadata["dimer_sector"] = "odd"
            metadata["halfcut_start_mode"] = "half"
            metadata["dimer_convention"] = "signed alternating mean of nearest-neighbor spin correlations; magnitude plotted"
        else
            metadata["zero_filter"] = "legacy normalized exp(50*|0><0|), finite-strength projector approximation"
        end
        open(io -> TOML.print(io, metadata; sorted=true), joinpath(stage, "metadata.toml"), "w")
        mv(stage, destination)
        return destination
    finally
        isdir(stage) && rm(stage; recursive=true)
    end
end

function main(runner, model, engine_path, args=ARGS)
    if args in (["--help"], ["-h"])
        println("Usage: julia --project=. examples/$model/run.jl [NEW_OUTPUT_DIRECTORY]")
        return
    end
    length(args) <= 1 || throw(ArgumentError("Expected at most one output directory"))
    !isempty(args) && startswith(only(args), "--") && throw(ArgumentError("Unknown option"))
    output = isempty(args) ? joinpath("results", "$(model)_example") : only(args)
    destination = run_demo(runner, model, joinpath(@__DIR__, model, "config.toml"), output, engine_path)
    println("Validated single-trajectory example: $destination")
end

end
