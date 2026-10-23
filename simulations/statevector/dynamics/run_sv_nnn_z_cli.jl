include("AKLTStatevectorNNNZ.jl")
using .AKLTStatevectorNNNZ

if !(length(ARGS) in (12, 13, 14, 15))
    error("expected 12-15 args: L cutoff bc p_nn n_steps iterations self_av_space_it ti tf interval seed filename [observable_mode] [halfcut_start_mode] [halfcut_start_seed]")
end

L = parse(Int, ARGS[1])
cutoffin = parse(Float64, ARGS[2])
bc = Symbol(ARGS[3])
p_nn = parse(Float64, ARGS[4])
n_steps = parse(Int, ARGS[5])
iterations = parse(Int, ARGS[6])
self_av_space_it = parse(Int, ARGS[7])
ti = parse(Int, ARGS[8])
tf = parse(Int, ARGS[9])
interval = parse(Int, ARGS[10])
master_seed = parse(Int, ARGS[11])
filename = ARGS[12]
observable_mode = length(ARGS) >= 13 ? Symbol(lowercase(ARGS[13])) : :all
halfcut_start_mode = length(ARGS) >= 14 ? ARGS[14] : "random"
halfcut_start_seed = length(ARGS) >= 15 ? parse(Int, ARGS[15]) : 0

AKLTStatevectorNNNZ.runy(
    L, cutoffin, bc, p_nn, n_steps, iterations, self_av_space_it,
    ti, tf, interval, master_seed, filename,
    observable_mode, halfcut_start_mode, halfcut_start_seed,
)
