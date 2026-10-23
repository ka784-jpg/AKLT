using LinearAlgebra

BLAS.set_num_threads(1)
Threads.nthreads() == 1 || error("Julia thread cap violated: $(Threads.nthreads())")
BLAS.get_num_threads() == 1 || error("BLAS thread cap violated: $(BLAS.get_num_threads())")

length(ARGS) == 10 || error(
    "usage: run_order.jl SOURCE L P_NN NSTEPS TI TF INTERVAL CONTROLLER_SEED OUTPUT_DIR OUTPUT_TAG"
)

source_file = ARGS[1]
L = parse(Int, ARGS[2])
p_nn = parse(Float64, ARGS[3])
n_steps = parse(Int, ARGS[4])
ti = parse(Int, ARGS[5])
tf = parse(Int, ARGS[6])
interval = parse(Int, ARGS[7])
controller_seed = parse(Int, ARGS[8])
output_dir = ARGS[9]
output_tag = ARGS[10]

isfile(source_file) || error("missing source file: $source_file")
mkpath(output_dir)
include(source_file)
using .AKLTStatevectorNNNZ

AKLTStatevectorNNNZ.runy(
    L, 0.0, :periodic, p_nn, n_steps, 1, L,
    ti, tf, interval, controller_seed, joinpath(output_dir, output_tag),
    :order_only, "random", 0,
)
