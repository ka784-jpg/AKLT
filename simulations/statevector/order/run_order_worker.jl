#!/usr/bin/env julia

using LinearAlgebra
BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "AKLTStatevectorOrderOnly.jl"))
using .AKLTStatevectorOrderOnly

length(ARGS) == 5 || error("usage: run_order_worker.jl L p_nn p_nnn seed outbase")

L = parse(Int, ARGS[1])
p_nn = parse(Float64, ARGS[2])
p_nnn = parse(Float64, ARGS[3])
seed = parse(Int, ARGS[4])
outbase = ARGS[5]

@assert iseven(L)
@assert 0.0 <= p_nn <= 1.0
@assert 0.0 <= p_nnn <= 1.0
@assert isapprox(p_nn + p_nnn, 1.0; atol=1.0e-12)
@assert BLAS.get_num_threads() == 1

ti = 2 * L
tf = 2 * L + 4
self_av_space_it = div(L, 2)

AKLTStatevectorOrderOnly.runy(
    L,
    0.0,
    :periodic,
    p_nn,
    tf,
    1,
    self_av_space_it,
    ti,
    tf,
    1,
    seed,
    outbase,
)
