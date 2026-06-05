#!/usr/bin/env julia
# ===============================================================
# benchmark_julia_tensor.jl
# Polyhedron Gravity Benchmark — Tensor Comparison
# ===============================================================

using Printf, Dates, DelimitedFiles
using LinearAlgebra
using Base.Threads

# Prevent BLAS oversubscription
BLAS.set_num_threads(1)

# ---------------------------------------------------------------
# Include both modules (updated paths)
# ---------------------------------------------------------------
include("./PolyGravitation/Polyhedron_Gravitation.jl")        # Float64 module
include("./PolyGravitationBig/Polyhedron_Gravitation_Big.jl")  # BigFloat module

using .Polyhedron_Gravitation: build_polyhedron, gravity_tensor
using .Polyhedron_Gravitation_Big: build_polyhedron_big, gravity_tensor_big

println()
println("=== Polyhedron Gravity Tensor Benchmark ===")
println("Date: ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
println("Threads: ", Threads.nthreads())
println("---------------------------------------------------------------")

# ----------------------------- Load data -----------------------------
vertices = readdlm("../data/icosahedron_vertices.csv", ',')
faces    = Int.(readdlm("../data/icosahedron_faces.csv", ','))
points   = readdlm("../data/eval_points_100k_plus_vertices.csv", ',')

Nverts, Nfaces, Npts = size(vertices,1), size(faces,1), size(points,1)
println("Vertices: $Nverts | Faces: $Nfaces | Points: $Npts")

# ----------------------------- Utility -------------------------------
function write_time(fname::AbstractString; total::Float64, perpt::Float64)
    open(fname, "w") do io
        @printf(io, "total_time_sec: %.6f\n", total)
        @printf(io, "time_per_point_sec: %.12e\n", perpt)
        @printf(io, "time_per_point_us: %.6f\n", perpt * 1e6)
    end
end

warm_n = min(16, Npts)

# =====================================================================
#                     Float64 (FAST)
# =====================================================================
println("\n[Float64-FAST] Tensor Benchmarking...")

# Build model with tensor-only precomputations
model_f64 = build_polyhedron(Float64.(vertices), faces;
    G=1.0, rho=1.0, eps=0.0, mode=:tensor)

# warm-up
gravity_tensor(model_f64, Matrix(@view points[1:warm_n, :]))

# timed run
t0 = time()
T_f64 = gravity_tensor(model_f64, points)
t_elapsed_f64 = time() - t0
t_per_point_f64 = t_elapsed_f64 / Npts

@printf("Float64 time: %.3f s | per-pt: %.3f µs/pt\n",
        t_elapsed_f64, t_per_point_f64 * 1e6)

open("../data/T_julia_float64_fast.csv", "w") do io
    for i in 1:Npts
        @printf(io, "%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e\n",
            T_f64[i,1,1],T_f64[i,1,2],T_f64[i,1,3],
            T_f64[i,2,1],T_f64[i,2,2],T_f64[i,2,3],
            T_f64[i,3,1],T_f64[i,3,2],T_f64[i,3,3])
    end
end
write_time("../data/time_julia_tensor_float64_fast.txt";
           total=t_elapsed_f64, perpt=t_per_point_f64)

# =====================================================================
#                     BigFloat (50-digit)
# =====================================================================
const BIG50_DIGITS = 50
setprecision(BigFloat, BIG50_DIGITS; base=10)

println("\n[BigFloat-50] Tensor Benchmarking...")

# Build model (tensor mode)
model_bf50 = build_polyhedron_big(BigFloat.(vertices), faces;
    G=BigFloat(1.0), rho=BigFloat(1.0), eps=BigFloat(0),
    mode=:tensor)

points_bf50 = BigFloat.(points)

# warm-up
gravity_tensor_big(model_bf50, Matrix(@view points_bf50[1:warm_n, :]))

# timed run
t0 = time()
T_bf50 = gravity_tensor_big(model_bf50, points_bf50)
t_elapsed_bf50 = time() - t0
t_per_point_bf50 = t_elapsed_bf50 / Npts

@printf("BigFloat-50 time: %.3f s | per-pt: %.3f µs/pt\n",
        t_elapsed_bf50, t_per_point_bf50 * 1e6)

open("../data/T_julia_bigfloat_50.csv", "w") do io
    for i in 1:Npts
        for r in 1:3, c in 1:3
            @printf(io, "%.*f%s", BIG50_DIGITS, T_bf50[i,r,c],
                    (r==3 && c==3) ? "\n" : ",")
        end
    end
end
write_time("../data/time_julia_tensor_bigfloat_50.txt";
           total=t_elapsed_bf50, perpt=t_per_point_bf50)

# =====================================================================
#                     BigFloat (250-digit reference)
# =====================================================================
const BIG250_DIGITS = 250
setprecision(BigFloat, BIG250_DIGITS; base=10)

println("\n[BigFloat-Ref] Tensor Benchmarking (250 digits)...")

# Build model (tensor mode)
model_bf250 = build_polyhedron_big(BigFloat.(vertices), faces;
    G=BigFloat(1.0), rho=BigFloat(1.0), eps=BigFloat(0),
    mode=:tensor)

points_bf250 = BigFloat.(points)

# warm-up
gravity_tensor_big(model_bf250, Matrix(@view points_bf250[1:warm_n, :]))

# timed run
t0 = time()
T_bf250 = gravity_tensor_big(model_bf250, points_bf250)
t_elapsed_bf250 = time() - t0
t_per_point_bf250 = t_elapsed_bf250 / Npts

@printf("BigFloat-250 time: %.3f s | per-pt: %.3f µs/pt\n",
        t_elapsed_bf250, t_per_point_bf250 * 1e6)

open("../data/T_ref_bigfloat_250digits.txt", "w") do io
    for i in 1:Npts
        for r in 1:3, c in 1:3
            @printf(io, "%.*f%s", BIG250_DIGITS, T_bf250[i,r,c],
                    (r==3 && c==3) ? "\n" : ",")
        end
    end
end
write_time("../data/time_julia_tensor_bigfloat.txt";
           total=t_elapsed_bf250, perpt=t_per_point_bf250)

# =====================================================================
# Summary
# =====================================================================
println("\n=== Tensor Benchmark Summary ===")
@printf("%-24s %12s %16s\n", "Program", "Time (s)", "Time per pt (µs/pt)")
@printf("%-24s %12.5f %16.3f\n", "Julia Float64-FAST", t_elapsed_f64,   t_per_point_f64*1e6)
@printf("%-24s %12.5f %16.3f\n", "Julia BigFloat-50",  t_elapsed_bf50,  t_per_point_bf50*1e6)
@printf("%-24s %12.5f %16.3f\n", "Julia BigFloat-250", t_elapsed_bf250, t_per_point_bf250*1e6)

println("\nResults saved in ../data/:")
println("  • T_julia_float64_fast.csv")
println("  • time_julia_tensor_float64_fast.txt")
println("  • T_julia_bigfloat_50.csv")
println("  • time_julia_tensor_bigfloat_50.txt")
println("  • T_ref_bigfloat_250digits.txt")
println("  • time_julia_tensor_bigfloat.txt")
println("---------------------------------------------------------------")
println("Done.")
