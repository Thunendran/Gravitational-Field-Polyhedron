#!/usr/bin/env julia
# ===============================================================
# benchmark_julia_potential.jl
# Polyhedron Gravity Benchmark — Potential Comparison
# ===============================================================

using Printf, Dates, DelimitedFiles
using LinearAlgebra
using Base.Threads

# Prevent BLAS oversubscription
BLAS.set_num_threads(1)

# ---------------------------------------------------------------
# Include both gravity models (correct folders)
# ---------------------------------------------------------------
include("./PolyGravitation/Polyhedron_Gravitation.jl")       # Float64 module
include("./PolyGravitationBig/Polyhedron_Gravitation_Big.jl") # BigFloat module

using .Polyhedron_Gravitation: build_polyhedron, potential
using .Polyhedron_Gravitation_Big: build_polyhedron_big, potential_big

println()
println("=== Polyhedron Gravity Potential Benchmark ===")
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

warm_n = min(64, Npts)

# =====================================================================
#                     Float64 (FAST)
# =====================================================================
println("\n[Float64-FAST] Potential Benchmarking...")

# Build model (potential-only mode)
model_f64 = build_polyhedron(Float64.(vertices), faces;
    G=1.0, rho=1.0, eps=0.0, mode=:potential)

# warm-up
potential(model_f64, Matrix(@view points[1:warm_n, :]))

# timed run
t0 = time()
U_f64 = potential(model_f64, points)
t_elapsed_f64 = time() - t0
t_per_point_f64 = t_elapsed_f64 / Npts

@printf("Float64 time: %.3f s | per-pt: %.3f µs/pt\n",
        t_elapsed_f64, t_per_point_f64 * 1e6)

writedlm("../data/U_julia_float64_fast.csv", U_f64, ',')
write_time("../data/time_julia_float64_fast.txt";
           total=t_elapsed_f64, perpt=t_per_point_f64)

# =====================================================================
#                     BigFloat (50-digit)
# =====================================================================
const BIG50_DIGITS = 50
setprecision(BigFloat, BIG50_DIGITS; base=10)

println("\n[BigFloat-50] Potential Benchmarking...")

# Build model (potential-only)
model_bf50 = build_polyhedron_big(BigFloat.(vertices), faces;
    G=BigFloat(1.0), rho=BigFloat(1.0), eps=BigFloat(0),
    mode=:potential)

points_bf50 = BigFloat.(points)

# warm-up
potential_big(model_bf50, Matrix(@view points_bf50[1:warm_n, :]))

# timed run
t0 = time()
U_bf50 = potential_big(model_bf50, points_bf50)
t_elapsed_bf50 = time() - t0
t_per_point_bf50 = t_elapsed_bf50 / Npts

@printf("BigFloat-50 time: %.3f s | per-pt: %.3f µs/pt\n",
        t_elapsed_bf50, t_per_point_bf50 * 1e6)

# Save as one value per line with 50 digits
open("../data/U_julia_bigfloat_50.csv", "w") do io
    for v in U_bf50
        @printf(io, "%.*f\n", BIG50_DIGITS, v)
    end
end

write_time("../data/time_julia_bigfloat_50.txt";
           total=t_elapsed_bf50, perpt=t_per_point_bf50)

# =====================================================================
#                     BigFloat (250-digit reference)
# =====================================================================
const BIG250_DIGITS = 250
setprecision(BigFloat, BIG250_DIGITS; base=10)

println("\n[BigFloat-Ref] Potential Benchmarking (250 digits)...")

# Build model (potential-only)
model_bf250 = build_polyhedron_big(BigFloat.(vertices), faces;
    G=BigFloat(1.0), rho=BigFloat(1.0), eps=BigFloat(0),
    mode=:potential)

points_bf250 = BigFloat.(points)

# warm-up
potential_big(model_bf250, Matrix(@view points_bf250[1:warm_n, :]))

# timed run
t0 = time()
U_bf250 = potential_big(model_bf250, points_bf250)
t_elapsed_bf250 = time() - t0
t_per_point_bf250 = t_elapsed_bf250 / Npts

@printf("BigFloat-250 time: %.3f s | per-pt: %.3f µs/pt\n",
        t_elapsed_bf250, t_per_point_bf250 * 1e6)

# Save reference with full digits (one per line)
open("../data/U_ref_bigfloat_250digits.txt", "w") do io
    for v in U_bf250
        @printf(io, "%.*f\n", BIG250_DIGITS, v)
    end
end

write_time("../data/time_julia_bigfloat.txt";
           total=t_elapsed_bf250, perpt=t_per_point_bf250)

# =====================================================================
# Summary
# =====================================================================
println("\n=== Potential Benchmark Summary ===")
@printf("%-24s %12s %16s\n", "Program", "Time (s)", "Time per pt (µs/pt)")
@printf("%-24s %12.5f %16.3f\n", "Julia Float64-FAST", t_elapsed_f64,   t_per_point_f64*1e6)
@printf("%-24s %12.5f %16.3f\n", "Julia BigFloat-50",  t_elapsed_bf50,  t_per_point_bf50*1e6)
@printf("%-24s %12.5f %16.3f\n", "Julia BigFloat-250", t_elapsed_bf250, t_per_point_bf250*1e6)

println("\nResults saved in ../data/:")
println("  • U_julia_float64_fast.csv")
println("  • time_julia_float64_fast.txt")
println("  • U_julia_bigfloat_50.csv")
println("  • time_julia_bigfloat_50.txt")
println("  • U_ref_bigfloat_250digits.txt")
println("  • time_julia_bigfloat.txt")
println("---------------------------------------------------------------")
println("Done.")
