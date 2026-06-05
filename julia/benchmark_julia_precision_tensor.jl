#!/usr/bin/env julia
# ===============================================================
# precision_scaling_tensor.jl
# Polyhedron Gravity Tensor — Precision vs Accuracy & Runtime
# ===============================================================

using Printf, Dates, DelimitedFiles, LinearAlgebra, Statistics
using Base.Threads
using Plots
using Measures

gr(dpi = 300)
BLAS.set_num_threads(1)

# ---------------------------------------------------------------
# Include BigFloat gravity module
# ---------------------------------------------------------------
include("PolyGravityBig/PolyhedronGravityBigMT.jl")
using .PolyhedronGravityBigMT: build_polyhedron_big, tensor_big

println()
println("=== Polyhedron Gravity Tensor: Precision Scaling ===")
println("Date: ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
println("Threads: ", Threads.nthreads())
println("---------------------------------------------------------------")

vertices = readdlm("../data/icosahedron_vertices.csv", ',')
faces    = Int.(readdlm("../data/icosahedron_faces.csv", ','))
points   = readdlm("../data/eval_points_100k_plus_vertices.csv", ',')

Nverts, Nfaces, Npts = size(vertices,1), size(faces,1), size(points,1)
println("Vertices: $Nverts | Faces: $Nfaces | Points: $Npts")

prec_levels = 20:20:200
ref_prec = 250
results = NamedTuple[]
warm_n = min(64, Npts)

# ---------------------------------------------------------------
# Reference computation
# ---------------------------------------------------------------
setprecision(BigFloat, ref_prec; base=10)
println("\n[Reference] Computing 250-digit tensor...")

model_ref = build_polyhedron_big(BigFloat.(vertices), faces;
    G=BigFloat(1.0), rho=BigFloat(1.0), eps=BigFloat(0))
points_ref = BigFloat.(points)

tensor_big(model_ref, Matrix(@view points_ref[1:warm_n, :]))

t0 = time()
T_ref = tensor_big(model_ref, points_ref)
t_ref_total = time() - t0
t_ref_per_point = t_ref_total / Npts

@printf("Reference (250-digit): %.3f s total | %.3e s/pt\n", t_ref_total, t_ref_per_point)

open("../data/T_ref_bigfloat_250digits.txt", "w") do io
    for row in eachrow(T_ref)
        @printf(io, "%.*f,%.*f,%.*f,%.*f,%.*f,%.*f,%.*f,%.*f,%.*f\n",
            ref_prec, row[1], ref_prec, row[2], ref_prec, row[3],
            ref_prec, row[4], ref_prec, row[5], ref_prec, row[6],
            ref_prec, row[7], ref_prec, row[8], ref_prec, row[9])
    end
end

# ---------------------------------------------------------------
# Sweep arbitrary precisions
# ---------------------------------------------------------------
for p in prec_levels
    setprecision(BigFloat, p; base=10)
    println("\n[Precision = $p digits] Computing tensor...")

    model = build_polyhedron_big(BigFloat.(vertices), faces;
        G=BigFloat(1.0), rho=BigFloat(1.0), eps=BigFloat(0))
    pts_bf = BigFloat.(points)

    tensor_big(model, Matrix(@view pts_bf[1:warm_n, :]))

    t0 = time()
    T = tensor_big(model, pts_bf)
    t_elapsed = time() - t0
    t_per_point = t_elapsed / Npts

    rel_err = BigFloat[]
    for i in eachindex(T[:,1])
        for j in 1:9
            if T_ref[i,j] != 0
                push!(rel_err, abs((T_ref[i,j] - T[i,j]) / T_ref[i,j]))
            end
        end
    end

    mean_rel_err = mean(rel_err)
    log_rel_err = log10(mean_rel_err)

    @printf(" -> %3d digits: total %.3f s | %.3e s/pt | log₁₀(rel.err)=%.6f\n",
        p, t_elapsed, t_per_point, Float64(log_rel_err))

    push!(results, (
        precision = p,
        total_time = BigFloat(t_elapsed),
        time_per_point = BigFloat(t_per_point),
        mean_rel_error = mean_rel_err,
        log_rel_error = log_rel_err
    ))
end

# ---------------------------------------------------------------
# Save results
# ---------------------------------------------------------------
out_txt = "../data/tensor_precision_scaling.txt"
open(out_txt, "w") do io
    println(io, "# Precision Scaling Benchmark for Polyhedron Tensor")
    println(io, "# Columns: precision_digits, total_time_sec, time_per_point_sec, mean_rel_error, log10_rel_error")
    for r in results
        @printf(io, "%d,%0.6f,%0.12e,%0.12e,%0.6f\n",
            r.precision,
            Float64(r.total_time),
            Float64(r.time_per_point),
            Float64(r.mean_rel_error),
            Float64(r.log_rel_error))
    end
end
println("\nSaved results to: $out_txt")

# ---------------------------------------------------------------
# Extract for plotting
# ---------------------------------------------------------------
prec     = [r.precision for r in results]
logerr   = [Float64(r.log_rel_error) for r in results]
times_us = [Float64(r.time_per_point * 1e6) for r in results]

# ---------------------------------------------------------------
# Plot (publication style)
# ---------------------------------------------------------------
blue_custom = RGB(0/255, 174/255, 235/255)

plt = plot(
    prec, logerr;
    xlabel = "Precision (digits)",
    ylabel = "log₁₀(Relative Error)",
    lw = 5, marker = :circle,
    color = blue_custom, label = false,
    title = "Polyhedron Gravity Gradient Tensor: Precision vs Runtime and Relative Error",
    titlefont = font(21, "Arial", halign = :center, valign = :top),
    xguidefont = font(20), yguidefont = font(20),
    xtickfont = font(18), ytickfont = font(18),
    grid = true, framestyle = :box,
    size = (1600, 1000),
    left_margin = 20mm, right_margin = 30mm,
    bottom_margin = 25mm, top_margin = 5mm,
)

plot!(twinx(), prec, times_us;
    ylabel = "Time per point (µs)",
    lw = 4, ls = :dash, marker = :diamond,
    color = :red, label = false,
    yguidefont = font(20), ytickfont = font(18))

plot!(plt; framestyle = :box, foreground_color_border = :black)
plot!(plt; titlefont = font(23, "Arial", valign = :bottom))

legend_y = minimum(logerr) - 0.8
legend_x_center = mean(prec)

plot!([legend_x_center - 80, legend_x_center - 60], [legend_y, legend_y]; color=blue_custom, lw=4, label=false)
scatter!([legend_x_center - 70], [legend_y], m=:circle, ms=8, color=blue_custom, label=false)
plot!([legend_x_center + 10, legend_x_center + 30], [legend_y, legend_y]; color=:red, lw=4, ls=:dash, label=false)
scatter!([legend_x_center + 20], [legend_y], m=:diamond, ms=8, color=:red, label=false)
annotate!(legend_x_center - 50, legend_y, text("log₁₀(Relative Error)", 18, :left))
annotate!(legend_x_center + 40, legend_y, text("Time per point (µs)", 18, :left))

savefig(plt, "../data/tensor_precision_scaling_300dpi.png")
println("Saved plot: ../data/tensor_precision_scaling_300dpi.png")
