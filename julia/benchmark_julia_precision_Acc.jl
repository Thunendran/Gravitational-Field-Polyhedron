#!/usr/bin/env julia
# ===============================================================
# precision_scaling_accel.jl (Updated for zero-safe comparison)
# Polyhedron Gravity Acceleration — Precision vs Accuracy & Runtime
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
using .PolyhedronGravityBigMT: build_polyhedron_big, acceleration_big

println()
println("=== Polyhedron Gravity Acceleration: Precision Scaling ===")
println("Date: ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
println("Threads: ", Threads.nthreads())
println("---------------------------------------------------------------")

# ----------------------------- Load data -----------------------------
vertices = readdlm("../data/icosahedron_vertices.csv", ',')
faces    = Int.(readdlm("../data/icosahedron_faces.csv", ','))
points   = readdlm("../data/eval_points_100k_plus_vertices.csv", ',')

Nverts, Nfaces, Npts = size(vertices,1), size(faces,1), size(points,1)
println("Vertices: $Nverts | Faces: $Nfaces | Points: $Npts")

# ----------------------------- Parameters ----------------------------
prec_levels = 20:20:200
ref_prec    = 250
results     = NamedTuple[]
warm_n      = min(64, Npts)

# ----------------------------- Reference (250) -----------------------
setprecision(BigFloat, ref_prec; base=10)
println("\n[Reference] Computing 250-digit acceleration...")

model_ref = build_polyhedron_big(BigFloat.(vertices), faces;
    G=BigFloat(1.0), rho=BigFloat(1.0), eps=BigFloat(0))
points_ref = BigFloat.(points)

# Warm-up
acceleration_big(model_ref, Matrix(@view points_ref[1:warm_n, :]))

t0 = time()
A_ref = acceleration_big(model_ref, points_ref)   # Npts × 3 (BigFloat@250)
t_ref_total = time() - t0
t_ref_per_point = t_ref_total / Npts

@printf("Reference (250-digit): %.3f s total | %.3e s/pt\n",
        t_ref_total, t_ref_per_point)
println("---------------------------------------------------------------")

# Optional: save reference
open("../data/A_ref_bigfloat_250digits.txt", "w") do io
    for row in eachrow(A_ref)
        @printf(io, "%.*f,%.*f,%.*f\n",
            ref_prec, row[1], ref_prec, row[2], ref_prec, row[3])
    end
end

# ----------------------------- Sweep precisions ----------------------
for p in prec_levels
    setprecision(BigFloat, p; base=10)
    println("\n[Precision = $p digits] Computing acceleration...")

    model_p = build_polyhedron_big(BigFloat.(vertices), faces;
        G=BigFloat(1.0), rho=BigFloat(1.0), eps=BigFloat(0))
    pts_p = BigFloat.(points)

    # Warm-up
    acceleration_big(model_p, Matrix(@view pts_p[1:warm_n, :]))

    # Timed run
    t0 = time()
    A_p = acceleration_big(model_p, pts_p)   # Npts × 3 (BigFloat@p)
    t_elapsed   = time() - t0
    t_per_point = t_elapsed / Npts

    # ---------- Relative error computation ----------
    τ = BigFloat(10)^(-p)   # precision-dependent floor
    comp_means = BigFloat[0, 0, 0]

    for j in 1:3
        errs = BigFloat[]
        @inbounds for i in 1:Npts
            ref_val = BigFloat(A_ref[i, j])  # reference rounded to current p
            test_val = A_p[i, j]
            aref = abs(ref_val)
            aval = abs(test_val)

            # Skip if both are effectively zero (below τ)
            if aref < τ && aval < τ
                continue
            end

            den = max(aref, τ)
            push!(errs, abs(ref_val - test_val) / den)
        end

        comp_means[j] = isempty(errs) ? BigFloat(0) : mean(errs)
    end

    overall_mean_rel = mean(comp_means)
    log_rel_err      = (overall_mean_rel == 0) ? -Inf : log10(overall_mean_rel)

    @printf(" -> %3d digits: total %.3f s | %.3e s/pt | log₁₀(rel.err)=%s\n",
            p, t_elapsed, t_per_point,
            overall_mean_rel == 0 ? "-Inf" : @sprintf("%.6f", Float64(log_rel_err)))

    push!(results, (
        precision        = p,
        total_time       = BigFloat(t_elapsed),
        time_per_point   = BigFloat(t_per_point),
        mean_rel_error   = overall_mean_rel,
        mean_rel_error_x = comp_means[1],
        mean_rel_error_y = comp_means[2],
        mean_rel_error_z = comp_means[3],
        log_rel_error    = log_rel_err
    ))
end

# ----------------------------- Save results --------------------------
out_txt = "../data/accel_precision_scaling.txt"
open(out_txt, "w") do io
    println(io, "# Precision Scaling Benchmark for Polyhedron Acceleration")
    println(io, "# Columns: precision_digits, total_time_sec, time_per_point_sec, mean_rel_error, mean_rel_err_x, mean_rel_err_y, mean_rel_err_z, log10_rel_error")
    for r in results
        @printf(io, "%d,%0.6f,%0.12e,%0.12e,%0.12e,%0.12e,%0.12e,%s\n",
            r.precision,
            Float64(r.total_time),
            Float64(r.time_per_point),
            Float64(r.mean_rel_error),
            Float64(r.mean_rel_error_x),
            Float64(r.mean_rel_error_y),
            Float64(r.mean_rel_error_z),
            (isfinite(r.log_rel_error) ? @sprintf("%0.6f", Float64(r.log_rel_error)) : "-Inf"))
    end
end
println("\nSaved results to: $out_txt")

# ----------------------------- Plot (publication style) --------------
prec     = [r.precision for r in results]
logerr   = [isfinite(r.log_rel_error) ? Float64(r.log_rel_error) : -Inf for r in results]
times_us = [Float64(r.time_per_point * 1e6) for r in results]

blue_custom = RGB(0/255, 174/255, 235/255)

plt = plot(
    prec, logerr;
    xlabel = "Precision (digits)",
    ylabel = "log₁₀(Relative Error)",
    lw = 5,
    marker = :circle,
    color = blue_custom,
    label = false,
    title = "Polyhedron Gravitational Acceleration: Precision vs Runtime and Relative Error",
    titlefont = font(21, "Arial", halign = :center, valign = :top),
    xguidefont = font(20),
    yguidefont = font(20),
    xtickfont = font(18),
    ytickfont = font(18),
    grid = true,
    framestyle = :box,
    size = (1600, 1000),
    left_margin = 20mm,
    right_margin = 30mm,
    bottom_margin = 25mm,
    top_margin = 5mm,
)

plot!(
    twinx(), prec, times_us;
    ylabel = "Time per point (µs)",
    lw = 4, ls = :dash, marker = :diamond,
    color = :red, label = false,
    yguidefont = font(20), ytickfont = font(18),
)

plot!(plt; framestyle = :box, foreground_color_border = :black)
plot!(plt; titlefont = font(23, "Arial", valign = :bottom))

# Manual legend (bottom-center)
legend_y = (minimum(filter(!isinf, logerr)) - 0.8)
legend_x_center = mean(prec)

plot!([legend_x_center - 80, legend_x_center - 60], [legend_y, legend_y];
      color=blue_custom, lw=4, label=false)
scatter!([legend_x_center - 70], [legend_y], m=:circle, ms=8, color=blue_custom, label=false)

plot!([legend_x_center + 10, legend_x_center + 30], [legend_y, legend_y];
      color=:red, lw=4, ls=:dash, label=false)
scatter!([legend_x_center + 20], [legend_y], m=:diamond, ms=8, color=:red, label=false)

annotate!(legend_x_center - 50, legend_y, text("log₁₀(Relative Error)", 18, :left))
annotate!(legend_x_center + 40, legend_y, text("Time per point (µs)", 18, :left))

savefig(plt, "../data/accel_precision_scaling_300dpi.png")
println("Saved plot: ../data/accel_precision_scaling_300dpi.png")
