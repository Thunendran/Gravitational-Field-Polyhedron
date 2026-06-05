#!/usr/bin/env julia
# ===============================================================
# precision_scaling_potential.jl
# Polyhedron Gravity Potential — Precision vs Accuracy & Runtime
# ===============================================================

using Printf, Dates, DelimitedFiles, LinearAlgebra, Statistics
using Base.Threads
using Plots
using Measures

gr(dpi = 300)

# Prevent BLAS oversubscription
BLAS.set_num_threads(1)

# ---------------------------------------------------------------
# Include BigFloat gravity module (correct folder + name)
# ---------------------------------------------------------------
include("./PolyGravitationBig/Polyhedron_Gravitation_Big.jl")
using .Polyhedron_Gravitation_Big: build_polyhedron_big, potential_big

println()
println("=== Polyhedron Gravity Potential: Precision Scaling ===")
println("Date: ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
println("Threads: ", Threads.nthreads())
println("---------------------------------------------------------------")

# ----------------------------- Load data -----------------------------
vertices = readdlm("../data/icosahedron_vertices.csv", ',')
faces    = Int.(readdlm("../data/icosahedron_faces.csv", ','))
points   = readdlm("../data/eval_points_100k_plus_vertices.csv", ',')

Nverts, Nfaces, Npts = size(vertices,1), size(faces,1), size(points,1)
println("Vertices: $Nverts | Faces: $Nfaces | Points: $Npts")

# ---------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------
prec_levels = 20:20:200
ref_prec    = 250
results     = NamedTuple[]
warm_n      = min(64, Npts)

# ---------------------------------------------------------------
# Reference computation (250-digit)
# ---------------------------------------------------------------
setprecision(BigFloat, ref_prec; base=10)
println("\n[Reference] Computing 250-digit potential...")

# Build polyhedron in potential-only mode
model_ref = build_polyhedron_big(BigFloat.(vertices), faces;
    G=BigFloat(1.0), rho=BigFloat(1.0), eps=BigFloat(0),
    mode=:potential)

points_ref = BigFloat.(points)

# Warm-up
potential_big(model_ref, Matrix(@view points_ref[1:warm_n, :]))

t0 = time()
U_ref = potential_big(model_ref, points_ref)
t_ref_total = time() - t0
t_ref_per_point = t_ref_total / Npts

@printf("Reference (250-digit): %.3f s total, %.3e s/pt\n",
        t_ref_total, t_ref_per_point)
println("---------------------------------------------------------------")

# Save reference potential
open("../data/U_ref_bigfloat_250digits.txt", "w") do io
    for v in U_ref
        @printf(io, "%.*f\n", ref_prec, v)
    end
end

# ---------------------------------------------------------------
# Sweep arbitrary precisions (20:20:200)
# ---------------------------------------------------------------
for p in prec_levels
    setprecision(BigFloat, p; base=10)
    println("\n[Precision = $p digits] Computing potential...")

    # Build model (potential mode only)
    model = build_polyhedron_big(BigFloat.(vertices), faces;
        G=BigFloat(1.0), rho=BigFloat(1.0), eps=BigFloat(0),
        mode=:potential)

    pts_bf = BigFloat.(points)

    # Warm-up
    potential_big(model, Matrix(@view pts_bf[1:warm_n, :]))

    # Timed run
    t0 = time()
    U = potential_big(model, pts_bf)
    t_elapsed = time() - t0
    t_per_point = t_elapsed / Npts

    # === Relative Error Calculation ===
    rel_err = BigFloat[]
    for i in eachindex(U)
        if U_ref[i] != 0
            push!(rel_err, abs((U_ref[i] - U[i]) / U_ref[i]))
        end
    end

    mean_rel_err = mean(rel_err)
    log_rel_err  = log10(mean_rel_err)

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
out_txt = "../data/potential_precision_scaling.txt"
open(out_txt, "w") do io
    println(io, "# Precision Scaling Benchmark for Polyhedron Potential")
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
# Extract arrays from results for plotting
# ---------------------------------------------------------------
prec     = [r.precision for r in results]
logerr   = [Float64(r.log_rel_error) for r in results]
times_us = [Float64(r.time_per_point * 1e6) for r in results]

# ---------------------------------------------------------------
# Linear fit: Time per point (µs) vs Precision (digits)
# ---------------------------------------------------------------
X = [ones(length(prec)) prec]
coeffs = X \ times_us
a, b = coeffs[2], coeffs[1]
times_fit = X * coeffs
rmse = sqrt(mean((times_us .- times_fit).^2))

@printf("\nLinear Fit (Time per point vs Precision): time = %.4f * prec + %.4f\n", a, b)
@printf("RMSE (µs): %.4f\n", rmse)

# ---------------------------------------------------------------
# Plot results (publication style)
# ---------------------------------------------------------------
blue_custom = RGB(0/255, 174/255, 235/255)

plt = plot(
    prec, logerr;
    xlabel = "Precision (digits)",
    ylabel = "log₁₀(Relative Error)",
    lw = 5,
    marker = :circle,
    color = blue_custom,
    label = false,
    title = "Polyhedron Gravitational Potential: Precision vs Runtime and Relative Error",
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

# --- Secondary y-axis for time ---
plt2 = twinx(plt)

plot!(
    plt2, prec, times_us;
    ylabel = "Time per point (µs)",
    lw = 4,
    ls = :dash,
    marker = :diamond,
    color = :red,
    label = false,
    yguidefont = font(20),
    ytickfont = font(18),
)

# --- Linear fit (orange dotted) ---
plot!(
    plt2, prec, times_fit;
    lw = 3,
    ls = :dot,
    color = :orange,
    label = false,
)

# --- Annotate fit equation and RMSE ---
annotate!(
    plt2,
    maximum(prec) * 0.55,
    maximum(times_us) * 0.9,
    text(@sprintf("Fit: y = %.2f·x + %.1f\nRMSE = %.2f µs", a, b, rmse), 15, :orange)
)

# --- Final formatting ---
plot!(plt; framestyle = :box, foreground_color_border = :black)
plot!(plt; titlefont = font(23, "Arial", valign = :bottom))

# --- Manual legend ---
legend_y = minimum(logerr) - 0.8
legend_x_center = mean(prec)

# Blue curve
plot!(plt, [legend_x_center - 80, legend_x_center - 60],
      [legend_y, legend_y]; color=blue_custom, lw=4, label=false)
scatter!(plt, [legend_x_center - 70], [legend_y],
         m=:circle, ms=8, color=blue_custom, label=false)

# Red dashed line
plot!(plt, [legend_x_center + 10, legend_x_center + 30],
      [legend_y, legend_y]; color=:red, lw=4, ls=:dash, label=false)
scatter!(plt, [legend_x_center + 20], [legend_y],
         m=:diamond, ms=8, color=:red, label=false)

# Legend text
annotate!(plt, legend_x_center - 50, legend_y, text("log₁₀(Relative Error)", 18, :left))
annotate!(plt, legend_x_center + 40, legend_y, text("Time per point (µs)", 18, :left))

# --- Save image ---
savefig(plt, "../data/potential_precision_scaling_300dpi.png")
println("Saved plot: ../data/potential_precision_scaling_300dpi.png")

println("---------------------------------------------------------------")
println("Done.")
