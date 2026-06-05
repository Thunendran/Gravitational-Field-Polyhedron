# ===============================================================
# test_polyhedron_gravitation_compare.jl
# Compare BigFloat vs Float64 Polyhedron Gravity implementations
#
# Author: Thunendran Periyandy (2025)
# Purpose:
#   Cross-verify results between the standard (Float64) and
#   high-precision (BigFloat) implementations of the
#   PolyGravitation framework.
# ===============================================================

# ----------------------------------------------------------------------
# Include module files
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Include module files (with correct relative paths)
# ----------------------------------------------------------------------
include("./PolyGravitation/Polyhedron_Gravitation.jl")
include("./PolyGravitationBig/Polyhedron_Gravitation_Big.jl")


# Import functions from each module
using .Polyhedron_Gravitation:
    build_polyhedron, potential, acceleration, gravity_tensor

using .Polyhedron_Gravitation_Big:
    build_polyhedron_big, potential_big, acceleration_big, gravity_tensor_big

using Printf, Dates, LinearAlgebra, Base.Threads

# ----------------------------------------------------------------------
# Setup and output
# ----------------------------------------------------------------------
setprecision(BigFloat, 250)  # high-precision arithmetic (≈75 decimal digits)

outfile = open("polyhedron_gravity_comparison.txt", "w")

function logf(msg)
    println(msg)
    println(outfile, msg)
end

logf("\n--- Polyhedron Gravity Comparison Test ---")
logf("Date: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
logf("Number of Julia threads: $(Threads.nthreads())")
logf("BigFloat precision: $(precision(BigFloat)) bits")
logf("---------------------------------------------------------------")

# ----------------------------------------------------------------------
# Geometry Definition (Simple Tetrahedron)
# ----------------------------------------------------------------------
vertices = [
    0.0 0.0 0.0;
    1.0 0.0 0.0;
    0.0 1.0 0.0;
    0.0 0.0 1.0
]

faces = [
    1 3 2;
    1 2 4;
    1 4 3;
    2 3 4
]

# ----------------------------------------------------------------------
# Model Initialization
# ----------------------------------------------------------------------
logf("\nInitializing models...")

model_big = build_polyhedron_big(BigFloat.(vertices), faces;
    G=BigFloat(1.0), rho=BigFloat(1.0), eps=BigFloat(0))

model_f64 = build_polyhedron(Float64.(vertices), faces;
    G=1.0, rho=1.0, eps=0.0)

logf("Models successfully initialized.")

# ----------------------------------------------------------------------
# Test Points (Exterior, Interior, Vertex)
# ----------------------------------------------------------------------
test_points = [
    2.0 2.0 2.0;
    0.25 0.25 0.25;
    0.0 0.0 0.0
]

# ----------------------------------------------------------------------
# Evaluation Loop
# ----------------------------------------------------------------------
for i in 1:size(test_points, 1)
    p_f64 = test_points[i, :]
    p_big = BigFloat.(p_f64)

    logf("\n============================================================")
    msg = @sprintf("Point: [%.6f, %.6f, %.6f]", p_f64[1], p_f64[2], p_f64[3])
    logf(msg)
    logf("============================================================")

    # --- BigFloat Computation ---
    t1 = time()
    U_big = potential_big(model_big, p_big)
    g_big = acceleration_big(model_big, p_big)
    Γ_big = gravity_tensor_big(model_big, p_big)
    t_big = time() - t1

    # --- Float64 Computation ---
    t2 = time()
    U_f64 = potential(model_f64, p_f64)
    g_f64 = acceleration(model_f64, p_f64)
    Γ_f64 = gravity_tensor(model_f64, p_f64)
    t_f64 = time() - t2

    # --- Differences ---
    ΔU = Float64(abs(U_big - BigFloat(U_f64)))
    Δg = maximum(abs.(Float64.(g_big) .- g_f64))
    ΔΓ = maximum(abs.(Float64.(Γ_big) .- Γ_f64))

    # ------------------------------------------------------------------
    # Logging Results
    # ------------------------------------------------------------------
    logf("\nPotential Comparison:")
    @printf(outfile, "  BigFloat: %.12e\n  Float64:  %.12e\n  ΔU: %.3e\n",
        Float64(U_big), U_f64, ΔU)

    logf("\nAcceleration Comparison (g):")
    for j in 1:3
        @printf(outfile,
            "  g[%d] BigFloat: % .12e  Float64: % .12e  Δ: %.3e\n",
            j, Float64(g_big[j]), g_f64[j], abs(Float64(g_big[j]) - g_f64[j]))
    end

    logf("\nGravity Tensor Comparison (Γ):")
    for r in 1:3, c in 1:3
        @printf(outfile,
            "  Γ[%d,%d] BigFloat: % .12e  Float64: % .12e  Δ: %.3e\n",
            r, c, Float64(Γ_big[r,c]), Γ_f64[r,c],
            abs(Float64(Γ_big[r,c]) - Γ_f64[r,c]))
    end

    logf("\nComputation Time:")
    @printf(outfile, "  BigFloat: %.3f s\n  Float64: %.3f s\n  Ratio: %.2fx\n",
        t_big, t_f64, t_big / t_f64)

    logf("------------------------------------------------------------")
end

close(outfile)

println("\n--- Comparison complete! ---")
println("Results saved to 'polyhedron_gravity_comparison.txt'")
