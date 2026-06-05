# ===============================================================
# Example_01_Gravitational_Field_Evaluation.jl
#
# Purpose:
#   Evaluate and record the full gravitational field (potential,
#   acceleration, and tensor) of a unit tetrahedron using the
#   PolyGravitation framework.
#
# Description:
#   Results are printed in machine-precision format and also saved
#   as a structured text document for publication or validation.
# ===============================================================

using LinearAlgebra
using StaticArrays
using Printf
using Statistics

# ---------------------------------------------------------------
# Load PolyGravitation framework
# ---------------------------------------------------------------
include(joinpath("PolyGravitation", "Polyhedron_Gravitation.jl"))
using .Polyhedron_Gravitation

# ---------------------------------------------------------------
# 1. Define Polyhedron Geometry (Unit Tetrahedron)
# ---------------------------------------------------------------
vertices = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
]

faces = [
    [1, 3, 2],
    [1, 2, 4],
    [1, 4, 3],
    [2, 3, 4]
]

# Helper function: convert vector list → N×3 matrix
to_matrix(vectors) = permutedims(hcat(vectors...))

# ---------------------------------------------------------------
# 2. Define Constants
# ---------------------------------------------------------------
G = 1.0
rho = 1.0

# ---------------------------------------------------------------
# 3. Define Test Points
# ---------------------------------------------------------------
p_interior  = mean(to_matrix(vertices); dims=1)[:]   # centroid (0.25, 0.25, 0.25)
p_exterior  = [2.0, 2.0, 2.0]                        # outside
p_near_face = [0.0, 0.0, 0.0]                        # near base

test_points = [p_interior, p_exterior, p_near_face]
labels = ["Interior (Centroid)", "Exterior (2,2,2)", "Near Base Face"]

# ---------------------------------------------------------------
# 4. Build Model and Compute Fields
# ---------------------------------------------------------------
model = build_polyhedron(
    to_matrix(vertices),
    to_matrix(faces),
    G = G,
    rho = rho,
    eps = 0.0
)

potentials   = [potential(model, p) for p in test_points]
accelerations = [acceleration(model, p) for p in test_points]
tensors      = [gravity_tensor(model, p) for p in test_points]

# ---------------------------------------------------------------
# 5. Print and Save Structured Output
# ---------------------------------------------------------------
output_file = "Example_01_Gravitational_Field_Evaluation.txt"
open(output_file, "w") do io
    function writeboth(str)
        print(str)
        write(io, str)
    end

    writeboth("======================================================================\n")
    writeboth("Gravitational Field Evaluation for Unit Tetrahedron\n")
    writeboth("======================================================================\n")
    writeboth("Sign Convention: Potential is negative. ∇²U = -4πGρ.\n\n")

    for (i, label) in enumerate(labels)
        p = test_points[i]
        U = potentials[i]
        g = accelerations[i]
        T = tensors[i]
        trT = tr(T)

        writeboth("----------------------------------------------------------------------\n")
        writeboth(@sprintf("## Point %d: %s at [%g %g %g] ##\n\n", i, label, p[1], p[2], p[3]))
        writeboth(@sprintf("Potential U: %.15g\n", U))
        writeboth(@sprintf("Acceleration g = -∇U: [% .8f % .8f % .8f]\n",
                            g[1], g[2], g[3]))

        writeboth("Tensor T = ∇∇U (units of G*ρ):\n")
        for r in 1:3
            formatted_row = [@sprintf("% .8f", T[r, c]) for c in 1:3]
            writeboth("[" * join(formatted_row, " ") * "]\n")
        end


        writeboth(@sprintf("Trace(T)/(Gρ): %.15g\n\n", trT / (G * rho)))
    end
end

println("======================================================================")
println("Gravitational Field Evaluation for Unit Tetrahedron")
println("======================================================================")
println("Sign Convention: Potential is negative. ∇²U = -4πGρ.\n")

for (i, label) in enumerate(labels)
    p = test_points[i]
    U = potentials[i]
    g = accelerations[i]
    T = tensors[i]
    trT = tr(T)

    println("----------------------------------------------------------------------")
    @printf("## Point %d: %s at [%g %g %g] ##\n\n", i, label, p[1], p[2], p[3])
    @printf("Potential U: %.15g\n", U)
    @printf("Acceleration g = -∇U: [% .8f % .8f % .8f]\n", g[1], g[2], g[3])
    println("Tensor T = ∇∇U (units of G*ρ):")
    for r in 1:3
        formatted_row = [@sprintf("% .8f", T[r, c]) for c in 1:3]
        println("[" * join(formatted_row, " ") * "]")
    end
    @printf("Trace(T)/(Gρ): %.15g\n\n", trT / (G * rho))
end

println("Results written to: $(abspath(output_file))")
