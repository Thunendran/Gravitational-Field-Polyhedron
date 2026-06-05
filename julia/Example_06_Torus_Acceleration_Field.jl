###############################################################
# Example 06: Gravitational Acceleration Field of a Torus (z = 0 Plane)
###############################################################

using LinearAlgebra
using Printf
using PyCall
using PyPlot

# Import the Polyhedron Gravitation module
include(joinpath("PolyGravitation", "Polyhedron_Gravitation.jl"))
using .Polyhedron_Gravitation

###############################################################
# 1) Torus Mesh Generation (same structure as Example 04)
###############################################################

function create_torus_tri_mesh(R1::Float64, R2::Float64; n_u::Int=100, n_v::Int=50)
    u = range(0, 2π; length=n_u+1)[1:end-1]
    v = range(0, 2π; length=n_v+1)[1:end-1]

    X = [(R1 + R2*cos(vv)) * cos(uu) for vv in v, uu in u]
    Y = [(R1 + R2*cos(vv)) * sin(uu) for vv in v, uu in u]
    Z = [R2 * sin(vv) for vv in v, uu in u]
    V = hcat(vec(X), vec(Y), vec(Z))

    idx(i, j) = (i-1)*n_u + j
    faces = Matrix{Int64}(undef, 2*n_v*n_u, 3)
    t = 1
    for i in 1:n_v
        ip = (i == n_v) ? 1 : i+1
        for j in 1:n_u
            jp = (j == n_u) ? 1 : j+1
            v0, v1, v2, v3 = idx(i,j), idx(i,jp), idx(ip,j), idx(ip,jp)
            faces[t,   :] .= [v0, v2, v1]
            faces[t+1, :] .= [v1, v2, v3]
            t += 2
        end
    end
    return V, faces
end

###############################################################
# 2) Setup Torus Geometry and Model
###############################################################

R1, R2 = 2.0, 1.0
a = R1 + R2 + 0.5
V, F = create_torus_tri_mesh(R1, R2; n_u=101, n_v=51)

# Build the model (disable face orientation for torus)
model = build_polyhedron(Matrix(V), Matrix(F);
                         G=1.0, rho=1.0, eps=0.0, orient_faces=false)
println("Torus model initialized.")

###############################################################
# 3) Define Grid in Plane z = 0
###############################################################

z_fixed = 0.0
x_vals = collect(range(-a, a; length=19))
y_vals = collect(range(-a, a; length=19))
X = repeat(x_vals', length(y_vals), 1)
Y = repeat(y_vals, 1, length(x_vals))
grid_points = hcat(vec(X), vec(Y), fill(z_fixed, length(x_vals)*length(y_vals)))

###############################################################
# 4) Compute Gravitational Acceleration Field
###############################################################

println("Computing gravitational acceleration field...")
accel = Polyhedron_Gravitation.acceleration(model, grid_points)
U = reshape(accel[:, 1], size(X))   # g_x
Vv = reshape(accel[:, 2], size(Y))  # g_y
println("Computation complete.")

###############################################################
# 5) Plot the Vector Field (Quiver)
###############################################################

mpl_patches = pyimport("matplotlib.patches")
Circle = mpl_patches."Circle"

fig, ax = subplots(figsize=(9, 8))
ax.quiver(X, Y, U, Vv, color="red", scale=80, pivot="middle", width=0.002)

# --- Draw torus cross-section as two circles ---
inner_radius = R1 - R2
outer_radius = R1 + R2
circle_inner = Circle((0.0, 0.0), inner_radius, edgecolor="black", facecolor="none", linestyle="-", linewidth=1.5)
circle_outer = Circle((0.0, 0.0), outer_radius, edgecolor="black", facecolor="none", linestyle="-", linewidth=1.5)
ax.add_patch(circle_inner)
ax.add_patch(circle_outer)

# --- Labels and Formatting ---
ax.set_title(raw"Gravitational Acceleration in the Plane $z=0$", fontsize=24, color="black")
ax.set_xlabel("x", fontsize=24, color="black")
ax.set_ylabel("y", fontsize=24, color="black")
ax.tick_params(axis="both", colors="black")
ax.set_xlim([-a, a])
ax.set_ylim([-a, a])
ax.set_aspect("equal")
ax.grid(false)

tight_layout()
savefig("gravitational_acceleration_field_torus.png", dpi=300, bbox_inches="tight")
println("Plot saved to gravitational_acceleration_field_torus.png")
show()
