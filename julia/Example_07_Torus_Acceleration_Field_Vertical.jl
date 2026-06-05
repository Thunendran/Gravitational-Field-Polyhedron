###############################################################
# Example 07: Gravitational Acceleration Field of a Torus (x = 0 Plane)
###############################################################

using LinearAlgebra
using Printf
using PyCall
using PyPlot

# Import Polyhedron Gravitation module
include(joinpath("PolyGravitation", "Polyhedron_Gravitation.jl"))
using .Polyhedron_Gravitation

###############################################################
# 1) Torus Mesh Generation
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

# Disable face orientation for torus
model = build_polyhedron(Matrix(V), Matrix(F);
                         G=1.0, rho=1.0, eps=0.0, orient_faces=false)
println("Torus model initialized.")

###############################################################
# 3) Define Grid in Vertical Plane (x = 0)
###############################################################
grid_res_y, grid_res_z = 21, 21
y_limit = R1 + R2 + 0.5
z_limit = R2 + 2.5

y_vals = collect(range(-y_limit, y_limit; length=grid_res_y))
z_vals = collect(range(-z_limit, z_limit; length=grid_res_z))
YY = repeat(y_vals', grid_res_z, 1)
ZZ = repeat(z_vals, 1, grid_res_y)

# Points lie in x=0 plane
grid_points = hcat(zeros(length(y_vals)*length(z_vals)),
                   vec(YY), vec(ZZ))

###############################################################
# 4) Compute Gravitational Acceleration
###############################################################
println("Computing gravitational acceleration field...")
accel = Polyhedron_Gravitation.acceleration(model, grid_points)
U = reshape(accel[:, 2], size(YY))   # g_y
Vv = reshape(accel[:, 3], size(ZZ))  # g_z
println("Computation complete.")

###############################################################
# 5) Plot the Quiver Field
###############################################################
mpl_patches = pyimport("matplotlib.patches")
Circle = mpl_patches."Circle"

fig, ax = subplots(figsize=(9, 8))
ax.quiver(YY, ZZ, U, Vv, color="red", scale=80, pivot="middle", width=0.002)

# --- Draw torus cross-section (two circles) ---
circle_left  = Circle((-R1, 0.0), R2, edgecolor="black", facecolor="none",
                      linestyle="-", linewidth=1.0)
circle_right = Circle((R1, 0.0), R2, edgecolor="black", facecolor="none",
                      linestyle="-", linewidth=1.0)
ax.add_patch(circle_left)
ax.add_patch(circle_right)

# --- Labels, formatting ---
ax.set_title(raw"Gravitational Acceleration in the Plane $x=0$", fontsize=24, color="black")
ax.set_xlabel("y-axis", fontsize=24, color="black")
ax.set_ylabel("z-axis", fontsize=24, color="black")
ax.tick_params(axis="both", colors="black")

ax.set_xlim([-3.5, 3.5])
ax.set_ylim([-3.0, 3.0])
ax.set_aspect("equal")
ax.grid(false)

tight_layout()
savefig("gravitational_acceleration_field_torus_vertical.png", dpi=300, bbox_inches="tight")
println("Plot saved to gravitational_acceleration_field_torus_vertical.png")
show()
