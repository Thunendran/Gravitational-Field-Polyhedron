###############################################################
# Example 04: Torus Laplacian Field (Julia, PyPlot)
# Mirrors the Python example: watertight torus mesh, gravity tensor,
# Laplacian = -trace(Γ), and discrete colormap with integer ticks.
###############################################################

using LinearAlgebra
using Printf
using PyCall
using PyPlot

# Bring in your module
include(joinpath("PolyGravitation", "Polyhedron_Gravitation.jl"))
using .Polyhedron_Gravitation   # exports: build_polyhedron, gravity_tensor

###############################################################
# 1) Torus Mesh Generation (watertight)
###############################################################

"""
create_torus_tri_mesh(R1, R2; n_u=100, n_v=50) -> (V, F)

- R1: major radius
- R2: minor radius
- n_u: samples around the z-axis (longitudinal)
- n_v: samples around the tube (meridional)

Returns:
- V: Matrix{Float64} of size (n_v*n_u)×3
- F: Matrix{Int64} of size (2*n_v*n_u)×3   (1-based indices)
"""
function create_torus_tri_mesh(R1::Float64, R2::Float64; n_u::Int=100, n_v::Int=50)
    u = range(0, 2π; length=n_u+1)[1:end-1]
    v = range(0, 2π; length=n_v+1)[1:end-1]

    X = [(R1 + R2*cos(vv)) * cos(uu) for vv in v, uu in u]
    Y = [(R1 + R2*cos(vv)) * sin(uu) for vv in v, uu in u]
    Z = [R2 * sin(vv) for vv in v, uu in u]

    V = hcat(vec(X), vec(Y), vec(Z))   # N×3 matrix
    idx(i, j) = (i-1)*n_u + j

    faces = Matrix{Int64}(undef, 2*n_v*n_u, 3)
    t = 1
    for i in 1:n_v
        ip = (i == n_v) ? 1 : i+1
        for j in 1:n_u
            jp = (j == n_u) ? 1 : j+1
            v0 = idx(i,  j)
            v1 = idx(i,  jp)
            v2 = idx(ip, j)
            v3 = idx(ip, jp)

            # assign as row vectors, not tuples
            faces[t,   :] .= [v0, v2, v1]
            faces[t+1, :] .= [v1, v2, v3]
            t += 2
        end
    end
    return V, faces
end

###############################################################
# 2) Geometry, Grid, and Model
###############################################################

R1, R2 = 2.0, 1.0
a = R1 + R2 + 1.0
V, F = create_torus_tri_mesh(R1, R2; n_u=100, n_v=50)

# Build the polyhedron model (pass a concrete Matrix, not Adjoint)
# Set orient_faces=false to mirror the Python's "orient_faces=False" usage
model = build_polyhedron(Matrix(V), Matrix(F);
                         G=1.0, rho=1.0, eps=0.0, orient_faces=false)
println("Model initialized.")

# Grid in z=0 plane
grid_res = 251
xs = collect(range(-a, a; length=grid_res))
ys = collect(range(-a, a; length=grid_res))
X = repeat(xs', grid_res, 1)
Y = repeat(ys, 1, grid_res)
grid_points = hcat(vec(X), vec(Y), zeros(grid_res^2))

###############################################################
# 3) Compute Laplacian = -trace(Γ)
###############################################################

println("Computing Laplacian on the grid...")
Γ = gravity_tensor(model, grid_points)           # size: (N, 3, 3)
# Laplacian is trace(Γ)
lap_vals = [tr(@view Γ[i, :, :]) for i in 1:size(grid_points, 1)]
L_grid = reshape(lap_vals, grid_res, grid_res) ./ (model.G * model.rho)
println("Computation complete.")

###############################################################
# 4) Plotting (discrete colormap + integer ticks)
###############################################################

# Matplotlib helpers via PyCall (same as Example 02 style)
np = pyimport("numpy")
mpl_colors = pyimport("matplotlib.colors")
mpl_patches = pyimport("matplotlib.patches")
ListedColormap = mpl_colors."ListedColormap"
BoundaryNorm = mpl_colors."BoundaryNorm"
Circle = mpl_patches."Circle"

# Custom discrete colormap (blue → cyan→yellow → red)
N = 50
colors = Any[(0.1, 0.4, 1.0)]
for i in 1:(N-2)
    r = (i-1)/(N-3)
    push!(colors, (r, 1.0, 1.0-r))
end
push!(colors, (1.0, 0.0, 0.0))
cmap_discrete = ListedColormap(colors)

vmin = -4 * π
vmax = 0.0
bounds = np.linspace(vmin, vmax, N + 1)
norm = BoundaryNorm(bounds, cmap_discrete[:N])

fig, ax = subplots(figsize=(10, 10))
im = ax.imshow(L_grid,
               extent=(-a, a, -a, a),
               origin="lower", cmap=cmap_discrete, norm=norm, interpolation="nearest")

# Colorbar with specific integer ticks (0, -2, -4, ..., -12)
cbar = colorbar(im, ax=ax, boundaries=bounds, fraction=0.046, pad=0.04)
cbar[:set_ticks]([0.0, -2.0, -4.0, -6.0, -8.0, -10.0, -12.0])
cbar[:set_label](raw"$\nabla^2 U / (G\rho)$", fontsize=14)

# Torus cross-section boundary in z=0: two white circles
inner_radius = R1 - R2
outer_radius = R1 + R2
circle_inner = Circle((0.0, 0.0), inner_radius, edgecolor="white", facecolor="none",
                      linestyle="-", linewidth=2.0)
circle_outer = Circle((0.0, 0.0), outer_radius, edgecolor="white", facecolor="none",
                      linestyle="-", linewidth=2.0)
ax.add_patch(circle_inner)
ax.add_patch(circle_outer)

# Titles, labels, annotations
ax.set_title(raw"Analytical Laplacian on a 251 × 251 grid in the plane $z = 0$", fontsize=20)
ax.set_xlabel("x-axis", fontsize=25)
ax.set_ylabel("y-axis", fontsize=25)
ax.set_aspect("equal")
ax.grid(false)

ax.text(R1, 0.0, raw"$\nabla^2 U = -4\pi$", color="white",
       fontsize=16, ha="center", va="center", fontweight="bold")
ax.text(0.0, 0.0, raw"$\nabla^2 U = 0$", color="white",
       fontsize=16, ha="center", va="center", fontweight="bold")
ax.text(a - 2.0, 3.0, raw"$\nabla^2 U = 0$", color="white",
       fontsize=16, ha="center", va="center", fontweight="bold")

tight_layout()
savefig("laplacian_torus_final_integer_ticks.png", dpi=300)
println("Plot saved to laplacian_torus_final_integer_ticks.png")
show()
