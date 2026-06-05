###############################################################
# Example 05: Torus Laplacian – Vertical (y–z) Slice at x = 0
# Converted faithfully from the working Python version.
###############################################################

using LinearAlgebra
using Printf
using PyCall
using PyPlot

# Bring in the Julia gravitational module
include(joinpath("PolyGravitation", "Polyhedron_Gravitation.jl"))
using .Polyhedron_Gravitation   # exports build_polyhedron, gravity_tensor

###############################################################
# 1) Torus Mesh Generation (identical to Example 04’s correct version)
###############################################################

function create_torus_tri_mesh(R1::Float64, R2::Float64; n_u::Int=100, n_v::Int=50)
    u = range(0, 2π; length=n_u+1)[1:end-1]
    v = range(0, 2π; length=n_v+1)[1:end-1]

    X = [(R1 + R2*cos(vv)) * cos(uu) for vv in v, uu in u]
    Y = [(R1 + R2*cos(vv)) * sin(uu) for vv in v, uu in u]
    Z = [R2 * sin(vv) for vv in v, uu in u]
    V = hcat(vec(X), vec(Y), vec(Z))   # (n_v*n_u)×3 matrix

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
# 2) Setup Geometry, Grid, and Model
###############################################################

R1, R2 = 2.0, 1.0
V, F = create_torus_tri_mesh(R1, R2; n_u=100, n_v=50)

model = build_polyhedron(Matrix(V), Matrix(F);
                         G=1.0, rho=1.0, eps=0.0, orient_faces=false)
println("Model initialized.")

# --- Define the vertical calculation grid (y–z plane at x=0) ---
ny, nz = 251, 126
y_range = collect(range(-3.5, 3.5; length=ny))
z_range = collect(range(-1.5, 1.5; length=nz))
Y = repeat(y_range', nz, 1)
Z = repeat(z_range, 1, ny)
grid_points = hcat(zeros(ny*nz), vec(Y), vec(Z))

@printf("Defined a %dx%d calculation grid on the x=0 plane.\n", ny, nz)

###############################################################
# 3) Compute Laplacian = −trace(Γ)
###############################################################

println("Computing Laplacian on the vertical grid...")
Γ = gravity_tensor(model, grid_points)
lap_vals = [tr(@view Γ[i, :, :]) for i in 1:size(grid_points,1)]
L_grid = reshape(lap_vals, nz, ny) ./ (model.G * model.rho)
println("Computation complete.")

###############################################################
# 4) Plotting
###############################################################

np = pyimport("numpy")
mpl_colors = pyimport("matplotlib.colors")
mpl_patches = pyimport("matplotlib.patches")
ListedColormap = mpl_colors."ListedColormap"
BoundaryNorm = mpl_colors."BoundaryNorm"
Circle = mpl_patches."Circle"

# --- Custom discrete colormap ---
N = 50
colors = Any[(0.1, 0.4, 1.0)]
for i in 1:(N-2)
    r = (i-1)/(N-3)
    push!(colors, (r, 1.0, 1.0 - r))
end
push!(colors, (1.0, 0.0, 0.0))
cmap_discrete = ListedColormap(colors)
vmin, vmax = -4π, 0.0
bounds = np.linspace(vmin, vmax, N + 1)
norm = BoundaryNorm(bounds, cmap_discrete[:N])

fig, ax = subplots(figsize=(10, 7))
im = ax.imshow(L_grid,
               extent=[-3.5, 3.5, -1.5, 1.5],
               origin="lower", cmap=cmap_discrete, norm=norm,
               interpolation="nearest")

# --- Colorbar ---
cbar = colorbar(im, ax=ax, boundaries=bounds, fraction=0.046, pad=0.04, shrink=0.40)
cbar[:set_ticks]([0.0, -4.0, -8.0, -12.0])
cbar[:set_label](raw"$\nabla^2 U / (G\rho)$", fontsize=14)

# --- White circles marking torus cross-section ---
circle_left  = Circle((-R1, 0.0), R2, edgecolor="white", facecolor="none", linestyle="-", linewidth=2.0)
circle_right = Circle(( R1, 0.0), R2, edgecolor="white", facecolor="none", linestyle="-", linewidth=2.0)
ax.add_patch(circle_left)
ax.add_patch(circle_right)

# --- Labels and Annotations ---
ax.set_title(raw"Analytical Laplacian on a 251 × 126 grid in the plane $x = 0$", fontsize=20)
ax.set_xlabel("y-axis", fontsize=20)
ax.set_ylabel("z-axis", fontsize=20)
ax.set_aspect("equal")
ax.grid(false)

ax.text(-R1, 0.0, raw"$\nabla^2 U = -4\pi$", color="white",
       fontsize=16, ha="center", va="center", fontweight="bold")
ax.text(R1, 0.0, raw"$\nabla^2 U = -4\pi$", color="white",
       fontsize=16, ha="center", va="center", fontweight="bold")
ax.text(0.0, 1.2, raw"$\nabla^2 U = 0$", color="white",
       fontsize=16, ha="center", va="center", fontweight="bold")

tight_layout()
savefig("laplacian_torus_vertical_slice_corrected.png", dpi=300)
println("Plot saved to laplacian_torus_vertical_slice_corrected.png")
show()
