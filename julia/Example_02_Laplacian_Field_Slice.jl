###############################################################
# Example 02: Laplacian Slice of a Tetrahedral Polyhedron
# Equivalent to the Python version using PolyhedronGravity.
###############################################################

using LinearAlgebra
using Printf
using PyCall
using PyPlot

# Import the Julia module
include(joinpath("PolyGravitation", "Polyhedron_Gravitation.jl"))
using .Polyhedron_Gravitation

###############################################################
# 1. Helper Functions
###############################################################

# --- 2D cross product (scalar z-component) ---
cross2D(o, a, b) = (a[1]-o[1]) * (b[2]-o[2]) - (a[2]-o[2]) * (b[1]-o[1])

# --- Convex hull (monotone chain, same as Python version) ---
function convex_hull_2d(points_xy::Matrix{Float64})
    P = copy(points_xy)
    if size(P, 1) < 3
        return P
    end

    # Sort points by x, then y (same as np.lexsort)
    P_sorted = sort(collect(eachrow(P)), by = r -> (r[1], r[2]))
    P = reduce(vcat, permutedims.(P_sorted))

    lower = Matrix{Float64}(undef, 0, 2)
    for i in 1:size(P, 1)
        while size(lower, 1) ≥ 2 && cross2D(lower[end-1, :], lower[end, :], P[i, :]) ≤ 0
            lower = lower[1:end-1, :]
        end
        lower = vcat(lower, P[i, :]')
    end

    upper = Matrix{Float64}(undef, 0, 2)
    for i in size(P, 1):-1:1
        while size(upper, 1) ≥ 2 && cross2D(upper[end-1, :], upper[end, :], P[i, :]) ≤ 0
            upper = upper[1:end-1, :]
        end
        upper = vcat(upper, P[i, :]')
    end

    hull = vcat(lower[1:end-1, :], upper[1:end-1, :])
    return hull
end

# --- Edge-plane intersection ---
function edge_intersection(v1::Vector{Float64}, v2::Vector{Float64}, z_plane::Float64)
    z1, z2 = v1[3], v2[3]
    if (z1 > z_plane && z2 < z_plane) || (z1 < z_plane && z2 > z_plane)
        t = (z_plane - z1) / (z2 - z1)
        return v1 .+ t .* (v2 .- v1)
    end
    return nothing
end

# --- Compute intersection polygon safely ---
function compute_intersection_polygon(vertices::Vector{Vector{Float64}}, z_plane::Float64)
    edges = [(1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
    intersection_points = Matrix{Float64}(undef, 0, 2)
    for (i, j) in edges
        pt = edge_intersection(vertices[i], vertices[j], z_plane)
        if pt !== nothing
            intersection_points = vcat(intersection_points, pt[1:2]')
        end
    end
    boundary_xy = convex_hull_2d(intersection_points)
    boundary_xy = vcat(boundary_xy, boundary_xy[1:1, :])  # close polygon
    return boundary_xy
end

###############################################################
# 2. Define Polyhedron Geometry
###############################################################

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

println("PolyhedronGravity model initialized.")

vertices_mat = reduce(vcat, permutedims.(vertices))
faces_mat    = reduce(vcat, permutedims.(faces))
model = build_polyhedron(vertices_mat, faces_mat; G=1.0, rho=1.0, eps=0.0)

###############################################################
# 3. Compute Laplacian on a 2D Grid
###############################################################

z_fixed = 0.25
x_range = (-0.2, 1.0)
y_range = (-0.2, 1.0)
resolution = 200

xs = range(x_range[1], x_range[2], length=resolution)
ys = range(y_range[1], y_range[2], length=resolution)
X = repeat(collect(xs)', resolution, 1)
Y = repeat(collect(ys), 1, resolution)
grid_points = hcat(vec(X), vec(Y), fill(z_fixed, resolution^2))

@printf("Computing Laplacian on a %dx%d grid at z=%.3f...\n", resolution, resolution, z_fixed)

tensors_ana = gravity_tensor(model, grid_points)
laplacian_ana = [tr(reshape(tensors_ana[i, :, :], (3,3))) for i in 1:size(grid_points,1)]
L_ana = reshape(laplacian_ana, (resolution, resolution)) ./ (model.G * model.rho)

println("Calculation complete.")

###############################################################
# 4. Compute Boundary Polygon
###############################################################

boundary_xy = compute_intersection_polygon(vertices, z_fixed)

###############################################################
# 5. Plot Analytical Laplacian Field
###############################################################

np = pyimport("numpy")
mpl_colors = pyimport("matplotlib.colors")
ListedColormap = mpl_colors["ListedColormap"]
BoundaryNorm = mpl_colors["BoundaryNorm"]

N_colors = 50
colors = Any[(0.1, 0.4, 1.0)]
for i in 1:(N_colors-2)
    push!(colors, (i/(N_colors-3), 1.0, 1.0 - i/(N_colors-3)))
end
push!(colors, (1.0, 0.0, 0.0))
cmap_discrete = ListedColormap(colors)
vmin, vmax = -4 * π, 0.0
bounds = np.linspace(vmin, vmax, N_colors + 1)
norm = BoundaryNorm(bounds, cmap_discrete[:N])

fig, ax = subplots(figsize=(8,7), constrained_layout=true)
fig.suptitle(@sprintf("Analytical Laplacian ∇²U / (Gρ) on a Plane at z = %.3f", z_fixed), fontsize=16)

im = ax.imshow(L_ana,
    extent=(x_range[1], x_range[2], y_range[1], y_range[2]),
    origin="lower", cmap=cmap_discrete, norm=norm)

ax.set_title("Calculated from Trace of Analytical Tensor", fontsize=12)
ax.set_xlabel("x-axis")
ax.set_ylabel("y-axis")

# --- Overlay the white boundary polygon ---
ax.plot(boundary_xy[:,1], boundary_xy[:,2], color="white", linewidth=2, linestyle="-")

fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

savefig("laplacian_tetra_slice_analytical.png", dpi=200)
println("Plot saved to laplacian_tetra_slice_analytical.png")

show()
