###############################################################
# Example 03: Analytical Laplacian of a Concave L-Shaped Body
###############################################################

using LinearAlgebra
using Printf
using PyCall
using PyPlot

# Include PolyhedronGravity model
include(joinpath("PolyGravitation", "Polyhedron_Gravitation.jl"))
using .Polyhedron_Gravitation

###############################################################
# 1. Build Concave L-Shaped Mesh
###############################################################

function build_L_mesh()
    occ = Set([(0,0,0), (1,0,0), (0,1,0)])  # occupied cubes

    dirs = [
        ("-x", (-1,0,0), [-1,0,0.]),
        ("+x", (1,0,0),  [1,0,0.]),
        ("-y", (0,-1,0), [0,-1,0.]),
        ("+y", (0,1,0),  [0,1,0.]),
        ("-z", (0,0,-1), [0,0,-1.]),
        ("+z", (0,0,1),  [0,0,1.])
    ]

    function face_quad(x,y,z,name)
        if name=="-x"; return [[x,y,z],[x,y,z+1],[x,y+1,z+1],[x,y+1,z]]; end
        if name=="+x"; return [[x+1,y,z],[x+1,y+1,z],[x+1,y+1,z+1],[x+1,y,z+1]]; end
        if name=="-y"; return [[x,y,z],[x+1,y,z],[x+1,y,z+1],[x,y,z+1]]; end
        if name=="+y"; return [[x,y+1,z],[x,y+1,z+1],[x+1,y+1,z+1],[x+1,y+1,z]]; end
        if name=="-z"; return [[x,y,z],[x,y+1,z],[x+1,y+1,z],[x+1,y,z]]; end
        if name=="+z"; return [[x,y,z+1],[x+1,y,z+1],[x+1,y+1,z+1],[x,y+1,z+1]]; end
    end

    verts = Vector{Vector{Float64}}()
    vidx = Dict{Tuple{Int,Int,Int},Int}()

    get_idx(p) = get!(vidx, p) do
        push!(verts, [Float64(p[1]), Float64(p[2]), Float64(p[3])])
        length(verts)
    end

    faces = Vector{NTuple{3,Int}}()
    for (x,y,z) in occ
        for (name, δ, outward) in dirs
            nx, ny, nz = x+δ[1], y+δ[2], z+δ[3]
            if (nx,ny,nz) in occ; continue; end
            quad = face_quad(x,y,z,name)
            triA = (get_idx(Tuple(quad[1])), get_idx(Tuple(quad[2])), get_idx(Tuple(quad[3])))
            triB = (get_idx(Tuple(quad[1])), get_idx(Tuple(quad[3])), get_idx(Tuple(quad[4])))

            for tri in (triA, triB)
                p0,p1,p2 = verts[tri[1]], verts[tri[2]], verts[tri[3]]
                n = cross(p1 .- p0, p2 .- p0)
                if dot(n, outward) < 0
                    tri = (tri[1], tri[3], tri[2])
                end
                push!(faces, tri)
            end
        end
    end
    return verts, faces
end

V, F = build_L_mesh()
@printf("Concave L mesh created: %d vertices, %d faces\n", length(V), length(F))

###############################################################
# 2. Instantiate Gravity Model
###############################################################

Vmat = reduce(vcat, permutedims.([collect(v) for v in V]))
Fmat = reduce(vcat, permutedims.([collect(f) for f in F]))
model = build_polyhedron(Vmat, Fmat; G=1.0, rho=1.0, eps=0.0)
println("PolyhedronGravity model initialized.")

###############################################################
# 3. Analytical Laplacian on a 2D Grid
###############################################################

z_fixed = 0.5
x_range = (-0.5, 2.5)
y_range = (-0.5, 2.5)
resolution = 300

xs = range(x_range[1], x_range[2], length=resolution)
ys = range(y_range[1], y_range[2], length=resolution)
X = repeat(collect(xs)', resolution, 1)
Y = repeat(collect(ys), 1, resolution)
grid_points = hcat(vec(X), vec(Y), fill(z_fixed, resolution^2))

@printf("Computing Laplacian on a %dx%d grid at z=%.3f...\n", resolution, resolution, z_fixed)
tensors = gravity_tensor(model, grid_points)
laplacian = [tr(reshape(tensors[i,:,:],(3,3))) for i in 1:size(grid_points,1)]
L = reshape(laplacian, (resolution, resolution)) ./ (model.G * model.rho)
println("Calculation complete.")

###############################################################
# 4. Cross-Section Outline Construction
###############################################################

function section_segments_from_mesh(Vm::Vector{Vector{Float64}}, Fm::Vector{NTuple{3,Int}}, z0::Float64; tol=0.0)
    segs = Vector{Tuple{Tuple{Float64,Float64},Tuple{Float64,Float64}}}()
    for (i,j,k) in Fm
        tri = [Vm[i], Vm[j], Vm[k]]
        zvals = [p[3] for p in tri]
        pts = Vector{Vector{Float64}}()
        for (a,b) in ((1,2),(2,3),(3,1))
            z1, z2 = zvals[a], zvals[b]
            if (z1 - z0)*(z2 - z0) < -tol
                t = (z0 - z1)/(z2 - z1)
                p = tri[a] .+ t .* (tri[b] .- tri[a])
                push!(pts, p[1:2])
            end
        end
        if length(pts) == 2
            push!(segs, (Tuple(pts[1]), Tuple(pts[2])))
        end
    end
    return segs
end

function stitch_segments_to_polylines(segs; tol=1e-9)
    if isempty(segs); return []; end
    function key(p)
        (round(p[1]/tol), round(p[2]/tol))
    end
    pts = Dict{Tuple{Int,Int},Vector{Float64}}()
    adj = Dict{Tuple{Int,Int},Vector{Tuple{Int,Int}}}()
    for (a,b) in segs
        ka, kb = key(a), key(b)
        pts[ka] = collect(a); pts[kb] = collect(b)
        push!(get!(adj, ka, Tuple{Int,Int}[]), kb)
        push!(get!(adj, kb, Tuple{Int,Int}[]), ka)
    end

    visited = Set{Tuple{Int,Int}}()
    polys = Vector{Matrix{Float64}}()
    for s in keys(adj)
        if s in visited; continue; end
        line = [pts[s]]
        prev, cur = s, adj[s][1]
        while cur != s
            push!(visited, cur)
            push!(line, pts[cur])
            nxts = filter(x -> x != prev, adj[cur])
            if isempty(nxts); break; end
            prev, cur = cur, nxts[1]
        end
        push!(line, pts[s])
        push!(polys, reduce(vcat, permutedims.(line)))
    end
    return polys
end

segs = section_segments_from_mesh(V, F, z_fixed)
polys = stitch_segments_to_polylines(segs)
println("Outline construction complete.")

###############################################################
# 5. Plot Analytical Laplacian Field (robust PyCall usage)
###############################################################

# Use the same approach as Example 2 (this avoids PyCall kwarg pitfalls)
np = pyimport("numpy")
plt = pyimport("matplotlib.pyplot")
mpl_colors = pyimport("matplotlib.colors")
ListedColormap = mpl_colors["ListedColormap"]
BoundaryNorm = mpl_colors["BoundaryNorm"]

# --- Build colormap identical to Python version ---
N_colors = 50
colors = Any[(0.1, 0.4, 1.0)]
for i in 1:(N_colors - 2)
    push!(colors, (i / (N_colors - 3), 1.0, 1.0 - i / (N_colors - 3)))
end
push!(colors, (1.0, 0.0, 0.0))
cmap_discrete = ListedColormap(colors)

# Range for ∇²U/(Gρ): should be −4π inside, 0 outside
vmin = -4 * π
vmax = 0.0
bounds = np.array(LinRange(vmin, vmax, N_colors + 1))  # explicit NumPy array
# Access number of colors via .N (attribute) — safer than [:N]
norm = BoundaryNorm(bounds, Int(cmap_discrete[:N]))

# Create figure using plt.* (like Example 2)
fig_ax = plt.subplots(1, 1; figsize=(10, 8), constrained_layout=true)
fig = fig_ax[1]; ax = fig_ax[2]
fig[:suptitle](@sprintf("Analytical Laplacian ∇²U / (Gρ) for Concave L-Body at z = %.3f", z_fixed), fontsize=16)

# Explicitly convert the field to NumPy for imshow
L_np = np.array(L)

# Plot the Laplacian field
im = ax[:imshow](
    L_np;
    extent=(x_range[1], x_range[2], y_range[1], y_range[2]),
    origin="lower",
    cmap=cmap_discrete,
    norm=norm,
)

ax[:set_title]("Calculated from Trace of Analytical Tensor", fontsize=12)
ax[:set_xlabel]("x-axis")
ax[:set_ylabel]("y-axis")

# Overlay each closed polyline outline (convert to NumPy arrays)
for poly in polys
    x_np = np.array(poly[:, 1])
    y_np = np.array(poly[:, 2])
    ax[:plot](x_np, y_np, color="white", linewidth=2, linestyle="-")
end

# Colorbar — consistent with Example 2 style
cb = fig[:colorbar](im; ax=ax, fraction=0.046, pad=0.04)
# If you want even-integer ticks only, you could use (uncomment):
# ticks_np = np.arange(Int(floor(vmin/2)*2), Int(ceil(vmax)), 2)
# cb[:set_ticks](ticks_np)
# cb[:ax][:set_yticklabels]([string(t) for t in ticks_np])
cb[:set_label]("∇²U / (Gρ)", fontsize=20)

plt[:savefig]("concave_L_mesh_laplacian_analytical.png", dpi=200)
println("Plot saved to concave_L_mesh_laplacian_analytical.png")
plt[:show]()



###############################################################
# 6. Potential Checks at Specific Points
###############################################################

pts_to_check = [
    [0.5, 0.5, 0.5],
    [1.5, 1.5, 0.5],
    [3.0, 3.0, 2.0]
]

U = potential(model, reduce(vcat, permutedims.(pts_to_check)))
println("\n--- Potential at Specific Points ---")
for (p, u) in zip(pts_to_check, U)
    @printf("  U(%.2f, %.2f, %.2f) = %.8f\n", p[1], p[2], p[3], u)
end
