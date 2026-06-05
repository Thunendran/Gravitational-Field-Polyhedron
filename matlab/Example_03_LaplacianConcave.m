% =======================================================================
% Example 03 — Analytical Laplacian of a Concave L-Shape
% =======================================================================
% Builds a concave L-shaped polyhedron from three adjacent unit cubes and
% computes the analytical Laplacian field (∇²U / (Gρ)) on a horizontal plane.
%
% Demonstrates:
%   • Procedural mesh generation for concave geometry
%   • Analytical Laplacian from gravity tensor trace
%   • Accurate boundary overlay from stitched edge–plane intersections
%
% Expected results:
%   Inside the solid: ∇²U / (Gρ) ≈ −4π
%   Outside the body: ≈ 0
% =======================================================================

clear; clc; close all;
addpath(genpath(pwd));

%% ----------------------------------------------------------------------
% 1) Build a Concave L-Shaped Mesh (outward-oriented triangles only)
% -----------------------------------------------------------------------
disp('Building concave L-shaped mesh...');

occ = [0 0 0; 1 0 0; 0 1 0];  % three cubes: (0,0,0), (1,0,0), (0,1,0)
dirs = { ...
    '-x', [-1 0 0], [-1 0 0]; ...
    '+x', [ 1 0 0], [ 1 0 0]; ...
    '-y', [ 0 -1 0], [ 0 -1 0]; ...
    '+y', [ 0  1 0], [ 0  1 0]; ...
    '-z', [ 0 0 -1], [ 0 0 -1]; ...
    '+z', [ 0 0  1], [ 0 0  1]  ...
};

% Face quad generator
function quad = face_quad(x, y, z, name)
    switch name
        case '-x'; quad = [x y z; x y z+1; x y+1 z+1; x y+1 z];
        case '+x'; quad = [x+1 y z; x+1 y+1 z; x+1 y+1 z+1; x+1 y z+1];
        case '-y'; quad = [x y z; x+1 y z; x+1 y z+1; x y z+1];
        case '+y'; quad = [x y+1 z; x y+1 z+1; x+1 y+1 z+1; x+1 y+1 z];
        case '-z'; quad = [x y z; x y+1 z; x+1 y+1 z; x+1 y z];
        case '+z'; quad = [x y z+1; x+1 y z+1; x+1 y+1 z+1; x y+1 z+1];
    end
end

verts = [];                       % Nx3
vidx  = containers.Map();         % key -> vertex index
faces = [];                       % Mx3 (int32 indices)

for ci = 1:size(occ,1)
    x = occ(ci,1); y = occ(ci,2); z = occ(ci,3);
    for d = 1:size(dirs,1)
        name    = dirs{d,1};
        delta   = dirs{d,2};
        outward = dirs{d,3};
        nx = x + delta(1); ny = y + delta(2); nz = z + delta(3);
        if any(all(occ == [nx ny nz], 2)), continue; end  % internal face → skip

        quad = face_quad(x,y,z,name);     % 4×3
        triA = quad([1 2 3],:);
        triB = quad([1 3 4],:);

        for tri = {triA, triB}
            T = tri{1};                   % 3×3
            idx = zeros(1,3);
            for t = 1:3
                key = sprintf('%.6f_%.6f_%.6f', T(t,1), T(t,2), T(t,3));
                if ~isKey(vidx, key)
                    vidx(key) = size(verts,1) + 1;
                    verts(end+1,:) = T(t,:); 
                end
                idx(t) = vidx(key);
            end
            p0 = verts(idx(1),:); p1 = verts(idx(2),:); p2 = verts(idx(3),:);
            n  = cross(p1 - p0, p2 - p0);
            if dot(n, outward) < 0
                idx([2 3]) = idx([3 2]); % enforce outward orientation
            end
            faces(end+1,:) = idx; 
        end
    end
end

vertices = double(verts);
faces    = int32(faces);
fprintf('Concave L-mesh created: %d vertices, %d faces\n', size(vertices,1), size(faces,1));

%% ----------------------------------------------------------------------
% 2) Instantiate Gravity Model
% -----------------------------------------------------------------------
model = PolyhedronGravitation(vertices, faces, ...
    'G', 1.0, 'Density', 1.0, 'Eps', 0.0, 'OrientFaces', true);
disp('PolyhedronGravitation model initialized.');

%% ----------------------------------------------------------------------
% 3) Analytical Laplacian on a 2D Slice (trace of Γ)
% -----------------------------------------------------------------------
z_fixed   = 0.5;
x_range   = [-0.5, 2.5];
y_range   = [-0.5, 2.5];
resolution = 300;

xs = linspace(x_range(1), x_range(2), resolution);
ys = linspace(y_range(1), y_range(2), resolution);
[X, Y] = meshgrid(xs, ys);
Z = ones(size(X)) * z_fixed;

grid_points = [X(:), Y(:), Z(:)];

fprintf('Computing Laplacian on a %dx%d grid at z = %.2f...\n', ...
    resolution, resolution, z_fixed);

tensors   = model.gravity_tensor(grid_points);
laplacian = zeros(size(grid_points,1),1);
for i = 1:size(grid_points,1)
    laplacian(i) = trace(squeeze(tensors(i,:,:)));
end
L = reshape(laplacian, size(X)) / (model.G * model.rho);
disp('Laplacian computation complete.');

%% ----------------------------------------------------------------------
% 4) Exact Cross-Section Boundary
%     - section_segments_from_mesh(): triangle–plane intersections
%     - stitch_segments_to_polylines(): connect segments into loops
% -----------------------------------------------------------------------
function segs = section_segments_from_mesh(Vm, Fm, z0, tol)
    if nargin < 4, tol = 0; end
    segs = [];  % K x 4: [x1 y1 x2 y2]
    for f = 1:size(Fm,1)
        tri = Vm(double(Fm(f,:)), :);  % 3 x 3
        z   = tri(:,3);
        pts = [];
        edges = [1 2; 2 3; 3 1];       % (0,1),(1,2),(2,0) in Python
        for e = 1:3
            a = edges(e,1); b = edges(e,2);
            z1 = z(a); z2 = z(b);
            if (z1 - z0) * (z2 - z0) < -tol  % opposite sides
                t  = (z0 - z1) / (z2 - z1);
                p  = tri(a,:) + t * (tri(b,:) - tri(a,:));
                pts = [pts; p(1:2)];
            end
        end
        if size(pts,1) == 2
            segs(end+1, :) = [pts(1,1) pts(1,2) pts(2,1) pts(2,2)]; %#ok<AGROW>
        end
    end
end

function polys = stitch_segments_to_polylines(segs, tol)
    % segs: K x 4  => [x1 y1 x2 y2]
    if isempty(segs), polys = {}; return; end
    if nargin < 2, tol = 1e-9; end

    % Quantize endpoints to build adjacency 
    key = @(p) sprintf('%ld_%ld', round(p(1)/tol), round(p(2)/tol));

    % Build point table and adjacency (undirected)
    pts = containers.Map();                         % key -> [x y]
    adj = containers.Map('KeyType','char','ValueType','any'); % key -> cell of neighbor keys

    for i = 1:size(segs,1)
        a = segs(i,1:2); b = segs(i,3:4);
        ka = key(a); kb = key(b);
        if ~isKey(pts,ka), pts(ka) = a; end
        if ~isKey(pts,kb), pts(kb) = b; end
        if ~isKey(adj,ka), adj(ka) = {}; end
        if ~isKey(adj,kb), adj(kb) = {}; end
        adj(ka) = [adj(ka), {kb}];
        adj(kb) = [adj(kb), {ka}];
    end

    % Walk loops
    allKeys = keys(adj);
    visited = containers.Map('KeyType','char','ValueType','logical');
    polys = {};
    for sIdx = 1:numel(allKeys)
        s = allKeys{sIdx};
        if isKey(visited,s) && visited(s), continue; end
        nbrs = adj(s);
        if isempty(nbrs), continue; end

        line = pts(s);                              % start point
        prev = '';                                  % none
        cur  = nbrs{1};                             % pick first neighbor

        % mark start visited
        visited(s) = true;

        maxhops = 100000; hops = 0;
        while ~strcmp(cur,s) && hops < maxhops
            visited(cur) = true;
            line = [line; pts(cur)]; 
            nbrsCur = adj(cur);
            % choose next neighbor != prev
            nextKey = '';
            for kk = 1:numel(nbrsCur)
                if ~strcmp(nbrsCur{kk}, prev)
                    nextKey = nbrsCur{kk};
                    break;
                end
            end
            if isempty(nextKey)
                break; % open polyline segment (unexpected for closed section)
            end
            prev = cur;
            cur  = nextKey;
            hops = hops + 1;
        end

        % close if looped
        if strcmp(cur, s)
            line = [line; pts(s)]; 
        end

        polys{end+1} = line; 
    end
end

% Build segments 
segs  = section_segments_from_mesh(vertices, faces, z_fixed, 0);
polys = stitch_segments_to_polylines(segs, 1e-9);

%% ----------------------------------------------------------------------
% 5) Plot Analytical Laplacian Field + Accurate Boundary Overlay
% -----------------------------------------------------------------------
figure('Name','Concave L-Shape Laplacian','Color','w','Position',[100 100 880 740]);
hold on;
title(sprintf('Analytical Laplacian ∇²U / (Gρ) at z = %.2f', z_fixed), ...
    'FontSize',14,'Interpreter','tex');
xlabel('x-axis'); ylabel('y-axis');

% colormap (deep blue → cyan → yellow → red)
N_colors = 50;
colors = zeros(N_colors,3);
colors(1,:) = [0.1, 0.4, 1.0];
for i = 2:N_colors-1
    t = (i-2)/(N_colors-3);
    colors(i,:) = [t, 1.0, 1.0 - t];
end
colors(end,:) = [1.0, 0.0, 0.0];
colormap(colors);

vmin = -4*pi; vmax = 0.0;
imagesc(xs, ys, L, [vmin vmax]);
set(gca, 'YDir','normal');
axis equal tight;

cb = colorbar;
cb.Label.String = '\nabla^2U / (G\rho)';
cb.FontSize = 12;

% Draw each stitched polyline in white
for p = 1:numel(polys)
    poly = polys{p};
    if size(poly,1) >= 2
        plot(poly(:,1), poly(:,2), 'w-', 'LineWidth', 2);
    end
end

xlim(x_range); ylim(y_range);
grid on; box on; hold off;

saveas(gcf, 'concave_L_mesh_laplacian_analytical.png');
disp('Plot saved to "concave_L_mesh_laplacian_analytical.png".');

%% ----------------------------------------------------------------------
% 6) Sanity Check — Potential at Sample Points
% -----------------------------------------------------------------------
pts_to_check = [
    0.5, 0.5, 0.5;   % inside
    1.5, 1.5, 0.5;   % cavity
    3.0, 3.0, 2.0    % far field
];
U = model.potential(pts_to_check);
fprintf('\n--- Potential at Specific Points ---\n');
for i = 1:size(pts_to_check,1)
    fprintf('  U([%.2f, %.2f, %.2f]) = %.8f\n', pts_to_check(i,1), ...
        pts_to_check(i,2), pts_to_check(i,3), U(i));
end

%% ----------------------------------------------------------------------
% 7) Cleanup
% -----------------------------------------------------------------------
model.close();
