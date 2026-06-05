% =======================================================================
% Example_04_TorusLaplacianVertical.m
% Analytical Laplacian of a Torus (Vertical y–z Plane Slice)
% Using PolyhedronGravitation Framework
% =======================================================================
% Computes ∇²U / (Gρ) on a 251×126 grid slice at x=0.
%
% Inside torus:   ∇²U / (Gρ) ≈ −4π
% Outside torus:  ≈ 0
% =======================================================================

clear; clc; close all;
addpath(genpath(pwd));

% ==============================================================================
% 1. Setup Geometry, Grid, and Model
% ==============================================================================

R1 = 2.0;   % Major radius
R2 = 1.0;   % Minor radius

% --- Generate the torus mesh ---
[vertices, faces] = create_torus_tri_mesh(R1, R2, 100, 50);

% --- Initialize PolyhedronGravitation model ---
model = PolyhedronGravitation(vertices, faces, ...
    'Density', 1.0, 'OrientFaces', false, 'G', 1.0);
disp('Model initialized.');

% ==============================================================================
% 2. Define a 251 × 126 Vertical Grid (y–z plane, x = 0)
% ==============================================================================
Ny = 251; Nz = 126;   % grid resolution
y_limit = R1 + R2 + 0.5;
z_limit = R2 + 0.5;

y_range = linspace(-y_limit, y_limit, Ny);
z_range = linspace(-z_limit, z_limit, Nz);
[YY, ZZ] = meshgrid(y_range, z_range);

grid_points = [zeros(numel(YY), 1), YY(:), ZZ(:)];
fprintf('Defined a %dx%d calculation grid on the x=0 plane.\n', Ny, Nz);

% ==============================================================================
% 3. Compute Analytical Laplacian (Trace of Gravity Tensor)
% ==============================================================================
disp('Computing Laplacian on the vertical grid...');
tensors = model.gravity_tensor(grid_points);

num_points = size(tensors, 1);
laplacian_values = zeros(num_points, 1);
for i = 1:num_points
    laplacian_values(i) = trace(squeeze(tensors(i, :, :)));
end

laplacian_grid = reshape(laplacian_values, [Nz, Ny]);
disp('Computation complete.');

% ==============================================================================
% 4. Plotting — Color Map and Torus Boundaries
% ==============================================================================

% --- Colormap ---
N_colors = 50;
colors_rgb = zeros(N_colors, 3);
colors_rgb(1, :) = [0.1, 0.4, 1]; % blue
for i = 2:(N_colors - 1)
    ratio = (i - 2) / (N_colors - 3);
    colors_rgb(i, :) = [ratio, 1.0, 1.0 - ratio];
end
colors_rgb(N_colors, :) = [1, 0, 0]; % red
vmin = -4 * pi;
vmax = 0.0;

figure('Position', [100, 100, 900, 600]);
ax = gca;
imagesc(y_range, z_range, laplacian_grid);
set(ax, 'YDir', 'normal');
colormap(ax, colors_rgb);
caxis([vmin, vmax]);
hold on;

% --- Draw torus cross-section (two circles) ---
theta = linspace(0, 2*pi, 200);
plot(-R1 + R2 * cos(theta), R2 * sin(theta), 'w-', 'LineWidth', 2);
plot( R1 + R2 * cos(theta), R2 * sin(theta), 'w-', 'LineWidth', 2);

% --- Labels, annotations, and colorbar ---
title(['Analytical Laplacian on a 251', char(215), '126 grid (x = 0 Plane)'], ...
    'FontSize', 18);   
xlabel('y-axis', 'FontSize', 14);
ylabel('z-axis', 'FontSize', 14);
axis equal;
axis tight;
grid off;

text(-R1, 0, '$\nabla^2 U = -4\pi$', 'Color', 'white', ...
    'FontSize', 14, 'HorizontalAlignment', 'center', ...
    'FontWeight', 'bold', 'Interpreter', 'latex');
text( R1, 0, '$\nabla^2 U = -4\pi$', 'Color', 'white', ...
    'FontSize', 14, 'HorizontalAlignment', 'center', ...
    'FontWeight', 'bold', 'Interpreter', 'latex');
text(0, z_limit - 0.2, '$\nabla^2 U = 0$', 'Color', 'white', ...
    'FontSize', 16, 'HorizontalAlignment', 'center', ...
    'FontWeight', 'bold', 'Interpreter', 'latex');

% --- Add Colorbar ---
cbar = colorbar;
cbar.Ticks = [-12, -8, -4, 0];
ylabel(cbar, '$\nabla^2 U / (G\rho)$', 'FontSize', 14, 'Interpreter', 'latex');

% --- Save Figure ---
saveas(gcf, 'laplacian_torus_vertical_slice_251x126.png');
disp('Plot saved to laplacian_torus_vertical_slice_251x126.png');

% ==============================================================================
% 5. Torus Mesh Generator
% ==============================================================================
function [V, F] = create_torus_tri_mesh(R1, R2, n_u, n_v)
%CREATE_TORUS_TRI_MESH  Build torus surface mesh using periodic Delaunay
%   R1 = major radius
%   R2 = minor radius
%   n_u, n_v = sampling resolution

    %% Base parameter grid
    u = (0:n_u-1) * (2*pi/n_u);
    v = (0:n_v-1) * (2*pi/n_v);
    [U0, V0] = meshgrid(u, v);   % size n_v x n_u
    N = numel(U0);

    %% 3×3 tiling in (u,v) space
    UE = []; VE = []; map_idx = [];
    for su = -1:1
        for sv = -1:1
            tmpU = U0 + 2*pi*su;
            tmpV = V0 + 2*pi*sv;
            UE   = [UE; tmpU(:)];
            VE   = [VE; tmpV(:)];
            map_idx = [map_idx; (1:N).'];
        end
    end

    %% Delaunay triangulation with periodic filtering
    dt = delaunayTriangulation(UE, VE);
    T  = dt.ConnectivityList;
    Uc = mean(UE(T), 2);
    Vc = mean(VE(T), 2);
    keep = (Uc >= 0 & Uc < 2*pi & Vc >= 0 & Vc < 2*pi);
    T   = T(keep, :);

    % Map tiled indices back to base grid
    F = [map_idx(T(:,1)) map_idx(T(:,2)) map_idx(T(:,3))];
    F_sorted = sort(F, 2);
    [~, ia] = unique(F_sorted, 'rows', 'stable');
    F = F(ia, :);

    %% Evaluate torus vertices in 3D
    X = (R1 + R2*cos(V0)) .* cos(U0);
    Y = (R1 + R2*cos(V0)) .* sin(U0);
    Z =  R2 * sin(V0);
    V = [X(:), Y(:), Z(:)];
end
