% =======================================================================
% Example_05_TorusLaplacian.m
% Analytical Laplacian of a Torus (z = 0 Plane)
% Using PolyhedronGravitation Framework
% =======================================================================
clear; clc; close all;

% ==============================================================================
% 1. Setup Geometry, Grid, and Model
% ==============================================================================

% --- Define Torus Geometry ---
R1 = 2.0;
R2 = 1.0;
a = R1 + R2 + 1.0; % Grid limit

% --- Generate the Mesh ---
[vertices, faces] = create_torus_tri_mesh(R1, R2, 100, 50);

% --- Instantiate the model with face orientation turned OFF ---
model = PolyhedronGravitation(vertices, faces, ...
    'Density', 1.0, 'OrientFaces', false, 'G', 1.0);
disp('Model initialized.');

% --- Define the calculation grid (z = 0 plane) ---
grid_res = 251;
x_range = linspace(-a, a, grid_res);
y_range = linspace(-a, a, grid_res);
[XX, YY] = meshgrid(x_range, y_range);

% --- Create (N x 3) grid points ---
grid_points = [XX(:), YY(:), zeros(numel(XX), 1)];
fprintf('Defined a %dx%d calculation grid on the z=0 plane.\n', grid_res, grid_res);

% ==============================================================================
% 2. Compute the Correct Laplacian
% ==============================================================================
disp('Computing Laplacian on the grid...');

tensors = model.gravity_tensor(grid_points);

% The Laplacian ∇²U is the trace of the gravity tensor
num_points = size(tensors, 1);
laplacian_values = zeros(num_points, 1);
for i = 1:num_points
    laplacian_values(i) = trace(squeeze(tensors(i, :, :)));
end

laplacian_grid = reshape(laplacian_values, [grid_res, grid_res]);
disp('Computation complete.');

% ==============================================================================
% 3. Plotting with Custom Style
% ==============================================================================

% --- Custom colormap and normalization ---
N_colors = 50;
colors_rgb = zeros(N_colors, 3);
colors_rgb(1, :) = [0.1, 0.4, 1];
for i = 2:(N_colors - 1)
    ratio = (i - 2) / (N_colors - 3);
    colors_rgb(i, :) = [ratio, 1.0, 1.0 - ratio];
end
colors_rgb(N_colors, :) = [1, 0, 0];
vmin = -4 * pi;
vmax = 0.0;

% --- Create the plot ---
figure('Position', [100, 100, 800, 800]);
ax = gca;
imagesc(x_range, y_range, laplacian_grid');
set(ax, 'YDir', 'normal');
colormap(ax, colors_rgb);
caxis([vmin, vmax]);
hold on;

% --- Plot torus boundary ---
inner_radius = R1 - R2;
outer_radius = R1 + R2;
theta = linspace(0, 2*pi, 200);
plot(inner_radius * cos(theta), inner_radius * sin(theta), 'w-', 'LineWidth', 2);
plot(outer_radius * cos(theta), outer_radius * sin(theta), 'w-', 'LineWidth', 2);

% --- Labels and Annotations ---
title('Analytical Laplacian of Torus in the z = 0 Plane', ...
    'FontSize', 18, 'Interpreter', 'latex');
xlabel('x-axis', 'FontSize', 14);
ylabel('y-axis', 'FontSize', 14);
axis equal;
grid off;
text(R1, 0, '$\nabla^2 U = -4\pi$', 'Color', 'white', ...
    'FontSize', 12, 'HorizontalAlignment', 'center', ...
    'FontWeight', 'bold', 'Interpreter', 'latex');
text(0, 0, '$\nabla^2 U = 0$', 'Color', 'white', ...
    'FontSize', 12, 'HorizontalAlignment', 'center', ...
    'FontWeight', 'bold', 'Interpreter', 'latex');

hold off;

% --- Add Colorbar ---
cbar = colorbar;
cbar.Ticks = [-12, -10, -8, -6, -4, -2, 0];
ylabel(cbar, '$\nabla^2 U / (G\rho)$', ...
    'FontSize', 14, 'Interpreter', 'latex');

% --- Save Figure ---
saveas(gcf, 'laplacian_torus_matlab.png');
disp('Plot saved to laplacian_torus_matlab.png');

% ==============================================================================
% 4. Torus Mesh Generator
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
