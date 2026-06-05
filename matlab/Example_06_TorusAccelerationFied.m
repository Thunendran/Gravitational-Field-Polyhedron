% =======================================================================
% Example 06 — Analytical Gravitational Acceleration Field of a Torus
% =======================================================================

clear; clc; close all;
addpath(genpath(pwd));

% -----------------------------------------------------------------------
% 1. Torus Geometry and Mesh Generation
% -----------------------------------------------------------------------
R1 = 2.0;
R2 = 1.0;
a  = R1 + R2 + 0.5;

[vertices, faces] = create_torus_tri_mesh(R1, R2, 101, 51);

model = PolyhedronGravitation(vertices, faces, ...
    'Density', 1.0, 'OrientFaces', false, 'G', 1.0);
disp('PolyhedronGravitation torus model initialized.');

% -----------------------------------------------------------------------
% 2. Define Grid in z = 0 Plane
% -----------------------------------------------------------------------
z_fixed = 0.0;
n_grid = 19; % odd → includes 0 center
x_vals = linspace(-a, a, n_grid);
y_vals = linspace(-a, a, n_grid);
[XX, YY] = meshgrid(x_vals, y_vals);

grid_points = [XX(:), YY(:), z_fixed * ones(numel(XX), 1)];
fprintf('Defined %d×%d grid on the z = 0 plane.\n', n_grid, n_grid);

% -----------------------------------------------------------------------
% 3. Compute Acceleration Field
% -----------------------------------------------------------------------
disp('Computing gravitational acceleration field...');
accel = model.acceleration(grid_points);

U = reshape(accel(:,1), size(XX));
V = reshape(accel(:,2), size(YY));
disp('Computation complete.');

% -----------------------------------------------------------------------
% 4. Visualization (Quiver + Center Point)
% -----------------------------------------------------------------------
figure('Position', [100, 100, 850, 750]);
hold on;

% --- Draw quiver field ---
q = quiver(XX, YY, U, V, 1.0, 'Color', 'r', 'LineWidth', 1.0, ...
    'AutoScale', 'off', 'MaxHeadSize', 0.6);

% --- Torus cross-section boundaries ---
inner_radius = R1 - R2;
outer_radius = R1 + R2;
theta = linspace(0, 2*pi, 400);
plot(inner_radius * cos(theta), inner_radius * sin(theta), ...
    'k-', 'LineWidth', 1.2);
plot(outer_radius * cos(theta), outer_radius * sin(theta), ...
    'k-', 'LineWidth', 1.2);

% --- Identify and mark the zero-acceleration (center) point ---
mag = hypot(U, V);
[~, idx_min] = min(mag(:));
x0 = XX(idx_min);
y0 = YY(idx_min);
plot(x0, y0, 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 2);

% -----------------------------------------------------------------------
% 5. Formatting
% -----------------------------------------------------------------------
title('Gravitational Acceleration Field in the Plane z = 0', ...
    'FontSize', 20, 'FontWeight', 'bold');
xlabel('x-axis', 'FontSize', 16);
ylabel('y-axis', 'FontSize', 16);
axis equal;
axis([-a, a, -a, a]);
grid off;
set(gca, 'Box', 'on', 'LineWidth', 1.0, 'Layer', 'top');

hold off;

% -----------------------------------------------------------------------
% 6. Save Figure
% -----------------------------------------------------------------------
saveas(gcf, 'gravitational_acceleration_field_torus.png');
disp('Plot saved to gravitational_acceleration_field_torus.png');

% -----------------------------------------------------------------------
% 7. Torus Mesh Generator
% -----------------------------------------------------------------------
function [V, F] = create_torus_tri_mesh(R1, R2, n_u, n_v)
    u = (0:n_u-1) * (2*pi/n_u);
    v = (0:n_v-1) * (2*pi/n_v);
    [U0, V0] = meshgrid(u, v);
    N = numel(U0);
    UE = []; VE = []; map_idx = [];
    for su = -1:1
        for sv = -1:1
            tmpU = U0 + 2*pi*su;
            tmpV = V0 + 2*pi*sv;
            UE = [UE; tmpU(:)];
            VE = [VE; tmpV(:)];
            map_idx = [map_idx; (1:N).'];
        end
    end
    dt = delaunayTriangulation(UE, VE);
    T = dt.ConnectivityList;
    Uc = mean(UE(T), 2);
    Vc = mean(VE(T), 2);
    keep = (Uc >= 0 & Uc < 2*pi & Vc >= 0 & Vc < 2*pi);
    T = T(keep, :);
    F = [map_idx(T(:,1)) map_idx(T(:,2)) map_idx(T(:,3))];
    F_sorted = sort(F, 2);
    [~, ia] = unique(F_sorted, 'rows', 'stable');
    F = F(ia, :);
    X = (R1 + R2*cos(V0)) .* cos(U0);
    Y = (R1 + R2*cos(V0)) .* sin(U0);
    Z =  R2 * sin(V0);
    V = [X(:), Y(:), Z(:)];
end
