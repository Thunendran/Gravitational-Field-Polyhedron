% =======================================================================
% Example 07 — Analytical Gravitational Acceleration Field of a Torus
%                (Vertical Slice: x = 0 Plane)
% =======================================================================
% Computes the analytical gravitational acceleration vectors of a torus
% on the vertical (y–z) plane using the PolyhedronGravitation model.
%
% Demonstrates:
%   • Vertical-plane field evaluation
%   • Vector quiver visualization across cross-section
%   • Handling of non-star-shaped geometry (OrientFaces = false)
% =======================================================================

clear; clc; close all;
addpath(genpath(pwd));

% -----------------------------------------------------------------------
% 1. Torus Geometry and Mesh Generation
% -----------------------------------------------------------------------
R1 = 2.0;   % Major radius
R2 = 1.0;   % Minor radius
a  = R1 + R2 + 0.5;

[vertices, faces] = create_torus_tri_mesh(R1, R2, 101, 51);

model = PolyhedronGravitation(vertices, faces, ...
    'Density', 1.0, 'OrientFaces', false, 'G', 1.0);
disp('PolyhedronGravitation torus model initialized.');

% -----------------------------------------------------------------------
% 2. Define Grid in Vertical Plane (x = 0)
% -----------------------------------------------------------------------
grid_res_y = 21;
grid_res_z = 21;
y_limit = R1 + R2 + 0.5;
z_limit = R2 + 2.5;

y_range = linspace(-y_limit, y_limit, grid_res_y);
z_range = linspace(-z_limit, z_limit, grid_res_z);
[YY, ZZ] = meshgrid(y_range, z_range);

% Evaluation points: x = 0 plane
grid_points = [zeros(numel(YY), 1), YY(:), ZZ(:)];
fprintf('Defined a %dx%d grid on the x=0 plane.\n', grid_res_y, grid_res_z);

% -----------------------------------------------------------------------
% 3. Compute Gravitational Acceleration Field
% -----------------------------------------------------------------------
disp('Computing gravitational acceleration field...');
accel = model.acceleration(grid_points);

% Gravitational field direction 
U = reshape(accel(:,2), size(YY));  % g_y component
V = reshape(accel(:,3), size(ZZ));  % g_z component
disp('Computation complete.');

% -----------------------------------------------------------------------
% 4. Visualization (Quiver Field)
% -----------------------------------------------------------------------
figure('Position', [100, 100, 850, 750]);
hold on;

% --- Draw quiver arrows ---
q = quiver(YY, ZZ, U, V, 1.0, 'Color', 'r', ...
    'LineWidth', 1.0, 'AutoScale', 'off', 'MaxHeadSize', 0.6);

% --- Draw torus cross-section boundaries ---
theta = linspace(0, 2*pi, 400);
plot(-R1 + R2*cos(theta), R2*sin(theta), 'k-', 'LineWidth', 1.5);
plot( R1 + R2*cos(theta), R2*sin(theta), 'k-', 'LineWidth', 1.5);

% -----------------------------------------------------------------------
% 5. Formatting
% -----------------------------------------------------------------------
title('Gravitational Acceleration Field in the Plane x = 0', ...
    'FontSize', 20, 'FontWeight', 'bold');
xlabel('y-axis', 'FontSize', 16);
ylabel('z-axis', 'FontSize', 16);
axis equal;
axis([-3.5, 3.5, -3.0, 3.0]);
grid off;
set(gca, 'Box', 'on', 'LineWidth', 1.0, 'Layer', 'top');

hold off;

% -----------------------------------------------------------------------
% 6. Save Figure
% -----------------------------------------------------------------------
saveas(gcf, 'gravitational_acceleration_field_torus_vertical.png');
disp('Plot saved to gravitational_acceleration_field_torus_vertical.png');

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
