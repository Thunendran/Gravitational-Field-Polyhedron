% =======================================================================
% Example 08 — Gravitational Potential and Zero-Acceleration Locus (z = 0)
% =======================================================================
% Computes the gravitational potential and detects the zero-acceleration
% (gₓ = gᵧ = 0) circle for a solid torus using PolyhedronGravitation.
%
% Demonstrates:
%   • Combined potential + acceleration field computation
%   • Detection of zero-g regions via sign change logic
%   • Circle fit through zero-acceleration points
% =======================================================================

clear; clc; close all;
addpath(genpath(pwd));

% -----------------------------------------------------------------------
% 1. Torus Geometry and Mesh Generation
% -----------------------------------------------------------------------
R1 = 2.0;   % major radius
R2 = 1.0;   % minor radius
a  = R1 + R2 + 0.5;

[vertices, faces] = create_torus_tri_mesh(R1, R2, 101, 51);

% Non-star-shaped geometry → disable face orientation
model = PolyhedronGravitation(vertices, faces, ...
    'Density', 1.0, 'G', 1.0, 'OrientFaces', false);
disp('PolyhedronGravitation torus model initialized.');

% -----------------------------------------------------------------------
% 2. Define Grid in Plane z = 0
% -----------------------------------------------------------------------
z_fixed = 0.0;
N = 101;
x_vals = linspace(-a, a, N);
y_vals = linspace(-a, a, N);
[XX, YY] = meshgrid(x_vals, y_vals);
grid_points = [XX(:), YY(:), z_fixed * ones(numel(XX), 1)];

% -----------------------------------------------------------------------
% 3. Compute Potential and Acceleration
% -----------------------------------------------------------------------
disp('Computing potential and acceleration fields...');
Phi = reshape(model.potential(grid_points), size(XX));

accel = model.acceleration(grid_points);
Ax = reshape(accel(:,1), size(XX));
Ay = reshape(accel(:,2), size(YY));
disp('Computation complete.');

% -----------------------------------------------------------------------
% 4. Detect Zero-Acceleration Cells
% -----------------------------------------------------------------------
Ax00 = Ax(1:end-1,1:end-1);
Ax10 = Ax(2:end,1:end-1);
Ax01 = Ax(1:end-1,2:end);
Ax11 = Ax(2:end,2:end);

Ay00 = Ay(1:end-1,1:end-1);
Ay10 = Ay(2:end,1:end-1);
Ay01 = Ay(1:end-1,2:end);
Ay11 = Ay(2:end,2:end);

mask_Ax = (min(cat(3,Ax00,Ax10,Ax01,Ax11),[],3) <= 0) & ...
          (max(cat(3,Ax00,Ax10,Ax01,Ax11),[],3) >= 0);
mask_Ay = (min(cat(3,Ay00,Ay10,Ay01,Ay11),[],3) <= 0) & ...
          (max(cat(3,Ay00,Ay10,Ay01,Ay11),[],3) >= 0);
cell_mask = mask_Ax & mask_Ay;

% Cell centers of zero-acceleration candidates
Xc = 0.25 * (XX(1:end-1,1:end-1) + XX(2:end,1:end-1) + ...
             XX(1:end-1,2:end) + XX(2:end,2:end));
Yc = 0.25 * (YY(1:end-1,1:end-1) + YY(2:end,1:end-1) + ...
             YY(1:end-1,2:end) + YY(2:end,2:end));
cand_x = Xc(cell_mask);
cand_y = Yc(cell_mask);

% Cluster nearby duplicates
dx = x_vals(2) - x_vals(1);
dy = y_vals(2) - y_vals(1);
merge_radius2 = (1.5 * max(dx, dy))^2;
x_zeros = [];
y_zeros = [];
for k = 1:numel(cand_x)
    if all((cand_x(k)-x_zeros).^2 + (cand_y(k)-y_zeros).^2 > merge_radius2)
        x_zeros(end+1) = cand_x(k);
        y_zeros(end+1) = cand_y(k);
    end
end

% -----------------------------------------------------------------------
% 5. Fit Symmetry-Based Zero-g Circle
% -----------------------------------------------------------------------
R = mean(sqrt(x_zeros.^2 + y_zeros.^2));
fprintf('Symmetry-enforced zero-g circle radius = %.4f\n', R);

theta = linspace(0, 2*pi, 200);
circle_x = R * cos(theta);
circle_y = R * sin(theta);

% -----------------------------------------------------------------------
% 6. Visualization — Contour + Zero-g Circle
% -----------------------------------------------------------------------
figure('Position', [100, 100, 850, 750]);

% --- Custom colormap ---
cmap = ajet(64);
contourf(XX, YY, Phi, 30, 'LineColor', 'none');
colormap(cmap);

% --- Colorbar  ---
c = colorbar;
ylabel(c, 'Gravitational Potential', 'FontSize', 14);

% --- Draw torus cross-section (two circles) ---
inner_radius = R1 - R2;
outer_radius = R1 + R2;
theta_c = linspace(0, 2*pi, 400);
hold on;
plot(inner_radius*cos(theta_c), inner_radius*sin(theta_c), ...
    'k-', 'LineWidth', 1.5);
plot(outer_radius*cos(theta_c), outer_radius*sin(theta_c), ...
    'k-', 'LineWidth', 1.5);

% --- Zero-g locus and origin ---
plot(circle_x, circle_y, 'k.', 'MarkerSize', 4);
plot(0, 0, 'ko', 'MarkerSize', 3, 'MarkerFaceColor', 'k');

% --- Labels and Formatting ---
title('Gravitational Potential and Zero-g Locus (Plane z = 0)', ...
    'FontSize', 20, 'FontWeight', 'bold');
xlabel('x-axis', 'FontSize', 16);
ylabel('y-axis', 'FontSize', 16);
axis equal;
axis([-a, a, -a, a]);
grid off;
set(gca, 'Box', 'on', 'LineWidth', 1.0, 'Layer', 'top');

saveas(gcf, 'gravitational_potential_field_torus_zeroG.png');
disp('Plot saved to gravitational_potential_field_torus_zeroG.png');

% -----------------------------------------------------------------------
% 7. ajet Colormap
% -----------------------------------------------------------------------
function cmap = ajet(n)
    % Generate softened jet colormap
    m = floor(n/6);
    j = jet(n + 2*m);
    cmap = j(m+1:end-m, :);
end

% -----------------------------------------------------------------------
% 8. Torus Mesh Generator
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
    Uc = mean(UE(T),2);
    Vc = mean(VE(T),2);
    keep = (Uc >= 0 & Uc < 2*pi & Vc >= 0 & Vc < 2*pi);
    T = T(keep,:);
    F = [map_idx(T(:,1)) map_idx(T(:,2)) map_idx(T(:,3))];
    F_sorted = sort(F,2);
    [~, ia] = unique(F_sorted,'rows','stable');
    F = F(ia,:);
    X = (R1 + R2*cos(V0)) .* cos(U0);
    Y = (R1 + R2*cos(V0)) .* sin(U0);
    Z =  R2 * sin(V0);
    V = [X(:), Y(:), Z(:)];
end
