% =======================================================================
% Example 02 — Laplacian Slice of a Tetrahedron
% =======================================================================
% This example computes and visualizes the Laplacian (∇²U / (Gρ))
% on a 2D plane slice through a tetrahedral mass using the
% analytical polyhedron gravity model.
%
% Demonstrates:
%   - Using gravity_tensor() to derive the Laplacian analytically
%   - Generating a color map of ∇²U inside/outside the body
%   - Drawing the tetrahedron cross-section overlay
%
% Expected Results:
%   Inside the tetrahedron: ∇²U / (Gρ) ≈ -4π
%   Outside the body:      ∇²U / (Gρ) ≈ 0
% =======================================================================

clear; clc; close all;
addpath(genpath(pwd));

% ---------------------------------------------------------------
% Geometry and Model
% ---------------------------------------------------------------
vertices = [
    0.0, 0.0, 0.0;
    1.0, 0.0, 0.0;
    0.0, 1.0, 0.0;
    0.0, 0.0, 1.0
];

faces = int32([
    1 3 2;
    1 2 4;
    1 4 3;
    2 3 4
]);

model = PolyhedronGravitation(vertices, faces, ...
    'G', 1.0, 'Density', 1.0, 'Eps', 0.0, 'OrientFaces', true);

disp('PolyhedronGravitation model initialized.');

% ---------------------------------------------------------------
% 2. Compute Laplacian Field (trace of Gamma)
% ---------------------------------------------------------------
z_fixed = 0.25;
x_range = [-0.2, 1.0];
y_range = [-0.2, 1.0];
resolution = 200;

xs = linspace(x_range(1), x_range(2), resolution);
ys = linspace(y_range(1), y_range(2), resolution);
[X, Y] = meshgrid(xs, ys);
Z = ones(size(X)) * z_fixed;
grid_points = [X(:), Y(:), Z(:)];

fprintf('Computing Laplacian on a %dx%d grid at z = %.2f...\n', ...
    resolution, resolution, z_fixed);

tensors_ana = model.gravity_tensor(grid_points);
laplacian_ana = zeros(size(grid_points, 1), 1);
for k = 1:size(grid_points, 1)
    Gamma = squeeze(tensors_ana(k,:,:));
    laplacian_ana(k) = trace(Gamma);
end
L_ana = reshape(laplacian_ana, size(X)) / (model.G * model.rho);
disp('Laplacian computation complete.');

% ---------------------------------------------------------------
% 3. Compute Boundary Intersection
% ---------------------------------------------------------------
edges = [
    1 2; 1 3; 1 4; 2 3; 2 4; 3 4
];
intersection_points = [];
for e = 1:size(edges,1)
    v1 = vertices(edges(e,1),:);
    v2 = vertices(edges(e,2),:);
    z1 = v1(3); z2 = v2(3);
    if (z1 > z_fixed && z2 < z_fixed) || (z1 < z_fixed && z2 > z_fixed)
        t = (z_fixed - z1) / (z2 - z1);
        pt = v1 + t*(v2 - v1);
        intersection_points = [intersection_points; pt(1:2)];
    end
end

if size(intersection_points,1) >= 3
    k = convhull(intersection_points(:,1), intersection_points(:,2));
    boundary = intersection_points(k,:);
    boundary = [boundary; boundary(1,:)]; % close loop
else
    boundary = intersection_points;
end

% ---------------------------------------------------------------
% 4. Plotting 
% ---------------------------------------------------------------
figure('Name','Laplacian Slice','Color','w','Position',[100 100 750 640]);
hold on;
title(sprintf('Analytical Laplacian ∇²U / (Gρ) at z = %.2f', z_fixed), ...
    'FontSize',14,'Interpreter','tex');
xlabel('x-axis'); ylabel('y-axis');

% --- Colormap: Deep Blue → Cyan → Yellow → Red (50 discrete levels)
N_colors = 50;
colors = zeros(N_colors, 3);

% First entry: deep blue
colors(1,:) = [0.1, 0.4, 1.0];
% Middle range: blue→cyan→yellow
for i = 2:N_colors-1
    t = (i-2)/(N_colors-3);
    colors(i,:) = [t, 1.0, 1.0 - t];
end
% Last entry: red
colors(end,:) = [1.0, 0.0, 0.0];
colormap(colors);

vmin = -4*pi;
vmax = 0.0;

imagesc(xs, ys, L_ana, [vmin vmax]);
set(gca, 'YDir','normal');
axis equal tight;
cb = colorbar;
cb.Label.String = '\nabla^2U / (G\rho)';
cb.FontSize = 12;

% Overlay tetrahedron slice boundary
if ~isempty(boundary)
    plot(boundary(:,1), boundary(:,2), 'w-', 'LineWidth', 2);
end

xlim(x_range);
ylim(y_range);
grid on;
box on;
hold off;

saveas(gcf, 'laplacian_tetra_slice_analytical.png');
disp('Plot saved to "laplacian_tetra_slice_analytical.png".');

% ---------------------------------------------------------------
% 5. Cleanup
% ---------------------------------------------------------------
model.close();
