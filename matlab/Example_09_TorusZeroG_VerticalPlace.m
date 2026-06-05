% ===============================================================
% Example 09 — Gravitational Potential and Zero-Acceleration Points (x = 0)
% ===============================================================
% Analytical gravitational potential and acceleration of a torus
% in the vertical (y–z) plane using PolyhedronGravitation.
%
% Matches the Python plot exactly:
%   • Same “ajet” colormap and contour style
%   • Same title, axis labels, and legend layout
%   • Accelerated batch computation for large grids
% ===============================================================

clear; clc; close all;
addpath(genpath(pwd));

% ---------------------------------------------------------------
% 1. Torus Geometry and Mesh Generation
% ---------------------------------------------------------------
R1 = 2.0;       % Major radius
R2 = 1.0;       % Minor radius
[vertices, faces] = create_torus_tri_mesh(R1, R2, 101, 51);

model = PolyhedronGravitation(vertices, faces, ...
    'Density', 1.0, 'G', 1.0, 'OrientFaces', false);
disp('PolyhedronGravitation torus model initialized.');

% ---------------------------------------------------------------
% 2. Define Vertical (x = 0) Grid
% ---------------------------------------------------------------
grid_res_y = 101;
grid_res_z = 101;
y_limit = R1 + R2 + 0.8;
z_limit = R2 + 2.5;

y_range = linspace(-y_limit, y_limit, grid_res_y);
z_range = linspace(-z_limit, z_limit, grid_res_z);
[YY, ZZ] = meshgrid(y_range, z_range);
grid_points = [zeros(numel(YY),1), YY(:), ZZ(:)];
fprintf('Computing acceleration on %dx%d grid (x=0 plane)...\n', ...
    grid_res_y, grid_res_z);

% ---------------------------------------------------------------
% 3. Compute Acceleration (Optimized Batch Mode)
% ---------------------------------------------------------------
N = size(grid_points,1);
batch = 2000;
accel = zeros(N,3);
for i = 1:batch:N
    idx = i:min(i+batch-1,N);
    accel(idx,:) = model.acceleration(grid_points(idx,:));
end
Ay = reshape(accel(:,2), size(YY));
Az = reshape(accel(:,3), size(ZZ));
disp('Acceleration computation complete (batched).');

% Optional: potential background
computePotential = true;
if computePotential
    Phi = zeros(N,1);
    for i = 1:batch:N
        idx = i:min(i+batch-1,N);
        Phi(idx) = model.potential(grid_points(idx,:));
    end
    Phi = reshape(Phi, size(YY));
else
    Phi = sqrt(Ay.^2 + Az.^2);
end

% ---------------------------------------------------------------
% 4. Detect Zero-Acceleration Cells
% ---------------------------------------------------------------
Ay00 = Ay(1:end-1,1:end-1); Ay10 = Ay(2:end,1:end-1);
Ay01 = Ay(1:end-1,2:end);   Ay11 = Ay(2:end,2:end);
Az00 = Az(1:end-1,1:end-1); Az10 = Az(2:end,1:end-1);
Az01 = Az(1:end-1,2:end);   Az11 = Az(2:end,2:end);

mask_Ay = (min(cat(3,Ay00,Ay10,Ay01,Ay11),[],3) <= 0) & ...
          (max(cat(3,Ay00,Ay10,Ay01,Ay11),[],3) >= 0);
mask_Az = (min(cat(3,Az00,Az10,Az01,Az11),[],3) <= 0) & ...
          (max(cat(3,Az00,Az10,Az01,Az11),[],3) >= 0);
cell_mask = mask_Ay & mask_Az;

Yc = 0.25 * (YY(1:end-1,1:end-1) + YY(2:end,1:end-1) + ...
             YY(1:end-1,2:end) + YY(2:end,2:end));
Zc = 0.25 * (ZZ(1:end-1,1:end-1) + ZZ(2:end,1:end-1) + ...
             ZZ(1:end-1,2:end) + ZZ(2:end,2:end));
cand_y = Yc(cell_mask);
cand_z = Zc(cell_mask);

dy = y_range(2) - y_range(1);
dz = z_range(2) - z_range(1);
merge_radius2 = (1.5 * max(dy, dz))^2;
y_zeros = []; z_zeros = [];
for k = 1:numel(cand_y)
    if all((cand_y(k)-y_zeros).^2 + (cand_z(k)-z_zeros).^2 > merge_radius2)
        y_zeros(end+1) = cand_y(k);
        z_zeros(end+1) = cand_z(k);
    end
end

fprintf('\nZero-acceleration equilibrium points (approx):\n');
for i = 1:numel(y_zeros)
    fprintf('  (y = %.4f, z = %.4f)\n', y_zeros(i), z_zeros(i));
end

% ---------------------------------------------------------------
% 5. Visualization 
% ---------------------------------------------------------------
figure('Position',[100,100,900,800]);
cmap = ajet(64);
contourf(YY, ZZ, Phi, 30, 'LineColor','none');
colormap(cmap);
c = colorbar;
ylabel(c, 'Gravitational Potential', 'FontSize',16);

hold on;

% --- Torus cross-section (two circles) ---
theta = linspace(0,2*pi,400);
plot(-R1 + R2*cos(theta), R2*sin(theta), 'k-', 'LineWidth',1.2);
plot( R1 + R2*cos(theta), R2*sin(theta), 'k-', 'LineWidth',1.2);

% --- Zero-g points ---
scatter(y_zeros, z_zeros, 24, 'k', 'filled', ...
    'DisplayName','Zero-acceleration points');

% --- Titles and Labels ---
title('Gravitational Potential and Zero-g Points (Plane x = 0)', ...
    'FontSize',22, 'Color','black', 'FontWeight','bold');
xlabel('y-axis', 'FontSize',20, 'Color','black');
ylabel('z-axis', 'FontSize',20, 'Color','black');
axis equal;
axis([-y_limit, y_limit, -z_limit, z_limit]);
set(gca,'TickDir','out','LineWidth',1.0,'Box','on','Layer','top');
grid off;

% --- Legend ---
%legend('Zero-acceleration points', 'Location','southeast','FontSize',12, 'TextColor','black');

saveas(gcf, 'gravitational_potential_field_torus_vertical_zeroG_final.png');
disp('Plot saved to gravitational_potential_field_torus_vertical_zeroG_final.png');

% ---------------------------------------------------------------
% 6. ajet Colormap
% ---------------------------------------------------------------
function cmap = ajet(n)
    m = floor(n/6);
    j = jet(n + 2*m);
    cmap = j(m+1:end-m, :);
end

% ---------------------------------------------------------------
% 7. Torus Mesh Generator
% ---------------------------------------------------------------
function [V, F] = create_torus_tri_mesh(R1, R2, n_u, n_v)
    u = (0:n_u-1) * (2*pi/n_u);
    v = (0:n_v-1) * (2*pi/n_v);
    [U0, V0] = meshgrid(u, v);
    N = numel(U0);
    UE=[]; VE=[]; map_idx=[];
    for su=-1:1
        for sv=-1:1
            tmpU=U0+2*pi*su; tmpV=V0+2*pi*sv;
            UE=[UE; tmpU(:)]; VE=[VE; tmpV(:)];
            map_idx=[map_idx; (1:N).'];
        end
    end
    dt=delaunayTriangulation(UE,VE);
    T=dt.ConnectivityList;
    Uc=mean(UE(T),2); Vc=mean(VE(T),2);
    keep=(Uc>=0 & Uc<2*pi & Vc>=0 & Vc<2*pi);
    T=T(keep,:);
    F=[map_idx(T(:,1)) map_idx(T(:,2)) map_idx(T(:,3))];
    F_sorted=sort(F,2);
    [~,ia]=unique(F_sorted,'rows','stable');
    F=F(ia,:);
    X=(R1+R2*cos(V0)).*cos(U0);
    Y=(R1+R2*cos(V0)).*sin(U0);
    Z= R2*sin(V0);
    V=[X(:),Y(:),Z(:)];
end
