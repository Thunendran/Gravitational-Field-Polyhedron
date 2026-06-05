% torus_tri_mesh.m
clear; clc; close all;

%% 1) Torus parameters
R1 = 2.0;        % major radius
R2 = 1.0;        % minor radius
n_u = 96;        % samples around major circle
n_v = 48;        % samples around tube

%% 2) Generate mesh
[V, F] = create_torus_tri_mesh(R1, R2, n_u, n_v);

%% 3) Save mesh to files
save('torus_vertices.mat','V');
save('torus_faces.mat','F');
writematrix(V,'V.txt','Delimiter',' ');
writematrix(F,'F.txt','Delimiter',' ');
disp('Mesh saved to torus_vertices.mat, torus_faces.mat, V.txt, and F.txt');

%% 4) Plot solid green torus with black grid lines
figure('Position',[50,50,1200,900]); 

patch('Vertices', V, 'Faces', F, ...
      'FaceColor', [0, 0.9, 0], ...   % solid green
      'EdgeColor', 'k', ...           % black mesh edges
      'LineWidth', 0.5, ...
      'FaceLighting','flat');

axis equal vis3d
xlabel('x'); ylabel('y'); zlabel('z');
title('Torus', 'FontSize', 16);
grid on; view(45,30);
camlight headlight; lighting flat;

% --- White background for export ---
set(gcf, 'Color', 'w');

% --- Save figure as PNG (300 dpi) ---
print(gcf, 'torus_mesh.png', '-dpng', '-r300');

%% 5) Helper Function
function [V, F] = create_torus_tri_mesh(R1, R2, n_u, n_v)
    % ---- 1) Base parameter grid ----
    u = (0:n_u-1) * (2*pi/n_u);
    v = (0:n_v-1) * (2*pi/n_v);
    [U0, V0] = meshgrid(u, v);
    N = numel(U0);

    % ---- 2) 3×3 tiling ----
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

    % ---- 3) Periodic Delaunay ----
    dt = delaunayTriangulation(UE, VE);
    T  = dt.ConnectivityList;
    Uc = mean(UE(T), 2);
    Vc = mean(VE(T), 2);
    keep = (Uc >= 0 & Uc < 2*pi & Vc >= 0 & Vc < 2*pi);
    T   = T(keep, :);

    F = [map_idx(T(:,1)) map_idx(T(:,2)) map_idx(T(:,3))];
    F_sorted = sort(F, 2);
    [~, ia] = unique(F_sorted, 'rows', 'stable');
    F = F(ia, :);

    % ---- 4) Evaluate torus vertices ----
    X = (R1 + R2*cos(V0)) .* cos(U0);
    Y = (R1 + R2*cos(V0)) .* sin(U0);
    Z =  R2 * sin(V0);
    V = [X(:), Y(:), Z(:)];
end
