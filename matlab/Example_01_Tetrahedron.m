% ===============================================================
% Example 01 — Gravitation of a Simple Tetrahedron
% ===============================================================
clear; 
clc;

% ---------------------------------------------------------------
% Add the PolyGravitation folder to path
% ---------------------------------------------------------------
addpath(genpath('PolyGravitation'));  % adjust path if needed

% ---------------------------------------------------------------
% Geometry definition (simple tetrahedron)
% ---------------------------------------------------------------
vertices = [
    0, 0, 0;
    1, 0, 0;
    0, 1, 0;
    0, 0, 1
];
faces = int32([
    1, 3, 2;
    1, 2, 4;
    3, 4, 1;
    2, 3, 4
]);

% ---------------------------------------------------------------
% Model instantiation
% ---------------------------------------------------------------
% Constructor syntax:
%   PolyhedronGravitation(vertices, faces, 'G',1.0, 'Density',1.0, ...
%                         'Eps',0.0, 'OrientFaces',true, 'NumThreads',[])
model = PolyhedronGravitation(vertices, faces, ...
    'G',1.0, 'Density',1.0, 'Eps',0.0, 'OrientFaces',true);

% ---------------------------------------------------------------
% Test evaluation points
% ---------------------------------------------------------------
test_points = [
    2, 2, 2;               % External point
    0.25, 0.25, 0.25;      % Internal point
    0, 0, 0;               % Vertex (singularity)
    0, 1, 0;               % Vertex (singularity)
    0, 1.0000000001, 0     % Near-singularity
];

% ---------------------------------------------------------------
% Output setup
% ---------------------------------------------------------------
fid = fopen('gravitation_results.txt','w');
fprintf(fid,'--- Polyhedral Gravitation: Tetrahedron ---\n');

% ---------------------------------------------------------------
% Evaluation loop
% ---------------------------------------------------------------
for k = 1:size(test_points,1)
    P = test_points(k,:); % each point is row (Nx3)

    % Compute gravitational potential, acceleration, and tensor
    U = model.potential(P, 10);
    g = model.acceleration(P, 10);
    T = model.gravity_tensor(P, 10);

    % -----------------------------------------------------------
    % Output formatting
    % -----------------------------------------------------------
    fprintf(fid,'\n%s\n',repmat('=',1,60));
    fprintf(fid,'Point: [%.6f, %.6f, %.6f]\n', P);
    fprintf(fid,'%s\n',repmat('=',1,60));
    fprintf(fid,'U: %.12f\n', U);
    fprintf(fid,'g: [% .12e  % .12e  % .12e]\n', g(1), g(2), g(3));
    fprintf(fid,'Γ rows:\n');
    for i = 1:3
        fprintf(fid,'  [% .12e  % .12e  % .12e]\n', T(i,1), T(i,2), T(i,3));
    end
end

% ---------------------------------------------------------------
% Cleanup
% ---------------------------------------------------------------
fclose(fid);
fprintf('Results saved to "gravitation_results.txt"\n');

% Optional: close MATLAB parallel pool if open
delete(gcp('nocreate'));
