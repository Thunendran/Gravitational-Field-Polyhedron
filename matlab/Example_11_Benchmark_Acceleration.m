% =======================================================================
% Example 11 — MATLAB Parallel Benchmark (Gravitational Acceleration)
% =======================================================================
% Computes the gravitational acceleration for a general polyhedral model
% using the unified PolyhedronGravitation framework.
%
% Reads from:  Polyhedron/data/
% Writes to:   A_matlab_parallel.csv and time_accel_matlab_parallel.txt
%
% Notes:
%   • Uses internal chunked evaluation (NumThreads = 10)
%   • Keeps MATLAB parallel pool open for subsequent benchmarks
% =======================================================================

clear; clc;

% -----------------------------------------------------------------------
% 1. Path and data setup
% -----------------------------------------------------------------------
addpath(genpath('PolyGravitation'));  % ensure scripts are accessible

thisFile = mfilename('fullpath');
root = fileparts(fileparts(thisFile));      % → .../Polyhedron
dataDir = fullfile(root, 'data');

fprintf('\n--- MATLAB Parallel Benchmark (PolyhedronGravitation: Acceleration) ---\n');

req = {'icosahedron_vertices.csv', ...
       'icosahedron_faces.csv', ...
       'eval_points_100k_plus_vertices.csv'};

for k = 1:numel(req)
    if ~isfile(fullfile(dataDir, req{k}))
        error('Missing %s. Please run the Python data generator first.', req{k});
    end
end

% -----------------------------------------------------------------------
% 2. Load data
% -----------------------------------------------------------------------
vertices = readmatrix(fullfile(dataDir, 'icosahedron_vertices.csv'));
faces    = readmatrix(fullfile(dataDir, 'icosahedron_faces.csv'));
Pts      = readmatrix(fullfile(dataDir, 'eval_points_100k_plus_vertices.csv'));

fprintf('Vertices: %d | Faces: %d | Evaluation Points: %d\n', ...
        size(vertices,1), size(faces,1), size(Pts,1));

% -----------------------------------------------------------------------
% 3. Initialize model
% -----------------------------------------------------------------------
model = PolyhedronGravitation(vertices, faces, ...
    'Density', 1.0, 'G', 1.0, 'OrientFaces', true);
disp('PolyhedronGravitation model initialized.');

% -----------------------------------------------------------------------
% 4. Ensure parallel pool is active
% -----------------------------------------------------------------------
p = gcp('nocreate');
if isempty(p)
    parpool;   % start pool once, keep open
end

% -----------------------------------------------------------------------
% 5. Compute acceleration (chunked evaluation)
% -----------------------------------------------------------------------
fprintf('\nComputing acceleration on %d evaluation points...\n', size(Pts,1));

N = size(Pts,1);
A = zeros(N,3);
block_size = 20000;  % adjusted for 100k points

tic;
for i = 1:block_size:N
    i_end = min(i + block_size - 1, N);
    block = Pts(i:i_end, :);
    % Compute using 10 internal threads
    A(i:i_end, :) = model.acceleration(block, 10);
end
t_elapsed = toc;

disp('Computation complete.');

% -----------------------------------------------------------------------
% 6. Save benchmark results
% -----------------------------------------------------------------------
A_outfile = fullfile(dataDir, 'A_matlab_parallel.csv');
T_outfile = fullfile(dataDir, 'time_accel_matlab_parallel.txt');

writematrix(A, A_outfile);
fid = fopen(T_outfile, 'w');
fprintf(fid, 'matlab_parallel_time_sec: %.6f\n', t_elapsed);
fprintf(fid, 'time_per_point_sec: %.12e\n', t_elapsed / N);
fprintf(fid, 'time_per_point_us: %.6f\n', (t_elapsed / N) * 1e6);
fclose(fid);

fprintf('\nMATLAB acceleration benchmark completed successfully.\n');
fprintf('Elapsed time: %.3f seconds (%.3f µs/pt)\n', ...
        t_elapsed, (t_elapsed / N) * 1e6);
fprintf('Results saved to:\n  %s\n  %s\n', A_outfile, T_outfile);

% -----------------------------------------------------------------------
% 7. Notes
% -----------------------------------------------------------------------
% • This benchmark uses the general PolyhedronGravitation model.
% • The MATLAB parallel pool remains open for reuse by subsequent examples.
