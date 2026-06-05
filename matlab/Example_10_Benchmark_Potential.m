% =======================================================================
% Example 10 — MATLAB Benchmark (PolyhedronGravitation, Machine Precision)
% =======================================================================
% Evaluates the gravitational potential for a general polyhedral model
% (e.g., icosahedron) using the unified PolyhedronGravitation framework.
%
% Reads from:  Polyhedron/data/
% Writes to:   U_matlab_parallel.csv and time_matlab_parallel.txt
%
% Notes:
%   • Uses internal chunk-based parallelization (NumThreads).
%   • Keeps MATLAB parallel pool open for subsequent benchmarks.
% =======================================================================

clear; clc;

% -----------------------------------------------------------------------
% 1. Path and data setup
% -----------------------------------------------------------------------
addpath(genpath('PolyGravitation'));  % ensure model classes are accessible

thisFile = mfilename('fullpath');
root = fileparts(fileparts(thisFile));      % → .../Polyhedron
dataDir = fullfile(root, 'data');

fprintf('\n--- MATLAB Parallel Benchmark (PolyhedronGravitation) ---\n');

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
    parpool;   % start pool once (kept open for reuse)
end

% -----------------------------------------------------------------------
% 5. Compute potential (chunked evaluation)
% -----------------------------------------------------------------------
fprintf('\nComputing potential on %d evaluation points...\n', size(Pts,1));

N = size(Pts,1);
U = zeros(N,1);
block_size = 20000;  % adjust based on system memory / cores

tic;
for i = 1:block_size:N
    i_end = min(i + block_size - 1, N);
    block = Pts(i:i_end, :);
    % Compute using 10 internal threads (no parfor here)
    U(i:i_end) = model.potential(block, 10);
end
t_elapsed = toc;

disp('Computation complete.');

% -----------------------------------------------------------------------
% 6. Save benchmark results
% -----------------------------------------------------------------------
U_outfile = fullfile(dataDir, 'U_matlab_parallel.csv');
T_outfile = fullfile(dataDir, 'time_matlab_parallel.txt');

writematrix(U, U_outfile);
fid = fopen(T_outfile, 'w');
fprintf(fid, 'matlab_parallel_time_sec: %.6f\n', t_elapsed);
fclose(fid);

fprintf('\nMATLAB benchmark completed successfully.\n');
fprintf('Elapsed time: %.3f seconds\n', t_elapsed);
fprintf('Results saved to:\n  %s\n  %s\n', U_outfile, T_outfile);

% -----------------------------------------------------------------------
% 7. Note
% -----------------------------------------------------------------------
% The parallel pool remains open intentionally for subsequent runs.
% Do NOT close it automatically in this script.
