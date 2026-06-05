% =======================================================================
% Example 12 — MATLAB Parallel Benchmark (Gravity Tensor)
% =======================================================================
% Computes the gravitational gravity tensor (3×3) for a general
% polyhedral model using the PolyhedronGravitation framework.
%
% Reads from:  Polyhedron/data/
% Writes to:   T_matlab_parallel.csv and time_tensor_matlab_parallel.txt
%
% Notes:
%   • Uses internal chunked evaluation (NumThreads = 10)
%   • Each tensor is flattened to 9 columns (row-major order)
%   • Keeps MATLAB parallel pool open for subsequent benchmarks
% =======================================================================

clear; clc;

% -----------------------------------------------------------------------
% 1. Path and data setup
% -----------------------------------------------------------------------
addpath(genpath('PolyGravitation'));  % ensure framework scripts available

thisFile = mfilename('fullpath');
root = fileparts(fileparts(thisFile));      % → .../Polyhedron
dataDir = fullfile(root, 'data');

fprintf('\n--- MATLAB Parallel Benchmark (PolyhedronGravitation: Gravity Tensor) ---\n');

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
    parpool;   % start once, keep open
end

% -----------------------------------------------------------------------
% 5. Compute gravity tensor (chunked evaluation)
% -----------------------------------------------------------------------
fprintf('\nComputing gravity tensor on %d evaluation points...\n', size(Pts,1));

N = size(Pts,1);
T_flat = zeros(N, 9);
block_size = 15000;  % adjusted for heavier tensor computation

tic;
for i = 1:block_size:N
    i_end = min(i + block_size - 1, N);
    block = Pts(i:i_end, :);

    % Compute tensor for this block (returns [M × 3 × 3])
    T_block = model.gravity_tensor(block, 10);

    % Flatten to N×9
    M = size(T_block,1);
    for j = 1:M
        Tij = squeeze(T_block(j,:,:));
        T_flat(i + j - 1, :) = Tij(:)';
    end
end
t_elapsed = toc;

disp('Computation complete.');

% -----------------------------------------------------------------------
% 6. Save benchmark results
% -----------------------------------------------------------------------
T_outfile = fullfile(dataDir, 'T_matlab_parallel.csv');
T_timefile = fullfile(dataDir, 'time_tensor_matlab_parallel.txt');

writematrix(T_flat, T_outfile);

fid = fopen(T_timefile, 'w');
fprintf(fid, 'matlab_parallel_time_sec: %.6f\n', t_elapsed);
fprintf(fid, 'time_per_point_sec: %.12e\n', t_elapsed / N);
fprintf(fid, 'time_per_point_us: %.6f\n', (t_elapsed / N) * 1e6);
fclose(fid);

fprintf('\nMATLAB tensor benchmark completed successfully.\n');
fprintf('Elapsed time: %.3f seconds (%.3f µs/pt)\n', ...
        t_elapsed, (t_elapsed / N) * 1e6);
fprintf('Results saved to:\n  %s\n  %s\n', T_outfile, T_timefile);

% -----------------------------------------------------------------------
% 7. Notes
% -----------------------------------------------------------------------
% • Each tensor is flattened row-wise for CSV compatibility (N×9).
% • The MATLAB parallel pool remains open intentionally for reuse.
