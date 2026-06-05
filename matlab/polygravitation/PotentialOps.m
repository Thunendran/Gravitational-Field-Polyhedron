% =======================================================================
% PotentialOps.m
% Gravitational potential of a homogeneous polyhedron
%
% Manuscript alignment:
%   Implements Eq. (10)–(13) of Periyandy & Bevis (2025)
%   "The Gravitational Field of a Homogeneous Polyhedron".
%
%   The gravitational potential at field point P is:
%
%       U(P) = -Gρ * Σ_f [ (1/2) Σ_e (Δz · (det_mn L_mn) - Δz² S_ℓ) ]
%
%   where:
%       Δz = (r_i · n̂_f)                 (Eq. 4)
%       det_ij = ((r_i × r_j) · n̂_f)     (Eq. 5)
%       L_mn = ln((r_m + r_n + L_mn)/(r_m + r_n - L_mn)) / L_mn  (Eq. 7)
%       S_ℓ  = atan2(Δz ||n_raw||, −(det_a det_b + Δz² c_ℓ)/r_ℓ) (Eq. 8)
%
% Purpose:
%   Compute U(P) for arbitrary batches of field points {P_k}, 
%   using face-wise accumulation of logarithmic and arctangent kernels.
%
% Numerical design:
%   • Vectorized across batches of points and faces.
%   • Numerically safe log_term / arctan_term (see helpers.m)
%   • Guarded by eps=0 to avoid divisions by zero or degenerate edges.
%
% Parallelism:
%   • Points are chunked and processed with parfor (user-configurable NumThreads).
%   • No nested parallelism (each parfor worker executes potential_chunk).
%
% Data dependencies:
%   • Requires ensure_potential_precomp() from PolyhedronBase:
%         - L_ij, L_jk, L_ki, inv_L_*, dot_ij_ki, dot_jk_ij, dot_ki_jk
%
% Output:
%   • Scalar potential U [N×1] (double precision)
%
% References:
%   Periyandy & Bevis (2025), Eq. (4)–(9), (10)–(13), Sections 2.3–3.1.
% =======================================================================

classdef PotentialOps < handle
    methods (Access = protected)
        % ---------------------------------------------------------------
        % potential_chunk
        % Evaluate the gravitational potential for a batch of field points.
        % ---------------------------------------------------------------
        function out_chunk = potential_chunk(obj, P)
            % P : (B x 3)  field point batch
            B = size(P,1);
            F = size(obj.Vi,1);
            out_chunk = zeros(B,1);

            % --- Face vertex triplets and normals ---
            Vi = obj.Vi; Vj = obj.Vj; Vk = obj.Vk;
            nh  = obj.n_hat;                 % (F x 3)
            nhBF3 = reshape(nh, [1, F, 3]);  % (1 x F x 3)
            nraw   = obj.n_raw_norm;         % (F x 1)
            nrawBF = reshape(nraw, [1, F]);  % (1 x F)

            % --- Precomputed geometric invariants (Eq. 7–8) ---
            inv_L_ij = obj.inv_L_ij; inv_L_jk = obj.inv_L_jk; inv_L_ki = obj.inv_L_ki;
            L_ij = obj.L_ij; L_jk = obj.L_jk; L_ki = obj.L_ki;
            dot_ij_ki = obj.dot_ij_ki; dot_jk_ij = obj.dot_jk_ij; dot_ki_jk = obj.dot_ki_jk;
            epsval = obj.eps;

            % --- Relative vectors from P to each vertex ---
            % Shapes: (B x F x 3)
            Pi = permute(Vi, [3 1 2]) - permute(P, [1 3 2]);
            Pj = permute(Vj, [3 1 2]) - permute(P, [1 3 2]);
            Pk = permute(Vk, [3 1 2]) - permute(P, [1 3 2]);

            % --- Distances to vertices (B x F) ---
            r_i = vecnorm(Pi, 2, 3);
            r_j = vecnorm(Pj, 2, 3);
            r_k = vecnorm(Pk, 2, 3);

            % --- Δz and determinants (Eq. 4–5) ---
            diff_z = sum(Pi .* nhBF3, 3);          % Δz = (Pi · n̂)
            dz2    = diff_z .* diff_z;             % (Δz)^2
            det_ij = sum(cross(Pi, Pj, 3) .* nhBF3, 3);
            det_jk = sum(cross(Pj, Pk, 3) .* nhBF3, 3);
            det_ki = sum(cross(Pk, Pi, 3) .* nhBF3, 3);

            % --- Logarithmic edge terms (Eq. 7) ---
            L12 = helpers.log_term(r_i, r_j, L_ij, epsval) .* reshape(inv_L_ij, 1, []);
            L23 = helpers.log_term(r_j, r_k, L_jk, epsval) .* reshape(inv_L_jk, 1, []);
            L31 = helpers.log_term(r_k, r_i, L_ki, epsval) .* reshape(inv_L_ki, 1, []);

            % (avoid NaN/Inf propagation)
            L12(~isfinite(L12)) = 0.0;
            L23(~isfinite(L23)) = 0.0;
            L31(~isfinite(L31)) = 0.0;

            % --- Arctangent kernels (Eq. 8) ---
            % replicate dot_* vectors across batch dimension (B x F)
            S1 = helpers.arctan_term(diff_z .* nrawBF, det_ki, det_ij, dz2, ...
                repmat(reshape(dot_ij_ki,1,[]), B,1), r_i, epsval);
            S2 = helpers.arctan_term(diff_z .* nrawBF, det_ij, det_jk, dz2, ...
                repmat(reshape(dot_jk_ij,1,[]), B,1), r_j, epsval);
            S3 = helpers.arctan_term(diff_z .* nrawBF, det_jk, det_ki, dz2, ...
                repmat(reshape(dot_ki_jk,1,[]), B,1), r_k, epsval);

            % --- Aggregate terms per face (Eq. 10–11) ---
            term1 = diff_z .* (det_ij .* L12 + det_jk .* L23 + det_ki .* L31);
            term2 = dz2   .* (S1 + S2 + S3 - sign(diff_z) * pi);
            total = 0.5 * sum(term1 - term2, 2);  % (B x 1)

            % --- Apply physical constants ---
            out_chunk(:) = -obj.G * obj.rho * total;
        end
    end

    methods
        % ---------------------------------------------------------------
        % potential
        % Public API entry point: computes U for arbitrary N×3 points.
        % ---------------------------------------------------------------
        function out = potential(obj, points, block_size)
            if nargin < 3 || isempty(block_size)
                block_size = 1000;  % default block size
            end

            % Ensure minimal precomputations are available
            obj.ensure_potential_precomp();

            pts = double(points);
            if size(pts,2) ~= 3
                error('potential: points must be Nx3');
            end
            N = size(pts,1);
            out = zeros(N,1);

            % Sequential or parallel chunk execution
            if N <= block_size || obj.NumThreads == 1
                out(:) = obj.potential_chunk(pts);
                if N == 1, out = out(1); end
                return
            end

            chunks = obj.iter_chunks(N, block_size);
            out_local = cell(numel(chunks),1);
            parfor (k = 1:numel(chunks), obj.NumThreads)
                s = chunks{k}(1); e = chunks{k}(2);
                out_local{k} = obj.potential_chunk(pts(s:e,:));
            end
            for k = 1:numel(chunks)
                s = chunks{k}(1); e = chunks{k}(2);
                out(s:e) = out_local{k};
            end
            if N == 1, out = out(1); end
        end
    end
end
