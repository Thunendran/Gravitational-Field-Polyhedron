% =======================================================================
% AccelOps.m
% Gravitational acceleration of a homogeneous polyhedron
%
% Manuscript alignment:
%   Implements Eq. (14)–(17) of Periyandy & Bevis (2025)
%   "The Gravitational Field of a Homogeneous Polyhedron".
%
%   The gravitational acceleration at field point P is:
%
%       g(P) = G·ρ [ Σ_f ω_f(P) (F_f r_f)  −  Σ_e L_e(P) (E_e r_u) ]
%
%   where:
%       ω_f(P)   = solid angle subtended by face f at P         (Eq. 14)
%       L_e(P)   = ln((r_i + r_j + L_e)/(r_i + r_j − L_e))     (Eq. 15)
%       F_f      = n̂_f ⊗ n̂_f                                 (Eq. 16)
%       E_e      = n̂_a ⊗ t_a + n̂_b ⊗ t_b  (adjacent-face edge dyad)
%       r_f      = v_i − P,   r_u = u − P                      (Eq. 17)
%
% Purpose:
%   Compute the gravitational acceleration vector field g(P) using the
%   exact dyadic formulation, summing the solid-angle and edge-logarithmic
%   contributions per face and per edge.
%
% Numerical design:
%   • Double precision arithmetic throughout.
%   • Uses per-point (per-batch) accumulation to avoid reshape/squeeze
%     broadcast inconsistencies in MATLAB’s 3D arrays.
%   • Epsilon (eps = 0) only guards logarithmic degeneracies (Eq. 15).
%
% Parallelism:
%   • Chunk-level parallelism with parfor, no nested threading.
%   • Each worker computes acceleration_chunk for its batch.
%
% Data dependencies:
%   • Requires ensure_accel_tensor_precomp() from PolyhedronBase:
%         - Face dyads Fdyads (Nf×3×3)
%         - Edge topology: edge_u, edge_v, edge_E, edge_len
%
% Output:
%   • Vector acceleration g [N×3] (double precision)
%
% References:
%   Periyandy & Bevis (2025), Sections 3.2–3.3, Eq. (14)–(17).
% =======================================================================

classdef AccelOps < handle
    methods (Access = protected)
        % ---------------------------------------------------------------
        % acceleration_chunk
        % Evaluate the gravitational acceleration for a batch of points.
        %
        % Implements: g(P) = G·ρ [ Σ_f ω_f(P) (F_f r_f) − Σ_e L_e(P) (E_e r_u) ]
        % ---------------------------------------------------------------
        function out_chunk = acceleration_chunk(obj, P)
            % P : (B x 3)
            B = size(P,1);
            out_chunk = zeros(B,3);

            % --- Geometry and precomputed structures ---
            Vi = obj.Vi; Vj = obj.Vj; Vk = obj.Vk;   % (F x 3)
            Fdyads   = obj.Fdyads;                  % (F x 3 x 3)
            edge_u   = obj.edge_u;                  % (E x 3)
            edge_v   = obj.edge_v;                  % (E x 3)
            edge_E   = obj.edge_E;                  % (E x 3 x 3)
            edge_len = obj.edge_len(:);             % (E x 1)
            epsval   = obj.eps;

            F = size(Vi,1);
            E = size(edge_u,1);

            % -----------------------------------------------------------
            % Process each field point sequentially within the chunk
            % (per-point loop, per PolyhedronParallel design)
            % -----------------------------------------------------------
            for b = 1:B
                Pb = P(b,:);           % (1 x 3)
                Pb_col = Pb.';         % (3 x 1)

                % ===================== Faces ===========================
                % r_f = Vi - P
                r_f = Vi - Pb;  % (F x 3)

                % ω_f (solid angle)  [Eq. (14)]
                omega_f_row = helpers.solid_angle_tri(Pb, Vi, Vj, Vk, epsval); % (1 x F)

                % Accumulate face contribution B = Σ_f ω_f * (F_f * r_f)
                Bsum = zeros(3,1);
                for f = 1:F
                    Ff = squeeze(Fdyads(f,:,:));  % (3 x 3)
                    rf = r_f(f,:).';              % (3 x 1)
                    Bsum = Bsum + omega_f_row(1,f) * (Ff * rf);
                end

                % ===================== Edges ===========================
                % r_u = edge_u - P,  r_v = edge_v - P
                r_u = edge_u - Pb;  % (E x 3)
                r_v = edge_v - Pb;  % (E x 3)

                r_i_norm = vecnorm(r_u, 2, 2);  % (E x 1)
                r_j_norm = vecnorm(r_v, 2, 2);  % (E x 1)

                % log_e (Eq. 15) evaluated per edge
                log_e_row = helpers.log_term(r_i_norm.', r_j_norm.', edge_len, epsval); % (1 x E)

                % Accumulate edge contribution A = Σ_e log_e * (E_e * r_u)
                Asum = zeros(3,1);
                for e = 1:E
                    Ee = squeeze(edge_E(e,:,:));  % (3 x 3)
                    ru = r_u(e,:).';              % (3 x 1)
                    Asum = Asum + log_e_row(1,e) * (Ee * ru);
                end

                % ==================== Combine ==========================
                % g = Gρ (Bsum − Asum)
                out_chunk(b,:) = (obj.G * obj.rho) * (Bsum - Asum).';
            end
        end
    end

    methods
        % ---------------------------------------------------------------
        % acceleration
        % Public API for computing gravitational acceleration vectors.
        % ---------------------------------------------------------------
        function out = acceleration(obj, points, block_size)
            % points: (N x 3)
            % block_size: chunk size for parallel execution
            if nargin < 3 || isempty(block_size)
                block_size = 1000;
            end

            % Ensure precomputations (face & edge dyads)
            obj.ensure_accel_tensor_precomp();

            pts = double(points);
            if size(pts,2) ~= 3
                error('acceleration: points must be Nx3');
            end
            N = size(pts,1);
            out = zeros(N,3);

            % ---- Serial mode (small workloads) ----
            if N <= block_size || obj.NumThreads == 1
                out(:,:) = obj.acceleration_chunk(pts);
                if N == 1, out = out(1,:); end
                return;
            end

            % ---- Parallel mode (large workloads) ----
            chunks = obj.iter_chunks(N, block_size);
            out_local = cell(numel(chunks),1);
            parfor (k = 1:numel(chunks), obj.NumThreads)
                s = chunks{k}(1);
                e = chunks{k}(2);
                out_local{k} = obj.acceleration_chunk(pts(s:e,:));
            end

            % Assemble chunk outputs
            for k = 1:numel(chunks)
                s = chunks{k}(1);
                e = chunks{k}(2);
                out(s:e,:) = out_local{k};
            end
            if N == 1, out = out(1,:); end
        end
    end
end
