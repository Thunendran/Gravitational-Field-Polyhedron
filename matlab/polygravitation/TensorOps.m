% =======================================================================
% TensorOps.m
% Gravitational gravity tensor (second derivative field)
%
% Manuscript alignment:
%   Implements Eq. (18)–(22) of Periyandy & Bevis (2025)
%   "The Gravitational Field of a Homogeneous Polyhedron".
%
%   The gravity tensor at field point P is:
%
%       Γ(P) = G·ρ [ Σ_e L_e(P) E_e  −  Σ_f ω_f(P) F_f ]
%
%   where:
%       L_e(P) = ln((r_i + r_j + L_e)/(r_i + r_j − L_e))  (Eq. 19)
%       ω_f(P) = solid angle subtended by face f at P      (Eq. 18)
%       F_f    = n̂_f ⊗ n̂_f                               (Eq. 20)
%       E_e    = n̂_a ⊗ t_a + n̂_b ⊗ t_b                   (Eq. 21)
%
%   Γ(P) is symmetric (Γ_xy = Γ_yx, etc.) and represents
%   the Hessian of the potential field U(P):
%
%       Γ_ij = ∂²U / ∂x_i ∂x_j                             (Eq. 22)
%
% Purpose:
%   Compute the full gravitational tensor (Γ) at one or more
%   field points using the closed-form dyadic summation.
%
% Design:
%   • Uses face- and edge-based dyadic accumulation.
%   • Fully vectorized across points (batch mode) but retains
%     simple summation loops for clarity and symmetry.
%
% Numerical design:
%   • Double precision arithmetic.
%   • eps = 0 guards near-zero divisions only.
%
% Parallelism:
%   • parfor applied at the outer level (gravity_tensor), not inside chunk.
%   • Each chunk independently evaluates tensor_chunk for its points.
%
% Data dependencies:
%   • Requires ensure_accel_tensor_precomp() from PolyhedronBase:
%         - Fdyads, edge_E, edge_len, edge_u, edge_v
%
% Output:
%   • Γ (N x 3 x 3) : symmetric gravity tensor for each field point.
%
% References:
%   Periyandy & Bevis (2025), Sections 3.4–3.5, Eq. (18)–(22).
% =======================================================================

classdef TensorOps < handle
    methods (Access = public)
        % ---------------------------------------------------------------
        % tensor_chunk
        % Evaluate the gravitational tensor Γ(P) for a batch of points.
        % ---------------------------------------------------------------
        function out_chunk = tensor_chunk(obj, P)
            % P : (B x 3)
            B = size(P,1);
            out_chunk = zeros(B,3,3);

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
            % Process each field point sequentially within this chunk
            % -----------------------------------------------------------
            for b = 1:B
                Pb = P(b,:);

                % ===================== Faces ===========================
                omega_f = helpers.solid_angle_tri(Pb, Vi, Vj, Vk, epsval); % (1 x F)

                % ===================== Edges ===========================
                r_u = edge_u - Pb;   % (E x 3)
                r_v = edge_v - Pb;   % (E x 3)
                r_i_norm = vecnorm(r_u, 2, 2);
                r_j_norm = vecnorm(r_v, 2, 2);
                log_e = helpers.log_term(r_i_norm.', r_j_norm.', edge_len, epsval); % (1 x E)

                % ===================== Accumulation ====================
                face_sum = zeros(3,3);
                edge_sum = zeros(3,3);
                for f = 1:F
                    Ff = squeeze(Fdyads(f,:,:));
                    face_sum = face_sum + omega_f(1,f) * Ff;
                end
                for e = 1:E
                    Ee = squeeze(edge_E(e,:,:));
                    edge_sum = edge_sum + log_e(1,e) * Ee;
                end

                % ===================== Combine =========================
                out_chunk(b,:,:) = (obj.G * obj.rho) * (edge_sum - face_sum);
            end
        end
    end

    methods
        % ---------------------------------------------------------------
        % gravity_tensor
        % Compute the gravitational tensor Γ(P) at one or more field points.
        % ---------------------------------------------------------------
        function out = gravity_tensor(obj, points, block_size)
            if nargin < 3 || isempty(block_size)
                block_size = 100;
            end

            % Ensure precomputations are available
            obj.ensure_accel_tensor_precomp();

            pts = double(points);
            if size(pts,2) ~= 3
                error('gravity_tensor: points must be Nx3');
            end
            N = size(pts,1);
            out = zeros(N,3,3);

            % ---- Serial mode ----
            if N <= block_size || obj.NumThreads == 1
                out(:,:,:) = obj.tensor_chunk(pts);
                if N == 1, out = squeeze(out(1,:,:)); end
                return;
            end

            % ---- Parallel mode ----
            chunks = obj.iter_chunks(N, block_size);
            out_local = cell(numel(chunks),1);

            parfor (k = 1:numel(chunks), obj.NumThreads)
                s = chunks{k}(1);
                e = chunks{k}(2);
                out_local{k} = obj.tensor_chunk(pts(s:e,:));
            end

            % Assemble chunk outputs
            for k = 1:numel(chunks)
                s = chunks{k}(1);
                e = chunks{k}(2);
                out(s:e,:,:) = out_local{k};
            end

            if N == 1
                out = squeeze(out(1,:,:));
            end
        end
    end
end
