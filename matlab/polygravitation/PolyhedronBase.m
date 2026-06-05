% =======================================================================
% PolyhedronBase.m
% Core geometry container + selective precomputations + chunk scheduling
%
% Manuscript alignment:
%   - Implements the geometric setup and selective precomputations described
%     in Sections 2.1–2.5 of Periyandy & Bevis (2025):
%       Eq. (2)   : face normals n̂_f = (v2−v1)×(v3−v1) / ||·||
%       Eq. (3–6) : edge vectors e_ij, e_jk, e_ki and scalar products
%       Eq. (7–9) : geometric invariants for potential, acceleration, tensor
%
% Design principles:
%   • Common geometry (faces, vertex triplets, normals) is always computed.
%   • Potential precomputations: edge lengths, safe inverses, dot-products.
%   • Accel/tensor precomputations: face dyads + edge dyads/topology.
%   • No redundant work — each ensure_* method sets a flag once complete.
%
% Parallel control:
%   • NumThreads property used by chunk-based parallel execution.
%   • Threading granularity managed in higher-level ops (PotentialOps etc).
%
% Numerical model:
%   • Machine precision (float64).
%   • Epsilon guard (eps = 0) avoids divisions by near-zero lengths only.
%   • Orientation optionally enforced by centroid test (Eq. (1)).
%
% =======================================================================

classdef PolyhedronBase < handle
    properties
        % ------------------ Physical constants -------------------------
        G   (1,1) double = 1.0       % Gravitational constant
        rho (1,1) double = 1.0       % Density
        eps (1,1) double = 0.0       % Numerical epsilon 

        % ------------------ Mesh storage -------------------------------
        V double                     % (Nv x 3) vertices
        F int32                      % (Nf x 3) faces (triangular indices)

        % ------------------ Common precomputations ---------------------
        Vi double                    % (Nf x 3) vertex i of each face
        Vj double                    % (Nf x 3) vertex j of each face
        Vk double                    % (Nf x 3) vertex k of each face
        n_hat double                 % (Nf x 3) unit normals
        n_raw_norm double            % (Nf x 1) raw normal magnitudes

        % ------------------ Potential precomputations ------------------
        L_ij double; L_jk double; L_ki double
        inv_L_ij double; inv_L_jk double; inv_L_ki double
        dot_ij_ki double; dot_jk_ij double; dot_ki_jk double

        % ------------------ Accel/tensor precomputations ----------------
        Fdyads double                % (Nf x 3 x 3)
        edge_i int32; edge_j int32   % (Ne x 1) node indices
        edge_len double              % (Ne x 1)
        edge_E double                % (Ne x 3 x 3)
        edge_u double                % (Ne x 3)
        edge_v double                % (Ne x 3)

        % ------------------ Precomp flags -------------------------------
        have_potential (1,1) logical = false
        have_accel     (1,1) logical = false
        have_tensor    (1,1) logical = false

        % ------------------ Parallel configuration ---------------------
        NumThreads (1,1) double = NaN
    end

    methods
        % ===============================================================
        % Constructor: setup physical constants, mesh, and normals.
        % ===============================================================
        function obj = PolyhedronBase(vertices, faces, varargin)
            p = inputParser;
            p.addParameter('G', 1.0);
            p.addParameter('Density', 1.0);
            p.addParameter('Eps', 0.0);
            p.addParameter('OrientFaces', true);
            p.addParameter('NumThreads', []);
            p.parse(varargin{:});
            S = p.Results;

            % Physical constants
            obj.G   = double(S.G);
            obj.rho = double(S.Density);
            obj.eps = double(S.Eps);

            % Mesh
            obj.V = double(vertices);
            obj.F = int32(faces);

            % Common precomputation (normals, triplets)
            obj.precompute_common(S.OrientFaces);

            % Thread configuration (shared with higher-level ops)
            if isempty(S.NumThreads)
                try
                    pool = gcp('nocreate');
                    if isempty(pool)
                        obj.NumThreads = max(1, feature('numcores'));
                    else
                        obj.NumThreads = max(1, pool.NumWorkers);
                    end
                catch
                    obj.NumThreads = 1;
                end
            else
                obj.NumThreads = max(1, double(S.NumThreads));
            end
        end

        function delete(~), end
        function close(~),  end

        % ===============================================================
        % iter_chunks: returns { [s,e], ... } for N-point partitioning
        % Used by parfor-based methods (PotentialOps, AccelOps, TensorOps)
        % ===============================================================
        function chunks = iter_chunks(obj, N, block_size)
            if N <= 0
                chunks = {};
                return;
            end
            num_chunks = max(obj.NumThreads, min(N, ceil(N / block_size)));
            eff_bs = max(1, ceil(N / num_chunks));
            s = 1; chunks = {};
            while s <= N
                e = min(s + eff_bs - 1, N);
                chunks{end+1} = [s, e]; 
                s = e + 1;
            end
        end

        % ===============================================================
        % precompute_common
        % Compute per-face normals and oriented vertex triplets
        % ===============================================================
        function precompute_common(obj, orient_faces)
            F = obj.F;
            V = obj.V;

            Fi = F(:,1); Fj = F(:,2); Fk = F(:,3);
            v0 = V(Fi,:); v1 = V(Fj,:); v2 = V(Fk,:);
            normals_raw = cross(v1 - v0, v2 - v0, 2);

            if orient_faces
                centroid = mean(V, 1);
                face_centers = (v0 + v1 + v2) / 3.0;
                flip = dot(normals_raw, face_centers - centroid, 2) < 0.0;
                if any(flip)
                    obj.F(flip,:) = obj.F(flip,[1 3 2]);
                    Fi = obj.F(:,1); Fj = obj.F(:,2); Fk = obj.F(:,3);
                    v0 = V(Fi,:); v1 = V(Fj,:); v2 = V(Fk,:);
                    normals_raw = cross(v1 - v0, v2 - v0, 2);
                end
            end

            obj.Vi = V(obj.F(:,1),:);
            obj.Vj = V(obj.F(:,2),:);
            obj.Vk = V(obj.F(:,3),:);

            obj.n_raw_norm = vecnorm(normals_raw, 2, 2);
            nhat = zeros(size(normals_raw));
            valid = obj.n_raw_norm > obj.eps;
            nhat(valid,:) = normals_raw(valid,:) ./ obj.n_raw_norm(valid);
            obj.n_hat = nhat;
        end

        % ===============================================================
        % ensure_potential_precomp
        % Prepare potential-only invariants (Eq. 7–8)
        % ===============================================================
        function ensure_potential_precomp(obj)
            if obj.have_potential, return; end

            e_ij = obj.Vj - obj.Vi;
            e_jk = obj.Vk - obj.Vj;
            e_ki = obj.Vi - obj.Vk;

            obj.L_ij = vecnorm(e_ij, 2, 2);
            obj.L_jk = vecnorm(e_jk, 2, 2);
            obj.L_ki = vecnorm(e_ki, 2, 2);

            obj.inv_L_ij = zeros(size(obj.L_ij));
            obj.inv_L_jk = zeros(size(obj.L_jk));
            obj.inv_L_ki = zeros(size(obj.L_ki));

            nz_ij = obj.L_ij > obj.eps;
            nz_jk = obj.L_jk > obj.eps;
            nz_ki = obj.L_ki > obj.eps;

            obj.inv_L_ij(nz_ij) = 1.0 ./ obj.L_ij(nz_ij);
            obj.inv_L_jk(nz_jk) = 1.0 ./ obj.L_jk(nz_jk);
            obj.inv_L_ki(nz_ki) = 1.0 ./ obj.L_ki(nz_ki);

            obj.dot_ij_ki = sum(e_ij .* e_ki, 2);
            obj.dot_jk_ij = sum(e_jk .* e_ij, 2);
            obj.dot_ki_jk = sum(e_ki .* e_jk, 2);

            obj.have_potential = true;
        end

        % ===============================================================
        % ensure_accel_tensor_precomp
        % Compute face dyads and edge dyads shared by accel/tensor ops
        % ===============================================================
        function ensure_accel_tensor_precomp(obj)
            if obj.have_accel, return; end

            nhat = obj.n_hat;
            Fcount = size(nhat,1);
            obj.Fdyads = zeros(Fcount,3,3);
            for f = 1:Fcount
                n = nhat(f,:);
                obj.Fdyads(f,:,:) = n' * n;
            end

            % --- Edge connectivity and dyad construction ---
            edge_to_faces = containers.Map('KeyType','char','ValueType','any');
            edge_map = containers.Map('KeyType','char','ValueType','any');
            F = obj.F;

            for f_idx = 1:size(F,1)
                a = F(f_idx,1); b = F(f_idx,2); c = F(f_idx,3);
                edges = [a b; b c; c a];
                for ei = 1:3
                    u = edges(ei,1); v = edges(ei,2);
                    key = sprintf('%d_%d', min(u,v), max(u,v));
                    if ~isKey(edge_to_faces, key)
                        edge_to_faces(key) = [f_idx];
                    else
                        edge_to_faces(key) = [edge_to_faces(key) f_idx];
                    end
                    edge_map(key) = [u v];
                end
            end

            edge_i = []; edge_j = []; edge_len = []; edge_E = {};
            V = obj.V; nh = obj.n_hat;
            for key = keys(edge_to_faces)
                key = key{1};
                faces = edge_to_faces(key);
                if numel(faces) ~= 2, continue; end
                uv = edge_map(key); u = uv(1); v = uv(2);
                fa = faces(1); fb = faces(2);
                na = nh(fa,:); nb = nh(fb,:);

                evec = V(v,:) - V(u,:);
                L = norm(evec);
                if L <= obj.eps, continue; end
                ehat = evec / L;

                face_a = F(fa,:);
                off = face_a(~ismember(face_a, [u v]));
                off_pos = find(face_a == off, 1, 'first');
                after_off = face_a(mod(off_pos,3) + 1);

                tA = cross(na, ehat);
                tB = cross(nb, ehat);
                if after_off == u
                    tA = -tA;
                else
                    tB = -tB;
                end
                Ee = (na' * tA) + (nb' * tB);  % 3×3 dyad

                edge_i(end+1) = u; 
                edge_j(end+1) = v; 
                edge_len(end+1) = L; 
                edge_E{end+1} = Ee; 
            end

            obj.edge_i = int32(edge_i(:));
            obj.edge_j = int32(edge_j(:));
            obj.edge_len = edge_len(:);
            obj.edge_E = permute(cat(3, edge_E{:}), [3 1 2]);
            obj.edge_u = V(obj.edge_i,:);
            obj.edge_v = V(obj.edge_j,:);

            obj.have_accel  = true;
            obj.have_tensor = true;
        end
    end
end
