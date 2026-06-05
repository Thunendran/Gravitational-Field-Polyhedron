% =======================================================================
% helpers.m
% Supporting mathematical utilities for Polyhedral Gravitation
%
% Manuscript alignment (equations & symbols):
%   • Safe edge logarithm  L_e  (Eq. (7)): 
%       L_e(P) = ln((r_a + r_b + L_ab) / (r_a + r_b - L_ab))
%     Implemented as log_term(ra, rb, rab, epsval).
%
%   • Stable arctangent kernel S_ℓ (Eq. (8)):
%       S_ℓ = atan2( Δz ||n_raw|| , -(det_mn det_nℓ + Δz^2 c_ℓ)/r_ℓ )
%     Implemented as arctan_term(numerator, det_a, det_b, dz2, dot_ab, r, epsval)
%     where:
%       numerator = Δz ||n_raw||,
%       det_a, det_b are the projected 2× areas (Eq. (5)),
%       dz2 = (Δz)^2 (Eq. (4)),
%       dot_ab = c_ℓ (Eq. (6)).
%
%   • Signed solid angle ω_f(P) (Eqs. (9), (20)):
%       ω_f(P) = ∬_f ( (x - P) · n̂_f ) / ||x - P||^3  dS(x)
%     Implemented as solid_angle_tri(P, v1, v2, v3, epsval).
%
% Numerical design (manuscript Section 2.7):
%   • All routines are singularity-resistant:
%       - log_term only evaluates logs where denominators are positive and L_ab > eps.
%       - arctan_term uses atan2 with guarded denominator scaling.
%       - solid_angle_tri uses a triple product / robust denominator with an ε addend.
%
% Shapes:
%   • P     : (B x 3) batch of field points
%   • v1,v2,v3 : (F x 3) face vertices (Vi, Vj, Vk)
%   • ra,rb : (B x F) distances to edge endpoints
%   • rab   : (F x 1) or (1 x F) edge lengths L_ab
%   • Outputs are sized to match the per-face/batch layout:
%       - log_term  : (B x F)     (or (1 x F) if B = 1 in arclog use)
%       - arctan_term: (B x F)
%       - solid_angle_tri: (B x F)
%
% Precision:
%   • All computations are in double precision. Set epsval = 0 for strict
%   evaluation.
%
% References:
%   “The Gravitational Field of a Homogeneous Polyhedron”
%   Periyandy & Bevis (2025) — equations cited above.
% =======================================================================

classdef helpers
    methods (Static)
        % ---------------------------------------------------------------
        % log_term
        % Safe edge logarithm:
        %   log_term(ra, rb, rab, epsval) = ln((ra + rb + rab) / (ra + rb - rab))
        % Vectorized over batches (rows) and faces/edges (cols).
        %
        % Inputs:
        %   ra, rb : (B x F) distances
        %   rab    : (F x 1) or (1 x F) edge lengths L_ab
        %   epsval : scalar ε ≥ 0
        %
        % Output:
        %   out    : (B x F)
        % ---------------------------------------------------------------
        function out = log_term(ra, rb, rab, epsval)
            % Ensure rab is a row (1 x F) for broadcasting with (B x F)
            if iscolumn(rab); rab = rab.'; end
            sum_ab = ra + rb;                      % (B x F)
            rab_b  = rab;                          % (1 x F)
            num    = sum_ab + rab_b;               % (B x F)
            den    = sum_ab - rab_b;               % (B x F)

            % Valid where L_ab > eps and denominator > eps
            ok  = (rab_b > epsval) & (den > epsval);  % (B x F), implicit expansion

            out = zeros(size(ra), 'like', ra);
            if any(ok(:))
                % Only evaluate log where well-conditioned; elsewhere leave 0
                out(ok) = log(num(ok) ./ den(ok));
            end
        end

        % ---------------------------------------------------------------
        % arctan_term
        % Stable arctangent kernel used by the potential face-sum:
        %
        % Inputs:
        %   numerator = Δz * ||n_raw||          : (B x F)
        %   det_a, det_b                        : (B x F) projected areas (Eq. (5))
        %   dz2 = (Δz)^2                         : (B x F)
        %   dot_ab = c_ℓ                         : (1 x F) or (B x F)
        %   r  = r_ℓ                             : (B x F)
        %   epsval                               : scalar ε ≥ 0
        %
        % Output:
        %   out                                   : (B x F)
        % ---------------------------------------------------------------
        function out = arctan_term(numerator, det_a, det_b, dz2, dot_ab, r, epsval)
            % Make sure dot_ab can broadcast across (B x F)
            if isvector(dot_ab) && (size(dot_ab,1)==1 || size(dot_ab,2)==1)
                % force row
                dot_ab = dot_ab(:).';   % (1 x F)
            end

            out = zeros(size(r), 'like', r);
            ok  = (r > epsval);

            % denom = -((det_b * det_a) + dz2 * dot_ab)
            denom = - ( (det_b .* det_a) + dz2 .* dot_ab );  % (B x F)

            % safe division; only divide where r > eps
            x = zeros(size(r), 'like', r);
            y = zeros(size(r), 'like', r);
            x(ok) = denom(ok) ./ r(ok);
            y(ok) = numerator(ok);

            out(ok) = atan2(y(ok), x(ok));
        end

        % ---------------------------------------------------------------
        % solid_angle_tri
        % Signed solid angle of triangle (v1,v2,v3) as seen from points P.
        %
        % Inputs:
        %   P      : (B x 3)
        %   v1,v2,v3 : (F x 3)
        %   epsval : scalar ε ≥ 0   (added to denominator to avoid 0/0)
        %
        % Output:
        %   omega  : (B x F)
        % ---------------------------------------------------------------
        function omega = solid_angle_tri(P, v1, v2, v3, epsval)
            % Relative vectors r1,r2,r3 : (B x F x 3)
            r1 = permute(v1, [3 1 2]) - permute(P, [1 3 2]);
            r2 = permute(v2, [3 1 2]) - permute(P, [1 3 2]);
            r3 = permute(v3, [3 1 2]) - permute(P, [1 3 2]);

            % Norms (B x F)
            r1n = vecnorm(r1, 2, 3);
            r2n = vecnorm(r2, 2, 3);
            r3n = vecnorm(r3, 2, 3);

            % Scalar triple product (B x F)
            triple = dot(cross(r2, r3, 3), r1, 3);

            % Denominator (B x F) — robust form per manuscript Section 2.7
            denom = (r1n .* r2n .* r3n) ...
                  + (dot(r1, r2, 3) .* r3n) ...
                  + (dot(r2, r3, 3) .* r1n) ...
                  + (dot(r3, r1, 3) .* r2n);

            % Add epsval to avoid 0/0 when r’s become collinear and norms vanish
            omega = 2.0 * atan2(triple, denom + epsval);
        end
    end
end
