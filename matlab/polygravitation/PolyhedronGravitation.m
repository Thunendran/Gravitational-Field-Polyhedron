% =======================================================================
% PolyhedronGravitation.m
% Unified interface for polyhedral gravitational field computations
%
% Manuscript alignment:
%   Implements Section 3 ("Unified Field Evaluation") of
%   Periyandy & Bevis (2025), where the gravitational potential (U),
%   acceleration (g), and gravity tensor (Γ) are computed from a
%   homogeneous polyhedron mesh.
%
% Purpose:
%   Provides a single, user-facing class that aggregates the core base
%   geometry and all operator modules:
%       - PotentialOps   : scalar potential   U(P)
%       - AccelOps       : vector acceleration g(P)
%       - TensorOps      : symmetric tensor   Γ(P)
%       - PolyhedronBase : geometry + precomputation management
%
% Usage:
%   model = PolyhedronGravitation(V, F, ...
%              'G', 1, ...
%              'Density', 1, ...
%              'Eps', 0.0, ...
%              'OrientFaces', true, ...
%              'NumThreads', 8);
%
%   U  = model.potential(points);
%   g  = model.acceleration(points);
%   Γ  = model.gravity_tensor(points);
%
% Design principles (see manuscript §2.7 and §3.2):
%   • Selective precomputations:
%       - potential()            → only edge lengths & dot products (Eq. 7–8)
%       - acceleration/tensor()  → face dyads + edge dyads/topology (Eq. 9–12)
%   • Parallelism:
%       - High-level methods internally chunk points (parfor friendly)
%       - No nested parallel loops (thread-safe structure)
%   • Numerical stability:
%       - Epsilon (eps=0) only guards divisions by ~0 (no analytic bias)
%       - Double precision throughout
%
% File organization:
%   PolyhedronGravitation.m  (this core file)
%   PotentialOps.m           (U)
%   AccelOps.m               (g)
%   TensorOps.m              (Γ)
%   PolyhedronBase.m         (geometry + precompute)
%   helpers.m                (safe log/arctan/solid-angle kernels)
%
% References:
%   “The Gravitational Field of a Homogeneous Polyhedron”
%   Periyandy & Bevis (2025), Sections 2–3.
% =======================================================================

classdef PolyhedronGravitation < TensorOps & AccelOps & PotentialOps & PolyhedronBase & handle


    methods
        % ---------------------------------------------------------------
        % Constructor: initializes geometry and shared parameters.
        %
        % Inherits the PolyhedronBase constructor which performs:
        %   - vertex/face loading
        %   - normal orientation 
        %   - NumThreads selection
        % ---------------------------------------------------------------
        function obj = PolyhedronGravitation(V, F, varargin)
            obj@PolyhedronBase(V, F, varargin{:});
        end
    end
end
