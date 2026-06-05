=======================================================================
PolyGravitation — MATLAB Analytical Polyhedral Gravitation Framework
=======================================================================

Authors:     Thunendran Periyandy and Michael Bevis
Affiliations: The Ohio State University; Sabaragamuwa University of Sri Lanka
Manuscript:  "The Gravitational Field of a Homogeneous Polyhedron" (2025)
Language:    MATLAB R2023a 
=======================================================================

DESCRIPTION
=======================================================================
This repository implements the singularity-free analytical formulation
for the gravitational potential, acceleration, and gravity-gradient tensor
of a homogeneous polyhedron, as described in the manuscript:

   Periyandy & Bevis (2025), "The Gravitational Field of a Homogeneous Polyhedron"

The formulation analytically regularizes the logarithmic and arctangent
terms, ensuring continuity and numerical stability across interior,
boundary, and exterior domains. It employs dyadic tensor derivatives,
vectorized matrix operations, and MATLAB parallelization to deliver
physically consistent and computationally efficient results.


=======================================================================
FOLDER STRUCTURE
=======================================================================

PolyGravitation/
│
├── helpers.m
│     Core analytical kernels: logarithmic, arctangent, and solid angle terms
│
├── PolyhedronBase.m
│     Geometry preprocessing, connectivity, and reusable face/edge topology
│
├── PotentialOps.m
│     Closed-form gravitational potential computation (singularity-free)
│
├── AccelOps.m
│     Analytical gravitational acceleration (dyadic formulation)
│
├── TensorOps.m
│     Analytical gravity tensor (∇∇ᵗU) with full physical consistency
│
├── PolyhedronGravitation.m
│     Unified user-facing API with support for parallel and chunked evaluation
│
├── data/
│     Input and benchmark data:
│       - icosahedron_vertices.csv
│       - icosahedron_faces.csv
│       - eval_points_100k_plus_vertices.csv
│       - U_matlab_parallel.csv
│       - A_matlab_parallel.csv
│       - T_matlab_parallel.csv
│       - time_matlab_parallel.txt
│       - time_accel_matlab_parallel.txt
│       - time_tensor_matlab_parallel.txt
│
└── examples/

=======================================================================
IMPLEMENTATION NOTES
=======================================================================

• All routines are vectorized and parallelized using MATLAB’s parfor constructs.
• Parallel pool (parpool) remains open between runs to minimize overhead.
• Geometric quantities are precomputed for performance and reused across evaluations.
• Data CSV files must reside in the /data directory.
• Output benchmarks are saved automatically into the same /data folder.
• Compatible with large-scale (≥10⁵ point) evaluations on multicore CPUs.

=======================================================================
CITATION
=======================================================================
If you use this framework, please cite:

   Periyandy, T. & Bevis, M. (2025).
   "The Gravitational Field of a Homogeneous Polyhedron."
   Division of Geodetic Science, The Ohio State University.

=======================================================================
END OF FILE
=======================================================================
