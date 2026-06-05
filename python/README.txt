======================================================================
README.txt
======================================================================
Title: PolyGravitation - Analytical Gravitational Field of a Homogeneous Polyhedron
Authors: Thunendran Periyandy and Michael Bevis
Affiliations:
  - Division of Geodetic Science, School of Earth Sciences, The Ohio State University
  - Faculty of Geomatics, Sabaragamuwa University of Sri Lanka
Correspondence: thunendran@gmail.com
Date: November 2025
======================================================================

1. OVERVIEW
----------------------------------------------------------------------
This package provides a complete analytical implementation of the 
gravitational potential, acceleration, and gravity gradient tensor 
of a homogeneous polyhedron. 

The implementation corresponds to the formulations described in the 
manuscript:
    Periyandy, T. & Bevis, M. (2025)
    "The Gravitational Field of a Homogeneous Polyhedron"

The model is singularity-free, numerically stable, and valid for 
points located inside, on the surface of, or outside a polyhedral 
body.

This code is written in pure Python (NumPy-based), fully vectorized, 
and parallelized using a thread pool for multi-core evaluation.

----------------------------------------------------------------------

2. DIRECTORY STRUCTURE
----------------------------------------------------------------------
The main folder "PolyGravitation" contains the following scripts:

PolyGravitation/
│
├── base.py
│   Core geometry and threading management.
│   - Stores vertex and face arrays (V, F)
│   - Computes face normals and vertex triplets
│   - Handles optional face orientation
│   - Provides selective precomputation routines:
│       _ensure_potential_precomp()   → edge lengths, inverses, dot products
│       _ensure_accel_tensor_precomp()→ face/edge dyads and topology
│   - Manages ThreadPoolExecutor for point-chunk parallelism

├── core.py
│   Main user interface (class PolyhedronGravitation).
│   - Combines all mixins: PotentialOps, AccelOps, TensorOps, and PolyhedronBase
│   - Provides unified API:
│        model.potential(points)
│        model.acceleration(points)
│        model.gravity_tensor(points)
│   - Handles thread management and precomputation control

├── potential.py
│   Implements the analytical face-summation formula for the scalar 
│   gravitational potential:
│       U(P) = -Gρ * Σ_f [ ... face terms ... ]
│   - Uses only potential-specific precomputations (edges, dot products)
│   - Parallelized over point chunks via ThreadPoolExecutor
│   - Returns array of potentials for all evaluation points

├── acceleration.py
│   Implements the dyadic acceleration formulation:
│       g(P) = Gρ [ Σ_f ω_f(P) (F_f r_f) − Σ_e L_e(P) (E_e r_u) ]
│   - Uses face dyads and edge dyads computed in base.py
│   - Parallelized over point chunks
│   - Returns acceleration vectors (N x 3)

├── tensor.py
│   Computes the gravity gradient tensor (second derivative of potential):
│       Γ(P) = Gρ [ Σ_e L_e(P) E_e − Σ_f ω_f(P) F_f ]
│   - Shares precomputed face and edge dyads with acceleration
│   - Returns symmetric 3x3 matrices per point (N x 3 x 3)

├── helpers.py
│   Provides numerically stable helper functions used in all computations:
│       log_term()        → safe logarithmic kernel
│       arctan_term()     → stable arctangent kernel
│       solid_angle_tri() → signed solid angle of a triangle as seen from P
│   - Vectorized for batch operations
│   - Used to avoid singularities near faces and edges

----------------------------------------------------------------------

3. INSTALLATION
----------------------------------------------------------------------
Requirements:
    Python >= 3.9
    numpy (latest stable version)

To use the package, copy the "PolyGravitation" folder into your 
project or working directory, then import as follows:

    from PolyGravitation.core import PolyhedronGravitation

----------------------------------------------------------------------

4. USAGE EXAMPLE
----------------------------------------------------------------------
Example: compute potential, acceleration, and gravity tensor of a cube.

    import numpy as np
    from PolyGravitation.core import PolyhedronGravitation

    # Define vertices and faces
    V = np.array([
        [-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
        [-1,-1, 1],[1,-1, 1],[1,1, 1],[-1,1, 1]
    ])
    F = np.array([
        [0,1,2],[0,2,3],
        [4,5,6],[4,6,7],
        [0,1,5],[0,5,4],
        [1,2,6],[1,6,5],
        [2,3,7],[2,7,6],
        [3,0,4],[3,4,7]
    ])

    # Initialize model
    model = PolyhedronGravitation(V, F, G=6.674e-11, density=2500.0)

    # Points of evaluation
    P = np.array([[0,0,5],[2,0,0],[0,0,0]])

    # Compute potential, acceleration, tensor
    U = model.potential(P)
    g = model.acceleration(P)
    Gt = model.gravity_tensor(P)

    print("Potential:", U)
    print("Acceleration:", g)
    print("Tensor:", Gt)

    model.close()  # Properly close thread pool

----------------------------------------------------------------------

5. DESIGN PRINCIPLES
----------------------------------------------------------------------
- Double precision (float64) used throughout.
- Memory layout: all arrays C-contiguous for BLAS optimization.
- Thread-safe: one global thread pool, no nested threads.
- Adaptive computation:
    • small N → serial evaluation
    • large N → parallel evaluation
- Automatic face orientation based on centroid normal direction.
- Selective caching of precomputed invariants:
    Only recomputes what each operation (potential/accel/tensor) requires.

----------------------------------------------------------------------

6. JUPYTER NOTEBOOK
----------------------------------------------------------------------
The Jupyter notebook accompanying this package reproduces all numerical
tests, validation results, and figures discussed in the manuscript.
It demonstrates:

    - Evaluation inside, on, and outside polyhedron
    - Laplacian and Poisson verification tests
    - Singularity elimination comparisons
    - Benchmarking and performance scaling

----------------------------------------------------------------------

7. VALIDATION SUMMARY
----------------------------------------------------------------------
- Results verified against analytical test cases (sphere, cube)
- Laplacian consistency:
      ∇²U = -4πGρ (inside), 0 (outside)
- Machine precision stability (10⁻¹⁴ to 10⁻¹⁶)
- Run
