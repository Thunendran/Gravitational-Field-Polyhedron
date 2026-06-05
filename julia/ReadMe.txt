===============================================================
README — PolyGravitation Julia Framework
===============================================================

Author:  Thunendran Periyandy
Advisor: Michael Bevis
Project:  The Gravitational Field of a Homogeneous Polyhedron
Date:     November 2025

---------------------------------------------------------------
Overview
---------------------------------------------------------------
This Julia framework implements a unified and high-precision model
for evaluating the gravitational field (potential, acceleration,
and gravity tensor) of a homogeneous polyhedron.

Two precision modes are supported:
  • PolyGravitation     →  Float64 (fast machine-precision)
  • PolyGravitationBig  →  BigFloat (arbitrary precision)

The framework provides modular structure, benchmark scripts,
and reproducible examples for scientific and geophysical studies.

---------------------------------------------------------------
Directory Structure
---------------------------------------------------------------

📁 Root Folder
│
├── PolyGravitation/
│   ├── Polyhedron_Gravitation.jl         # Main Float64 interface
│   ├── PolyhedronBase.jl                 # Core geometry + precomputations
│   ├── PolyhedronPotential.jl            # Potential field routines
│   ├── PolyhedronAcceleration.jl         # Acceleration routines
│   ├── PolyhedronTensor.jl               # Gravity tensor routines
│   └── PolyhedronHelpers.jl              # Vector and geometry utilities
│
├── PolyGravitationBig/
│   ├── Polyhedron_Gravitation_Big.jl     # Main BigFloat interface
│   ├── PolyhedronBase_Big.jl             # High-precision geometry setup
│   ├── PolyhedronPotential_Big.jl        # BigFloat potential
│   ├── PolyhedronAcceleration_Big.jl     # BigFloat acceleration
│   ├── PolyhedronTensor_Big.jl           # BigFloat tensor
│   └── PolyhedronHelpers_Big.jl          # High-precision utilities
│
├── data/
│   ├── icosahedron_vertices.csv          # Vertex list
│   ├── icosahedron_faces.csv             # Face topology
│   ├── eval_points_100k_plus_vertices.csv # Evaluation points
│   └── (Output benchmark results and reference data)
│
├── benchmark_julia.jl                    # Potential benchmark
├── benchmark_julia_accel.jl              # Acceleration benchmark
├── benchmark_julia_tensor.jl             # Tensor benchmark
├── precision_scaling_potential.jl        # Precision vs runtime scaling
│
└── Example_*.jl                          # Example field and Laplacian tests

---------------------------------------------------------------
Benchmark Scripts
---------------------------------------------------------------

1️ benchmark_julia_Float64_Big50_Big250.jl
   • Compares Float64, BigFloat(50), and BigFloat(250) potential evaluations.
   • Uses selective precomputations (mode=:potential).

2️ benchmark_julia_accel.jl
   • Benchmarks acceleration computations for the same polyhedron geometry.
   • Runs all three precision levels (Float64, 50-digit, 250-digit).
   • Uses mode=:acceleration.

3️• Evaluates the full gravity gradient tensor.
   • Tests both speed and precision scaling.
   • Uses mode=:tensor.

4️ precision_scaling_potential.jl
   • Sweeps BigFloat precision from 20–200 digits.
   • Measures runtime vs mean relative error compared to a 250-digit reference.
   • Generates log-log scaling plots.

---------------------------------------------------------------
Usage
---------------------------------------------------------------
Run benchmarks directly from the Julia REPL or terminal:

    julia benchmark_julia.jl
    julia benchmark_julia_accel.jl
    julia benchmark_julia_tensor.jl
    julia precision_scaling_potential.jl

All results and timing summaries are automatically saved in:
    ../data/

---------------------------------------------------------------
Notes
---------------------------------------------------------------
• Each script automatically controls BLAS threading
  to ensure consistent timing results.
• The PolyGravitation modules handle mode-specific
  precomputations (potential, acceleration, tensor).
• BigFloat precision can be adjusted using setprecision().

---------------------------------------------------------------
End of README
===============================================================
