# PolyGravitation

**Analytical Gravitational Field of a Homogeneous Polyhedron**

Multi-language implementation (Python, MATLAB, Julia) of the singularity-free closed-form formulation for the gravitational potential, acceleration, and gravity-gradient tensor of a homogeneous polyhedron.

---

## Reference

> Periyandy, T. & Bevis, M. (2025).  
> *The Gravitational Field of a Homogeneous Polyhedron.*  
> Division of Geodetic Science, School of Earth Sciences, The Ohio State University.

**Authors:**  
- Thunendran Periyandy — Sabaragamuwa University of Sri Lanka / The Ohio State University  
- Michael Bevis — The Ohio State University  

Correspondence: thunendran@gmail.com

---

## Overview

This repository provides the complete supplementary code for the manuscript above. All three implementations compute:

| Quantity | Symbol | Formula type |
|---|---|---|
| Gravitational potential | U(P) | Face-summation integral |
| Gravitational acceleration | g(P) | Dyadic edge/face formulation |
| Gravity gradient tensor | Γ(P) | Second derivative of U |

The formulation is:
- **Singularity-free** — regularised logarithmic and arctangent kernels
- **Valid everywhere** — interior, surface, and exterior evaluation points
- **Machine-precision** — float64 (10⁻¹⁵) in Python/MATLAB/Julia standard; arbitrary precision in Julia BigFloat variant
- **Vectorised and parallelised** in all three languages

---

## Repository Structure

```
PolyGravitation/
├── python/
│   ├── polygravitation/          # Core Python package (float64)
│   │   ├── core.py               # PolyhedronGravitation class (main API)
│   │   ├── base.py               # Geometry, threading, precomputation
│   │   ├── potential.py          # Gravitational potential U(P)
│   │   ├── acceleration.py       # Acceleration g(P)
│   │   ├── tensor.py             # Gravity tensor Γ(P)
│   │   └── helpers.py            # Stable log/arctan/solid-angle kernels
│   ├── Examples.ipynb            # All worked examples (Jupyter)
│   ├── BenchmarkTest.ipynb       # Speed benchmarks
│   ├── benchmark_accel_tensor.ipynb
│   └── *.png / *.pdf             # Output figures
│
├── matlab/
│   ├── polygravitation/          # Core MATLAB package
│   │   ├── PolyhedronGravitation.m   # Unified user API
│   │   ├── PolyhedronBase.m          # Geometry preprocessing
│   │   ├── PotentialOps.m            # Potential computation
│   │   ├── AccelOps.m                # Acceleration computation
│   │   ├── TensorOps.m               # Tensor computation
│   │   └── helpers.m                 # Kernel functions
│   ├── Example_01_Tetrahedron.m  … Example_12_Benchmark_GravityTensor.m
│   └── *.png                     # Output figures
│
├── julia/
│   ├── PolyGravitation/          # Standard float64 Julia module
│   │   ├── Polyhedron_Gravitation.jl
│   │   ├── PolyhedronBase.jl
│   │   ├── PolyhedronPotential.jl
│   │   ├── PolyhedronAcceleration.jl
│   │   ├── PolyhedronTensor.jl
│   │   └── PolyhedronHelpers.jl
│   ├── PolyGravitationBig/       # Arbitrary-precision BigFloat variant
│   │   └── (mirrors above with _Big suffix)
│   ├── Example_01 … Example_08   # Worked example scripts
│   └── benchmark_julia_*.jl      # Performance and precision benchmarks
│
└── data/
    ├── icosahedron_vertices.csv
    ├── icosahedron_faces.csv
    ├── eval_points_100k_plus_vertices.csv
    ├── A_ref_bigfloat_250digits.txt   # Reference solution (250-digit precision)
    └── summary_runtime_accuracy.txt
```

---

## Performance Summary

Reference solution: Julia BigFloat at 250 significant digits.

| Implementation | Runtime (s/point) | Mean error | RMS error |
|---|---|---|---|
| Python (float64) | 1.98 × 10⁻⁶ | 3.21 × 10⁻¹⁵ | 3.61 × 10⁻¹⁵ |
| MATLAB (float64) | 6.04 × 10⁻⁶ | 4.81 × 10⁻¹⁵ | 5.73 × 10⁻¹⁵ |
| Julia (float64) | 4.90 × 10⁻⁷ | 3.02 × 10⁻¹⁵ | 3.42 × 10⁻¹⁵ |
| Julia BigFloat (50 digits) | 2.00 × 10⁻⁴ | 5.56 × 10⁻⁵⁰ | 7.22 × 10⁻⁵⁰ |

---

## Requirements

### Python
- Python >= 3.9
- NumPy (latest stable)

### MATLAB
- MATLAB R2023a or later
- Parallel Computing Toolbox (for `parfor` benchmarks)

### Julia
- Julia >= 1.9
- Standard library only (`LinearAlgebra`, `Statistics`)
- No external packages required

---

## Quick Start

### Python

```python
import numpy as np
from polygravitation import PolyhedronGravitation

# Cube: 8 vertices, 12 triangular faces
V = np.array([
    [-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
    [-1,-1, 1],[1,-1, 1],[1,1, 1],[-1,1, 1]
], dtype=float)
F = np.array([
    [0,1,2],[0,2,3],[4,5,6],[4,6,7],
    [0,1,5],[0,5,4],[1,2,6],[1,6,5],
    [2,3,7],[2,7,6],[3,0,4],[3,4,7]
])

model = PolyhedronGravitation(V, F, G=6.674e-11, density=2500.0)

P = np.array([[0,0,5],[2,0,0],[0,0,0]])
U = model.potential(P)       # Gravitational potential
g = model.acceleration(P)   # Acceleration vector (N×3)
T = model.gravity_tensor(P) # Gradient tensor (N×3×3)

model.close()
```

### MATLAB

```matlab
addpath('matlab/polygravitation')

% Build polyhedron model
model = PolyhedronGravitation(V, F, G, density);

% Evaluate
U = model.potential(P);
g = model.acceleration(P);
T = model.gravity_tensor(P);
```

### Julia

```julia
include("julia/PolyGravitation/Polyhedron_Gravitation.jl")
using .Polyhedron_Gravitation

poly = build_polyhedron(V, F; G=6.674e-11, density=2500.0)
U = potential(poly, P)
g = acceleration(poly, P)
T = gravity_tensor(poly, P)
```

---

## Examples

| # | Description |
|---|---|
| 01 | Basic gravitational field evaluation (tetrahedron) |
| 02 | Laplacian field slice |
| 03 | Concave L-shape Laplacian verification |
| 04–05 | Torus Laplacian (horizontal and vertical slices) |
| 06–07 | Torus acceleration field (horizontal and vertical) |
| 08 | Zero-g circle fit on torus |
| 09 | Torus zero-g vertical plane |
| 10–12 | Benchmark: potential, acceleration, gravity tensor |
| 13 | Cross-language speed comparison |

All examples are available as Jupyter notebooks (Python), `.m` scripts (MATLAB), and `.jl` scripts (Julia).

---

## Validation

The implementations are verified by:
- **Laplacian test**: ∇²U = −4πGρ (inside), 0 (outside)
- **Poisson consistency**: ∇ · g = −4πGρ (inside), 0 (outside)
- **Trace of gradient tensor**: tr(Γ) = −4πGρ (inside)
- **Cross-language agreement**: mean error < 10⁻¹⁴ across all three implementations
- **Reference benchmark**: Julia BigFloat at 250-digit precision

---

## Citation

If you use this code, please cite:

```
Periyandy, T. & Bevis, M. (2025).
The Gravitational Field of a Homogeneous Polyhedron.
Division of Geodetic Science, School of Earth Sciences,
The Ohio State University.
```

---

## License

This code is made available as supplementary material to the manuscript.
Please contact the authors for reuse beyond academic research.
