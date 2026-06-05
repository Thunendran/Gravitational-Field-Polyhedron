#!/usr/bin/env python3
# ===============================================================
# Example_13_SpeedComparison_PolyhedronModels.py
# ===============================================================
# SPEED COMPARISON:
#   • Periyandy & Bevis (2025, multi-threaded vectorized)
#   • Werner–Arribas (legacy, serial loop)
#   • Finite-difference numerical baseline
# ===============================================================

import os, time, numpy as np, matplotlib.pyplot as plt, psutil

# ---------------------------------------------------------------
# 0. Configure threading BEFORE importing math-heavy libraries
# ---------------------------------------------------------------
num_threads = 8  # <=== set number of CPU threads to use
os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
os.environ["MKL_NUM_THREADS"] = str(num_threads)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)

print(f"Configured all math backends to use {num_threads} threads.\n")

# ---------------------------------------------------------------
# 1. Import models
# ---------------------------------------------------------------
from polygrav import PolyhedronGravitation
from Polyhedron_WS.GP_Polyhedron_WS import Polyhedron as Polyhedron_WS


# ===============================================================
# 2. Numerical Helper Functions (Finite Differences)
# ===============================================================
def g_num_class(model, points):
    """Compute numerical acceleration via finite differences."""
    h = 1e-7
    accelerations = np.zeros_like(points)
    for i, p in enumerate(points):
        x, y, z = p
        φ = lambda pt: model.potential(pt)
        gx = -(φ([x + h, y, z]) - φ([x - h, y, z])) / (2 * h)
        gy = -(φ([x, y + h, z]) - φ([x, y - h, z])) / (2 * h)
        gz = -(φ([x, y, z + h]) - φ([x, y, z - h])) / (2 * h)
        accelerations[i, :] = [gx, gy, gz]
    return accelerations


def tensor_num_class(model, points):
    """Compute numerical gravity tensor via finite differences."""
    h = 1e-7
    tensors = np.zeros((points.shape[0], 3, 3))
    for i, p in enumerate(points):
        x, y, z = p
        g_xp = model.acceleration([x + h, y, z])
        g_xm = model.acceleration([x - h, y, z])
        g_yp = model.acceleration([x, y + h, z])
        g_ym = model.acceleration([x, y - h, z])
        g_zp = model.acceleration([x, y, z + h])
        g_zm = model.acceleration([x, y, z - h])
        col_x = (g_xp - g_xm) / (2 * h)
        col_y = (g_yp - g_ym) / (2 * h)
        col_z = (g_zp - g_zm) / (2 * h)
        tensors[i, :, :] = np.array([col_x, col_y, col_z]).T
    return tensors


# ===============================================================
# 3. Geometry Setup — Unit Cube
# ===============================================================
vertices = np.array([
    [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5], [ 0.5,  0.5, -0.5], [-0.5,  0.5, -0.5],
    [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5], [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5]
])
faces = np.array([
    [0, 3, 2], [0, 2, 1], [4, 5, 6], [4, 6, 7],
    [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
    [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]
], dtype=np.int32)

# Instantiate analytical models
model_pb = PolyhedronGravitation(vertices=vertices, faces=faces,
                                 density=1.0, G=1.0, orient_faces=True)
model_ws = Polyhedron_WS(vertices=vertices, face_indexes=faces, density=1.0)

print("Initialized analytical models:")
print(" • Periyandy & Bevis (vectorized, multi-threaded)")
print(" • Werner–Arribas (legacy, serial loop)\n")


# ===============================================================
# 4. Benchmark Setup
# ===============================================================
# Smoother and wider evaluation range (50 → 1,000,000, 100 steps)
sizes = np.logspace(np.log10(50), np.log10(1_000_000), 100, dtype=int)

results = {
    "Periyandy & Bevis": {"Potential": [], "Acceleration": [], "Tensor": []},
    "Werner–Arribas": {"Potential": [], "Acceleration": [], "Tensor": []},
    "Numerical": {"Acceleration": [], "Tensor": []}
}
num_accel_sizes, num_tensor_sizes = [], []


def measure_time(label, func):
    """Utility to measure runtime and print results."""
    t0 = time.perf_counter()
    func()
    dt = time.perf_counter() - t0
    print(f"  {label:<25s}: {dt:8.3f} s")
    return dt


# ===============================================================
# 5. Run Benchmarks
# ===============================================================
print(f"--- Starting Speed Benchmark (50 → 1,000,000 points, {len(sizes)} steps) ---\n")

for n in sizes:
    print(f"Running {n:>8d} points ...")

    points = np.random.rand(n, 3) * 5.0 + 2.0  # points outside cube

    # --- Periyandy & Bevis (vectorized, multi-threaded) ---
    results["Periyandy & Bevis"]["Potential"].append(
        measure_time("PB Potential", lambda: model_pb.potential(points))
    )
    results["Periyandy & Bevis"]["Acceleration"].append(
        measure_time("PB Acceleration", lambda: model_pb.acceleration(points))
    )
    results["Periyandy & Bevis"]["Tensor"].append(
        measure_time("PB Tensor", lambda: model_pb.gravity_tensor(points))
    )

    # --- Werner–Arribas (serial, loop) ---
    results["Werner–Arribas"]["Potential"].append(
        measure_time("WA Potential", lambda: [model_ws.U(p) for p in points])
    )
    results["Werner–Arribas"]["Acceleration"].append(
        measure_time("WA Acceleration", lambda: [model_ws.g(p) for p in points])
    )
    results["Werner–Arribas"]["Tensor"].append(
        measure_time("WA Tensor", lambda: [model_ws.gravity_gradients(p) for p in points])
    )

    # --- Numerical FD (subset only for cost reasons) ---
    if n <= 5000:
        results["Numerical"]["Acceleration"].append(
            measure_time("Numerical Acceleration", lambda: g_num_class(model_pb, points))
        )
        num_accel_sizes.append(n)
    if n <= 500:
        results["Numerical"]["Tensor"].append(
            measure_time("Numerical Tensor", lambda: tensor_num_class(model_pb, points))
        )
        num_tensor_sizes.append(n)

    print()

print("--- Benchmark Complete ---\n")


# ===============================================================
# 6. Plot Results (clean scatter plots)
# ===============================================================
fig, axes = plt.subplots(3, 1, figsize=(9, 12), sharex=True)

pb_color = np.array([0, 174, 235]) / 255   # Periyandy & Bevis blue
wa_color = "green"                         # Werner–Arribas green
num_color = "red"                          # Numerical red
marker_size = 25

for ax in axes:
    ax.minorticks_on()
    ax.grid(True, which="major", ls="--", alpha=0.4)
    ax.grid(True, which="minor", ls=":", alpha=0.2)

# --- Potential ---
ax = axes[0]
ax.scatter(sizes, results["Periyandy & Bevis"]["Potential"], s=marker_size,
           color=pb_color, label="Periyandy & Bevis")
ax.scatter(sizes, results["Werner–Arribas"]["Potential"], s=marker_size,
           color=wa_color, label="Werner–Arribas")
ax.set_ylabel("Time (s)")
ax.set_title("Potential Computation Speed")
ax.legend(loc="lower right")

# --- Acceleration ---
ax = axes[1]
ax.scatter(sizes, results["Periyandy & Bevis"]["Acceleration"], s=marker_size,
           color=pb_color, label="Periyandy & Bevis")
ax.scatter(sizes, results["Werner–Arribas"]["Acceleration"], s=marker_size,
           color=wa_color, label="Werner–Arribas")
ax.scatter(num_accel_sizes, results["Numerical"]["Acceleration"], s=marker_size,
           color=num_color, label="Numerical FD")
ax.set_ylabel("Time (s)")
ax.set_title("Acceleration Computation Speed")
ax.legend(loc="lower right")

# --- Tensor ---
ax = axes[2]
ax.scatter(sizes, results["Periyandy & Bevis"]["Tensor"], s=marker_size,
           color=pb_color, label="Periyandy & Bevis")
ax.scatter(sizes, results["Werner–Arribas"]["Tensor"], s=marker_size,
           color=wa_color, label="Werner–Arribas")
ax.scatter(num_tensor_sizes, results["Numerical"]["Tensor"], s=marker_size,
           color=num_color, label="Numerical FD")
ax.set_xlabel("Number of Observation Points")
ax.set_ylabel("Time (s)")
ax.set_title("Tensor Computation Speed")
ax.legend(loc="lower right")

# Log scale
for ax in axes:
    ax.set_xscale("log")
    ax.set_yscale("log")

fig.tight_layout()
fig.savefig("gravity_speed_test_subplots.pdf", dpi=300, bbox_inches="tight")
plt.show()

print("Plot saved to 'gravity_speed_test_subplots.pdf'.")
print(f"Threads used: {num_threads}")
print(f"Logical cores detected: {psutil.cpu_count(logical=True)}")
