# Mesh-FEM Perforated-Fin Toy Benchmark

## Objective

This benchmark tests whether a point-cloud neural operator changes its output
when the same cooling-fin geometry is represented with a different surface-point
density.  Geometry, boundary conditions, FEM solution, and query locations are
kept independent of the encoder sampling shift.

## Geometry and Mesh

Each case is a single watertight solid: a tapered, mildly wavy vertical fin
with three circular through-holes. Height, width, thickness, taper, waviness,
and hole centers/radii are deterministic functions of the case seed. Gmsh
constructs the CAD solid, subtracts the holes, and produces an adaptive
tetrahedral mesh. Element sizes are smaller near hole rims and the lower fin,
so the native mesh-derived source cloud has meaningful nonuniform density.

## Physics

The solid solves a deterministic nonlinear steady heat-conduction problem:

```text
-div(k(T) grad(T)) = q_geo(x; holes)             in Omega
k(T) = k0 (1 + a T^2), k0 = 1, a = 2.5
-k(T) grad(T) dot n = hT + rT^3                  on Gamma_boundary
h = 1.25, r = 3.5
```

The source `q_geo` is a sum of three smooth through-thickness volumetric lobes
centered on the three visible hole centers. Its amplitude is fixed, so every
case-to-case change is inferable from geometry alone, not hidden random
forcing. Every exterior surface loses heat by convection and nonlinear
radiation to ambient temperature zero. This removes the previous fixed
hot-base shortcut: fields are nontrivial throughout the solid and depend on
hole paths, nonlinear conductivity, and nonlinear boundary loss.

The outputs are volume temperature `T` and surface outward heat flux
`q_n=-k(T) grad(T) dot n`. The nonlinear terms remain positive and monotone,
so each geometry has a unique deterministic solution.

## Numerical Reliability

The solver uses first-order tetrahedral FEM (`scikit-fem`) and a Picard method.
Each iteration solves a symmetric positive-definite linearization using AMG
preconditioned conjugate gradients with a relative residual threshold of
`2e-8`, then under-relaxes until the relative iterate change is below `2e-7`.
A built-in convergence test resolves the same deterministic case on production
and 1.45x-coarser meshes, samples common in-domain points, and requires the
temperature relative L2 difference to remain below `1%`.

Gmsh refines around every hole rim and the heated-base junction using native
distance/threshold fields.  The generator audits the resulting surface mesh
and rejects cases unless the p95/p05 triangle-area ratio is at least `30`.
This produces a physically motivated, strongly non-uniform native point cloud
without changing the area-uniform reference query distribution.

The generator saves the actual mesh, nodal temperature, and face flux in every
case folder.  It additionally exports cases `0`, `1`, and `2` as surface VTP
and volume VTU files for direct ParaView inspection.

## Sampling Protocol

Native encoder points are sampled uniformly over *surface triangles*, not by
surface area.  Fine adaptive triangles consequently contribute more points per
unit area and faithfully expose native mesh-density bias.  Surface supervision
points are sampled area-uniformly; volume supervision points are sampled
tetra-volume-uniformly.  Their coordinates and FEM targets are independent of
the native encoder cloud.

The generated native cloud has 262K points.  SMART Base uses a reproducible
uniform 131K subset.  SATLOSS retains the full cloud and draws two independent
131K shifted views per batch using KDE beta, sine-x, or sine-y sampling.  Both
models use the DrivAerML SMART hierarchy exactly: 64K surface and 64K volume
queries, a 16K internal geometry subsample, and 4K latent anchors.

## High-Throughput Generation

The generator isolates each Gmsh/FEM case in its own process and separately
parallelizes KDE-cache construction.  On `servus05`, use all 40 physical cores
with one Gmsh/BLAS/kNN thread per process; this avoids the severe nested-thread
oversubscription caused by letting every kNN task use all logical CPUs.

```bash
cd /home/parsa/smart_parsa

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 SMART_KNN_N_JOBS=1 \
PYTHONPATH=/home/parsa/smart_parsa/smart \
/home/parsa/miniconda3/envs/smart/bin/python \
smart/scripts/generate_toy_perforated_fin_benchmark.py \
  --output-dir /mnt/ssdraid/parsa/toy_perforated_fin_nonlinear_fem_v2 \
  --results-dir /home/parsa/smart_parsa/results/toy_perforated_fin_nonlinear_fem \
  --train-cases 128 --validation-cases 32 \
  --geometry-points 262144 --surface-points 65536 --volume-points 65536 \
  --mesh-workers 40 --density-workers 40 --gmsh-threads 1 \
  --export-cases 0,1,2 \
  --verify-mesh-convergence
```

`--workers 0` is also an automatic physical-core setting.  The separate stage
controls are kept so a more memory-constrained machine can lower only the
mesh/FEM or the density-cache concurrency.
