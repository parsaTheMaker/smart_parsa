# Toy-SATLOSS: Complete Analytic Benchmark Specification

## Purpose

This is a deterministic, self-generated 3D manufactured-solution benchmark.
It isolates one causal question: when the geometry and output query points are
unchanged, do changes in the **encoder input-point distribution** change a
neural operator's prediction, and does SATLOSS reduce that change?

It is an exact **manufactured Poisson-PDE** problem, not CFD or FEA for a
real material.  The forcing and conductivity are analytically constructed from
the geometry, so the solution is known exactly.  It supports a point-sampling
robustness claim and a manufactured-PDE solution error; it must not be reported
as real-material accuracy or conservation evidence.

## Determinism

Case `i` is generated from `SeedSequence([master_seed, i, 1701])`.  Its exact
parameters are stored in `case_metadata.json`.  Let `d in S^2` be a global unit
direction.  A QR-sampled rotation `Q` is sign-corrected so `det(Q)=+1`; the
local direction is `u=dQ=(u_1,u_2,u_3)`.

```text
a_k       ~ Uniform(0.72, 1.28),          k=1,2,3
h_k       ~ Uniform(-0.22, 0.22),         k=1,2,3
b         ~ Uniform(S^2)                  bump direction
A_b       ~ Uniform(-0.12, 0.18)          bump amplitude
kappa_b   ~ Uniform(7, 14)                bump sharpness
```

## Geometry

The ellipsoidal base radial function is

```text
R_E(u) = [ (u_1/a_1)^2 + (u_2/a_2)^2 + (u_3/a_3)^2 ]^(-1/2).
```

The smooth angular deformation and localized bump are

```text
D(u) = h_1 (u_1^2-u_2^2) + h_2 (2u_2u_3)
     + h_3 (u_1^3-3u_1u_2^2),
B(u) = A_b exp[kappa_b (u . b - 1)].
```

The final boundary is the star-shaped surface

```text
R(u) = R_E(u) clip(1+D(u)+B(u), 0.55, 1.45),
x_s(d) = R(dQ)d.
```

The clip guarantees a positive radius.  Cases are smooth rotated anisotropic
solids with low-order shape variation and a local convex/concave feature.

## Exact Geometry-Parameterized Poisson PDE

Let `B={xi in R^3 : ||xi||<1}` be the unit reference ball.  The world-space
radial geometry map and its Jacobian are

```text
T_theta(xi)=Q R(xi/||xi||)xi,  T_theta(0)=0,
J_theta(xi)=dT_theta/dxi.
```

Its image is exactly the generated solid `Omega_theta`.  Define the positive,
low-frequency physical loading factor

```text
P(x)=1+0.25x_1-0.18x_2+0.12x_3.
```

The manufactured solution is

```text
u_theta(x)=[1-||xi||^2]P(x),  xi=T_theta^(-1)(x).
```

It exactly solves the standard isotropic Poisson problem

```text
-Delta_x u_theta(x)=f_theta(x),  x in Omega_theta,
u_theta(x)=0,                     x on dOmega_theta,
K(x)=I,
f_theta(x)=-Delta_x{[1-||T_theta^(-1)(x)||^2]P(x)}.
```

`f_theta` is the analytic manufactured forcing induced by the visible
geometry.  It is not an independent random field.  The code evaluates the
known exact solution directly; an expanded Laplacian is unnecessary for target
generation and less stable than the defining equation.

For interior coordinate `x`, the stored volume target is evaluated as

```text
r=||x||, d=x/max(r,epsilon), u=dQ,
xi=u r/R(u), q^2=||xi||^2,
y_v(x)=(1-q^2)P(x).
```

The stored surface target is physical outward normal flux

```text
y_s=n.grad_x u_theta,
n=J_theta^(-T)xi/||J_theta^(-T)xi||,  ||xi||=1.
```

Because the zero-boundary factor cancels derivatives of `P` at the boundary,
the implemented closed form is

```text
y_s(x_s)=-2P(x_s)||J_theta^(-T)xi||,  ||xi||=1.
```

The surface flux and volume solution are observables of the same standard
Poisson PDE with identity conductivity.

## Learnability And Identifiability

There is no case ID, random forcing coefficient, or hidden target parameter
that changes the PDE solution without changing the geometry.  The rotation,
axes, harmonic deformation, and bump parameters define `Omega_theta` and its
radial map `T_theta`; they therefore deterministically define `f_theta`,
`u_theta`, and the boundary flux.  Except for measure-zero geometric symmetries,
the randomly anisotropic/deformed surface identifies this parameter set from
its point cloud.  The learning task is therefore a deterministic map from
geometry and query coordinate to Poisson solution/flux.

The virtual-meshing parameters `(c_d,c_f,phi_d,A_d,F_d)` change only the
encoder-point sampling density.  They do not enter `R`, `f_theta`, `u_theta`,
or the surface flux.  This is the deliberate intervention: a robust
operator should retain the same output under those input-density changes.

## Stored Distributions

| File | Role | Distribution | Default points |
|---|---|---|---:|
| `geometry_coords.npy` | encoder source | case-specific nonuniform cloud | 131,072 |
| `surface_coords.npy`, `surface_data.npy` | surface target/query | independent isotropic-direction surface cloud | 65,536 |
| `volume_coords.npy`, `volume_data.npy` | volume target/query | independent radial-volume cloud | 65,536 |

The separation is central.  All models and all shifts use the same stored
surface and volume queries.  Only encoder input points change, so changing
evaluation quadrature cannot create an apparent robustness gain.

## Native Virtual-Meshing Cloud

For isotropic candidate directions `d_j`, draw

```text
c_d,c_f ~ Uniform(S^2), phi_d ~ Uniform(0,1),
A_d ~ Uniform(1.3,2.1), F_d ~ Uniform(1.0,2.0).
```

Define a smooth virtual-meshing log-weight and sampling probability:

```text
g(d) = A_d sin[2pi(d . c_d + phi_d)] + F_d exp[10(d . c_f - 1)],
p_j = exp[g(d_j)-max_l g(d_l)] / sum_l exp[g(d_l)-max_m g(d_m)].
```

The 131,072 native encoder points are selected without replacement from the
candidates using `p_j`, then mapped to `x_s(d_j)`.  This mimics smooth
adaptive-meshing density and a local refinement region without changing the
geometry or its targets.

The reference surface uses independent isotropic directions mapped to the
surface.  It is isotropic in direction, not strictly area-uniform after the
radial map.  This is controlled because the exact same cloud is shared by all
models and sampling shifts.

For volume points, candidate directions are selected proportional to `R(dQ)^3`;
then `s ~ Uniform(0,1)` and `x_v=R(dQ)s^(1/3)d`.  This is the standard
star-shaped radial construction for uniform volume sampling, up to finite
candidate importance-sampling error.

## Train-Only Normalization

Only training cases define coordinate and target statistics:

```text
x_norm = (x-x_min)/max(x_max-x_min,1e-12),
y_norm = (y-mu_train)/max(sigma_train,1e-12).
```

`x_min,x_max` are global componentwise training bounds.  Surface and volume
targets have separate scalar mean and standard deviation.  Validation examples
never enter any normalization statistic.  Reported errors are denormalized.

## KDE-16 Density Cache

For globally normalized native source points, let `N_16(j)` be the 16 nearest
neighbors.  The cached density uses

```text
h^2   = mean_{j,l in N_16(j)} ||x_j-x_l||^2,
rho_j = (1/16) sum_{l in N_16(j)} exp[-||x_j-x_l||^2/h^2],
ell_j = log(max(rho_j,tiny)).
```

`ell_j` is written as float16 in `geometry_log_density_k16_kde.npy`, exactly
aligned with `geometry_coords.npy`.  It chooses SATLOSS view points only; it
is not given to SMART as a feature.

## Vanilla SMART Protocol

`toy_satloss.yaml` uses the existing SMART architecture with:

```text
encoder input                       16,384
surface / volume queries             8,192 / 8,192
latent dimension / latent points         192 / 512
per-block source subsample           2,048, with replacement
encoder-decoder blocks                   6
optimizer / learning rate            AdamW / 2e-4
schedule                             cosine with 10% warm-up
batch size / epochs                  4 / 250
precision                            AMP float16.
```

The baseline uses `geometry_epoch_seeded_sampling=True`: each case uses a
reproducible uniform-without-replacement 16K subset of its 131K native source
per epoch.  For a scalar field over points `i`,

```text
L_rel(y_hat,y)=sqrt[sum_i(y_hat_i-y_i)^2/max(sum_i y_i^2,1e-5)],
L_supervised=L_rel(surface)+L_rel(volume).
```

## SATLOSS Protocol

`toy_satloss7.yaml` uses the same architecture, targets, normalizations,
optimizer, query budgets, and epochs.  It retains all 131K source points and
draws two independently sampled 16K views.  A single family is selected per
batch with probability one third each: inverse-KDE beta, sine-y, or sine-x.
Both views use that family but independently draw an intensity from `Uniform(0,1)`
and independently choose points.

For beta intensity `beta`, sampling without replacement uses

```text
w_j(beta)=exp(-beta ell_j), p_j(beta)=w_j(beta)/sum_l w_l(beta).
```

`beta=0` is uniform and high beta favors sparse native regions.  For sine axis
`a`, define

```text
t_j=(x_{j,a}-min_l x_{l,a})/max(max_l x_{l,a}-min_l x_{l,a},1e-8),
w_j=sin^2(pi t_j)+1e-6.
```

At mixture `alpha`, `round(alpha*16384)` weighted points are selected without
replacement and the remaining budget is uniform from unselected points.
`alpha=0` is uniform; `alpha=1` fully underrepresents the chosen-axis extremes.

With supervised losses `L_1,L_2`, prediction consistency is symmetric detached
Smooth-L1 (`beta=0.1`) over surface and volume predictions:

```text
C = 1/2 SmoothL1(y_hat^1,stopgrad(y_hat^2);beta=0.1)
  + 1/2 SmoothL1(y_hat^2,stopgrad(y_hat^1);beta=0.1),
L_SATLOSS = 0.2L_1 + 0.2L_2 + 0.6C.
```

There is no learned weighting, uncertainty weighting, GradNorm, ConFIG, soft
worst-case selection, or consistency warm-up in this specific toy protocol.

## Evaluation

Beta, sine-x, and sine-y use levels `{0,0.25,0.5,0.75,1}` and a fixed 4K
encoder budget.  The matched stored query clouds yield

```text
E_s=||y_hat_s-y_s||_2/max(||y_s||_2,1e-12),
E_v=||y_hat_v-y_v||_2/max(||y_v||_2,1e-12),
E_global=0.5(E_s+E_v).
```

The comparison writes tidy per-case data to `toy_sampling_results.csv`, shift
curves, linear/log endpoint bars, a density-shift histogram, and `protocol.json`.
Endpoint annotations show `100(E_base-E_SATLOSS)/E_base`; a positive number is
an error reduction.

## Inspection Files

`visualize_toy_satloss_examples.py` reconstructs the exact analytic surface
from metadata and writes a valid triangular VTP containing
`manufactured_surface` and `native_encoder_log_density`.  It also writes a
volume-point VTP containing `manufactured_volume` and high-resolution PNGs.
The VTP surface is for inspection only; training consumes stored point clouds.

## Interpretation Limits

This benchmark can establish controlled sampling-density sensitivity, any
SATLOSS reduction of it, and error against an exact manufactured Poisson-PDE
solution.  It cannot establish real CFD/FEA material fidelity, topology
robustness, missing-geometry robustness, remeshing validity, discretization
convergence, or physical conservation.  Those require the real dataset
experiments.
