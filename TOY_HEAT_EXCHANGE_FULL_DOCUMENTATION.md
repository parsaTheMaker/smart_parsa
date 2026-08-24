# Nonlinear Toy Heat-Exchange Benchmark

## 1. Purpose and Scope

This document specifies the `toy_heat_exchange_fem_v1` benchmark used in this repository. It is a geometry-conditioned, steady-state, nonlinear heat-conduction problem designed to test whether an unstructured neural operator remains reliable when the *point sampling of the same physical geometry* changes.

The benchmark has two goals:

1. Learn a deterministic physical map from a triangulated three-dimensional solid surface to a surface heat-flux field and an interior temperature field.
2. Measure whether that learned map changes when the encoder sees a different point distribution representing the same solid, and measure whether SATLOSS reduces that sensitivity.

The dataset is not an industrial calibration study. All quantities are nondimensional, deliberately controlled, and physically consistent. This makes the geometry-to-field relationship identifiable: no unobserved operating condition, material parameter, heat-source amplitude, or boundary-temperature parameter varies from case to case.

The implemented map is

\[
\mathcal{G}: \mathcal{S} \longmapsto
\left(
q_n\vert_{\partial\Omega},\; \theta\vert_{\Omega}
\right),
\]

where:

- \(\Omega\subset\mathbb{R}^3\) is the solid fin domain;
- \(\mathcal{S}=\partial\Omega\) is the watertight triangular surface supplied to the encoder;
- \(q_n\) is the signed outward conductive/exterior heat flux on the surface;
- \(\theta\) is the dimensionless steady temperature in the solid.

There are no per-case scalar conditioning inputs. Geometry alone determines the solution.

## 2. Current Generated Corpus

The active corpus is stored at:

```text
/mnt/ssdraid/parsa/toy_heat_exchange_fem_v1
```

Its manifest records:

| Property | Value |
|---|---:|
| Generator version | `toy_heat_exchange_fem_v1` |
| Global random seed | `42` |
| Training cases | 256 (`case_00000` to `case_00255`) |
| Validation cases | 32 (`case_00256` to `case_00287`) |
| Native geometry cloud per case | 524,288 points |
| Surface supervision queries per case | 65,536 points |
| Volume supervision queries per case | 65,536 points |
| Surface output | Signed `outward_heat_flux` |
| Volume output | `temperature` |
| Coordinate dimension | 3 |
| Explicit case parameters passed to the model | None |

Measured over all 288 completed cases:

| Quantity | Minimum | Median | Mean | Maximum |
|---|---:|---:|---:|---:|
| FEM nodes | 118,427 | 288,893 | 297,359 | 532,178 |
| Tetrahedra | 485,518 | 1,281,610 | 1,326,120 | 2,441,090 |
| Boundary triangles | 137,768 | 279,471 | 284,397 | 469,214 |
| Surface triangle-area \(p_{95}/p_{05}\) | 14.14 | 21.61 | 25.55 | 73.42 |
| Nonlinear iterations | 14 | 14 | 14 | 14 |
| Channel count | 1 | 3 | 3.01 | 5 |

The large range in triangle areas is intentional. It comes from physically motivated adaptive refinement near channels, small ligaments, and wavy boundaries. It is not an artificial post-processing density perturbation.

## 3. Geometry Definition

### 3.1 Coordinate system and base solid

Each case is an extruded solid fin. The longitudinal profile lies in the \((x,z)\) plane and is extruded through the thickness direction \(y\):

\[
\Omega = \left\{(x,y,z): z\in[0,H],\; x\in[x_L(z),x_R(z)],\; y\in[-t/2,t/2]\right\}
\setminus \bigcup_j C_j.
\]

The global axes are:

| Axis | Meaning |
|---|---|
| \(x\) | Width direction, with wavy left/right side walls |
| \(y\) | Extrusion/thickness direction; channels pass through this direction |
| \(z\) | Height direction |

The randomly sampled geometric ranges are:

| Parameter | Range |
|---|---:|
| Height \(H\) | \([1.25,1.75]\) |
| Thickness \(t\) | \([0.20,0.32]\) |
| Nominal left boundary | \([-0.67,-0.52]\) |
| Nominal right boundary | \([0.52,0.67]\) |

For the currently generated corpus, the measured global training-coordinate bounds are:

\[
x\in[-0.7425687, 0.7673534],\qquad
y\in[-0.15984373, 0.15984373],\qquad
z\in[0, 1.7495427].
\]

### 3.2 Wavy exterior walls

The side-wall displacement is smooth and zero at the lower and upper ends. Let

\[
s=\frac{z}{H},\qquad E(s)=\sin(\pi s).
\]

For coefficients \((a_1,a_2,n_1,n_2,\phi_1,\phi_2)\), the implemented displacement is

\[
w(z)=E(s)\left[
a_1\sin(2\pi n_1s+\phi_1)
+a_2\sin(2\pi n_2s+\phi_2)
\right].
\]

The coefficient ranges are:

| Coefficient | Range |
|---|---:|
| \(a_1\) | \([0.025,0.075]\) |
| \(a_2\) | \([0.010,0.045]\) |
| \(n_1\) | Integer in \(\{1,2,3\}\) |
| \(n_2\) | Integer in \(\{3,4,5,6\}\) |
| \(\phi_1,\phi_2\) | \([0,2\pi]\) |

One of three equally sampled waviness categories is used:

\[
(x_L,x_R)=
\begin{cases}
(x_{L,0}+w_L(z),\;x_{R,0}+w_R(z)) & \text{both sides wavy},\\
(x_{L,0}+w_L(z),\;x_{R,0}) & \text{left side wavy},\\
(x_{L,0},\;x_{R,0}+w_R(z)) & \text{right side wavy}.
\end{cases}
\]

In the current corpus the category counts are 97 both-wavy, 94 left-wavy, and 97 right-wavy cases.

The sine envelope ensures that the geometry joins smoothly to the bottom and top end faces. This prevents a discontinuous side-wall deformation from introducing a nonphysical CAD feature solely for data variation.

### 3.3 Internal through-channels

Every fin contains between one and five through-channels \(C_j\). A channel extends through the entire thickness direction \(y\), so its wall is exposed to a hot-fluid surrogate.

Each channel is independently selected as:

| Channel type | Probability | Parameters |
|---|---:|---|
| Circular | 0.55 | Radius \(r\in[0.055,0.115]\) |
| Rectangular | 0.45 | Half-width \(h_x\in[0.050,0.115]\), half-height \(h_z\in[0.050,0.125]\) |

For the current corpus, 482 circular and 385 rectangular channels were generated.

Channel placement is rejection sampled, not blindly randomized. Before a channel is accepted, the generator verifies all of the following:

1. The full channel lies inside the locally wavy profile.
2. It stays at least `0.035` coordinate units away from the exterior side walls.
3. It stays away from the top and bottom end faces.
4. It does not overlap any previously accepted channel, including the conservative clearance of a rectangular channel corner.
5. The variable side-wall profile is checked along the channel's entire \(z\)-extent, not only at its centre.

If a requested channel cannot be placed after 400 trials, the case simply has fewer channels. A valid lower-channel-count geometry is preferred over forcing overlap, self-intersection, a zero-thickness ligament, or a non-manifold Boolean result.

## 4. Physics Model

### 4.1 Nondimensional temperature

The solved state is a nondimensional temperature \(\theta\). It can be interpreted as a normalized temperature rise:

\[
\theta = \frac{T-T_\infty}{T_h-T_\infty},
\]

where \(T_h\) is the common hot channel-wall temperature and \(T_\infty\) is the ambient temperature. The code solves directly in \(\theta\); it does not define a particular material, length, or temperature unit system. This is deliberate: the benchmark tests the learning problem rather than a calibrated engineering component.

All cases use the same physical coefficients and the same hot-wall condition. Therefore the geometry is the only variable input.

### 4.2 Nonlinear steady conduction equation

Within the solid, the governing equation is

\[
-\nabla\cdot\left(k(\theta)\nabla\theta\right)=0
\qquad\text{in }\Omega,
\]

with temperature-dependent conductivity

\[
k(\theta)=1+a\theta^2,
\qquad a=1.8.
\]

This nonlinear conductivity increases in hotter material. Consequently, channel placement, channel shape, local ligament thickness, and the wavy outer-wall geometry change both the heat path and the local effective conduction response.

There is no volumetric source term. Heat enters the solid through the hot internal channel walls and leaves through the exterior boundary.

### 4.3 Boundary partition

The boundary is divided into two disjoint physical regions:

\[
\partial\Omega=\Gamma_h\cup\Gamma_e,
\qquad \Gamma_h\cap\Gamma_e=\varnothing.
\]

| Boundary | Physical role | Implemented condition |
|---|---|---|
| \(\Gamma_h\) | Circular/rectangular channel walls | Isothermal Dirichlet: \(\theta=1\) |
| \(\Gamma_e\) | All exterior faces, including front/back, side walls, top, and bottom | Convective-radiative Robin condition |

The hot channel-wall condition is exact at finite-element degrees of freedom:

\[
\theta=1 \qquad \text{on }\Gamma_h.
\]

No hidden channel temperature, fluid temperature, or heat flux varies per case. The network sees the channel shape and placement through the geometry only.

On the exterior boundary, with \(\mathbf n\) the outward normal of the solid,

\[
-k(\theta)\nabla\theta\cdot\mathbf n
=
\mathrm{Bi}_{\mathrm{ext}}\theta
+R\left[(\theta+\tau)^4-\tau^4\right]
\qquad\text{on }\Gamma_e,
\]

where the fixed coefficients are:

| Symbol | Implemented value | Role |
|---|---:|---|
| \(\mathrm{Bi}_{\mathrm{ext}}\) | `0.35` | Convective heat-transfer coefficient |
| \(R\) | `0.070` | Radiation coefficient |
| \(\tau\) | `1.5` | Ambient-temperature ratio in the nondimensional radiation term |
| \(a\) | `1.8` | Nonlinear-conductivity coefficient |

The radiation term is a temperature-difference form. It is zero at \(\theta=0\), and its positive secant coefficient is used during iteration to maintain a positive-definite linear subproblem.

### 4.4 Flux sign convention and output meaning

The surface target is named `outward_heat_flux` and uses the solid's outward normal.

- On exterior faces, \(q_n>0\) means heat leaves the solid to the environment.
- On channel walls, heat enters the solid from the prescribed hot wall. Under the same outward-solid-normal convention this is typically \(q_n<0\).

The generator explicitly checks that the mean channel-wall flux is negative and the mean exterior flux is positive. A case that violates these signs is rejected.

## 5. FEM Discretization and Nonlinear Solution

### 5.1 CAD and mesh generation

Gmsh/OpenCASCADE constructs the domain as follows:

1. Sample 97 points along the wavy \((x,z)\) profile.
2. Build a closed planar profile and extrude it by \(t\) in \(y\).
3. Boolean-subtract all circular cylinders and rectangular prisms that form the through-channels.
4. Generate a tetrahedral mesh with Gmsh 3-D algorithm 10.
5. Run Netgen mesh optimization.

The mesh is adaptive for physical resolution rather than for data augmentation:

| Refinement mechanism | Purpose |
|---|---|
| Distance field around channel faces | Resolves hot walls and narrow ligaments between nearby channels |
| Distance field near wavy exterior side faces | Resolves crests, troughs, and nearby boundary gradients |
| Gmsh curvature sizing | Adds curvature-sensitive refinement |
| Global limits | `mesh_h_min = 0.0025`, `mesh_h_max = 0.028` |

The recorded triangle-area \(p_{95}/p_{05}\) ratio is a diagnostic of genuine adaptive mesh variation. It is not an acceptance gate: FEM validity depends on finite coordinates, valid topology, positive tetrahedral volumes, a successful nonlinear solve, and physical boundary checks.

### 5.2 Finite elements

The solver uses continuous piecewise-linear tetrahedral finite elements:

\[
V_h=\{v_h\in C^0(\Omega):v_h\vert_K\text{ is affine for every tetrahedron }K\}.
\]

The channel-wall degrees of freedom are constrained to \(\theta=1\) exactly. The exterior Robin term is assembled on exterior facets. The unknown field is therefore continuous and piecewise affine in the volume.

### 5.3 Picard fixed-point iteration

The nonlinear conduction and radiation terms are solved by damped Picard iteration. At iteration \(m\):

\[
k^{(m)}=1+a(\theta^{(m)})^2,
\]

\[
R^{(m)}=R\left[(\theta^{(m)}+\tau)^4-\tau^4\right],
\]

and the exterior radiative flux is represented by the nonnegative secant coefficient

\[
c_{\mathrm{rad}}^{(m)}=
\begin{cases}
R^{(m)}/\theta^{(m)}, & \theta^{(m)}>10^{-8},\\
4R\tau^3, & \text{otherwise}.
\end{cases}
\]

This yields a symmetric positive-definite linear problem at each iteration. The implementation uses:

- smoothed-aggregation algebraic multigrid (`pyamg`) as a V-cycle preconditioner;
- conjugate gradients with relative tolerance \(10^{-10}\) and at most 20,000 iterations;
- sparse direct solve fallback only if CG reports failure;
- damping \(\theta^{(m+1)}\leftarrow0.70\,\theta_{\mathrm{updated}}+0.30\,\theta^{(m)}\);
- exact restoration of \(\theta=1\) at channel-wall degrees of freedom after every update;
- nonlinear stopping condition

\[
\frac{\lVert\theta^{(m+1)}-\theta^{(m)}\rVert_2}
{\max(\lVert\theta^{(m+1)}\rVert_2,10^{-12})}
<2\times10^{-7}.
\]

The maximum nonlinear iteration count is 60. In the current corpus every accepted case converged in 14 iterations. The recorded linear relative residuals range from approximately \(2.0\times10^{-11}\) to \(9.8\times10^{-11}\), while final nonlinear relative changes range from approximately \(1.1\times10^{-7}\) to \(1.6\times10^{-7}\).

### 5.4 Numerical and physical acceptance checks

Before a case is persisted, the generator verifies:

1. All tetrahedral volumes are finite and greater than \(10^{-14}\).
2. Boundary triangles are finite and have positive area.
3. Every channel has enough classified wall facets.
4. Both the channel and exterior boundary sets are nonempty.
5. The linear-system residual is finite and no greater than \(2\times10^{-8}\).
6. The nonlinear iteration converges within the configured limit.
7. \(0\leq\theta\leq1\) within a \(10^{-7}\) numerical tolerance.
8. Channel-wall temperature error is no greater than \(10^{-12}\).
9. Mean channel-wall flux has the heat-injection sign and mean exterior flux has the heat-rejection sign.

Recoverable CAD/meshing/solve failures are retried with a distinct deterministic seed. Invalid geometry is never silently stored as a valid sample.

## 6. Preprocessed Representation

Each completed case directory contains the following arrays:

| File | Shape | Meaning |
|---|---|---|
| `geometry_coords.npy` | \((524288,3)\) | Native encoder surface cloud |
| `geometry_log_density_k16_kde.npy` | \((524288,)\) | Cached KDE-16 log sampling density of native encoder points |
| `surface_coords.npy` | \((65536,3)\) | Surface query coordinates |
| `surface_data.npy` | \((65536,1)\) | Signed outward heat-flux targets |
| `volume_coords.npy` | \((65536,3)\) | Volume query coordinates |
| `volume_data.npy` | \((65536,1)\) | Temperature targets |
| `surface_mesh_points.npy` | \((N_v,3)\) | Full FEM boundary-mesh vertices |
| `surface_mesh_faces.npy` | \((N_f,3)\) | Full FEM boundary triangles |
| `surface_fem_face_flux.npy` | \((N_f,)\) | Flux per boundary face |
| `volume_mesh_tetra.npy` | \((N_t,4)\) | FEM tetrahedral connectivity |
| `fem_nodal_temperature.npy` | \((N_v,)\) | FEM nodal temperature |
| `case_metadata.json` | JSON | Geometry parameters, mesh statistics, physics settings, convergence data |
| `_COMPLETE.json` | JSON | Atomic completion marker |

At dataset root:

| File | Meaning |
|---|---|
| `preprocessed_manifest.json` | Version, split membership, point budgets, seed |
| `surface_stats_toy_heat_exchange_fem_train_stats_v1.npy` | Training-only surface-target mean and standard deviation |
| `volume_stats_toy_heat_exchange_fem_train_stats_v1.npy` | Training-only volume-target mean and standard deviation |
| `position_stats_toy_heat_exchange_fem_train_stats_v1.npy` | Training-only global coordinate lower/upper bounds |

## 7. Sampling Semantics

### 7.1 Native encoder geometry cloud

The native geometry cloud is sampled by first choosing a **boundary triangle uniformly by triangle index**, then sampling a point uniformly inside that triangle using exponential/Dirichlet barycentric weights.

For selected triangle vertices \((\mathbf v_1,\mathbf v_2,\mathbf v_3)\), independent \(u_i\sim\mathrm{Uniform}(0,1)\) are transformed as

\[
w_i=\frac{-\log(\max(u_i,10^{-12}))}
{\sum_{j=1}^3-\log(\max(u_j,10^{-12}))},
\qquad
\mathbf x=\sum_{i=1}^3w_i\mathbf v_i.
\]

This makes each point uniform inside its selected triangle, but triangles are selected uniformly rather than by area. Consequently, adaptive FEM refinement creates a deliberately nonuniform *encoder point density*: refined channel and curved-wall regions contain more source points. This is the physically realistic sampling distribution whose changes SATLOSS is intended to address.

### 7.2 Supervision clouds

The supervised clouds intentionally use different, unbiased sampling rules:

| Cloud | Sampling rule | Target interpolation |
|---|---|---|
| Surface query cloud | Triangle selected proportional to area, then uniform barycentric point | Constant face flux of selected FEM boundary triangle |
| Volume query cloud | Tetrahedron selected proportional to volume, then uniform barycentric point | P1 barycentric interpolation of nodal temperature |

Thus, queries are area-uniform on the surface and volume-uniform in the solid. The same query points and ground-truth fields are used when comparing distinct encoder point distributions. Only the encoder geometry changes.

### 7.3 Coordinate normalization

Coordinates use fixed global training bounds, not per-case normalization:

\[
\widetilde{\mathbf x}=
\frac{\mathbf x-\mathbf x_{\min}^{\mathrm{train}}}
{\mathbf x_{\max}^{\mathrm{train}}-\mathbf x_{\min}^{\mathrm{train}}}.
\]

This preserves relative size, thickness, channel placement, and shape variation across cases. Per-case normalization would remove part of the geometry variation that the operator should learn.

### 7.4 Target normalization

Each target group is standardized using statistics computed from the training split only:

\[
\widetilde y=\frac{y-\mu_{\mathrm{train}}}{\sigma_{\mathrm{train}}}.
\]

For the current corpus:

| Target | Training mean | Training standard deviation |
|---|---:|---:|
| Surface outward heat flux | 0.10420842 | 2.89798100 |
| Volume temperature | 0.44438280 | 0.26352927 |

Metrics and saved physical predictions should be denormalized before physical interpretation.

## 8. Baseline SMART Training Protocol

The base SMART configuration is `toy_heat_exchange.yaml`.

| Setting | Value |
|---|---:|
| Encoder input points | 65,536, uniformly subsampled without replacement from native 524,288 points |
| Surface queries/batch | 32,768 |
| Volume queries/batch | 32,768 |
| Native-to-encoder fraction | \(1/8\) |
| Native-to-query fraction | \(1/16\) per query type |
| SMART internal support points | 8,192 |
| SMART latent anchors | 2,048 |
| Internal hierarchy | \(65536\rightarrow8192\rightarrow2048\) |
| Latent width | 256 |
| Encoder/decoder blocks | 6 |
| Optimizer | AdamW |
| Learning rate | \(2\times10^{-4}\) |
| Scheduler | Cosine, 20% warm-up fraction |
| Precision | AMP FP16 |
| Base epochs | 300 |
| Base batch size | 1 |

The ratios are aligned with the DrivAerML SMART hierarchy after accounting for the heat-exchange native geometry cloud:

\[
524288\rightarrow65536\rightarrow8192\rightarrow2048.
\]

The base model sees a fresh uniform encoder subsample each epoch through deterministic epoch-seeded sampling. It is trained with supervised relative \(L_2\) loss on both surface flux and volume temperature:

\[
\mathcal L_{\mathrm{sup}}=
\operatorname{RelL2}(\hat q_n,q_n)+
\operatorname{RelL2}(\hat\theta,\theta).
\]

For a field \(y\), the implementation computes

\[
\operatorname{RelL2}(\hat y,y)=
\operatorname{mean}_{b,c}
\left[
\sqrt{
\frac{\sum_p(\hat y_{bpc}-y_{bpc})^2}
{\max(\sum_p y_{bpc}^2,10^{-5})}
}
\right].
\]

## 9. SATLOSS7 Protocol

### 9.1 Purpose

SATLOSS7 is not a different architecture. For SMART, it retains the same model and changes only the training objective and encoder-view construction. The aim is to make predictions less dependent on the specific density distribution of input points representing the same geometry.

The YAML configuration does not hard-code an initialization checkpoint, so SATLOSS7 can be run from scratch or as a weight-only continuation of a matched completed base model. For a controlled base-versus-SATLOSS comparison, the intended protocol is the latter: load only matched base weights through `experiment.init_ckpt`, and initialize the SATLOSS optimizer and scheduler afresh. The current SMART SATLOSS configuration sets 300 epochs; other model families use the shared 150-epoch continuation configuration unless their leaf configuration overrides it.

### 9.2 Two geometry views

For each batch, the full native 524,288-point encoder cloud is retained by the dataset. Two independently sampled 65,536-point views are then drawn. Both views use the same shift family for that batch, but their shift intensities and random samples are independent.

The three families are selected with probabilities

\[
P(\mathrm{beta})=P(\mathrm{sine\text{-}y})=P(\mathrm{sine\text{-}x})\approx\frac13.
\]

#### KDE inverse-density family

The cached KDE-16 log density \(\ell_i=\log\hat\rho_i\) is computed on the normalized native geometry cloud. For each view independently,

\[
\beta\sim\mathrm{Uniform}(0,1),
\qquad
p_i\propto\exp(-\beta\ell_i)=\hat\rho_i^{-\beta}.
\]

Points are sampled without replacement using \(p_i\). At \(\beta=0\), the sampling is uniform over source points. Larger \(\beta\) favors points that are sparse in the native cloud and therefore changes the representation of refined regions.

#### Sine-\(x\) and sine-\(y\) families

For one spatial axis, each point receives a normalized per-view coordinate

\[
u_i=\frac{x_{i,a}-\min_jx_{j,a}}
{\max_jx_{j,a}-\min_jx_{j,a}},
\]

with weighted-selection probability

\[
p_i\propto\sin^2(\pi u_i)+10^{-6}.
\]

For each view independently, \(\alpha\sim\mathrm{Uniform}(0,1)\). A fraction \(\alpha\) of the 65,536 points is drawn without replacement from the sine-weighted distribution; the remaining points are drawn uniformly without replacement from the points not already selected.

- Sine-\(x\) alters support along the fin-width direction.
- Sine-\(y\) alters support through the thin extrusion direction.

These shifts preserve the geometry and point count while making the centre of the selected coordinate range denser and its ends sparser. They are deliberately distinct from masking: no geometric region is removed as an explicit occlusion operation.

### 9.3 Loss function

Both shifted views are queried at the same surface and volume query coordinates and supervised against the same ground truth:

\[
\mathcal L_1=\mathcal L_{\mathrm{sup}}(\hat y_1,y),
\qquad
\mathcal L_2=\mathcal L_{\mathrm{sup}}(\hat y_2,y).
\]

The prediction-consistency term compares the two outputs with symmetric stop-gradient Smooth-\(L_1\) loss:

\[
\mathcal L_c=
\frac12\left[
\operatorname{Huber}_{\delta}(\hat y_1,\operatorname{sg}(\hat y_2))+
\operatorname{Huber}_{\delta}(\hat y_2,\operatorname{sg}(\hat y_1))
\right],
\]

computed separately for surface and volume groups and averaged. Here \(\operatorname{sg}\) denotes stop-gradient and \(\delta=0.1\) is the Smooth-\(L_1\) transition parameter.

The final fixed-weight SATLOSS7 objective is

\[
\boxed{
\mathcal L_{\mathrm{SATLOSS7}}
=0.2\mathcal L_1+0.2\mathcal L_2+0.6\mathcal L_c
}
\]

with no consistency warm-up. The two supervised terms prevent a trivial agreement solution; the consistency term makes agreement under input-density shifts a primary learning signal.

### 9.4 What SATLOSS7 does and does not claim

SATLOSS7 does **not** make the operator mathematically invariant to arbitrary changes in topology, missing geometry, noise, or a different physical boundary-value problem. It trains the operator to reduce sensitivity to the specified sampling-distribution changes while preserving the same geometry and targets. The remeshing tests extend this check to physically plausible changes in triangular discretization.

## 10. Remeshing Robustness Evaluation

### 10.1 Principle

The remeshing experiments change only the encoder geometry source. Surface and volume queries remain the original area-uniform and volume-uniform query arrays, with their original ground truth. Therefore a performance change can be attributed to the geometry representation, not target resampling.

The source VTP is the exact adaptive FEM boundary mesh, not the 524,288-point native cloud. A fixed-size encoder cloud is sampled from each remeshed triangle mesh by selecting triangles uniformly and sampling barycentrically inside them, matching the native-cloud convention.

### 10.2 Remeshing methods and target reductions

The evaluation infrastructure supports factors 5 and 10, corresponding to approximately fivefold and tenfold reductions in triangle count.

| Label | Backend/method | Main character |
|---|---|---|
| Angle | VTK `DecimatePro` | Feature-angle-aware mesh decimation |
| Isotropic | Isotropic remeshing backend | More uniform triangles with target edge length calibrated from surface area and target triangle count |
| Voxel | Uniform voxel/quadric clustering | Spatial clustering followed by quadric-style geometric reduction |

Each output is checked for finite vertices, triangles only, positive triangle area, zero boundary edges, and zero non-manifold edges. Degenerate triangles are repaired before final validation where possible. The source topology and physical geometry are not intentionally altered beyond the chosen remeshing approximation.

For the current benchmark comparisons, use only the specifically activated geometry-source methods. If a script is invoked with `--active-geometry-sources isotropic`, then angle and voxel files are not part of that result even if they exist on disk.

### 10.3 Evaluation metrics

For every source distribution, model, and case, predictions are denormalized and evaluated as:

\[
e_{\mathrm{surface}}=
\frac{\lVert\hat q_n-q_n\rVert_2}{\lVert q_n\rVert_2},
\qquad
e_{\mathrm{volume}}=
\frac{\lVert\hat\theta-\theta\rVert_2}{\lVert\theta\rVert_2},
\]

\[
e_{\mathrm{global}}=
\frac12(e_{\mathrm{surface}}+e_{\mathrm{volume}}).
\]

For a base/SATLOSS paired comparison, the displayed relative change is conventionally

\[
100\times\frac{e_{\mathrm{SATLOSS}}-e_{\mathrm{base}}}{e_{\mathrm{base}}}.
\]

Negative values indicate lower error for SATLOSS.

## 11. Other Operator Families in the Toy Benchmark

The same data adapter, coordinate normalization, query budgets, targets, base/SATLOSS protocol, and evaluation definitions are available for all model families below. Their architecture changes, but their physics data and paired sampling experiment do not.

| Model | Base configuration | Important configured capacity |
|---|---|---|
| SMART | `toy_heat_exchange.yaml` | 256 latent width; 8,192 support points; 2,048 anchors; 6 blocks |
| MSPT | `toy_heat_exchange_mspt.yaml` | 4 blocks; hidden size 192; 4 heads; 128 latent tokens; batch 2 |
| LNO | `toy_heat_exchange_lno.yaml` | 8 blocks; 256 modes/dimension; 8 heads; 2 layers; batch 4 |
| PointNet++ SSG | `toy_heat_exchange_pointnet2_ssg.yaml` | 2,048 then 512 set-abstraction centroids; ball radii 0.03/0.06 in globally normalized coordinates; batch 4 |
| Transolver++ | `toy_heat_exchange_transolverpp.yaml` | 4 layers; width 256; 8 heads; 32 slices; batch 2 |
| PointTransformerV3 | `toy_heat_exchange_point_transformer_v3.yaml` | Density-sensitive sparse hierarchy; 11M-class configuration; local query support 4; batch 1 |

Each SATLOSS7 configuration inherits the paired base architecture and the common protocol described above. A fair paired comparison requires that base and SATLOSS checkpoints use the same model architecture, data split, target normalization, query budget, and evaluation input budget.

## 12. Reproducibility and Determinism

### 12.1 Seeds

The generator has root seed `42`. Each case uses a deterministic seed sequence based on the root seed, case identifier, generator namespace, and retry index. Native-cloud sampling uses a separate deterministic seed sequence. Therefore rerunning a completed case with the same version and arguments reproduces its geometry and sampled arrays, subject to numerical-library and platform reproducibility.

The dataset's epoch-seeded sampling uses a seed sequence based on:

\[
(\text{split seed},\;\text{epoch},\;\text{case id},\;\text{stream id}).
\]

For SATLOSS7, the view samplers use deterministic seeds based on training seed, epoch, batch index, and view identity.

### 12.2 Parallel generation and cleanup safety

FEM cases are generated in spawned worker processes. Native numerical libraries are pinned to one thread per worker. Workers use parent-death signaling and one-case process recycling to bound Gmsh/AMG native allocations and prevent orphaned native workers after terminal interruption. The generator writes arrays atomically using a `.partial` temporary file followed by rename and writes `_COMPLETE.json` only after all case files are valid.

### 12.3 Ground-truth inspection files

The generator can export selected cases as:

```text
results/toy_heat_exchange/heat_exchange_case_XXXXX_surface_ground_truth.vtp
results/toy_heat_exchange/heat_exchange_case_XXXXX_volume_ground_truth.vtu
```

The surface VTP contains nodal `temperature` and a nodal average of `outward_heat_flux`; the volume VTU contains nodal `temperature` on the tetrahedral mesh. These are for inspection and do not replace the stored point-query arrays used for training.

## 13. Recommended Reporting Language

The following concise description is accurate for a paper or report:

> We construct a nonlinear three-dimensional heat-exchange benchmark in which geometry is the sole input. Each sample is a watertight extruded fin with randomly parameterized wavy side walls and one to five non-overlapping circular or rectangular through-channels. Temperature solves a steady nonlinear conduction equation with temperature-dependent conductivity, fixed hot temperature on channel walls, and convective-radiative heat loss on the exterior. The native encoder cloud is sampled uniformly over adaptive FEM triangles and therefore inherits physically motivated density variation, whereas surface and volume supervision queries are sampled uniformly with respect to surface area and volume. This permits controlled evaluation of whether neural-operator predictions depend on the point distribution used to represent the same geometry.

For the SATLOSS contribution:

> SATLOSS7 trains two independently sampled, equal-budget representations of the same geometry under a shared sampling-shift family. Each view remains supervised by the same physical ground truth, while a symmetric stop-gradient prediction-consistency term penalizes disagreement between view predictions. The fixed objective is \(0.2\mathcal L_1+0.2\mathcal L_2+0.6\mathcal L_c\). Consequently, any robustness gain is evaluated under unchanged geometry, query locations, and targets, isolating sensitivity to the encoder point distribution.

## 14. Limits and Interpretation

1. The physics is a nondimensional heat-conduction model, not a material-specific engineering certification model.
2. The problem is steady state; it does not model transient thermal storage, fluid dynamics within channels, contact resistance, or conjugate fluid-solid heat transfer.
3. Channel walls are prescribed-temperature boundaries. They represent a hot-fluid effect but do not solve the fluid state.
4. The surface flux target is piecewise constant per FEM face before query sampling, while volume temperature is P1-interpolated within tetrahedra.
5. Remeshing changes the discrete geometric representation, not the physical CAD construction or ground-truth solution. It is a representation-robustness test, not a new PDE solve on every remesh.
6. SATLOSS7 targets robustness to the specified density and remeshing-related encoder shifts. It is not a guarantee against arbitrary distribution shift or missing geometry.

## 15. Artifact Map

| Artifact | Role |
|---|---|
| `smart/scripts/generate_toy_heat_exchange_benchmark.py` | CAD generation, FEM solve, sampling, statistics, KDE cache, VTP/VTU export |
| `smart/data/toy_heat_exchange_dataset.py` | Dataset loading, normalization, deterministic subsampling, cached density loading |
| `smart/config/toy_heat_exchange_common.yaml` | Shared data/training point budgets and optimizer settings |
| `smart/config/toy_heat_exchange.yaml` | SMART baseline architecture |
| `smart/config/toy_heat_exchange_satloss7_protocol.yaml` | Shared SATLOSS7 two-view sampling and loss protocol |
| `smart/config/toy_heat_exchange_satloss7.yaml` | SMART SATLOSS7 configuration |
| `smart/scripts/prepare_toy_heat_exchange_remeshing.py` | VTP export, topology-validated angle/isotropic/voxel remeshing, mesh galleries |
| `smart/scripts/compare_toy_heat_exchange_sampling_invariance.py` | SMART base/SATLOSS sampling and remeshing comparison |
| `smart/scripts/compare_toy_heat_exchange_all_models_sampling_invariance.py` | Multi-model paired comparison |

This map is included for reproducibility. The benchmark definition, equations, sampling rules, and protocol are fully specified in this document.

## 16. Reference Commands

The following commands reproduce the principal benchmark stages with the current corpus scale. They are included as operational documentation; checkpoint names and GPU assignments can be changed without altering the benchmark definition.

### 16.1 Generate the FEM corpus

```bash
cd /home/parsa/smart_parsa

PYTHONPATH=/home/parsa/smart_parsa/smart \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 SMART_KNN_N_JOBS=1 \
/home/parsa/miniconda3/envs/smart/bin/python \
smart/scripts/generate_toy_heat_exchange_benchmark.py \
  --output-dir /mnt/ssdraid/parsa/toy_heat_exchange_fem_v1 \
  --results-dir /home/parsa/smart_parsa/results/toy_heat_exchange \
  --train-cases 256 \
  --validation-cases 32 \
  --geometry-points 524288 \
  --surface-points 65536 \
  --volume-points 65536 \
  --mesh-workers 24 \
  --density-workers 1 \
  --max-cases-per-worker 1 \
  --gmsh-threads 1 \
  --export-cases 0,1,2
```

### 16.2 Prepare remeshed evaluation surfaces

```bash
cd /home/parsa/smart_parsa

PYTHONPATH=/home/parsa/smart_parsa/smart \
PYTHONUNBUFFERED=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 VTK_SMP_MAX_THREADS=1 \
/home/parsa/miniconda3/envs/smart/bin/python \
smart/scripts/prepare_toy_heat_exchange_remeshing.py \
  --data-root /mnt/ssdraid/parsa/toy_heat_exchange_fem_v1 \
  --split all \
  --factors 5,10 \
  --workers 16 \
  --surface-vtp-dir /mnt/ssdraid/parsa/toy_heat_exchange_surface_vtp \
  --angle-output-dir /mnt/ssdraid/parsa/toy_heat_exchange_surface_vtp_angle \
  --isotropic-output-dir /mnt/ssdraid/parsa/toy_heat_exchange_surface_vtp_isotropic \
  --voxel-output-dir /mnt/ssdraid/parsa/toy_heat_exchange_surface_vtp_voxel \
  --results-dir /home/parsa/smart_parsa/results/toy_heat_exchange_remeshing \
  --example-count 3
```

### 16.3 Train SMART base

```bash
cd /home/parsa/smart_parsa

PYTHONPATH=/home/parsa/smart_parsa/smart \
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/parsa/miniconda3/envs/smart/bin/python \
smart/train_toy_heat_exchange.py \
  --config-name=toy_heat_exchange \
  experiment.model_tag=heat-exchange-base-ratio-aligned \
  experiment.name=TOY_HEAT_EXCHANGE_SMART_BASE \
  wandb.project=smart_toy_heat_exchange \
  wandb.entity=parsa-vatani99-technical-university-of-munich
```

### 16.4 Train SMART SATLOSS7 from base weights

```bash
cd /home/parsa/smart_parsa

PYTHONPATH=/home/parsa/smart_parsa/smart \
CUDA_VISIBLE_DEVICES=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/parsa/miniconda3/envs/smart/bin/python \
smart/train_toy_heat_exchange_satloss7.py \
  --config-name=toy_heat_exchange_satloss7 \
  experiment.model_tag=heat-exchange-satloss-ratio-aligned \
  experiment.name=TOY_HEAT_EXCHANGE_SMART_SATLOSS \
  experiment.init_ckpt=/absolute/path/to/matched_smart_base_best.pt \
  experiment.resume_ckpt= \
  experiment.resume_full_state=False \
  wandb.project=smart_toy_heat_exchange \
  wandb.entity=parsa-vatani99-technical-university-of-munich
```

`experiment.init_ckpt` loads only the base model weights. The SATLOSS optimizer, scheduler, and mixed-precision scaler remain fresh, as required by the matched continuation protocol.
