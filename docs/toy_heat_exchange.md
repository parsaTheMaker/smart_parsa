# Toy Heat Exchange Benchmark

`toy_heat_exchange` is a deterministic, geometry-only nonlinear steady-state
heat-conduction benchmark for testing sampling sensitivity of point-cloud
neural operators.  The input is the solid's surface point cloud; the targets
are temperature in the solid and outward heat flux on the solid boundary.
There are no learned or hidden per-case forcing variables.

## Geometry

Each solid is an extrusion in the thickness direction of a two-dimensional
profile.  Its left and/or right boundary has the form

\[
x_s(z) = x_{s,0} + \sin(\pi z/H)\left[a_1\sin(2\pi n_1z/H+\phi_1)
+a_2\sin(2\pi n_2z/H+\phi_2)\right],
\]

where `s` is the left or right side.  The category is one of both-wavy,
left-wavy/right-straight, or right-wavy/left-straight.  The sine envelope gives
an exactly straight join at the base and tip.

The solid contains one to five through-channels, each either a circular
cylinder or a rectangular prism.  The rejection sampler enforces a positive
clearance to the exterior, positive channel-to-channel clearance, and checks
the profile across each channel's vertical span.  Every generated CAD body is
one watertight manifold solid after Boolean subtraction.

## Governing Equation

The solution is nondimensional temperature

\[
\theta = \frac{T-T_\infty}{T_h-T_\infty},
\]

where the globally fixed hot-fluid and ambient temperatures are `T_h=500 K`
and `T_inf=300 K`.  In the solid domain `Omega`, the benchmark solves

\[
-\nabla\cdot\left[(1+a\theta^2)\nabla\theta\right] = 0,
\qquad a=1.8.
\]

On the *visible walls of every internal circular or rectangular channel*, the
solid is held at the same globally fixed hot temperature:

\[
\theta = 1.
\]

This is an exact Dirichlet condition, so channel-wall temperature is constant
at `T_h=500 K` in every case.  The injected heat flux is an output of the
solution, \(q=-k(\theta)\nabla\theta\cdot n\), and varies with channel shape,
location, neighbouring channels, and the exterior geometry.  It is not a
volumetric source and does not depend on any unobserved channel label.

On all exterior faces the solid loses heat by convection and radiation:

\[
-(1+a\theta^2)\nabla\theta\cdot n = \mathrm{Bi}_e\theta
+R\left[(\theta+\tau)^4-\tau^4\right],
\]

with `Bi_e=0.35`, `R=0.07`, and `tau=T_inf/(T_h-T_inf)=1.5`.  The fourth-power
term is the nondimensional Stefan-Boltzmann radiation law.  It, together with
temperature-dependent conductivity, creates a nonlinear but monotone thermal
problem with a bounded physical solution.

## Numerical Method And Audits

Gmsh OpenCASCADE produces a tetrahedral mesh with distance/threshold fields
that refine hot channel walls, narrow ligaments between channels, and wavy
exterior sides.  Curvature-based sizing further resolves wavy crests and
troughs.  Linear P1 finite elements are solved by Picard iteration.  At each
iteration the positive conductivity and radiation secant coefficient are
evaluated from the current temperature; the resulting SPD system uses algebraic
multigrid-preconditioned conjugate gradients.  Channel degrees of freedom are
set exactly to \(\theta=1\) in every nonlinear iteration.  The generator rejects
a case when a linear residual, nonlinear update, temperature bound, tetrahedron
volume, adaptive-mesh area ratio, channel-wall facet classification, exact wall
temperature, or heat-flux sign audit fails.

Native encoder points are sampled uniformly per *mesh triangle*, retaining the
strong nonuniform density induced by local mesh refinement.  Surface and
volume targets use area- and volume-uniform samples, respectively.  KDE-16
log-density is computed and cached from the normalized native surface cloud.
This intentionally separates point sampling from the deterministic physical
operator.

## SMART Protocol

The native source cloud has 524,288 points.  Vanilla SMART receives 131,072
points and uses the standard `131K -> 16K -> 4K` encoder hierarchy.  SATLOSS7
uses the entire source cloud to form two 131,072-point shifted views and trains
with the existing fixed `[0.2, 0.2, 0.6]` primary-supervision,
secondary-supervision, and prediction-consistency weighting.  Hence only the
sampling protocol changes between the two methods; geometry, targets, queries,
optimizer, and backbone remain shared.
