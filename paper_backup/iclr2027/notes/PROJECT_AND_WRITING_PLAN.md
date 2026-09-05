# Project And Writing Plan

## Central Claim

For a fixed continuous geometry \(\Gamma\), two finite point clouds \(X_A\) and \(X_B\) can represent the same geometry while having different sampling densities. A geometry-to-field surrogate \(F_\theta\) should satisfy

\[
F_\theta(X_A, Q, \lambda) \approx F_\theta(X_B, Q, \lambda),
\]

for a fixed query set \(Q\) and physical conditioning \(\lambda\). Existing point-based architectures need not satisfy this property because attention, fixed-radius aggregation, KNN support, and voxel occupancy all depend on the input sampling distribution.

SATLOSS trains on two shifted views of the same geometry. Both views receive supervised losses against the same target, while a prediction-consistency loss discourages view-dependent outputs. The empirical claim is not that every model is sensitive, but that this sensitivity is a recurring and measurable failure mode across common surrogate architecture families and can be reduced without sacrificing aligned accuracy.

## Main-Paper Narrative

1. **Problem.** Unstructured-surrogate inputs are discretizations, not the geometry itself. Point density can change through remeshing, scanning, preprocessing, adaptive meshing, or varying acquisition pipelines.
2. **Why it happens.** Give the compact theoretical argument for density-dependent attention/integration and local/voxel support. Keep detailed derivations in the appendix.
3. **Method.** Define density estimation, inverse-density view sampling, the two supervised view losses, and the prediction-consistency term. State exactly which components are fixed and which are ablated.
4. **Protocol.** Evaluate equal query budgets, shared seeds/geometry views across checkpoints, density-shift endpoints, sine-x/sine-y shifts, and independently remeshed meshes. Separate in-distribution accuracy from robustness.
5. **Evidence.** Show improvements across DrivAerML and non-aerodynamic/toy or other geometry-to-field data where available. Include multiple architecture families rather than presenting SMART alone.
6. **Ablations.** Show the need for consistency, density-neighborhood choice, and training-shift range. Make clear which ablations use a weight-only baseline initialization and which train from scratch.
7. **Limitations.** SATLOSS addresses representation sampling shifts, not arbitrary geometric out-of-distribution changes, missing physical conditioning, or solver/data errors.

## Evidence Required Before Submission

- A clear baseline-vs-SATLOSS table with aligned accuracy and endpoint-shift accuracy.
- One paper-quality visualization explaining the sampled views and one showing the resulting prediction stability.
- Remeshing evaluation that uses actual remeshed input meshes, not an additional analytic shift applied after remeshing.
- Averages over several geometries and seeds or views, with uncertainty reporting where meaningful.
- A consistency-loss ablation using the same base initialization/training budget whenever the comparison claims causal benefit.
- A reproducibility package: configs, checkpoint identifiers, deterministic evaluation settings, plotting scripts, and anonymous code instructions.
- An AI-use disclosure reviewed by all authors.

## Suggested Nine-Page Budget

| Section | Target pages |
| --- | ---: |
| Introduction and contributions | 1.0 |
| Related work | 0.75 |
| Problem and mechanism | 1.25 |
| SATLOSS method | 1.5 |
| Experimental protocol | 1.0 |
| Main results | 2.0 |
| Ablations and limitations | 1.0 |
| Reproducibility and ethics/AI statements | exempt, placed before references |

The table is a planning tool, not a conference rule. The official submission limit is nine main-text pages; references, appendices, required AI disclosure, and an optional ethics/reproducibility statement are treated separately under the current ICLR guidance.

## Presentation Inventory

The update presentations are retained in `../../presentations/`. The 2026-08-19 presentation contains the most mature mathematical framing: density-dependent attention, ball/KNN neighborhood arguments, voxel occupancy, inverse-density sampling, and the two-view objective. Earlier updates should be mined for experimental chronology and ablation rationale, but claims in the paper must be revalidated against the final saved artifacts.
