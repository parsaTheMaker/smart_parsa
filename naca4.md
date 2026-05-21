# NACA4-to-AirfRANS-Like Dataset Infographic

## 1) Dataset Identity
- **Dataset root**: `/mnt/data1/parsa/naca4_zenodo_airfrans_like/Dataset`
- **Derived from**: NACA 4-digit 2D RANS field shards (Zenodo record family around `4106752`, tar.xz shard format).
- **Purpose**: Steady-state CFD learning with a two-stage pipeline:
  - geometry pretraining
  - physics finetuning

## 2) Provenance / Source
Raw source format per case:
- `<case>/<case>.vtk`
- `<case>/<case>_walls.vtk`

Converted to per-case NPZ format under:
- `<dataset_root>/<case>/<case>.npz`

Post-processing that has already been applied:
- Mach-proxy filtering (`max(|U|)/340 <= 0.4`)
- Radius filtering around center `(0,0)` with `r <= 10`
- Corrupt-file cleanup from split manifests

## 3) Current Split Sizes (after cleanup)
From `manifest.json`:
- `full_train`: **959**
- `full_test`: **239**
- `scarce_train`: **959**
- `scarce_test`: **239**

Total usable cases in active manifests: **1198**.

## 4) Data Quality History
From conversion stats (`stats.csv`):
- Total processed original cases: **1211**
- Kept (`ok`): **1200**
- Filtered out due to `mach_gt_0.4`: **11**

From corruption cleanup:
- Corrupt/missing cases removed from manifests and data folders.
- Cleanup reports available in dataset root:
  - `bad_npz_cases.json`
  - `corrupt_removed_report.json`

## 5) File/Folder Structure
Inside dataset root:
- `manifest.json`
- `stats.csv`
- `radius_filter_report.csv`
- `bad_npz_cases.json`
- `corrupt_removed_report.json`
- Per-case directory:
  - `<case_id>/<case_id>.npz`

## 6) Per-Case NPZ Schema (required keys)
Each `<case>.npz` stores arrays:
- `position`: `(N, 2)` float32
- `velocity`: `(N, 2)` float32
- `pressure`: `(N, 1)` float32
- `nu_t`: `(N, 1)` float32 (present in files, but can be ignored)
- `sdf`: `(N, 1)` float32
- `normals`: `(N, 2)` float32
- `surface`: `(N,)` bool
- `inlet_velocity`: scalar float32
- `angle_of_attack`: scalar float32
- optional provenance tags

## 7) Recommended Training Mapping (nu_t ignored)
For a geometry+physics setup that ignores `nu_t`:
- Geometry attributes: `[sdf, normal_x, normal_y]` (3 channels)
- Physics targets: `[u, v, p]` (3 channels)

If using a volume/surface packed representation:
- Attributes packed as: `[vol_attr(3), surf_attr(3)]` -> 6 channels
- Values packed as: `[vol_val(3), surf_val(3)]` -> 6 channels
- Point order: volume points first, surface points second

## 8) Current Sampling / Domain Settings
Current practical settings used in training:
- `max_num_surface_points: 2048`
- `max_num_volume_points: 32768`
- `max_num_points: 34816`
- `min_domain: [-5.0, -5.0]`
- `max_domain: [5.0, 5.0]`

Evaluation ROI commonly used:
- `x,y in [-2, 2]`

## 9) Radius Filtering Impact (`r<=10`)
From `radius_filter_report.csv`:
- Rows (cases): **1200**
- Total points before: **556,690,736**
- Total points after: **442,363,079**
- Total removed: **114,327,657**
- Mean fractional removal: **0.1905**
- 95th percentile fractional removal: **0.2635**

## 10) Distribution Snapshot (200 train-case sample)
Computed directly from current NPZ files:

### Velocity `u`
- mean: `22.4165`
- std: `12.9409`
- p1/p99: `-4.1568 / 49.2991`
- min/max: `-60.5751 / 80.2586`

### Velocity `v`
- mean: `4.4761`
- std: `8.2287`
- p1/p99: `-8.3697 / 38.2338`
- min/max: `-22.5653 / 134.6288`

### Pressure `p`
- mean: `-79.0977`
- std: `310.5267`
- p1/p99: `-1155.9054 / 417.2200`
- min/max: `-9177.0596 / 450.1235`

### SDF
- mean: `0.8452`
- std: `1.8314`
- p1/p99: `7.73e-06 / 8.5970`
- min/max: `0.0 / 10.0118`

### Surface fraction in sampled points
- approx `0.00271` (~0.27%) in this snapshot.

## 11) Known Caveats
- Raw VTK files do not expose robust per-case Reynolds/AoA metadata in-file.
- `angle_of_attack` and `inlet_velocity` in converted NPZ are conversion-level metadata/proxies.
- Dataset is steady-state; use stationary modeling assumptions.
- Pressure has heavy tails; percentile-based diagnostics are recommended.
- Keep an integrity scan in any data pipeline (corrupt NPZs occurred and were cleaned).

## 12) Recommended Preflight Checks
1. Validate all manifest-listed NPZ files can be opened.
2. Validate required keys exist: `position, velocity, pressure, sdf, normals, surface`.
3. Validate both volume and surface subsets are non-empty per case.
4. Validate no NaN/Inf in required arrays.
5. Log per-split point-count quantiles to detect outliers.

## 13) Handoff Summary
- Use dataset root: `/mnt/data1/parsa/naca4_zenodo_airfrans_like/Dataset`
- Split sizes: train `959`, test `239`
- Physics channels (recommended): `u,v,p` (ignore `nu_t`)
- Geometry channels: `sdf,nx,ny`
- Domain used in training: `[-5,5]`
- Common evaluation ROI: `[-2,2]`
- Integrity cleanup already performed; still run preflight checks in any new environment.
