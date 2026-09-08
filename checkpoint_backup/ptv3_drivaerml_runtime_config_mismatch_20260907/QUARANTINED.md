# Quarantined DrivAerML PTV3 DeAL Checkpoints

The base PTV3 checkpoint was trained with the density-sensitive runtime
configuration, while this DeAL continuation was trained with the standard
PTV3 runtime configuration. The differences include sparse ordering, order
shuffling, and density-preserving voxel weighting. Those options do not alter
state-dict tensor shapes, so loading the DeAL weights under the density-sensitive
configuration silently produced invalid inference.

The DeAL checkpoints were restored to `checkpoints/` after their actual runtime
configuration was identified. They must be evaluated with
`drivaerml_point_transformer_v3_satloss7`, never with the density-sensitive
DeAL config. The paired canonical PTV3 result rows and the all-architecture
cohort registry remain in the results quarantine because they used the wrong
runtime configuration.

The restored pair can be used for its recorded inference setup. A controlled
base--DeAL architecture study still requires a DeAL continuation trained with
the same runtime architecture as the density-sensitive base checkpoint.
