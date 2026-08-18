"""Dataset adapter for the mesh-FEM perforated-fin SATLOSS benchmark."""

from data.toy_satloss_dataset import ToySATLossDataset


class ToyPerforatedFinDataset(ToySATLossDataset):
    """Reuse the proven toy loader with fin-specific statistics and field names."""

    CACHE_VERSION = "toy_perforated_fin_nonlinear_fem_train_stats_v2"
    SURFACE_FIELDS = ("outward_heat_flux",)
    VOLUME_FIELDS = ("temperature",)
