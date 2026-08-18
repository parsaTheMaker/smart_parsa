"""Dataset adapter for the nonlinear geometry-only heat-exchanger benchmark."""

from data.toy_satloss_dataset import ToySATLossDataset


class ToyHeatExchangeDataset(ToySATLossDataset):
    """Use the shared point-cloud loader with heat-exchanger field metadata."""

    CACHE_VERSION = "toy_heat_exchange_fem_train_stats_v1"
    SURFACE_FIELDS = ("outward_heat_flux",)
    VOLUME_FIELDS = ("temperature",)
