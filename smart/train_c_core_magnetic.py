"""Train one base surrogate architecture on the C-core magnetic benchmark."""

import hydra
from omegaconf import DictConfig

from models.ab_upt import ABUPT
from models.geo_fno import GeoFNO
from models.lno import LNO
from models.mspt import MSPT
from models.point_transformer_v3 import PointTransformerV3
from models.pointnet2_ssg import PointNet2SSG
from models.smart.smart import SMART
from models.transolverpp import TransolverPP
from utils.surface_volume_trainer import run_surface_volume_training


MODELS = {
    "SMART": SMART,
    "AB_UPT": ABUPT,
    "GEOFNO": GeoFNO,
    "POINTNET2_SSG": PointNet2SSG,
    "LNO": LNO,
    "MSPT": MSPT,
    "TRANSOLVERPP": TransolverPP,
    "POINT_TRANSFORMER_V3": PointTransformerV3,
}


@hydra.main(version_base="1.2", config_path="config", config_name="c_core_magnetic_smart")
def main(cfg: DictConfig):
    model_name = str(cfg.experiment.model_name).upper()
    matches = [constructor for prefix, constructor in MODELS.items() if model_name.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"Could not resolve exactly one model constructor from {model_name!r}.")
    run_surface_volume_training(cfg, matches[0], accepts_geo_log_density=False)


if __name__ == "__main__":
    main()
