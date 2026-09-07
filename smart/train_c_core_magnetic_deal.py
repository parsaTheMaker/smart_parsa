"""Continue a completed C-core base surrogate with the DeAL objective."""

import hydra
from omegaconf import DictConfig

from models.ab_upt import ABUPT
from models.geo_fno import GeoFNO
from models.lno import LNOWithLatent
from models.mspt import MSPT
from models.point_transformer_v3 import PointTransformerV3WithLatent
from models.pointnet2_ssg import PointNet2SSGWithLatent
from models.smart.smart import SMART
from models.transolverpp import TransolverPP
from train_consistency_common import run_consistency_training


MODELS = {
    "SMART": SMART,
    "AB_UPT": ABUPT,
    "GEOFNO": GeoFNO,
    "POINTNET2_SSG": PointNet2SSGWithLatent,
    "LNO": LNOWithLatent,
    "MSPT": MSPT,
    "TRANSOLVERPP": TransolverPP,
    "POINT_TRANSFORMER_V3": PointTransformerV3WithLatent,
}


@hydra.main(
    version_base="1.2",
    config_path="config",
    config_name="c_core_magnetic_smart_deal_from_base",
)
def main(cfg: DictConfig):
    model_name = str(cfg.experiment.model_name).upper()
    matches = [constructor for prefix, constructor in MODELS.items() if model_name.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"Could not resolve exactly one DeAL model constructor from {model_name!r}.")
    run_consistency_training(cfg, model_ctor=matches[0], model_requires_density=False)


if __name__ == "__main__":
    main()

