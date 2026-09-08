"""Continue a completed C-core base surrogate with the DeAL objective."""

import hydra
from omegaconf import DictConfig

from models.smart.smart import SMART
from train_consistency_common import run_consistency_training


def resolve_model_constructor(model_name: str):
    """Load only the requested architecture and keep SMART runnable on lean installs."""
    model_name = str(model_name).upper()
    if model_name.startswith("SMART"):
        return SMART
    if model_name.startswith("AB_UPT"):
        from models.ab_upt import ABUPT

        return ABUPT
    if model_name.startswith("GEOFNO"):
        from models.geo_fno import GeoFNO

        return GeoFNO
    if model_name.startswith("POINTNET2_SSG"):
        from models.pointnet2_ssg import PointNet2SSGWithLatent

        return PointNet2SSGWithLatent
    if model_name.startswith("LNO"):
        from models.lno import LNOWithLatent

        return LNOWithLatent
    if model_name.startswith("MSPT"):
        from models.mspt import MSPT

        return MSPT
    if model_name.startswith("TRANSOLVERPP"):
        from models.transolverpp import TransolverPP

        return TransolverPP
    if model_name.startswith("POINT_TRANSFORMER_V3"):
        from models.point_transformer_v3 import PointTransformerV3WithLatent

        return PointTransformerV3WithLatent
    raise ValueError(f"Unsupported C-core DeAL model_name: {model_name!r}")


@hydra.main(
    version_base="1.2",
    config_path="config",
    config_name="c_core_magnetic_smart_deal_from_base",
)
def main(cfg: DictConfig):
    model_ctor = resolve_model_constructor(cfg.experiment.model_name)
    run_consistency_training(cfg, model_ctor=model_ctor, model_requires_density=False)


if __name__ == "__main__":
    main()
