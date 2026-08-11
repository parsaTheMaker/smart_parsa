import hydra
from omegaconf import DictConfig

from models.shift_crash_smart import ShiftCrashSMART
from shift_crash_training import run_shift_crash_training


@hydra.main(version_base="1.2", config_path="config", config_name="shift_crash_conditioned")
def main(cfg: DictConfig):
    run_shift_crash_training(cfg, satloss7=False, model_cls=ShiftCrashSMART)


if __name__ == "__main__":
    main()

