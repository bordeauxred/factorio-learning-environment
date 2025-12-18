"""
SkyrL training entrypoint for Factorio.

Usage:
    python examples/factorio/main_factorio.py <config_overrides>
"""
import ray
from omegaconf import DictConfig
import hydra

from skyrl_train.utils.config_validator import validate_cfg
from skyrl_train.utils.ray_utils import initialize_ray
from skyrl_train.experiments.baseppo_exp import BasePPOExp
from skyrl_gym import register


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    """Register Factorio env and run training."""
    register(
        id="factorio",
        entry_point="fle.env.skyrl_wrapper:FactorioEnv",
    )
    exp = BasePPOExp(cfg)
    exp.run()


@hydra.main(config_path="configs", config_name="factorio", version_base=None)
def main(cfg: DictConfig) -> None:
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
