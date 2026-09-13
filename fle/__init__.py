"""Factorio Learning Environment (FLE) package."""

# Make submodules available
from fle import agents, env, eval, cluster, commons

# Auto-register all gym environments when FLE is imported
try:
    from fle.env.gym_env.registry import register_all_environments
    register_all_environments()
except ImportError:
    pass

# Auto-register in skyrl-gym if available
try:
    from skyrl_gym.envs.registration import register
    register(
        id="factorio",
        entry_point="fle.env.skyrl_wrapper:FactorioEnv",
    )
except ImportError:
    pass

__all__ = ["agents", "env", "eval", "cluster", "commons"]
