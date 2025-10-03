import os
from typing import Optional

import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ---------------------------------------------------------------------
# MuJoCo version aliasing:
# If an old Gym ID (e.g., Hopper-v3) is unavailable, fall back to v5.
# ---------------------------------------------------------------------
MUJOCO_ENV_VERSION_ALIAS = {
    "-v2": "-v5",
    "-v3": "-v5",
    "-v4": "-v5",
}


def _alias_env_id(env_id: str) -> str:
    """Map legacy MuJoCo env suffixes to v5 when older versions are unavailable."""
    for old_suffix, new_suffix in MUJOCO_ENV_VERSION_ALIAS.items():
        if env_id.endswith(old_suffix):
            return env_id[: -len(old_suffix)] + new_suffix
    return env_id


# ----------------------- Env constructors -------------------------------------

def make_base_env(env_id: str, seed: int = 0):
    """Create a single Gymnasium env (with RGB array render), with v5 fallback."""
    def _thunk():
        try:
            env = gym.make(env_id, render_mode="rgb_array")
        except Exception:
            # Fallback: map legacy v2/v3/v4 to v5 if available
            new_id = _alias_env_id(env_id)
            if new_id != env_id:
                env = gym.make(new_id, render_mode="rgb_array")
            else:
                raise
        env.reset(seed=seed)
        return env

    return _thunk


def get_vecnormalize_path(model_dir: str) -> Optional[str]:
    """
    Return path to VecNormalize stats if present.
    Checks the algo folder first, then its parent (some SB3 repos store it there).
    """
    pkl = os.path.join(model_dir, "vec_normalize.pkl")
    if os.path.exists(pkl):
        return pkl
    parent = os.path.dirname(model_dir)
    pkl2 = os.path.join(parent, "vec_normalize.pkl")
    return pkl2 if os.path.exists(pkl2) else None


def make_env(env_id: str, seed: int = 0, vecnorm_path: Optional[str] = None):
    """
    Create a 1-env VecEnv (DummyVecEnv). If vecnorm_path is provided and exists,
    load VecNormalize stats and set eval mode (no further updates).
    """
    env = DummyVecEnv([make_base_env(env_id, seed=seed)])
    if vecnorm_path and os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False
    return env