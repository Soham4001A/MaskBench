import os
import glob
from typing import Tuple, Optional
import zipfile
import pickle

import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO, A2C, SAC, TD3
from sb3_contrib import TRPO, TQC, ARS
from stable_baselines3.common.vec_env import VecEnv

ALGO_MAP = {
    "ppo": PPO,
    "a2c": A2C,
    "sac": SAC,
    "td3": TD3,
    "trpo": TRPO,
    "tqc": TQC,
    "ars": ARS,
}

KNOWN_ALGOS = tuple(ALGO_MAP.keys())
IGNORE_SUBSTR = ("metrics", "monitor", "replay", "stats")


def _candidate_model_zips(model_dir: str):
    zips = glob.glob(os.path.join(model_dir, "*.zip"))
    return sorted(
        zp for zp in zips
        if not any(s in os.path.basename(zp).lower() for s in IGNORE_SUBSTR)
    )


def _select_model_zip(model_dir: str, algo_hint: Optional[str] = None) -> Tuple[str, str]:
    cand = _candidate_model_zips(model_dir)
    if not cand:
        raise FileNotFoundError(f"No candidate policy .zip found in {model_dir}")

    if algo_hint:
        ah = algo_hint.lower()
        for zp in cand:
            if os.path.basename(zp).lower().startswith(f"{ah}-"):
                return ah, zp

    for zp in cand:
        base = os.path.basename(zp).lower()
        for algo in KNOWN_ALGOS:
            if base.startswith(f"{algo}-"):
                return algo, zp

    pref = os.path.basename(cand[0]).lower().split("-")[0]
    if pref in ALGO_MAP:
        return pref, cand[0]

    raise ValueError(f"Could not infer algorithm from zips in {model_dir}: {[os.path.basename(z) for z in cand]}")


def _as_float32_space(space):
    """Return an equivalent Box space with dtype=float32 if needed."""
    if isinstance(space, spaces.Box) and space.dtype != np.float32:
        low = space.low.astype(np.float32, copy=False)
        high = space.high.astype(np.float32, copy=False)
        return spaces.Box(low=low, high=high, shape=space.shape, dtype=np.float32)
    return space


def load_sb3_model(model_dir: str, env: VecEnv):
    """
    Robust load for Gym-era SB3 zips on Gymnasium:
      1) pick correct .zip
      2) load WITHOUT env (skip early space checks)
      3) replace legacy schedules with constants
      4) force model/policy spaces to match env, with dtype=float32
      5) attach env
    """
    algo_hint = os.path.basename(model_dir)
    algo, zip_path = _select_model_zip(model_dir, algo_hint=algo_hint)
    cls = ALGO_MAP[algo]

    custom_objects = {
        "learning_rate": 3e-4,
        "lr_schedule": 3e-4,
        "clip_range": 0.2,
    }

    # Load the model without an environment first to get its original observation and action spaces
    temp_model = cls.load(zip_path, env=None, device="cpu", custom_objects=custom_objects)
    original_model_obs_space = temp_model.observation_space
    original_model_action_space = temp_model.action_space

    # Temporarily change env.observation_space and env.action_space to match original_model_obs_space and original_model_action_space
    original_env_obs_space = env.observation_space
    original_env_action_space = env.action_space
    env.observation_space = original_model_obs_space
    env.action_space = original_model_action_space

    if algo == "trpo":
        model = cls.load(zip_path, env=env, device="cpu", custom_objects=custom_objects)
    else:
        model = cls.load(zip_path, env=None, device="cpu", custom_objects=custom_objects)
    
    # Revert env.observation_space and env.action_space
    env.observation_space = original_env_obs_space
    env.action_space = original_env_action_space

    original_obs_space = model.observation_space # This will be the model's observation space after loading

    # 2) coerce observation space to float32 to avoid SB3 vectorization ambiguity
    env_obs_space_f32 = _as_float32_space(env.observation_space)

    # 3) align spaces on model and policy
    model.observation_space = env_obs_space_f32
    model.action_space = env.action_space

    if hasattr(model, "policy"):
        try:
            model.policy.observation_space = env_obs_space_f32
        except Exception:
            pass
        try:
            model.policy.action_space = env.action_space
        except Exception:
            pass
        fe = getattr(model.policy, "features_extractor", None)
        if fe is not None and hasattr(fe, "observation_space"):
            try:
                fe.observation_space = env_obs_space_f32
            except Exception:
                pass

    # 4) attach env
    if algo != "trpo":
        model.set_env(env)
    
    return algo, model, original_obs_space

    # 2) coerce observation space to float32 to avoid SB3 vectorization ambiguity
    env_obs_space_f32 = _as_float32_space(env.observation_space)

    # 3) align spaces on model and policy
    model.observation_space = env_obs_space_f32
    model.action_space = env.action_space

    if hasattr(model, "policy"):
        try:
            model.policy.observation_space = env_obs_space_f32
        except Exception:
            pass
        try:
            model.policy.action_space = env.action_space
        except Exception:
            pass
        fe = getattr(model.policy, "features_extractor", None)
        if fe is not None and hasattr(fe, "observation_space"):
            try:
                fe.observation_space = env_obs_space_f32
            except Exception:
                pass

    # 4) attach env
    if algo != "trpo":
        model.set_env(env)
    return algo, model, original_obs_space