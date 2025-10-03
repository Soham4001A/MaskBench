import os
import numpy as np
import pandas as pd
from typing import Dict, List, Callable
from stable_baselines3.common.vec_env import VecEnv
from tqdm import trange

def evaluate_model(model, env: VecEnv, episodes: int = 10, max_steps: int = 1000) -> Dict[str, float]:
    returns = []
    for _ in trange(episodes, desc="eval episodes", leave=False):
        obs = env.reset()[0]
        done = np.array([False])
        ep_ret = 0.0
        steps = 0
        while not done.any() and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated | truncated
            ep_ret += float(reward.mean())
            steps += 1
        returns.append(ep_ret)
    return {"mean_return": float(np.mean(returns)), "std_return": float(np.std(returns)), "n_episodes": episodes}

def sweep_mask_probs(run_fn: Callable[[float], Dict[str, float]], probs: List[float]) -> pd.DataFrame:
    rows = []
    for p in probs:
        metrics = run_fn(p)
        rows.append({"mask_prob": p, **metrics})
    return pd.DataFrame(rows)
