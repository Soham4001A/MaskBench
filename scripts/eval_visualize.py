#!/usr/bin/env python3
import argparse
import numpy as np
from gymnasium import spaces

from src.common.env_utils import make_env, get_vecnormalize_path
from src.common.model_utils import load_sb3_model
from src.masking.utils import mask_table
from src.common.plot_utils import render_sweep_plot

def build_masker(mask_type: str, p: float):
    if mask_type == "channel":
        from src.masking.channel_mask import ChannelMask
        return ChannelMask(p=p, drop_ratio=0.3)
    elif mask_type == "randomized":
        from src.masking.random_mask import RandomMask
        return RandomMask(p=p, drop_ratio=0.3)
    else:
        raise ValueError(f"Unknown mask_type: {mask_type}")

def as_vec_obs(obs, target_space=None):
    x = np.asarray(obs, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, ...]
    elif x.ndim > 2 and x.shape[0] == 1:
        x = np.squeeze(x, axis=0)
        if x.ndim == 1:
            x = x[None, ...]
    return x

def unwrap_vecnormalize(env):
    v = env
    visited = set()
    while v is not None and id(v) not in visited:
        visited.add(id(v))
        if isinstance(v, VecNormalize):
            return v
        v = getattr(v, "venv", None)
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--algo", required=True, help="algo folder name, e.g., ppo/sac/td3/trpo/tqc/a2c/ars")
    parser.add_argument("--mask-type", choices=["channel", "randomized"], default="channel")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--ckpt_root", default="checkpoints/mujoco")
    parser.add_argument("--out", default="outputs/sweep.png")
    args = parser.parse_args()

    model_dir = os.path.join(args.ckpt_root, args.env_id, args.algo)
    assert os.path.isdir(model_dir), f"Missing model dir: {model_dir}"

    vecnorm_path = get_vecnormalize_path(model_dir)
    env = make_env(args.env_id, seed=0, vecnorm_path=vecnorm_path)

    if isinstance(env.observation_space, spaces.Box) and env.observation_space.dtype != np.float32:
        new_space = spaces.Box(
            low=env.observation_space.low.astype(np.float32),
            high=env.observation_space.high.astype(np.float32),
            shape=env.observation_space.shape,
            dtype=np.float32,
        )
        env.observation_space = new_space

    try:
        vn = unwrap_vecnormalize(env)
        if vn is not None:
            norm_obs_backup, norm_reward_backup = vn.norm_obs, vn.norm_reward
            vn.norm_obs, vn.norm_reward = False, False

            expected_obs_shape = tuple(vn.venv.observation_space.shape)
            current_obs_shape = tuple(getattr(getattr(vn, "obs_rms", None), "mean", np.array([])).shape)

            if current_obs_shape != expected_obs_shape or vn.observation_space.dtype != np.float32:
                new_space = spaces.Box(
                    low=vn.venv.observation_space.low.astype(np.float32),
                    high=vn.venv.observation_space.high.astype(np.float32),
                    shape=expected_obs_shape,
                    dtype=np.float32,
                )
                vn.observation_space = new_space
                vn.obs_rms = RunningMeanStd(shape=expected_obs_shape)
                vn.ret_rms = RunningMeanStd(shape=())

            vn.norm_obs, vn.norm_reward = norm_obs_backup, norm_reward_backup
            vn.training = False
    except Exception as e:
        print(f"[MaskBench] VecNormalize unwrap/fix skipped due to: {e}")

    algo, model, original_obs_space = load_sb3_model(model_dir, env)

    sweep_probs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    all_rewards = {}

    for prob in sweep_probs:
        print(f"Running sweep for mask_prob={prob}...")
        masker = build_masker(args.mask_type, p=prob)
        
        episode_rewards = []
        for ep in range(args.episodes):
            obs = env.reset()
            rewards = []
            for step in range(args.max_steps):
                if isinstance(obs, np.ndarray) and obs.ndim == 1:
                    masked_obs = masker.maybe_apply(obs.astype(np.float32, copy=False))
                else:
                    masked_obs = np.asarray(obs, dtype=np.float32).copy()
                    masked_obs[0] = masker.maybe_apply(masked_obs[0])

                batched_obs = as_vec_obs(masked_obs)

                expected_shape = original_obs_space.shape[0]
                if batched_obs.shape[1] < expected_shape:
                    padding = np.zeros((batched_obs.shape[0], expected_shape - batched_obs.shape[1]), dtype=np.float32)
                    batched_obs = np.concatenate([batched_obs, padding], axis=1)

                original_space = model.observation_space
                padded_space = spaces.Box(low=-np.inf, high=np.inf, shape=(expected_shape,), dtype=np.float32)
                model.observation_space = padded_space
                model.policy.observation_space = padded_space

                action, _ = model.predict(batched_obs, deterministic=True)

                model.observation_space = original_space
                model.policy.observation_space = original_space

                act = np.asarray(action, dtype=np.float32)
                if act.ndim == 1:
                    act = act[None, ...]
                
                step_result = env.step(act)
                if len(step_result) == 5:
                    next_obs, reward, terminated, truncated, info = step_result
                    done = terminated | truncated
                else:
                    next_obs, reward, done, info = step_result
                rewards.append(float(np.mean(reward)))
                obs = next_obs

                if done.any():
                    obs = env.reset()
            episode_rewards.append(rewards)
        
        mean_rewards = np.mean(episode_rewards, axis=0)
        all_rewards[prob] = mean_rewards

    title = f"{args.algo}-{args.mask_type}"
    render_sweep_plot(all_rewards, args.out, title)

if __name__ == "__main__":
    main()
