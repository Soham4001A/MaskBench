#!/usr/bin/env python3
import argparse, os, yaml, glob, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.common.env_utils import make_env, get_vecnormalize_path
from src.common.model_utils import load_sb3_model
from src.common.eval_utils import evaluate_model, sweep_mask_probs
from src.masking.channel_mask import ChannelMask
from src.masking.random_mask import RandomMask

def build_masker(mask_type: str, p: float):
    if mask_type == "channel":
        return ChannelMask(p=p, drop_ratio=0.3)
    elif mask_type == "randomized":
        return RandomMask(p=p, drop_ratio=0.3)
    else:
        raise ValueError(f"Unknown mask_type: {mask_type}")

def apply_mask_to_obs(obs, masker):
    # obs is (1, obs_dim) because of VecEnv
    vec = obs.copy()
    masked = masker.maybe_apply(vec[0])
    vec[0] = masked
    return vec

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--eval-config", default="configs/eval_config.yaml")
    parser.add_argument("--ckpt_root", default="checkpoints/mujoco")
    args = parser.parse_args()

    with open(args.eval_config, "r") as f:
        econf = yaml.safe_load(f)
    probs = econf.get("mask_probs", [0.0, 0.25, 0.5, 0.75, 1.0])

    env_dir = os.path.join(args.ckpt_root, args.env_id)
    algo_dirs = sorted(glob.glob(os.path.join(env_dir, "*")))
    assert algo_dirs, f"No checkpoints found under {env_dir}"

    results_dir = "outputs/results_csv"
    plots_dir = "outputs/plots"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    summary_tables = []
    for algo_path in algo_dirs:
        algo_name = os.path.basename(algo_path)
        # Build env with potential VecNormalize
        vecnorm_path = os.path.join(algo_path, "vec_normalize.pkl")
        env = make_env(args.env_id, seed=0, vecnorm_path=vecnorm_path)

        # Baseline (0.0 mask)
        def run_with_mask(p, mask_type="channel"):
            # Wrap step loop to inject masking into obs
            # We'll monkey-patch env.step to apply mask right after step
            from types import MethodType
            masker = build_masker(mask_type, p)

            orig_reset = env.reset
            orig_step = env.step

            def reset_with_mask(*a, **kw):
                obs, info = orig_reset(*a, **kw)
                obs = apply_mask_to_obs(obs, masker)
                return obs, info

            def step_with_mask(action):
                obs, rew, term, trunc, info = orig_step(action)
                obs = apply_mask_to_obs(obs, masker)
                return obs, rew, term, trunc, info

            env.reset = MethodType(reset_with_mask, env)
            env.step = MethodType(step_with_mask, env)

            algo, model = load_sb3_model(algo_path, env)
            metrics = evaluate_model(model, env, episodes=args.episodes, max_steps=args.max_steps)

            # Restore
            env.reset = orig_reset
            env.step = orig_step
            return metrics

        # Run both mask types + baseline
        df_channel = sweep_mask_probs(lambda p: run_with_mask(p, "channel"), probs)
        df_channel["algo"] = algo_name
        df_channel["mask_type"] = "channel"

        df_random = sweep_mask_probs(lambda p: run_with_mask(p, "randomized"), probs)
        df_random["algo"] = algo_name
        df_random["mask_type"] = "randomized"

        df = pd.concat([df_channel, df_random], ignore_index=True)
        csv_path = os.path.join(results_dir, f"{args.env_id}_{algo_name}.csv")
        df.to_csv(csv_path, index=False)
        summary_tables.append(df)

    # Plot: per algo, lines over mask_prob; dashed for baseline overlay
    all_df = pd.concat(summary_tables, ignore_index=True)
    for mtype in ["channel", "randomized"]:
        plt.figure(figsize=(8,5))
        for algo in sorted(all_df["algo"].unique()):
            sub = all_df[(all_df["algo"]==algo) & (all_df["mask_type"]==mtype)]
            plt.plot(sub["mask_prob"], sub["mean_return"], label=algo)
            # Baseline as dashed horizontal line at p=0
            base_val = sub[sub["mask_prob"]==0.0]["mean_return"].values
            if base_val.size>0:
                plt.hlines(base_val[0], 0, 1.0, linestyles="dashed")
        plt.title(f"{args.env_id} – {mtype} mask")
        plt.xlabel("Mask probability")
        plt.ylabel("Mean return")
        plt.legend()
        outp = os.path.join(plots_dir, f"{args.env_id}_{mtype}.png")
        plt.tight_layout()
        plt.savefig(outp)
        plt.close()

    print(f"Saved CSVs to {results_dir} and plots to {plots_dir}")

if __name__ == "__main__":
    main()
