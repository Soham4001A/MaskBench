#!/usr/bin/env python3
import argparse
import os
import yaml
from huggingface_hub import snapshot_download

def download_repo(hf_repo: str, target_dir: str):
    os.makedirs(target_dir, exist_ok=True)
    print(f"[HF] Downloading {hf_repo} -> {target_dir}")
    snapshot_download(repo_id=hf_repo, local_dir=target_dir, repo_type="model", ignore_patterns=["*.git*"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["mujoco"], required=True, help="Which domain to download (mujoco only for now)")
    parser.add_argument("--config", default="configs/mujoco_checkpoints.yaml")
    parser.add_argument("--env-id", default=None, help="Optional: only download for this env")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    envs = cfg["envs"]
    for env_id, algos in envs.items():
        if args.env_id and env_id != args.env_id:
            continue
        for algo, meta in algos.items():
            repo = meta["hf_repo"]
            out_dir = os.path.join("checkpoints", "mujoco", env_id, algo)
            download_repo(repo, out_dir)
    print("Done.")

if __name__ == "__main__":
    main()
