#!/usr/bin/env bash
set -euo pipefail

# Requires: huggingface-cli (pip install huggingface_hub) and HF auth (hf login)
# Usage:
#   bash scripts/download_models.sh mujoco [Ant-v3]
#
DOMAIN="${1:-mujoco}"
ENV_FILTER="${2:-}"

CONFIG="configs/${DOMAIN}_checkpoints.yaml"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "huggingface-cli not found. Install with: pip install huggingface_hub"
  exit 1
fi

python - <<'PY'
import sys, yaml, os, subprocess
domain = sys.argv[1]
env_filter = sys.argv[2] if len(sys.argv) > 2 else ""
cfg = yaml.safe_load(open(f"configs/{domain}_checkpoints.yaml"))
for env_id, algos in cfg["envs"].items():
    if env_filter and env_id != env_filter:
        continue
    for algo, meta in algos.items():
        repo = meta["hf_repo"]
        out_dir = os.path.join("checkpoints", domain, env_id, algo)
        os.makedirs(out_dir, exist_ok=True)
        print(f"[HF] Downloading {repo} -> {out_dir}")
        # Use huggingface-cli to snapshot the repo
        subprocess.check_call([
            "huggingface-cli", "download", repo, "--repo-type", "model",
            "--local-dir", out_dir, "--local-dir-use-symlinks", "False"
        ])
print("Done.")
PY
