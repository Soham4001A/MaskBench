# MaskBench 1.0 (MuJoCo Baseline)

MaskBench evaluates robustness of pre-trained RL agents under input masking.
This baseline release focuses on **MuJoCo** continuous-control checkpoints from the **SB3 RL Zoo** on Hugging Face.

## Features
- One-click **download** of published checkpoints (Hugging Face).
- Two masking modes: **Channel Masking** (drop entire observation dims) and **Randomized Masking** (elementwise).
- **Batch eval** across all checkpoints for a given env (plots + CSV).
- **Visual eval** for a single model (env video + live reward plot + mask table overlay).

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Download all configured MuJoCo checkpoints
python scripts/download_models.py --domain mujoco

# Run eval across all models for Hopper-v3 (baseline + mask sweeps)
python scripts/eval_all.py --env-id Hopper-v3 --episodes 10

# Visualize one model
python scripts/eval_visualize.py --env-id Hopper-v3 --algo ppo --mask-type channel --mask-prob 0.5 --episodes 1
```
See `configs/mujoco_checkpoints.yaml` and `configs/eval_config.yaml` for options.

### Notes
- Checkpoints come from SB3’s Hugging Face org. Make sure MuJoCo can render on your platform.
- TRPO/TQC/ARS require `sb3-contrib` (already listed in requirements).
