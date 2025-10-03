# MaskBench: Robust RL under Observation Masking

A lightweight benchmarking and visualization suite for evaluating **robustness of reinforcement learning (RL) policies under missing/occluded observations**. MaskBench plugs stochastic maskers between the environment and the policy, logs reward trajectories, and renders side‑by‑side visualizations (env frames + reward curve + human‑readable mask table) for quick inspection.

---

## Table of Contents

1. [Motivation & Backstory](#motivation--backstory)
2. [Key Features](#key-features)
3. [Repository Structure](#repository-structure)
4. [Installation](#installation)
5. [Quickstart](#quickstart)
6. [Concepts & Mathematics](#concepts--mathematics)

   * [Observation masking model](#observation-masking-model)
   * [ChannelMask](#channelmask)
   * [RandomMask](#randommask)
   * [Effective masking rate and variance](#effective-masking-rate-and-variance)
   * [Mask scheduling](#mask-scheduling)
   * [POMDP view & intuition](#pomdp-view--intuition)
7. [Implementation Details](#implementation-details)

   * [Runtime flow](#runtime-flow)
   * [SB3 + Gymnasium compatibility](#sb3--gymnasium-compatibility)
   * [Legacy checkpoint adapter](#legacy-checkpoint-adapter)
8. [CLI Usage](#cli-usage)
9. [Troubleshooting](#troubleshooting)
10. [Extending MaskBench](#extending-maskbench)
11. [Evaluation Protocols & Suggested Experiments](#evaluation-protocols--suggested-experiments)
12. [Roadmap](#roadmap)

---

## Motivation & Backstory

This repo grew out of a series of experiments and discussions around **robustness to occlusions and partial observations** for continuous‑control agents (e.g., MuJoCo Hopper/Ant) and, later, Atari/classic‑control. The core idea: **inject stochastic patterns of missing data** at evaluation time (and optionally during training) to quantify how well a policy maintains performance when sensors fail, features drop, or channels go offline.

Key drivers that led us here:

* Practical settings (robotics, AV, HIL sims) experience **intermittent feature loss** (sensor dropouts, comms glitches). We want an **easy, consistent harness** to test policies under these failures.
* Prior MaskBench iterations showed friction around **Gym→Gymnasium migration**, **VecNormalize stats drift**, and **env version changes (v3→v5)**. We baked solutions into this repo so evaluation “just works.”
* A need for **visual, interpretable outputs**: per‑step mask tables, cumulative reward plots, and side‑by‑side video to quickly see *what was missing* when a policy failed.

---

## Key Features

*   **Full Sweep Evaluation**: Automates evaluation across a range of mask probabilities (0.0, 0.2, ..., 1.0) for all specified models.
*   **Organized Plotting**: Generates clear cumulative reward plots for each model, saved in environment-specific directories with descriptive filenames.
*   **Animated Visualization**: For single model evaluations, generates a video showing the environment alongside a real-time plot of all sweep curves, highlighting the current run.
*   **Flipped Y-Axis**: Plots automatically invert their Y-axis if cumulative rewards trend negative, enhancing readability.
*   **Pluggable maskers**:

    *   `ChannelMask`: drop whole channels/features at once.
    *   `RandomMask`: drop individual entries randomly.
*   **Stochastic control** via two knobs:

    *   `p` (masking probability per step): turn masking on/off stochastically at each timestep.
    *   `drop_ratio` (severity): fraction of features (or entries) zeroed when masking triggers.
*   **Legacy compatibility**:

    *   Gym→Gymnasium env fallback (v3→v5 aliasing for MuJoCo).
    *   Auto‑fix **VecNormalize** stats shape mismatches.
    *   Robust **model‑expected‑dim adapter** to safely run v3 checkpoints in v5 envs (pads or slices observations).
*   **Shape/dtype hygiene**: always feed `float32` batched observations to SB3 policies, avoiding vectorization ambiguity.

---

## Repository Structure

```
MaskBench/
├─ scripts/
│  ├─ eval_visualize.py           # Single model sweep evaluation with animation
│  └─ eval_all.py                 # Full sweep evaluation across all models in config
├─ src/
│  ├─ common/
│  │  ├─ env_utils.py             # Env creation, version aliasing, VecNormalize loading
│  │  ├─ model_utils.py           # SB3 model loading helpers
│  │  └─ plot_utils.py            # Plotting utilities for sweep results
│  └─ masking/
│     ├─ channel_mask.py          # ChannelMask implementation
│     ├─ random_mask.py           # RandomMask implementation
│     └─ utils.py                 # mask_table helper for display
└─ checkpoints/
   └─ mujoco/<ENV>/<ALGO>/        # Expected layout for saved models (+ vec_normalize.pkl)
```

---

## Installation

> Tested with Python 3.11, Gymnasium, Stable‑Baselines3, MuJoCo.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e .
pip install -r requirements.txt  # provide SB3, gymnasium[mujoco], numpy, opencv-python, matplotlib, pillow
```

If you have legacy **Gym** checkpoints, Gymnasium will still load and warn. We alias v3/v4→v5 where needed.

---

## Quickstart

### Running a Full Sweep Evaluation (`eval_all.py`)

To evaluate all models defined in `configs/mujoco_checkpoints.yaml` across the full range of mask probabilities (0.0, 0.2, ..., 1.0) and generate plots:

```bash
PYTHONPATH="/Users/sohamsane/Documents/Coding Projects/MaskBench" "/Users/sohamsane/Documents/Coding Projects/MaskBench/.venv/bin/python" scripts/eval_all.py \
  --episodes 1 \
  --max-steps 1000 \
  --ckpt_root checkpoints/mujoco \
  --out-dir outputs
```

Plots will be saved in `outputs/<ENV_ID>/<ALGO>-<MASK_TYPE>.png`.

### Running a Smoke Test (`eval_all.py --smoke`)

To quickly verify that all models can be loaded and run without errors (runs for 10 steps with `mask_prob=0.0`):

```bash
PYTHONPATH="/Users/sohamsane/Documents/Coding Projects/MaskBench" "/Users/sohamsane/Documents/Coding Projects/MaskBench/.venv/bin/python" scripts/eval_all.py --smoke
```

### Running a Single Model Sweep with Animation (`eval_visualize.py`)

To run a sweep for a single model, generate a plot, and also produce an animated video of the baseline run:

```bash
PYTHONPATH="/Users/sohamsane/Documents/Coding Projects/MaskBench" "/Users/sohamsane/Documents/Coding Projects/MaskBench/.venv/bin/python" scripts/eval_visualize.py \
  --env-id Ant-v3 \
  --algo ppo \
  --mask-type channel \
  --episodes 1 \
  --max-steps 1000 \
  --out outputs/Ant-v3/ppo-channel-sweep.png \
  --video-out outputs/videos/ant_ppo_channel_baseline.mp4
```

**Checkpoint layout** (default search path):

```
checkpoints/mujoco/<ENV>/<ALGO>/
  ├─ model.zip              # SB3 policy
  └─ vec_normalize.pkl      # (optional) normalization stats
```

If `vec_normalize.pkl` is absent in `<ALGO>/`, MaskBench also checks the parent folder.

---

## Concepts & Mathematics

### Observation masking model

Let the (row) observation at step (t) be (\mathbf{x}_t \in \mathbb{R}^D). A binary mask (\mathbf{m}_t \in {0,1}^D) is sampled, and the masked observation is
[
\tilde{\mathbf{x}}_t = \mathbf{m}_t \odot \mathbf{x}_t,\quad \text{where } \odot \text{ is Hadamard product.}
]
MaskBench draws a Bernoulli switch per step:
[
U_t \sim \mathrm{Bernoulli}(p), \quad p \in [0,1],
]
where (p) is the **masking probability**. If (U_t=0), we set (\mathbf{m}_t=\mathbf{1}) and leave (\mathbf{x}_t) unchanged. If (U_t=1), we construct (\mathbf{m}_t) according to the selected masker with **severity** parameter (r\in[0,1]) (the *drop ratio*).

We maintain shape and dtype invariants:

* Input/Output shape: `(D,)` or `(N,D)` in batched form.
* Dtype: `float32` throughout the SB3 path.

### ChannelMask

**Intuition:** entire *features/channels* go down at once (e.g., a sensor bus drops specific indices).

Construction when (U_t=1):

* Sample a subset (S_t \subseteq [D] = {1,\dots,D}) with (|S_t| = \lfloor r D \rfloor), uniformly without replacement.
* Define
  [
  (\mathbf{m}_t)_i = \begin{cases}
  0, & i\in S_t, \
  1, & i\notin S_t.
  \end{cases}
  ]
* Masked obs: (\tilde{\mathbf{x}}_t = \mathbf{x}_t \odot \mathbf{m}_t).

**Expected zeros per step:** (\mathbb{E}[Z_t] = p,r,D).

**Correlation structure:** entries masked **within a step** are *perfectly correlated* (entire chosen channels are zeroed together). Across steps, selections are i.i.d. unless a schedule is used.

### RandomMask

**Intuition:** each entry can flicker independently, modeling fine‑grained packet loss or per‑feature dropouts.

Construction when (U_t=1):

* For each feature (i\in [D]), draw (B_{t,i} \sim \mathrm{Bernoulli}(q)) with (q=r).
* Set ((\mathbf{m}_t)*i = 1 - B*{t,i}) and compute (\tilde{\mathbf{x}}_t = \mathbf{x}_t \odot \mathbf{m}_t).

**Expected zeros per step:** (\mathbb{E}[Z_t] = p,q,D = p,r,D).

**Correlation structure:** entries are **independent** given (U_t=1), so masking is spatially uncorrelated (unlike ChannelMask).

### Effective masking rate and variance

Over many steps, the expected fraction of zeroed entries is
[
\rho_{\mathrm{eff}} \approx p, r.
]
Variance differs:

* **ChannelMask (without replacement)**, for (U_t=1): (Z_t = |S_t|) is (approximately) deterministic at (rD) (ignoring the floor), so per‑step variance arises only from (U_t).
* **RandomMask** adds binomial variance given (U_t=1): (Z_t,|,U_t=1 \sim \mathrm{Binomial}(D, r)), so
  [
  \mathrm{Var}(Z_t) = p, D r(1-r) + p(1-p) (rD)^2.
  ]
  This higher variance can stress policies differently than channel‑level masking.

### Mask scheduling

Instead of a fixed (p), you can run schedules, e.g. linear warm‑up:
[
p(t) = \min!\left(p_{\max},; p_0 + \alpha t\right),
]
or cosine schedules. Severity (r) can be scheduled similarly. Scheduling is useful when gradually stress‑testing or training with curriculum.

### POMDP view & intuition

Masking transforms an MDP into a **POMDP**: the agent receives (\tilde{\mathbf{x}}_t) instead of the full (\mathbf{x}_t). ChannelMask models *structured* partial observability (entire sensors go missing); RandomMask models *unstructured* noise‑like missingness.

* **Why this matters:** robust policies should either (a) carry stronger belief state (RNNs/filters), or (b) learn redundancies and invariances across features. Evaluation under controlled (p,r) makes these gaps visible.

---

## Implementation Details

### Runtime flow

For each evaluation run (either a single sweep or part of a full sweep):

1.  **Environment Setup**: Gymnasium MuJoCo v5 (or legacy alias) is created.
2.  **Observation Space Harmonization**: The environment's observation space is forced to `float32`.
3.  **VecNormalize Handling**: If present, `VecNormalize` stats are loaded. If shape/dtype mismatches occur (e.g., `Ant-v3` checkpoint on `Ant-v5`), stats are reinitialized and the wrapper's observation space is aligned.
4.  **Model Loading**: The SB3 model is loaded. To handle strict observation/action space checks during loading (especially for `TRPO` models), the environment's spaces are temporarily adjusted to match the model's original training spaces, then reverted after loading.
5.  **Evaluation Loop**: The agent is run for `max_steps`. If an episode terminates early, the environment is reset, and the evaluation continues until `max_steps` are reached.
6.  **Masking**: `Masker.maybe_apply(x_t)` is applied to observations before feeding them to the model.
7.  **Observation Padding**: Observations are padded to match the input dimension expected by the loaded model's policy network.
8.  **Action Prediction**: `SB3 Model.predict(x̃_t)` is called to get actions.
9.  **Environment Step**: `Env.step(a_t)` is executed.

*   **Batching**: we always send `(N,D)` `float32` observations into SB3 (`N=1`).
*   **Display**: `mask_table()` renders a compact per‑feature table comparing `orig` vs `masked` values (used in `eval_visualize.py` animation).

### SB3 + Gymnasium compatibility

*   **Gym→Gymnasium warnings** are expected when loading Gym‑trained models; evaluation still works.
*   We **alias** legacy MuJoCo IDs (`-v2/-v3/-v4`) to `-v5` in `env_utils.py` to keep CLIs compatible.
*   We cast observation spaces to **float32** consistently to avoid SB3’s vectorization ambiguity.

### Legacy checkpoint adapter

Real‑world frictions we address:

*   **VecNormalize shape/dtype mismatch** (e.g., Ant v3’s 112‑dim obs vs v5’s 105‑dim): we unwrap `VecNormalize`, compare saved stats shape/dtype to the **underlying env’s** current obs shape/dtype, and **reinitialize** stats if mismatched.
*   **Policy input width mismatch**: some legacy models hard‑code the input width. We infer the model’s expected flattened dim and **pad** incoming observations so `model.predict()` remains well‑posed. (For scientific apples‑to‑apples, prefer matching env versions at training & eval.)

---

## CLI Usage

### `scripts/eval_all.py`

This script automates the evaluation of all models defined in `configs/mujoco_checkpoints.yaml` across a range of mask probabilities.

```bash
PYTHONPATH="/Users/sohamsane/Documents/Coding Projects/MaskBench" "/Users/sohamsane/Documents/Coding Projects/MaskBench/.venv/bin/python" scripts/eval_all.py [OPTIONS]
```

**Options:**

*   `--config-file PATH`: Path to the config file (default: `configs/mujoco_checkpoints.yaml`).
*   `--episodes INT`: Number of episodes to run for each sweep (default: 1).
*   `--max-steps INT`: Maximum number of steps per episode (default: 1000).
*   `--ckpt_root PATH`: Root directory for model checkpoints (default: `checkpoints/mujoco`).
*   `--out-dir PATH`: Output directory for generated plots (default: `outputs`).
*   `--smoke`: Run a quick smoke test (10 steps, `mask_prob=0.0`) to verify setup.

**Example: Full Evaluation**

```bash
PYTHONPATH="/Users/sohamsane/Documents/Coding Projects/MaskBench" "/Users/sohamsane/Documents/Coding Projects/MaskBench/.venv/bin/python" scripts/eval_all.py \
  --episodes 1 \
  --max-steps 1000 \
  --ckpt_root checkpoints/mujoco \
  --out-dir outputs
```

**Example: Smoke Test**

```bash
PYTHONPATH="/Users/sohamsane/Documents/Coding Projects/MaskBench" "/Users/sohamsane/Documents/Coding Projects/MaskBench/.venv/bin/python" scripts/eval_all.py --smoke
```

### `scripts/eval_visualize.py`

This script runs a sweep for a single model, generates a plot, and produces an animated video of the baseline run.

```bash
PYTHONPATH="/Users/sohamsane/Documents/Coding Projects/MaskBench" "/Users/sohamsane/Documents/Coding Projects/MaskBench/.venv/bin/python" scripts/eval_visualize.py [OPTIONS]
```

**Options:**

*   `--env-id ENV_ID`: Environment ID (e.g., `Ant-v3`).
*   `--algo ALGO`: Algorithm folder name (e.g., `ppo`, `sac`).
*   `--mask-type {channel,randomized}`: Type of masking to apply (default: `channel`).
*   `--episodes INT`: Number of episodes to run for each sweep (default: 1).
*   `--max-steps INT`: Maximum number of steps per episode (default: 1000).
*   `--ckpt_root PATH`: Root directory for model checkpoints (default: `checkpoints/mujoco`).
*   `--out PATH`: Output path for the generated sweep plot (default: `outputs/sweep.png`).
*   `--video-out PATH`: Output path for the animated video (default: `outputs/videos/preview.mp4`).

**Example:**

```bash
PYTHONPATH="/Users/sohamsane/Documents/Coding Projects/MaskBench" "/Users/sohamsane/Documents/Coding Projects/MaskBench/.venv/bin/python" scripts/eval_visualize.py \
  --env-id Ant-v3 \
  --algo ppo \
  --mask-type channel \
  --episodes 1 \
  --max-steps 1000 \
  --out outputs/Ant-v3/ppo-channel-sweep.png \
  --video-out outputs/videos/ant_ppo_channel_baseline.mp4
```

### `scripts/make_plot_sheet.py`

This script combines all generated plots for a given environment into a single large image.

```bash
PYTHONPATH="/Users/sohamsane/Documents/Coding Projects/MaskBench" "/Users/sohamsane/Documents/Coding Projects/MaskBench/.venv/bin/python" scripts/make_plot_sheet.py [OPTIONS]
```

**Options:**

*   `--env-id ENV_ID`: Environment ID (e.g., `Ant-v3`).
*   `--output-file PATH`: Output path for the combined plot sheet (default: `outputs/plot_sheet.png`).

**Example:**

```bash
PYTHONPATH="/Users/sohamsane/Documents/Coding Projects/MaskBench" "/Users/sohamsane/Documents/Coding Projects/MaskBench/.venv/bin/python" scripts/make_plot_sheet.py \
  --env-id Ant-v3 \
  --output-file outputs/Ant-v3_all_plots.png
```

**Checkpoint layout** (default search path):

```
checkpoints/mujoco/<ENV>/<ALGO>/
  ├─ model.zip              # SB3 policy
  └─ vec_normalize.pkl      # (optional) normalization stats
```

If `vec_normalize.pkl` is absent in `<ALGO>/`, MaskBench also checks the parent folder.

---

## Concepts & Mathematics

### Observation masking model

Let the (row) observation at step (t) be (\mathbf{x}_t \in \mathbb{R}^D). A binary mask (\mathbf{m}_t \in {0,1}^D) is sampled, and the masked observation is
[
\tilde{\mathbf{x}}_t = \mathbf{m}_t \odot \mathbf{x}_t,\quad \text{where } \odot \text{ is Hadamard product.}
]
MaskBench draws a Bernoulli switch per step:
[
U_t \sim \mathrm{Bernoulli}(p), \quad p \in [0,1],
]
where (p) is the **masking probability**. If (U_t=0), we set (\mathbf{m}_t=\mathbf{1}) and leave (\mathbf{x}_t) unchanged. If (U_t=1), we construct (\mathbf{m}_t) according to the selected masker with **severity** parameter (r\in[0,1]) (the *drop ratio*).

We maintain shape and dtype invariants:

* Input/Output shape: `(D,)` or `(N,D)` in batched form.
* Dtype: `float32` throughout the SB3 path.

### ChannelMask

**Intuition:** entire *features/channels* go down at once (e.g., a sensor bus drops specific indices).

Construction when (U_t=1):

* Sample a subset (S_t \subseteq [D] = {1,\dots,D}) with (|S_t| = \lfloor r D \rfloor), uniformly without replacement.
* Define
  [
  (\mathbf{m}_t)_i = \begin{cases}
  0, & i\in S_t, \
  1, & i\notin S_t.
  \end{cases}
  ]
* Masked obs: (\tilde{\mathbf{x}}_t = \mathbf{x}_t \odot \mathbf{m}_t).

**Expected zeros per step:** (\mathbb{E}[Z_t] = p,r,D).

**Correlation structure:** entries masked **within a step** are *perfectly correlated* (entire chosen channels are zeroed together). Across steps, selections are i.i.d. unless a schedule is used.

### RandomMask

**Intuition:** each entry can flicker independently, modeling fine‑grained packet loss or per‑feature dropouts.

Construction when (U_t=1):

* For each feature (i\in [D]), draw (B_{t,i} \sim \mathrm{Bernoulli}(q)) with (q=r).
* Set ((\mathbf{m}_t)*i = 1 - B*{t,i}) and compute (\tilde{\mathbf{x}}_t = \mathbf{x}_t \odot \mathbf{m}_t).

**Expected zeros per step:** (\mathbb{E}[Z_t] = p,q,D = p,r,D).

**Correlation structure:** entries are **independent** given (U_t=1), so masking is spatially uncorrelated (unlike ChannelMask).

### Effective masking rate and variance

Over many steps, the expected fraction of zeroed entries is
[
\rho_{\mathrm{eff}} \approx p, r.
]
Variance differs:

* **ChannelMask (without replacement)**, for (U_t=1): (Z_t = |S_t|) is (approximately) deterministic at (rD) (ignoring the floor), so per‑step variance arises only from (U_t).
* **RandomMask** adds binomial variance given (U_t=1): (Z_t,|,U_t=1 \sim \mathrm{Binomial}(D, r)), so
  [
  \mathrm{Var}(Z_t) = p, D r(1-r) + p(1-p) (rD)^2.
  ]
  This higher variance can stress policies differently than channel‑level masking.

### Mask scheduling

Instead of a fixed (p), you can run schedules, e.g. linear warm‑up:
[
p(t) = \min!\left(p_{\max},; p_0 + \alpha t\right),
]
or cosine schedules. Severity (r) can be scheduled similarly. Scheduling is useful when gradually stress‑testing or training with curriculum.

### POMDP view & intuition

Masking transforms an MDP into a **POMDP**: the agent receives (\tilde{\mathbf{x}}_t) instead of the full (\mathbf{x}_t). ChannelMask models *structured* partial observability (entire sensors go missing); RandomMask models *unstructured* noise‑like missingness.

* **Why this matters:** robust policies should either (a) carry stronger belief state (RNNs/filters), or (b) learn redundancies and invariances across features. Evaluation under controlled (p,r) makes these gaps visible.

---

## Troubleshooting

### Gym vs Gymnasium warnings

*   Expected for old checkpoints. Prefer re‑saving models and stats under Gymnasium over time.

### My Ant‑v3 policy still won’t run on v5

*   Use the built‑in **adapter** that pads/slices to the model’s input width (already wired). For rigorous comparisons, train & eval on the **same** env version.

### ARS Algorithm Compatibility

*   Currently, `ARS` policies are skipped in `eval_all.py` due to compatibility issues with `model.predict()`.

---

## Extending MaskBench

Create a new masker by implementing `maybe_apply(obs: np.ndarray) -> np.ndarray`:

```python
class MyMask:
    def __init__(self, p: float, drop_ratio: float):
        self.p = p
        self.r = drop_ratio
    def maybe_apply(self, obs: np.ndarray) -> np.ndarray:
        if np.random.rand() > self.p:
            return obs
        x = obs.astype(np.float32, copy=True)
        # construct m_t in {0,1}^D here …
        return x * m  # same shape, float32
```

Ideas:

*   **BlockMask**: contiguous spans of features (structured missingness between Channel and Random).
*   **TemporalHold**: if a feature gets dropped, hold its last value for (K) steps (models delayed updates).
*   **GaussianNoise**: multiplicative or additive noise instead of hard zeros.
*   **Image‑domain masks** (for vision obs): random erasing, cutout, occlusion patches.

---

## Evaluation Protocols & Suggested Experiments

*   **Grid over (p, r)**: (p\in{0,0.25,0.5,0.75,1.0}), (r\in{0,0.1,0.3,0.5}). Report average return, time‑to‑failure, and instability (std dev across seeds).
*   **Masker ablation**: Channel vs Random at same (p,r) to isolate the effect of correlation in missingness.
*   **Curriculum schedule**: Start with small (p,r), ramp up each episode.
*   **Action divergence**: Compare (\pi(\mathbf{x}_t)) vs (\pi(\tilde{\mathbf{x}}_t)) to quantify sensitivity.
*   **Transfer**: train with masking (or DR) and test without; or vice‑versa.

---

## Roadmap

*   Integrate **CleanRL** policies and Atari/classic‑control benchmarks.
*   Add **Dict/visual observations** support and pixel‑domain maskers.
*   Ship a **metrics module** (action divergence, instability, recovery time).
*   Optional **RNN policy** wrappers to assess memory augmentation under POMDP masking).