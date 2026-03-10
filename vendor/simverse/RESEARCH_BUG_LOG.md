# ShapeDraw Research Bug Log

Last updated: 2026-02-11

## 1) Value Loss Oscillation Around ~0.5 (Critical)

- Status: Fixed
- Symptom:
  - `loss/value` in W&B oscillated tightly around `~0.5` with frequent up/down spikes.
- Root cause:
  - In PPO tensor-fastpath training, `returns` for the critic were built from **normalized** advantages.
  - This forced the critic target distribution toward unit variance, making value loss hover around `0.5 * MSE(~1)`.
- Code location:
  - `simverse/src/simverse/losses/ppo.py`
- Fix applied:
  - Compute `returns` from **raw** advantages:
    - `returns = advantages + sampled_values`
  - Keep normalization only for policy optimization:
    - `policy_advantages = normalize(advantages)`
    - PPO surrogate uses `policy_advantages`, critic uses `returns`.
- Why this worked:
  - Critic target is now on the correct scale of environment returns instead of normalized policy-training scale.

## 2) ShapeDraw Drawing Step Bottleneck (High)

- Status: Fixed
- Symptom:
  - Throughput was heavily capped when pen drawing was active.
- Root cause:
  - `_draw_disks` used a Python loop over active envs and built `arange/meshgrid` per env.
- Code location:
  - `simverse/src/simverse/envs/shape_draw/torch_env.py`
- Fix applied:
  - Replaced per-env loop with a batched vectorized disk mask and single indexed assignment for all active envs.
- Why this worked:
  - Removed Python-loop overhead and per-env tensor construction in the hot path.

## 3) Missing Average Episode Reward in W&B (Medium)

- Status: Fixed
- Symptom:
  - W&B logged only latest episode reward; no running average metric.
- Root cause:
  - `TrainingStats.log_wandb()` did not emit average episode reward.
- Code location:
  - `simverse/src/simverse/agent/stats.py`
  - `simverse/src/simverse/losses/ppo.py`
- Fix applied:
  - Added `episode/reward_avg` to W&B payload.
  - Added episode-end W&B logging right after `push_reward(...)`.
- Why this worked:
  - Average now updates once per episode and is visible as a stable progress signal.

## 4) Silent CPU Fallback During "GPU" Training (High)

- Status: Fixed in training entrypoint behavior
- Symptom:
  - Training could run on CPU when CUDA init failed, causing major slowdown.
- Root cause:
  - Device selection allowed fallback to CPU instead of hard-failing for ShapeDraw GPU runs.
- Code location:
  - `simverse/src/simverse/envs/shape_draw/train.py`
  - `simverse/src/simverse/envs/shape_draw/training_config.py`
- Fix applied:
  - Enforced GPU-only mode in ShapeDraw train path:
    - fail fast if CUDA unavailable
    - explicit CUDA + fp16 config
    - enabled GPU-friendly settings (`TF32`, cuDNN benchmark, optional `torch.compile`, fused Adam when available)
- Why this worked:
  - Prevents accidental slow CPU runs and keeps config aligned with GPU intent.

