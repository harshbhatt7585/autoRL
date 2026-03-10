# Simverse Env Package Contract

An environment package is considered complete when it includes:

- `env.py`
- `train.py`

Visualization should happen through the shared browser-based `renderer/` app. Environment packages
should not include local Python render entrypoints.

## Recommended Pattern

1. `env.py` holds core torch environment logic and inherits `SimEnv`.
2. `env.py` exposes:
   - a canonical `*Env` alias for the torch implementation
   - `create_env(...)` factory
3. `train.py` uses shared helpers from `simverse.training.utils`:
   - `resolve_torch_device`
   - `resolve_rollout_dtype`
   - `configure_torch_backend`
   - `compile_policy_models`
   - `build_adam_optimizers`
   - `build_ppo_training_config`
4. If the environment needs visualization, write replay JSON that the shared `renderer/` app can
   load and render in the browser.

Use `simverse.envs.scaffold.missing_required_files` to validate package completeness.
