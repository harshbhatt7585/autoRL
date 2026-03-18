# autoRL

This repository is for autonomous search over RL environments.

## Setup

To set up a new run, work with the user to:

1. **Agree on a run tag**: propose a short tag based on today's date, for example `mar10`. The branch `autorl/<tag>` should not already exist (`git branch --list 'autorl/<tag>'` should return nothing). This is a fresh run.
2. **Switch to the main line**: move to the current main line branch before branching (for example `git checkout main`).
3. **Create and switch to the run branch**: run `git checkout -b autorl/<tag>`.
4. **Verify active branch before editing**: confirm `git branch --show-current` is exactly `autorl/<tag>`. Do not edit code on any other branch.
5. **Read the in-scope files**: the repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `framework.py` — fixed evaluator, fixed score, hard budget caps. Do not modify.
   - `train.py` — fixed entrypoint. Do not modify.
   - `candidate/env.py` — the candidate environment file you modify.
   - `candidate/train.py` — the env-specific policy and PPO tuning file you modify.
6. **Verify the runtime exists**: confirm that `.venv` exists and that the vendored Simverse package is installed. If not, tell the human to run:
   - `uv venv`
   - `uv pip install -e vendor/simverse`
7. **Initialize `results.tsv`**: create it with just the header row if needed. The baseline will be recorded after the first run.
8. **Confirm and go**: once setup looks good, kick off the experimentation.

## Build brief

The checked-in `candidate/env.py` is intentionally empty. The real task is to
build and train a **combat mission simulation** with **multiple cooperating
and opposing agents**. This is the only in-scope environment family for this
run. Do not switch to trading, household chores, or unrelated tasks.

## Episode horizon

The fixed evaluator now reads the task horizon from `candidate/train.py`
through `training_overrides()["max_steps"]`.

- The environment owns its semantic episode length.
- The outer harness still owns `--train-episodes` and `--eval-episodes`.
- Do not treat `max_steps` as a free optimization knob. It must faithfully match
  the chosen task brief.
- Do not compress the combat mission into a tiny proxy just to make the score easier.

## Experimentation

The fixed score is:

- `score = mean_eval_return`

That is the average post-training episode return across all greedy evaluation
episodes and all seeds. Other reported metrics are diagnostics only.

Each experiment runs with an **agent-chosen episode budget**. You launch it as:

```bash
uv run python train.py --train-episodes <n> --eval-episodes <m> --device cuda --wandb
```

The evaluator has hard caps and fixed runtime settings:

- up to `1000` PPO training episodes
- up to `100` greedy evaluation episodes
- `2` random seeds
- `32` parallel environments per seed
- CUDA is available on an NVIDIA `T4` (`16GB` VRAM); prefer `--device cuda` for training runs.
- Always enable W&B logging (`--wandb`) so runs are fully recorded.

**Budget policy**:

- Start small. Use low episode counts for early exploration.
- Raise the budget only when the current accepted candidate is clearly learning and you need more signal.
- Never exceed the hard caps.
- Do not vary `seed_count` or `num_envs`. The only budget knobs are `--train-episodes` and `--eval-episodes`.
- `max_steps` belongs to the candidate hyperparameter constants in `candidate/train.py`, not the outer experiment loop.
- Compare runs only against other runs at the same budget.
- When you raise the budget, first re-run the current accepted candidate at the new budget and record that as the new baseline before testing new ideas.

Useful example ladder:

- `12/8`
- `32/8`
- `64/12`
- `128/16`
- `256/24`
- `512/48`
- `1000/100`

**What you CAN do:**

- Modify `candidate/env.py`.
- Modify `candidate/train.py`.
- Choose `--train-episodes` and `--eval-episodes` for each run, as long as they stay within the hard caps.
- Change the task layout, reward structure, transition dynamics, observation encoding, action semantics, and episode logic.
- Change the env-specific policy architecture and PPO hyperparameters used for that candidate environment.
- Simplify or redesign the environment completely, as long as it still satisfies the interface expected by the fixed evaluator.

**What you CANNOT do:**

- Modify `framework.py`.
- Modify the root `train.py`.
- Modify `vendor/simverse`.
- Install new packages or add dependencies.
- Exceed the hard budget caps.
- Change `seed_count` or `num_envs`.
- Use a fake shortened `max_steps` that violates the chosen task brief.
- Change the score definition.
- Inflate reward magnitudes or tweak reward scaling only to make returns look larger without real behavior improvement.
- Game evaluation by parameter hacks that change reported rewards instead of improving mission strategy, robustness, or success rate.

**The goal is simple: get the highest score.** The budget is a search control knob, not a loophole. Spend as little budget as possible while ideas are weak, and only ratchet it upward when the accepted candidate has earned it. The only valid comparison is against the current accepted baseline at the same budget.

**Multi-agent mission constraint**: keep the search focused on a combat mission simulation with multiple agents, while still satisfying the fixed evaluator interfaces in this repository.

**Simplicity criterion**: all else equal, simpler is better. A tiny score improvement that adds ugly complexity is not worth it. Conversely, deleting mechanics and getting the same or better score is a strong win. When deciding whether to keep a change, weigh the complexity cost against the score gain. Small hacky gains are weak. Cleaner env logic with equal performance is strong.

**The first run**: your very first run should always be the baseline, so you run the current `candidate/env.py` and `candidate/train.py` exactly as they are. Right now that baseline is just an empty starter canvas. After recording it, start building the combat mission environment.

## Combat mission environment brief

Build a tactical multi-agent combat simulator where allies coordinate under
partial information against adversaries. The mission should reward tactical
movement, target selection, survival, and objective completion.

Core episode structure:

- Multi-agent setting with at least one friendly team and one opposing team.
- Discrete action space with movement and combat-relevant actions.
- One episode should represent a full mission window with clear termination logic
  (objective complete, team eliminated, or horizon reached).
- Maintain deterministic seeding for reproducibility.

Mission design:

- Include a compact arena or grid with obstacles/cover and mission objectives.
- Support meaningful coordination pressure (crossfire, flanking, escort, capture,
  defend, or extraction).
- Keep transitions and reward shaping stable enough for PPO to learn.

Required state:

- Team health / alive flags.
- Positions and mission-objective state.
- Ammo/cooldown/status features if used.
- Remaining mission time.

Observation design:

- The fixed evaluator still expects `C,H,W`.
- Encode multi-agent tactical state into tensor channels (ally map, enemy map,
  obstacles, objective, time, and status channels).
- Keep channel semantics explicit and stationary.

Reward structure:

- Dense shaping for tactical progress (objective advancement, effective damage,
  survival, formation/positioning).
- Penalties for friendly losses, invalid actions, or reckless behavior.
- Terminal mission reward for win/lose outcomes and objective completion quality.
- Keep rewards finite and approximately bounded for stable PPO training.
- Reward tuning must preserve meaning; do not simply increase coefficients to artificially boost score.

Success signal:

- `info["success"]` must be `True` only for successful mission completion.
- Partial tactical progress may shape reward but should not count as success.

What good behavior should look like:

- Allies use cover, avoid unnecessary exposure, and coordinate engagements.
- Policy prioritizes objectives over random firefights.
- Team survival and mission completion improve together over training.

## Rendering requirements (Simverse-style)

Follow the Simverse renderer contract from `vendor/simverse/src/simverse/envs/README.md`:

- Do not build a standalone local Python render entrypoint for the candidate env.
- Visualization must be browser-based via the shared `renderer/` workflow.
- Write replay JSON per episode so the shared renderer backend/frontend can load it.

Replay API pattern to align with:

- `GET /<env>/snapshot`
- `GET /<env>/replays`
- `GET /<env>/replays/{replay_id}`

When implementing replay output, keep schema stable across episodes and include
enough frame metadata to drive grid playback (positions, health/status, actions,
objective state, and terminal outcome).


## How to create ENV

The candidate environment must obey these constraints:

- `create_env(config, num_envs, device, dtype)` returns a `SimEnv`.
- The environment supports multiple agents for combat missions.
- The environment exposes a discrete `action_space`.
- The environment exposes a 3D `observation_space.shape` in `C, H, W`.
- `reset()` returns a dict that contains an `obs` tensor.
- `step(action)` returns `(obs_dict, reward, done, info)` compatible with the Simverse PPO trainer.
- `info["success"]` must be an env-level boolean signal so solve rate is meaningful.
- Rewards should stay finite and roughly within `[-1.0, 1.0]`.
- Episodes must terminate within `config.max_steps`, and that value must match the intended task horizon declared through `training_overrides()["max_steps"]`.

The candidate training file must obey these constraints:

- It may define the policy architecture for this environment.
- It may tune optimizer and PPO hyperparameters for this environment.
- It may define task-level hyperparameter constants such as `max_steps` through `training_overrides()`.
- It must enable W&B logging for training records (`--wandb` run convention).
- It must not set the episode budget itself. The outer experiment loop controls `--train-episodes` and `--eval-episodes`.
- It must not redefine the score.

## Output format

When the script finishes it prints a summary like this:

```text
---
score: 1.865000
mean_eval_return: 1.865000
mean_solve_rate: 1.000000
mean_train_return: 4.908301
learning_gain: 4.067227
headroom_bonus: 1.000000
stability: 1.000000
complexity_penalty: 0.400000
```

The training log is noisy, so extract the key metrics from the log file:

```bash
grep "^score:\|^mean_solve_rate:\|^mean_eval_return:\|^train_episodes:\|^eval_episodes:\|^max_steps:" run.log
```

## Logging results

`train.py` now auto-appends one row to `results.tsv` for each completed run.
Successful runs default to `status=pending`. If you pass `--status keep` or
`--status discard`, that label is written instead. If evaluation crashes after
argument parsing, `train.py` appends a `crash` row with zero metrics.

You may also pass `--description "<idea>"` so the row records what the
experiment tried. Do not use commas in the TSV itself.

The TSV has a header row and 9 columns:

```text
commit	train_episodes	eval_episodes	max_steps	score	solve_rate	eval_return	status	description
```

1. git commit hash, short form if git exists, otherwise `nogit`
2. train episodes used, for example `128`
3. eval episodes used, for example `16`
4. task horizon from `max_steps`, for example `100`
5. score achieved, for example `1.865000` — use `0.000000` for crashes
6. solve rate achieved, for example `1.000000` — use `0.000000` for crashes
7. eval return achieved, for example `1.865000` — use `0.000000` for crashes
8. status: `pending`, `keep`, `discard`, or `crash`
9. short text description of what the experiment tried

Do not commit `results.tsv`. Leave it untracked.

## The experiment loop

The experiment runs on a dedicated branch, for example `autorl/mar10`.

LOOP FOREVER:

1. Look at the git state: current branch, current accepted commit, and current accepted budget.
2. Start with a small budget if this is a fresh run. If the accepted candidate is already strong at the current budget, ratchet upward and re-baseline the accepted commit first.
3. Tune `candidate/env.py`, `candidate/train.py`, or both with one experimental idea.
4. Commit the experiment if git is available.
5. Run the experiment: `uv run python train.py --train-episodes <n> --eval-episodes <m> --device cuda --wandb --description "<idea>" > run.log 2>&1`
6. Read out the results: `grep "^score:\|^mean_solve_rate:\|^mean_eval_return:\|^train_episodes:\|^eval_episodes:\|^max_steps:" run.log`
7. If the grep output is empty, the run crashed. Read `tail -n 50 run.log`, decide whether the bug is easy to fix, and either retry or mark the idea as a crash.
8. Confirm the row was written to `results.tsv`, then update its status to `keep` or `discard` if needed.
9. If score improved at the same budget, keep the change and advance from the new accepted commit.
10. If score is equal or worse, restore the repo to the last accepted state and move on.

The idea is simple: you are an autonomous environment researcher. If an idea works, keep it. If it does not, discard it. The accepted state moves forward only when the score improves, and the accepted budget only moves upward.

**Timeout**: each experiment should finish quickly. If a run hangs or takes far longer than normal for this machine, kill it and treat it as a failure.

**Crashes**: if a run crashes because of something trivial, fix it and re-run. If the idea itself is broken, log `crash`, revert, and move on.

**NEVER STOP**: once the loop has begun, do not pause to ask whether you should continue. Do not ask for permission after every experiment. Continue until the human interrupts you. If you run out of ideas, think harder: simplify the task, change the topology, change the observation encoding, remove unnecessary mechanics, combine previous near-misses, or try a more radical task redesign.
