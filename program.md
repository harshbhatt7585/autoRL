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
   - `python3 -m venv .venv`
   - `.venv/bin/pip install -e vendor/simverse`
7. **Initialize `results.tsv`**: create it with just the header row if needed. The baseline will be recorded after the first run.
8. **Confirm and go**: once setup looks good, kick off the experimentation.

## Build brief

The checked-in `candidate/env.py` is intentionally empty. The real task is to
replace it with exactly one environment family from the approved list below.
Do not invent a fourth family. Do not try to build both at once. Pick one,
implement it cleanly, and optimize it.

The two families below are the human-authored target tasks. Treat them as the
source of truth for what the next real environment should be.

## Episode horizon

The fixed evaluator now reads the task horizon from `candidate/train.py`
through `training_overrides()["max_steps"]`.

- The environment owns its semantic episode length.
- The outer harness still owns `--train-episodes` and `--eval-episodes`.
- Do not treat `max_steps` as a free optimization knob. It must faithfully match
  the chosen task brief.
- Do not compress a `100`-hour trading task or a `1000`-step household task
  down to a tiny proxy just to make the score easier.

## Experimentation

The fixed score is:

- `score = mean_eval_return`

That is the average post-training episode return across all greedy evaluation
episodes and all seeds. Other reported metrics are diagnostics only.

Each experiment runs with an **agent-chosen episode budget**. You launch it as:

```bash
.venv/bin/python train.py --train-episodes <n> --eval-episodes <m>
```

The evaluator has hard caps and fixed runtime settings:

- up to `1000` PPO training episodes
- up to `100` greedy evaluation episodes
- `2` random seeds
- `32` parallel environments per seed
- `cpu` by default

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

**The goal is simple: get the highest score.** The budget is a search control knob, not a loophole. Spend as little budget as possible while ideas are weak, and only ratchet it upward when the accepted candidate has earned it. The only valid comparison is against the current accepted baseline at the same budget.

**Single-agent constraint**: keep the search focused. The current evaluator expects a single-agent `SimEnv` with a discrete action space and a `C,H,W` observation tensor. Do not drift into multi-agent or continuous-control work unless the fixed evaluator is explicitly changed by the human.

**Simplicity criterion**: all else equal, simpler is better. A tiny score improvement that adds ugly complexity is not worth it. Conversely, deleting mechanics and getting the same or better score is a strong win. When deciding whether to keep a change, weigh the complexity cost against the score gain. Small hacky gains are weak. Cleaner env logic with equal performance is strong.

**The first run**: your very first run should always be the baseline, so you run the current `candidate/env.py` and `candidate/train.py` exactly as they are. Right now that baseline is just an empty starter canvas. After recording it, start building one of the approved environments below.

## Approved environment families

You may build exactly one of these two families.

### 1. Trading env

Build a stock-trading simulator over `5` imaginary companies. Each company
should have a distinct pattern family so the policy can learn recognizable
price behavior instead of pure noise. Good examples are:

- momentum / trend-up
- mean-reverting
- volatile boom-bust
- slow cyclical
- noisy flat / deceptive

Core episode structure:

- Each episode samples one company.
- The agent starts with `$500` cash and zero holdings.
- One episode represents `100` hours of trading.
- One environment step equals `1` trading hour.
- The action space is exactly `3` actions: `buy`, `sell`, `rest`.
- `buy` should purchase one unit if enough cash is available.
- `sell` should liquidate one unit if inventory is available.
- `rest` leaves the portfolio unchanged.

Dataset design:

- Do not pull real market data or external files.
- Generate a small synthetic dataset of `5` fictional companies directly in the env.
- Each company should have its own seeded price generator so its behavior is repeatable.
- The point is not realism for its own sake. The point is that the companies expose different learnable trading regimes.
- Good structure is: one base pattern family per company plus mild stochastic variation around that family.

Required state:

- current cash
- current holdings
- current price
- recent price history window
- remaining time
- realized / unrealized profit markers if useful

Observation design:

- Even though this is semantically a trading simulator, the fixed evaluator still expects `C,H,W`.
- Encode the price history, cash, holdings, and time features into a compact tensor.
- Keep the encoding simple and stationary. Do not hide extra information in arbitrary channels.
- The observation should be sufficient for the policy to infer trend, volatility, current exposure, and time remaining.

Reward structure:

- Use a dense trading reward plus a terminal portfolio reward.
- The dense part should mainly reflect marked-to-market portfolio delta:
  `(cash + holdings * price_t+1) - (cash + holdings * price_t)`, normalized by the initial `$500` and clipped into a small range.
- The terminal part should reflect final portfolio value relative to the starting `$500`.
- Invalid sells when holdings are zero should be penalized.
- Invalid buys when cash is insufficient should be penalized.
- Repeated pointless churn should carry a small penalty so the policy does not learn hyperactive flipping.
- Holding through a good trend should be allowed to outperform constant trading.
- Keep the total reward signal bounded and smooth enough that PPO can learn it.
- You may tune magnitudes, but the structure should stay:
  dense portfolio delta + invalid-action penalties + terminal account-value reward.

Success signal:

- `info["success"]` should become `True` when the final portfolio beats a clear baseline.
- Good baselines are: finishing above the initial `$500`, or beating simple buy-and-hold on that episode.
- A stronger version is to require the agent to beat both the starting bankroll and a simple scripted baseline.

What good behavior should look like:

- buy early on trend-up series
- rest or avoid traps on deceptive flat series
- sell into boom-bust peaks
- avoid panic trading on noise
- preserve capital when no edge is visible

### 2. Text household task env

Build a text-only home or lab simulator with grid-world navigation. The agent
must learn disciplined timing and task order over a day of chores.

Fixed environment structure:

- Home size is `10x10`.
- The agent is a single navigator moving through the house.
- One episode lasts `1000` steps.
- The house represents `24` hours.
- Roughly `41` steps correspond to `1` hour.
- There are exactly `8` chores scheduled across the day.
- Chores are fixed and ordered by hour.

World design:

- The house should have named functional regions such as kitchen, dining area, laundry area, living room, and utility area.
- Each chore should be attached to a specific tile or small set of tiles.
- The schedule should be fixed enough that the policy can learn routine and timing, not pure memorization of random task order.
- The semantic story is text-only, but the transition logic still lives in a spatial simulator.

Required chores can include:

- open fridge
- cook breakfast
- wash clothes
- place plate on table
- cook food
- clean floor
- turn off lights
- turn on lights

Action space:

- `up`
- `down`
- `left`
- `right`
- `rest`
- `start_task`

Task execution rules:

- A task only succeeds if the agent is standing on the correct grid location.
- `start_task` should fail if used at the wrong location, wrong time, or wrong task order.
- The task list is fixed per episode so the agent is learning discipline, not free-form exploration.
- Each chore should have a target hour or narrow time window.
- Doing the right task too early or too late should be worse than doing it on schedule.
- Movement alone should not complete chores. The agent must explicitly trigger `start_task`.

Observation design:

- The simulator is text-only in meaning, but the evaluator still expects `C,H,W`.
- Encode the house layout, agent position, task locations, current hour, active task index, and completion flags into channels.
- Keep the text semantics explicit in the state variables and comments, even if the model consumes tensor channels.
- The observation should let the policy infer where it is, what time it is, what chore is currently due, and what has already been completed.

Reward structure:

- The dominant reward should be disciplined schedule-following, not generic exploration.
- Give a strong positive reward when the correct chore is started at the correct location and within its intended time window.
- Give smaller shaping reward for arriving at the correct location shortly before the task is due.
- Penalize attempting the wrong task, attempting a task out of order, or using `start_task` on the wrong tile.
- Penalize being late relative to the task's scheduled hour.
- Penalize excessive resting or wandering while a scheduled task is pending.
- Add a mild per-step cost so the policy prefers efficient movement.
- Add a strong terminal bonus if all `8` chores are completed correctly and in order within the day.
- You may tune magnitudes, but the structure should stay:
  on-time in-order completion reward + navigation shaping + timing/order penalties + terminal routine-completion bonus.

Success signal:

- `info["success"]` should be `True` only if the full daily schedule is completed correctly.
- Partial completion should still help reward shaping, but should not count as full success.

What good behavior should look like:

- reaching the right room before the target hour
- starting the correct task only when aligned with schedule and location
- conserving movement instead of random wandering
- finishing the full chore list cleanly and in order

## How to create ENV

The candidate environment must obey these constraints:

- `create_env(config, num_envs, device, dtype)` returns a `SimEnv`.
- The environment is single-agent.
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
5. Run the experiment: `.venv/bin/python train.py --train-episodes <n> --eval-episodes <m> --description "<idea>" > run.log 2>&1`
6. Read out the results: `grep "^score:\|^mean_solve_rate:\|^mean_eval_return:\|^train_episodes:\|^eval_episodes:\|^max_steps:" run.log`
7. If the grep output is empty, the run crashed. Read `tail -n 50 run.log`, decide whether the bug is easy to fix, and either retry or mark the idea as a crash.
8. Confirm the row was written to `results.tsv`, then update its status to `keep` or `discard` if needed.
9. If score improved at the same budget, keep the change and advance from the new accepted commit.
10. If score is equal or worse, restore the repo to the last accepted state and move on.

The idea is simple: you are an autonomous environment researcher. If an idea works, keep it. If it does not, discard it. The accepted state moves forward only when the score improves, and the accepted budget only moves upward.

**Timeout**: each experiment should finish quickly. If a run hangs or takes far longer than normal for this machine, kill it and treat it as a failure.

**Crashes**: if a run crashes because of something trivial, fix it and re-run. If the idea itself is broken, log `crash`, revert, and move on.

**NEVER STOP**: once the loop has begun, do not pause to ask whether you should continue. Do not ask for permission after every experiment. Continue until the human interrupts you. If you run out of ideas, think harder: simplify the task, change the topology, change the observation encoding, remove unnecessary mechanics, combine previous near-misses, or try a more radical task redesign.
