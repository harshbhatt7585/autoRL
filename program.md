# autoRL

This repository is for autonomous search over RL environments.

## Setup

To set up a new run, work with the user to:

1. **Agree on a run tag**: propose a short tag based on today's date, for example `mar10`. If git is available, the branch `autorl/<tag>` should not already exist. This is a fresh run.
2. **Create the branch**: if the repo is under git, create `autorl/<tag>` from the current main line. If the repo is not under git, tell the human that git is strongly recommended before long runs.
3. **Read the in-scope files**: the repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `framework.py` — fixed evaluator, fixed score, hard budget caps. Do not modify.
   - `train.py` — fixed entrypoint. Do not modify.
   - `candidate/env.py` — the candidate environment file you modify.
   - `candidate/train.py` — the env-specific policy and PPO tuning file you modify.
4. **Verify the runtime exists**: confirm that `.venv` exists and that the vendored Simverse package is installed. If not, tell the human to run:
   - `python3 -m venv .venv`
   - `.venv/bin/pip install -e vendor/simverse`
5. **Initialize `results.tsv`**: create it with just the header row if needed. The baseline will be recorded after the first run.
6. **Confirm and go**: once setup looks good, kick off the experimentation.

## Build brief

Write the target environment brief here before starting a long run.

- `Task:` fill in the environment the agent should build.
- `Core loop:` fill in the player or policy behavior you want the task to reward.
- `Required mechanics:` list the mechanics that must exist.
- `Success condition:` define what counts as solving the task.
- `Constraints:` note any limits on size, horizon, action count, observation design, or style.

The current `candidate/env.py` is intentionally an empty starter canvas. It is
not the task. The first real job is to replace it with an environment that
matches this brief.

## Experimentation

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
- Do not vary `seed_count`, `num_envs`, or `max_steps`. The only budget knobs are `--train-episodes` and `--eval-episodes`.
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
- Change `seed_count`, `num_envs`, or `max_steps`.
- Change the score definition.

**The goal is simple: get the highest score.** The budget is a search control knob, not a loophole. Spend as little budget as possible while ideas are weak, and only ratchet it upward when the accepted candidate has earned it. The only valid comparison is against the current accepted baseline at the same budget.

**Single-agent constraint**: keep the search focused. The current evaluator expects a single-agent `SimEnv` with a discrete action space and a `C,H,W` observation tensor. Do not drift into multi-agent or continuous-control work unless the fixed evaluator is explicitly changed by the human.

**Simplicity criterion**: all else equal, simpler is better. A tiny score improvement that adds ugly complexity is not worth it. Conversely, deleting mechanics and getting the same or better score is a strong win. When deciding whether to keep a change, weigh the complexity cost against the score gain. Small hacky gains are weak. Cleaner env logic with equal performance is strong.

**The first run**: your very first run should always be the baseline, so you run the current `candidate/env.py` and `candidate/train.py` exactly as they are. Right now that baseline is just an empty starter canvas. After recording it, start building the environment described in the build brief above.

## ENVs to create
1. trading ENV -- Build a simulated marketplace where the agent must, where agent can search, compare, filter and buy stocks. Create a dataset of 5 imaginary companies which must have some pattern of stocks. Give reward based on how well the agent did in buying and selling in a small windows of time-frame. One episode reflect an interval for the stocks, where agent will buy and sell based on past history of stocks of the given company. Agent can take 3 actions: buying, selling and resting. Agent will get free $500 initally to invest on stocks. One interval is 100 hours of trading, where agent can take step every hour, that means the episodic length of simulated trading env is 100.

2. Text household task env: A text-only home or lab simulator, where agent can navigate over a home based on grid-world and can complete tasks: open fridge, cook breakfast, wash clothes, place plate on table, cook food, clean floor, turn off lights and turn on lights. The goal of the env is to make agent descipline and organised. Agent has 24 hours to do these home chores. ENV has a fixed and in-order list of items to do and at which hour, that means will get reward based if it figures out the descpline, what task to do at what time. Agent has given 8 tasks do in 24 hour. Agent can navigation in home so it can to be present at on the spot where these task should be done (these task are scatterd in home). The grid world should be 10x10 and one episode length is 1000. There are 1000/24 ~ 41 steps in an hour, that means after 41 steps agent complete 1 hour. Agent can take action: up, down, left, right, rest and start task (when over grid where task present). 

## How to create ENV

The candidate environment must obey these constraints:

- `create_env(config, num_envs, device, dtype)` returns a `SimEnv`.
- The environment is single-agent.
- The environment exposes a discrete `action_space`.
- The environment exposes a 3D `observation_space.shape` in `C, H, W`.
- `reset()` returns a dict that contains an `obs` tensor.
- `step(action)` returns `(obs_dict, reward, done, info)` compatible with the Simverse PPO trainer.
- Rewards should stay finite and roughly within `[-1.0, 1.0]`.
- Episodes must terminate within `config.max_steps`.

The candidate training file must obey these constraints:

- It may define the policy architecture for this environment.
- It may tune optimizer and PPO hyperparameters for this environment.
- It must not set the episode budget itself. The outer experiment loop controls `--train-episodes` and `--eval-episodes`.
- It must not redefine the score.

## Output format

When the script finishes it prints a summary like this:

```text
---
score: 99.600000
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
grep "^score:\|^mean_solve_rate:\|^mean_eval_return:\|^train_episodes:\|^eval_episodes:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` as tab-separated text. Do not use commas.

The TSV has a header row and 8 columns:

```text
commit	train_episodes	eval_episodes	score	solve_rate	eval_return	status	description
```

1. git commit hash, short form if git exists, otherwise `nogit`
2. train episodes used, for example `128`
3. eval episodes used, for example `16`
4. score achieved, for example `99.600000` — use `0.000000` for crashes
5. solve rate achieved, for example `1.000000` — use `0.000000` for crashes
6. eval return achieved, for example `1.865000` — use `0.000000` for crashes
7. status: `keep`, `discard`, or `crash`
8. short text description of what the experiment tried

Do not commit `results.tsv`. Leave it untracked.

## The experiment loop

The experiment runs on a dedicated branch, for example `autorl/mar10`.

LOOP FOREVER:

1. Look at the git state: current branch, current accepted commit, and current accepted budget.
2. Start with a small budget if this is a fresh run. If the accepted candidate is already strong at the current budget, ratchet upward and re-baseline the accepted commit first.
3. Tune `candidate/env.py`, `candidate/train.py`, or both with one experimental idea.
4. Commit the experiment if git is available.
5. Run the experiment: `.venv/bin/python train.py --train-episodes <n> --eval-episodes <m> > run.log 2>&1`
6. Read out the results: `grep "^score:\|^mean_solve_rate:\|^mean_eval_return:\|^train_episodes:\|^eval_episodes:" run.log`
7. If the grep output is empty, the run crashed. Read `tail -n 50 run.log`, decide whether the bug is easy to fix, and either retry or mark the idea as a crash.
8. Record the result in `results.tsv`, including the budget used.
9. If score improved at the same budget, keep the change and advance from the new accepted commit.
10. If score is equal or worse, restore the repo to the last accepted state and move on.

The idea is simple: you are an autonomous environment researcher. If an idea works, keep it. If it does not, discard it. The accepted state moves forward only when the score improves, and the accepted budget only moves upward.

**Timeout**: each experiment should finish quickly. If a run hangs or takes far longer than normal for this machine, kill it and treat it as a failure.

**Crashes**: if a run crashes because of something trivial, fix it and re-run. If the idea itself is broken, log `crash`, revert, and move on.

**NEVER STOP**: once the loop has begun, do not pause to ask whether you should continue. Do not ask for permission after every experiment. Continue until the human interrupts you. If you run out of ideas, think harder: simplify the task, change the topology, change the observation encoding, remove unnecessary mechanics, combine previous near-misses, or try a more radical task redesign.
