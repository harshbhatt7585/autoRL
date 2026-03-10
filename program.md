# autoRL

This repository is for autonomous search over RL environments.

## Setup

To set up a new run, work with the user to:

1. **Agree on a run tag**: propose a short tag based on today's date, for example `mar10`. If git is available, the branch `autorl/<tag>` should not already exist. This is a fresh run.
2. **Create the branch**: if the repo is under git, create `autorl/<tag>` from the current main line. If the repo is not under git, tell the human that git is strongly recommended before long runs.
3. **Read the in-scope files**: the repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `framework.py` — fixed evaluator, fixed score, fixed budget. Do not modify.
   - `train.py` — fixed entrypoint. Do not modify.
   - `candidate/env.py` — the candidate environment file you modify.
   - `candidate/train.py` — the env-specific policy and PPO tuning file you modify.
4. **Verify the runtime exists**: confirm that `.venv` exists and that the vendored Simverse package is installed. If not, tell the human to run:
   - `python3 -m venv .venv`
   - `.venv/bin/pip install -e vendor/simverse`
5. **Initialize `results.tsv`**: create it with just the header row if needed. The baseline will be recorded after the first run.
6. **Confirm and go**: once setup looks good, kick off the experimentation.

## Experimentation

Each experiment runs with a **fixed evaluation budget**. You launch it simply as:

```bash
.venv/bin/python train.py
```

The default evaluator budget is fixed inside the repo:

- `12` PPO training episodes
- `8` greedy evaluation episodes
- `2` random seeds
- `32` parallel environments per seed
- `cpu` by default

**What you CAN do:**

- Modify `candidate/env.py`.
- Modify `candidate/train.py`.
- Change the task layout, reward structure, transition dynamics, observation encoding, action semantics, and episode logic.
- Change the env-specific policy architecture and PPO hyperparameters used for that candidate environment.
- Simplify or redesign the environment completely, as long as it still satisfies the interface expected by the fixed evaluator.

**What you CANNOT do:**

- Modify `framework.py`.
- Modify the root `train.py`.
- Modify `vendor/simverse`.
- Install new packages or add dependencies.
- Change the evaluation budget.
- Change the score definition.

**The goal is simple: get the highest score.** Since the evaluator budget is fixed, you do not need to worry about making training longer. The only thing that matters is whether the candidate environment scores better under the same budget.

**Single-agent constraint**: keep the search focused. The current evaluator expects a single-agent `SimEnv` with a discrete action space and a `C,H,W` observation tensor. Do not drift into multi-agent or continuous-control work unless the fixed evaluator is explicitly changed by the human.

**Simplicity criterion**: all else equal, simpler is better. A tiny score improvement that adds ugly complexity is not worth it. Conversely, deleting mechanics and getting the same or better score is a strong win. When deciding whether to keep a change, weigh the complexity cost against the score gain. Small hacky gains are weak. Cleaner env logic with equal performance is strong.

**The first run**: your very first run should always be the baseline, so you run the current `candidate/env.py` and `candidate/train.py` exactly as they are.

## Environment contract

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
- It must not change the fixed evaluator budget.
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
grep "^score:\|^mean_solve_rate:\|^mean_eval_return:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` as tab-separated text. Do not use commas.

The TSV has a header row and 6 columns:

```text
commit	score	solve_rate	eval_return	status	description
```

1. git commit hash, short form if git exists, otherwise `nogit`
2. score achieved, for example `99.600000` — use `0.000000` for crashes
3. solve rate achieved, for example `1.000000` — use `0.000000` for crashes
4. eval return achieved, for example `1.865000` — use `0.000000` for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description of what the experiment tried

Do not commit `results.tsv`. Leave it untracked.

## The experiment loop

The experiment runs on a dedicated branch, for example `autorl/mar10`.

LOOP FOREVER:

1. Look at the git state: current branch and current accepted commit.
2. Tune `candidate/env.py`, `candidate/train.py`, or both with one experimental idea.
3. Commit the experiment if git is available.
4. Run the experiment: `.venv/bin/python train.py > run.log 2>&1`
5. Read out the results: `grep "^score:\|^mean_solve_rate:\|^mean_eval_return:" run.log`
6. If the grep output is empty, the run crashed. Read `tail -n 50 run.log`, decide whether the bug is easy to fix, and either retry or mark the idea as a crash.
7. Record the result in `results.tsv`.
8. If score improved, keep the change and advance from the new accepted commit.
9. If score is equal or worse, restore the repo to the last accepted state and move on.

The idea is simple: you are an autonomous environment researcher. If an idea works, keep it. If it does not, discard it. The accepted state moves forward only when the score improves.

**Timeout**: each experiment should finish quickly. If a run hangs or takes far longer than normal for this machine, kill it and treat it as a failure.

**Crashes**: if a run crashes because of something trivial, fix it and re-run. If the idea itself is broken, log `crash`, revert, and move on.

**NEVER STOP**: once the loop has begun, do not pause to ask whether you should continue. Do not ask for permission after every experiment. Continue until the human interrupts you. If you run out of ideas, think harder: simplify the task, change the topology, change the observation encoding, remove unnecessary mechanics, combine previous near-misses, or try a more radical task redesign.
