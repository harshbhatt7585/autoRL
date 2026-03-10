# autoRL

This repo is a minimal adaptation of Andrej Karpathy's
[`autoresearch`](https://github.com/karpathy/autoresearch) pattern for RL
environment invention, backed by a vendored copy of
[`simverse`](https://github.com/harshbhatt7585/simverse).

## What matters from `autoresearch`

The important design choice in Karpathy's repo is not "let the agent touch the
whole project". It is the opposite:

- keep the evaluator fixed
- keep the time budget fixed
- keep the mutable surface area tiny
- compare every run on the same metric

In the original repo, the agent edits `train.py` while `prepare.py` and the
metric stay fixed. For RL environment generation, the cleanest analogue is:

- `candidate/env.py` and `candidate/train.py` are the mutable candidate surface
- `framework.py` contains the fixed Simverse PPO evaluator and score
- root `train.py` evaluates the current candidate with a fixed budget

If you let the agent mutate both the environment and the evaluator at the same
time, scores drift and experiments stop being comparable.

## Repo layout

- `candidate/env.py`: the editable `SimEnv` candidate
- `candidate/train.py`: env-specific policy and PPO hyperparameters
- `framework.py`: fixed Simverse PPO evaluator and score
- `train.py`: fixed CLI entrypoint that prints comparable metrics
- `program.md`: instructions for the autonomous agent
- `vendor/simverse`: vendored upstream Simverse source

## Scoring philosophy

The score is intentionally not just "final reward". It combines:

- greedy solve rate after PPO training
- greedy evaluation return
- training gain from early episodes to late episodes
- headroom bonus, which penalizes trivial untrained policies
- stability across seeds
- a small penalty for oversized observation and action spaces

This pushes the agent toward environments that are solvable, learnable, and not
completely trivial.

## Quick start

Create a local virtual environment and install the vendored Simverse package:

```bash
python3 -m venv .venv
.venv/bin/pip install -e vendor/simverse
```

Run the fixed evaluator through that virtualenv:

```bash
.venv/bin/python train.py
```

The default evaluation budget is intentionally small:

- `12` PPO training episodes
- `8` greedy eval episodes
- `2` random seeds
- `32` parallel environments per seed

For an autonomous loop, initialize git first so the agent can keep or discard
environment mutations cleanly:

```bash
git init
git add .
git commit -m "Initial Simverse autoRL scaffold"
```

Then point your coding agent at `program.md`.

## Suggested next upgrades

This scaffold is intentionally small. The next useful upgrades are:

1. Promote the candidate into a full `simverse.envs.autorl_candidate` package.
2. Add a novelty archive so repeated key-door variants are penalized.
3. Emit replay JSON to the shared Simverse renderer for failure inspection.
4. Split search into proposer and judge agents once the single-agent loop is stable.
