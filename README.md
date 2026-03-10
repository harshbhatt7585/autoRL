# autoRL

This repo is for autonomous search over RL environments backed by a vendored
copy of [`simverse`](https://github.com/harshbhatt7585/simverse).

## Design principles

- keep the evaluator fixed
- keep the score fixed
- keep the mutable surface area tiny
- let the outer agent choose episode budget within hard caps
- keep the rest of the runtime stable enough that results stay interpretable

## Repo layout

- `candidate/env.py`: the editable `SimEnv` candidate, currently an intentionally empty starter canvas
- `candidate/train.py`: env-specific policy and PPO hyperparameters
- `framework.py`: fixed Simverse PPO evaluator and score
- `train.py`: fixed CLI entrypoint that prints comparable metrics
- `program.md`: instructions for the autonomous agent
- `vendor/simverse`: vendored upstream Simverse source

## Scoring philosophy

The evaluator reports greedy evaluation return, but the score itself combines:

- greedy solve rate after PPO training
- training gain from early episodes to late episodes
- headroom bonus, which penalizes trivial untrained policies
- stability across seeds
- normalized late-stage training return
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

The evaluator starts with small defaults but allows budget growth up to hard
caps:

- default `12` PPO training episodes
- default `8` greedy eval episodes
- hard cap `1000` PPO training episodes
- hard cap `100` greedy eval episodes
- `2` random seeds
- `32` parallel environments per seed

The intended workflow is to explore with small `--train-episodes` and
`--eval-episodes`, then ratchet them upward as the candidate gets stronger. The
rest of the evaluator should stay the same.

The checked-in candidate environment is intentionally minimal. The actual task
the agent should build belongs in `program.md`, not hardcoded in the baseline
starter env.

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
2. Add a novelty archive so repeated near-duplicate task variants are penalized.
3. Emit replay JSON to the shared Simverse renderer for failure inspection.
4. Split search into proposer and judge agents once the single-agent loop is stable.
