# Farmtila (Competitive v1)

Farmtila is now a simple 1v1 competitive environment designed for easy self-play training and direct policy comparison.

## Core Idea

Two agents collect seeds, then spend seeds to claim or steal tiles.
Final score is number of owned tiles at episode end.

## Action Space

Discrete (6):

1. `0` move up
2. `1` move down
3. `2` move left
4. `3` move right
5. `4` claim/steal tile (costs 1 seed)
6. `5` pickup/no-op (seeds are auto-collected on entry)

## State Layers

Observation has 5 channels:

1. `seed_grid`: `0/1` seed presence
2. `owner_grid`: `-1` unclaimed, `0/1` owner id
3. `farm_grid`: binary claimed mask (`0` empty, `1` claimed)
4. `agent_grid`: agent positions
5. `inventory_grid`: seeds currently held by agents

## Competitive Rules

1. Exactly 2 agents (symmetric start).
2. Stepping on seed collects it automatically.
3. Action `4` on a non-owned tile spends 1 seed and sets ownership to acting agent.
4. If tile was opponent-owned, it is stolen.
5. Episode ends at `max_steps` or when seed budget is exhausted, map has no seeds, and both inventories are empty.

## Rewards (Zero-Sum)

Rewards are explicitly zero-sum:

1. Per-step reward uses score-delta shaping:
- `score_delta = tiles_agent0 - tiles_agent1`
- each step gives reward based on change in this delta
- agent1 reward is exact negative of agent0 reward
2. Terminal reward:
- winner gets `+terminal_win_reward`
- loser gets `-terminal_win_reward`
- draw gets `0`

This makes competition clear and measurable.

## Win / Draw

- Win: own more tiles than opponent at episode end.
- Draw: both own the same number of tiles.

## Recommended Training Setup

1. Self-play with opponent pool (past checkpoints).
2. Fixed map size (for example `16x16`).
3. Evaluate with cross-play matrix and win-rate/Elo.
