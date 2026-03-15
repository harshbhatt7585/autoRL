from __future__ import annotations

# ruff: noqa: E402
import argparse
import contextlib
import importlib.metadata
import importlib.util
import io
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from statistics import mean, pstdev
from typing import Callable

os.environ.setdefault("MPLCONFIGDIR", str(Path("docs/benchmarks/.mpl-cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("docs/benchmarks/.cache").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from simverse.envs.battle_grid.agent import BattleGridAgent
from simverse.envs.battle_grid.config import BattleGridConfig
from simverse.envs.battle_grid.env import BattleGridEnv
from simverse.envs.battle_grid.env import create_env as create_battle_grid_env
from simverse.envs.maze_race.agent import MazeRaceAgent
from simverse.envs.maze_race.config import MazeRaceConfig
from simverse.envs.maze_race.env import MazeRaceEnv
from simverse.envs.maze_race.env import create_env as create_maze_race_env
from simverse.policies.simple import SimplePolicy
from simverse.training.ppo import PPOTrainer
from simverse.training.utils import (
    build_ppo_training_config,
    configure_torch_backend,
    resolve_rollout_dtype,
    run_ppo_training,
)

END_TO_END_SUITE = "end_to_end_training"
POLICY_UPDATE_SUITE = "policy_update_microbenchmark"
SUITE_TITLES = {
    END_TO_END_SUITE: "End-to-End Training Smoke",
    POLICY_UPDATE_SUITE: "PPO Policy Update Microbenchmark",
}
END_TO_END_NUM_ENVS = 512


@dataclass(frozen=True)
class BenchmarkCase:
    suite: str
    name: str
    label: str
    framework: str
    device: str
    unit: str
    sample_scale: float
    runner: Callable[[str], None]


@dataclass(frozen=True)
class PolicyUpdateSpec:
    name: str
    label: str
    obs_shape: tuple[int, int, int]
    action_count: int
    batch_size: int
    inner_steps: int


@dataclass(frozen=True)
class _ObsSpace:
    shape: tuple[int, int, int]


@dataclass(frozen=True)
class _ActionSpace:
    n: int


POLICY_UPDATE_SPECS: tuple[PolicyUpdateSpec, ...] = (
    PolicyUpdateSpec(
        name="battle_grid",
        label="Battle Grid PPO Update",
        obs_shape=(5, 13, 13),
        action_count=6,
        batch_size=2048,
        inner_steps=10,
    ),
    PolicyUpdateSpec(
        name="maze_race",
        label="Maze Race PPO Update",
        obs_shape=(5, 7, 7),
        action_count=5,
        batch_size=1024,
        inner_steps=16,
    ),
)

POLICY_UPDATE_SPEC_BY_NAME = {spec.name: spec for spec in POLICY_UPDATE_SPECS}


def battle_grid_agent_factory(
    agent_id: int,
    policy: torch.nn.Module,
    env: BattleGridEnv,
) -> BattleGridAgent:
    action_values = np.arange(getattr(env.action_space, "n", 6), dtype=np.int64)
    return BattleGridAgent(
        agent_id=agent_id,
        action_space=action_values,
        policy=policy,
        name=f"battle_agent_{agent_id}",
    )


def maze_race_agent_factory(
    agent_id: int,
    policy: torch.nn.Module,
    env: MazeRaceEnv,
) -> MazeRaceAgent:
    action_values = np.arange(getattr(env.action_space, "n", 5), dtype=np.int64)
    return MazeRaceAgent(
        agent_id=agent_id,
        action_space=action_values,
        policy=policy,
        name=f"maze_race_agent_{agent_id}",
    )


def run_battle_grid_smoke(device: str) -> None:
    dtype = resolve_rollout_dtype(device)
    configure_torch_backend(device)
    config = BattleGridConfig(
        width=13,
        height=13,
        num_agents=2,
        num_envs=END_TO_END_NUM_ENVS,
        max_steps=64,
        max_health=3,
        attack_damage=1,
        attack_range=1,
        step_penalty=0.01,
        damage_reward=0.05,
        kill_reward=1.0,
        death_penalty=1.0,
        timeout_win_reward=0.5,
        timeout_lose_penalty=0.5,
        draw_reward=0.0,
        policies=[],
    )
    env = create_battle_grid_env(config, num_envs=config.num_envs, device=device, dtype=dtype)
    training_config = build_ppo_training_config(
        num_agents=config.num_agents,
        num_envs=config.num_envs,
        max_steps=config.max_steps,
        episodes=1,
        training_epochs=1,
        lr=3e-4,
        batch_size=config.num_envs * 4,
        buffer_size=config.num_envs * config.num_agents * 8,
        device=device,
        dtype=dtype,
    )
    run_ppo_training(
        env=env,
        training_config=training_config,
        agent_factory=battle_grid_agent_factory,
        policy_factory=lambda obs_space, action_space: SimplePolicy(
            obs_space=obs_space,
            action_space=action_space,
        ),
        title="Battle Grid Benchmark",
        run_name="battle-grid-benchmark",
        episode_save_dir=None,
        use_wandb=False,
        use_compile=False,
        policy_name_prefix="battle_agent",
    )


def run_maze_race_smoke(device: str) -> None:
    dtype = resolve_rollout_dtype(device, cpu_dtype=torch.float32)
    configure_torch_backend(device)
    config = MazeRaceConfig(
        width=7,
        height=7,
        num_agents=2,
        num_envs=END_TO_END_NUM_ENVS,
        max_steps=64,
        win_reward=1.0,
        lose_penalty=1.0,
        draw_reward=0.0,
        policies=[],
    )
    env = create_maze_race_env(config, num_envs=config.num_envs, device=device, dtype=dtype)
    training_config = build_ppo_training_config(
        num_agents=config.num_agents,
        num_envs=config.num_envs,
        max_steps=config.max_steps,
        episodes=1,
        training_epochs=1,
        lr=3e-4,
        batch_size=config.num_envs * 2,
        buffer_size=config.num_envs * config.num_agents * 8,
        device=device,
        dtype=dtype,
    )
    run_ppo_training(
        env=env,
        training_config=training_config,
        agent_factory=maze_race_agent_factory,
        policy_factory=lambda obs_space, action_space: SimplePolicy(
            obs_space=obs_space,
            action_space=action_space,
        ),
        title="Maze Race Benchmark",
        run_name="maze-race-benchmark",
        episode_save_dir=None,
        use_wandb=False,
        use_compile=False,
        policy_name_prefix="maze_race_agent",
    )


def available_torch_devices() -> list[str]:
    devices = ["cpu"]
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def _result_from_samples(
    *,
    suite: str,
    name: str,
    label: str,
    framework: str,
    device: str,
    unit: str,
    samples: list[float],
) -> dict[str, object]:
    return {
        "suite": suite,
        "name": name,
        "label": label,
        "framework": framework,
        "device": device,
        "unit": unit,
        "samples": samples,
        "mean": mean(samples),
        "std": pstdev(samples) if len(samples) > 1 else 0.0,
    }


def benchmark_case(case: BenchmarkCase, repeats: int) -> dict[str, object]:
    samples: list[float] = []
    original_save_checkpoint = PPOTrainer.save_checkpoint
    try:
        PPOTrainer.save_checkpoint = lambda self, checkpoint_path: None
        with contextlib.redirect_stdout(io.StringIO()):
            case.runner(case.device)
        for _ in range(repeats):
            start = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                case.runner(case.device)
            samples.append((time.perf_counter() - start) * case.sample_scale)
    finally:
        PPOTrainer.save_checkpoint = original_save_checkpoint

    return _result_from_samples(
        suite=case.suite,
        name=case.name,
        label=case.label,
        framework=case.framework,
        device=case.device,
        unit=case.unit,
        samples=samples,
    )


def _sync_torch(device: str) -> None:
    if device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
    elif device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _build_torch_policy_inputs(spec: PolicyUpdateSpec, device: str) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(0)
    obs = torch.from_numpy(
        rng.standard_normal((spec.batch_size, *spec.obs_shape), dtype=np.float32)
    ).to(device=device, dtype=torch.float32)
    actions = torch.from_numpy(
        rng.integers(0, spec.action_count, size=(spec.batch_size,), dtype=np.int64)
    ).to(device=device)
    old_log_probs = torch.from_numpy(
        rng.normal(loc=0.0, scale=0.25, size=(spec.batch_size,)).astype(np.float32)
    ).to(device=device)
    advantages = torch.from_numpy(rng.normal(size=(spec.batch_size,)).astype(np.float32)).to(
        device=device
    )
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
    returns = torch.from_numpy(rng.normal(size=(spec.batch_size,)).astype(np.float32)).to(
        device=device
    )
    return {
        "obs": obs,
        "actions": actions,
        "old_log_probs": old_log_probs,
        "advantages": advantages,
        "returns": returns,
    }


def _run_torch_policy_update_steps(spec: PolicyUpdateSpec, device: str) -> None:
    torch.manual_seed(0)
    policy = SimplePolicy(
        obs_space=_ObsSpace(spec.obs_shape),
        action_space=_ActionSpace(spec.action_count),
    ).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    batch = _build_torch_policy_inputs(spec, device)
    clip_epsilon = 0.2
    entropy_coef = 0.01

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        logits, value = policy(batch["obs"])
        log_probs = torch.log_softmax(logits, dim=-1)
        chosen_log_probs = log_probs.gather(1, batch["actions"].unsqueeze(-1)).squeeze(-1)
        ratio = torch.exp(chosen_log_probs - batch["old_log_probs"])
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
        surr1 = ratio * batch["advantages"]
        surr2 = clipped_ratio * batch["advantages"]
        policy_loss = -torch.minimum(surr1, surr2).mean()
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        value_loss = 0.5 * (batch["returns"] - value.squeeze(-1)).pow(2).mean()
        loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy
        loss.backward()
        optimizer.step()

    for _ in range(3):
        step()
    _sync_torch(device)
    for _ in range(spec.inner_steps):
        step()
    _sync_torch(device)


def benchmark_torch_policy_update(
    spec: PolicyUpdateSpec,
    device: str,
    repeats: int,
) -> dict[str, object]:
    case = BenchmarkCase(
        suite=POLICY_UPDATE_SUITE,
        name=f"{spec.name}_torch_{device}",
        label=f"{spec.label} [Torch {device}]",
        framework="torch",
        device=device,
        unit="ms/update",
        sample_scale=1000.0 / spec.inner_steps,
        runner=partial(_run_torch_policy_update_steps, spec),
    )
    return benchmark_case(case, repeats)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _run_self_subprocess(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), *args],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )


def probe_mlx_runtime() -> dict[str, object]:
    completed = _run_self_subprocess("--mlx-probe")
    if completed.returncode != 0:
        return {
            "available": False,
            "error": completed.stderr.strip() or completed.stdout.strip() or "mlx probe failed",
        }
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {
            "available": False,
            "error": completed.stdout.strip() or "mlx probe returned invalid json",
        }
    return payload


def benchmark_mlx_policy_update(spec: PolicyUpdateSpec, repeats: int) -> dict[str, object]:
    completed = _run_self_subprocess("--mlx-policy-case", spec.name, "--repeats", str(repeats))
    if completed.returncode != 0:
        raise RuntimeError(
            completed.stderr.strip() or completed.stdout.strip() or "MLX benchmark failed"
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("MLX benchmark returned invalid JSON") from exc


def collect_environment_info() -> dict[str, object]:
    mps_built = bool(torch.backends.mps.is_built()) if hasattr(torch.backends, "mps") else False
    mps_available = (
        bool(torch.backends.mps.is_available()) if hasattr(torch.backends, "mps") else False
    )
    mlx_installed = importlib.util.find_spec("mlx") is not None
    mlx_version = None
    if mlx_installed:
        try:
            mlx_version = importlib.metadata.version("mlx")
        except Exception:
            mlx_version = "installed_but_metadata_unavailable"

    mlx_probe = probe_mlx_runtime() if mlx_installed else {"available": False, "error": None}
    if mlx_probe.get("version") is not None:
        mlx_version = mlx_probe["version"]

    if mlx_probe.get("available"):
        note = (
            "This benchmark includes real MLX PPO-update microbenchmarks for the policy hot path. "
            "End-to-end environment stepping and PPO training still run only through the "
            "PyTorch stack."
        )
    elif mlx_installed:
        note = (
            "MLX is installed, but the runtime could not initialize from this process, "
            "so MLX results are unavailable here."
        )
    else:
        note = (
            "MLX is not installed in this environment, so there is no MLX-vs-PyTorch policy-update "
            "comparison here."
        )

    return {
        "generated_at": datetime.now().isoformat(),
        "hostname": "local",
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "mps_built": mps_built,
        "mps_available": mps_available,
        "mlx_installed": mlx_installed,
        "mlx_runtime_available": bool(mlx_probe.get("available", False)),
        "mlx_version": mlx_version,
        "mlx_default_device": mlx_probe.get("default_device"),
        "mlx_runtime_error": mlx_probe.get("error"),
        "note": note,
    }


def _group_results(results: list[dict[str, object]]) -> list[tuple[str, list[dict[str, object]]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for item in results:
        suite = str(item["suite"])
        grouped.setdefault(suite, []).append(item)

    ordered_suites = [END_TO_END_SUITE, POLICY_UPDATE_SUITE]
    return [(suite, grouped[suite]) for suite in ordered_suites if suite in grouped]


def write_outputs(
    results: list[dict[str, object]], output_dir: Path, env_info: dict[str, object]
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"training_runtime_{timestamp}.json"
    png_path = output_dir / f"training_runtime_{timestamp}.png"

    payload = dict(env_info)
    payload["results"] = results
    json_path.write_text(json.dumps(payload, indent=2))

    grouped_results = _group_results(results)
    fig, axes = plt.subplots(
        len(grouped_results),
        1,
        figsize=(12, max(4, 4 * len(grouped_results))),
        squeeze=False,
    )
    palette = ["#2f6db2", "#c66b2d", "#2f9a64", "#9c4ec8", "#d1495b", "#7c6f64"]

    for ax, (suite, items) in zip(axes.flat, grouped_results, strict=False):
        labels = [str(item["label"]) for item in items]
        means = [float(item["mean"]) for item in items]
        stds = [float(item["std"]) for item in items]
        unit = str(items[0]["unit"])
        bars = ax.bar(labels, means, yerr=stds, capsize=6, color=palette[: len(labels)])
        ax.set_title(SUITE_TITLES.get(suite, suite.replace("_", " ").title()))
        ax.set_ylabel("Seconds" if unit == "sec" else unit)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", rotation=10)

        suffix = "s" if unit == "sec" else " ms"
        for bar, value in zip(bars, means, strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{value:.3f}{suffix}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    footer = (
        f"MPS built={env_info['mps_built']} available={env_info['mps_available']} | "
        f"MLX runtime available={env_info['mlx_runtime_available']} "
        f"device={env_info['mlx_default_device']}"
    )
    fig.text(0.01, 0.01, footer, fontsize=8, color="#555555")
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    return json_path, png_path


def print_terminal_summary(
    env_info: dict[str, object], results: list[dict[str, object]], json_path: Path, png_path: Path
) -> None:
    print("Benchmark environment")
    print(f"  generated_at: {env_info['generated_at']}")
    print(f"  hostname: {env_info['hostname']}")
    print(f"  python_version: {env_info['python_version']}")
    print(f"  torch_version: {env_info['torch_version']}")
    print(f"  mps_built: {env_info['mps_built']}")
    print(f"  mps_available: {env_info['mps_available']}")
    print(f"  mlx_installed: {env_info['mlx_installed']}")
    print(f"  mlx_runtime_available: {env_info['mlx_runtime_available']}")
    print(f"  mlx_version: {env_info['mlx_version']}")
    print(f"  mlx_default_device: {env_info['mlx_default_device']}")
    if env_info["mlx_runtime_error"]:
        print(f"  mlx_runtime_error: {env_info['mlx_runtime_error']}")
    print(f"  note: {env_info['note']}")
    print()

    for suite, items in _group_results(results):
        print(SUITE_TITLES.get(suite, suite))
        for item in items:
            print(
                f"  {item['label']}: mean={float(item['mean']):.3f} {item['unit']} "
                f"std={float(item['std']):.3f} samples={item['samples']}"
            )
        print()

    print(f"saved_json={json_path}")
    print(f"saved_png={png_path}")


def _run_mlx_probe_mode() -> None:
    import mlx.core as mx

    payload = {
        "available": True,
        "version": importlib.metadata.version("mlx"),
        "default_device": str(mx.default_device()),
    }
    print(json.dumps(payload))


def _run_mlx_policy_case_mode(spec_name: str, repeats: int) -> None:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    spec = POLICY_UPDATE_SPEC_BY_NAME[spec_name]
    rng = np.random.default_rng(0)

    class MLXSimplePolicy(nn.Module):
        def __init__(self, in_channels: int, action_count: int):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
            self.conv3 = nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1)
            self.conv4 = nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1)
            self.trunk = nn.Linear(128, 96)
            self.action_head = nn.Linear(96, action_count)
            self.value_head = nn.Linear(96, 1)

        def __call__(self, obs: mx.array) -> tuple[mx.array, mx.array]:
            x = mx.transpose(obs, (0, 2, 3, 1))
            x = nn.silu(self.conv1(x))
            x = nn.silu(self.conv2(x))
            x = nn.silu(self.conv3(x))
            x = nn.silu(self.conv4(x))
            x = mx.mean(x, axis=(1, 2))
            x = nn.silu(self.trunk(x))
            return self.action_head(x), self.value_head(x)

    model = MLXSimplePolicy(spec.obs_shape[0], spec.action_count)
    optimizer = optim.Adam(learning_rate=3e-4)

    obs = mx.array(rng.standard_normal((spec.batch_size, *spec.obs_shape), dtype=np.float32))
    actions = mx.array(rng.integers(0, spec.action_count, size=(spec.batch_size,), dtype=np.int32))
    old_log_probs = mx.array(
        rng.normal(loc=0.0, scale=0.25, size=(spec.batch_size,)).astype(np.float32)
    )
    advantages = mx.array(rng.normal(size=(spec.batch_size,)).astype(np.float32))
    advantages = (advantages - mx.mean(advantages)) / (mx.std(advantages) + 1e-6)
    returns = mx.array(rng.normal(size=(spec.batch_size,)).astype(np.float32))
    clip_epsilon = 0.2
    entropy_coef = 0.01

    def loss_fn(
        batch_obs: mx.array,
        batch_actions: mx.array,
        batch_old_log_probs: mx.array,
        batch_advantages: mx.array,
        batch_returns: mx.array,
    ) -> mx.array:
        logits, value = model(batch_obs)
        log_probs = nn.log_softmax(logits, axis=-1)
        selected = mx.take_along_axis(log_probs, batch_actions[:, None], axis=-1).squeeze(-1)
        ratio = mx.exp(selected - batch_old_log_probs)
        clipped_ratio = mx.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
        surr1 = ratio * batch_advantages
        surr2 = clipped_ratio * batch_advantages
        policy_loss = -mx.mean(mx.where(surr1 < surr2, surr1, surr2))
        probs = mx.softmax(logits, axis=-1)
        entropy = -mx.mean(mx.sum(probs * log_probs, axis=-1))
        value_loss = 0.5 * mx.mean(mx.square(batch_returns - mx.squeeze(value, axis=-1)))
        return policy_loss + 0.5 * value_loss - entropy_coef * entropy

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    mx.eval(model.parameters(), optimizer.state, obs, actions, old_log_probs, advantages, returns)

    def step() -> None:
        loss, grads = loss_and_grad(obs, actions, old_log_probs, advantages, returns)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters(), optimizer.state)

    for _ in range(3):
        step()
    mx.synchronize()

    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(spec.inner_steps):
            step()
        mx.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0 / spec.inner_steps)

    result = _result_from_samples(
        suite=POLICY_UPDATE_SUITE,
        name=f"{spec.name}_mlx_gpu",
        label=f"{spec.label} [MLX {mx.default_device()}]",
        framework="mlx",
        device=str(mx.default_device()),
        unit="ms/update",
        samples=samples,
    )
    print(json.dumps(result))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run quick Simverse training and policy benchmarks."
    )
    parser.add_argument("--mlx-probe", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--mlx-policy-case",
        choices=sorted(POLICY_UPDATE_SPEC_BY_NAME),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--repeats", type=int, default=3, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mlx_probe:
        _run_mlx_probe_mode()
        return
    if args.mlx_policy_case:
        _run_mlx_policy_case_mode(args.mlx_policy_case, args.repeats)
        return

    repeats = int(args.repeats)
    training_cases: list[BenchmarkCase] = []
    for device in available_torch_devices():
        training_cases.append(
            BenchmarkCase(
                suite=END_TO_END_SUITE,
                name=f"battle_grid_{device}",
                label=f"Battle Grid ({device})",
                framework="torch",
                device=device,
                unit="sec",
                sample_scale=1.0,
                runner=run_battle_grid_smoke,
            )
        )
        training_cases.append(
            BenchmarkCase(
                suite=END_TO_END_SUITE,
                name=f"maze_race_{device}",
                label=f"Maze Race 2P ({device})",
                framework="torch",
                device=device,
                unit="sec",
                sample_scale=1.0,
                runner=run_maze_race_smoke,
            )
        )

    if not training_cases:
        raise RuntimeError("No runnable benchmark cases were discovered")

    env_info = collect_environment_info()
    results = [benchmark_case(case, repeats=repeats) for case in training_cases]

    for spec in POLICY_UPDATE_SPECS:
        for device in available_torch_devices():
            results.append(benchmark_torch_policy_update(spec, device=device, repeats=repeats))

    if env_info["mlx_runtime_available"]:
        for spec in POLICY_UPDATE_SPECS:
            results.append(benchmark_mlx_policy_update(spec, repeats=repeats))

    json_path, png_path = write_outputs(results, Path("docs/benchmarks"), env_info)
    print_terminal_summary(env_info, results, json_path, png_path)


if __name__ == "__main__":
    main()
