import contextlib
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch

from simverse.core.agent import SimAgent
from simverse.core.env import SimEnv
from simverse.core.trainer import Trainer
from simverse.training.logging import training_logger
from simverse.training.stats import TrainingStats
from simverse.training.wandb import DEFAULT_WANDB_PROJECT
from simverse.utils.replay_buffer import ReplayBuffer

try:
    import wandb

    _WANDB_AVAILABLE = all(hasattr(wandb, attr) for attr in ("init", "log", "finish"))
except ImportError:
    wandb = None
    _WANDB_AVAILABLE = False


@dataclass
class RolloutProgress:
    training_start: float
    paused_time: float = 0.0
    last_active_time: float = 0.0
    total_agent_steps: int = 0
    last_logged_steps: int = 0


@dataclass
class EpisodeSummary:
    total_reward: float
    agent_rewards: np.ndarray
    steps: int
    next_observation: Dict[str, Any] | None = None
    all_done: bool = False
    active_mask: torch.Tensor | None = None


class PPOTrainer(Trainer):
    DEFAULT_BUFFER_SIZE = 10000
    DEFAULT_BATCH_SIZE = 32

    def __init__(
        self,
        optimizer: Optional[torch.optim.Optimizer] = None,
        optimizers: Optional[Dict[int, torch.optim.Optimizer]] = None,
        centralized_critic: Optional[torch.nn.Module] = None,
        centralized_critic_optimizer: Optional[torch.optim.Optimizer] = None,
        episodes: int = 1,
        training_epochs: int = 4,
        clip_epsilon: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        stats: Optional[TrainingStats] = None,
        config: Optional[Dict[str, Any]] = None,
        project_name: str = DEFAULT_WANDB_PROJECT,
        run_name: str = "ppo-training",
        episode_save_dir: str | None = None,
        device: Union[torch.device, str] = "cpu",
        batch_size: int = DEFAULT_BATCH_SIZE,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        dtype: torch.dtype = torch.float32,
        use_wandb: bool = True,
    ):
        super().__init__()

        if optimizer is None and not optimizers:
            raise ValueError(
                "PPOTrainer requires either a shared optimizer or per-agent optimizers"
            )
        if optimizer is not None and optimizers:
            raise ValueError("Provide only one of optimizer or optimizers")

        self.optimizer = optimizer
        self.optimizers = optimizers or {}
        self.centralized_critic = centralized_critic
        self.centralized_critic_optimizer = centralized_critic_optimizer
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.episodes = episodes
        self.training_epochs = training_epochs
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.stats = stats if stats is not None else TrainingStats()
        self.config = config or {}
        self.project_name = project_name
        self.run_name = run_name
        self._wandb_initialized = False
        self.use_wandb = use_wandb
        self.episode_save_dir = episode_save_dir
        self._env_metadata_cache: Dict[str, Any] | None = None
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.dtype = dtype
        # Keep trainable params in FP32 for optimizer stability; rollout tensors can use self.dtype.
        self.policy_dtype = torch.float32
        self.env_batch_size = 1
        self.entropy_coef = float(self.config.get("entropy_coef", 0.01))
        self.normalize_advantages = bool(self.config.get("normalize_advantages", True))
        self.max_grad_norm = float(self.config.get("max_grad_norm", 0.5))
        amp_config = self.config.get("use_amp")
        self.use_amp = bool(amp_config) if amp_config is not None else self.device.type == "cuda"
        amp_dtype_name = str(self.config.get("amp_dtype", "float16")).lower()
        self.amp_dtype = torch.bfloat16 if amp_dtype_name == "bfloat16" else torch.float16
        self._amp_enabled = self.use_amp and self.device.type == "cuda"
        self.grad_scaler = torch.amp.GradScaler(
            device="cuda",
            enabled=self._amp_enabled and self.amp_dtype == torch.float16,
        )
        self.use_ctde = bool(self.config.get("ctde", False))
        if self.use_ctde and self.centralized_critic is None:
            raise ValueError("CTDE requires a centralized_critic model")
        if self.use_ctde and self.centralized_critic_optimizer is None:
            raise ValueError("CTDE requires a centralized_critic_optimizer")
        configured_fastpath = self.config.get("torch_fastpath")
        if configured_fastpath is None:
            # MPS often regresses with heavy indexed writes used by the tensor fastpath.
            self.enable_torch_fastpath = self.device.type != "mps"
        else:
            self.enable_torch_fastpath = bool(configured_fastpath)
        self._tensor_buffers: Dict[int, Dict[str, torch.Tensor]] = {}
        self._tensor_buffer_sizes: Dict[int, int] = {}
        self._tensor_buffer_ptrs: Dict[int, int] = {}
        self._tensor_buffer_capacity = 0
        self._tensor_obs_shape: tuple[int, ...] | None = None
        self._tensor_final_values: Dict[int, torch.Tensor] = {}

    def _record_frame(self, frame_record: Dict[str, Any]) -> None:
        if self.episode_save_dir:
            self.stats.record_frame(frame_record)

    def _get_optimizer(self, agent_id: int) -> torch.optim.Optimizer:
        if self.optimizers:
            if agent_id not in self.optimizers:
                raise KeyError(f"Missing optimizer for agent {agent_id}")
            return self.optimizers[agent_id]
        if self.optimizer is None:
            raise RuntimeError("No optimizer configured for PPOTrainer")
        return self.optimizer

    def _move_policies_to_device(self) -> None:
        for agent in getattr(self, "agents", []):
            policy = getattr(agent, "policy", None)
            if policy is not None:
                policy.to(device=self.device, dtype=self.policy_dtype)
        if self.centralized_critic is not None:
            self.centralized_critic.to(device=self.device, dtype=self.policy_dtype)

    def _autocast_context(self):
        if self._amp_enabled:
            return torch.autocast(device_type="cuda", dtype=self.amp_dtype)
        return contextlib.nullcontext()

    def _optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        loss: torch.Tensor,
        model: torch.nn.Module,
    ) -> None:
        optimizer.zero_grad(set_to_none=True)
        if self.grad_scaler.is_enabled():
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(optimizer)
            if self.max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
            self.grad_scaler.step(optimizer)
            self.grad_scaler.update()
            return
        loss.backward()
        if self.max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
        optimizer.step()

    def _env_metadata(self) -> Dict[str, Any]:
        if self._env_metadata_cache is not None:
            return self._env_metadata_cache
        env = getattr(self, "env", None)
        config = getattr(env, "config", None)
        data: Dict[str, Any] = {}
        if config is not None:
            for attr in (
                "width",
                "height",
                "num_agents",
                "max_steps",
                "spawn_seed_every",
                "seeds_per_spawn",
                "total_seeds_per_episode",
            ):
                if hasattr(config, attr):
                    data[attr] = getattr(config, attr)
        self._env_metadata_cache = data
        return data

    def _format_rewards(self, rewards: Any) -> Any:
        if isinstance(rewards, dict):
            formatted = []
            for agent_id, value in rewards.items():
                try:
                    reward_value = float(value)
                except (TypeError, ValueError):
                    reward_value = 0.0
                formatted.append({"agent_id": agent_id, "reward": reward_value})
            return formatted
        try:
            return float(rewards)
        except (TypeError, ValueError):
            return rewards

    def _build_frame_record(
        self,
        observation: Dict[str, Any],
        actions: Dict[int, int],
        rewards: Any,
        info: Dict[str, Any],
        episode: int,
        step: int,
        done: bool,
    ) -> Dict[str, Any]:
        obs_array = observation.get("obs")
        serialized_obs = obs_array.tolist() if hasattr(obs_array, "tolist") else obs_array
        return {
            "episode": int(episode),
            "step": step,
            "observation": serialized_obs,
            "agents": observation.get("agents", []),
            "actions": [
                {"agent_id": agent_id, "action": action}
                for agent_id, action in sorted(actions.items())
            ],
            "rewards": self._format_rewards(rewards),
            "info": info,
            "done": bool(done),
        }

    def _record_episode_step(
        self,
        *,
        observation: Dict[str, Any],
        actions: Dict[int, int],
        reward_row: Any,
        info: Dict[str, Any],
        episode: int,
        step: int,
        done: bool,
        env_idx: int,
    ) -> None:
        frame_record = self._build_frame_record(
            self._extract_env_observation(observation, env_idx),
            actions,
            self._reward_row_to_dict(reward_row),
            info,
            episode,
            step,
            done,
        )
        self._record_frame(frame_record)

    def _obs_batch_array(self, observation: Dict[str, Any]) -> np.ndarray | torch.Tensor:
        obs_array = observation.get("obs")
        if isinstance(obs_array, torch.Tensor):
            arr = obs_array
        elif isinstance(obs_array, np.ndarray):
            arr = obs_array
        else:
            arr = np.asarray(obs_array)
        if arr.ndim == 3:
            if isinstance(arr, torch.Tensor):
                arr = arr.unsqueeze(0)
            else:
                arr = np.expand_dims(arr, axis=0)
        return arr

    def _prepare_obs_tensor(self, observation: Dict[str, Any]) -> torch.Tensor:
        obs_array = self._obs_batch_array(observation)
        if isinstance(obs_array, torch.Tensor):
            return obs_array.to(self.device, dtype=self.dtype)
        return torch.from_numpy(obs_array).to(self.dtype).to(self.device)

    def _prepare_local_obs_tensor(
        self,
        observation: Dict[str, Any],
        batch_size: int,
    ) -> torch.Tensor:
        local_obs = observation.get("local_obs")
        if local_obs is None:
            global_obs = self._prepare_obs_tensor(observation)
            return global_obs.unsqueeze(1).expand(
                -1,
                self.env.config.num_agents,
                *global_obs.shape[1:],
            )
        if isinstance(local_obs, torch.Tensor):
            local = local_obs.to(self.device, dtype=self.dtype)
        elif isinstance(local_obs, np.ndarray):
            local = torch.from_numpy(local_obs).to(self.device, dtype=self.dtype)
        else:
            local = torch.as_tensor(local_obs, device=self.device, dtype=self.dtype)
        if local.dim() == 4:
            local = local.unsqueeze(0)
        if local.shape[0] != batch_size:
            raise ValueError(
                f"Expected local_obs batch size {batch_size}, received {int(local.shape[0])}"
            )
        return local

    def _critic_value(self, global_observation: torch.Tensor) -> torch.Tensor:
        if self.centralized_critic is None:
            raise RuntimeError("Centralized critic is not configured")
        value = self.centralized_critic(global_observation)
        if isinstance(value, tuple):
            value = value[-1]
        if value.dim() == 1:
            value = value.unsqueeze(-1)
        return value.to(dtype=torch.float32, device=self.device)

    def _batch_size_from_obs(self, observation: Dict[str, Any]) -> int:
        return int(self._obs_batch_array(observation).shape[0])

    def _reward_to_array(self, reward: Any, batch_size: int) -> np.ndarray | torch.Tensor:
        if isinstance(reward, torch.Tensor):
            return reward.to(dtype=torch.float32, device=self.device)
        if isinstance(reward, np.ndarray):
            return reward.astype(np.float32, copy=False)

        reward_array = np.zeros((batch_size, self.env.config.num_agents), dtype=np.float32)

        def _assign(row_idx: int, value: Any) -> None:
            if isinstance(value, dict):
                for agent_id, agent_reward in value.items():
                    reward_array[row_idx, int(agent_id)] = float(agent_reward)
            else:
                reward_array[row_idx, :] = float(value)

        if isinstance(reward, list):
            for row_idx, value in enumerate(reward[:batch_size]):
                _assign(row_idx, value)
        else:
            _assign(0, reward)

        return reward_array

    def _done_to_array(self, done: Any, batch_size: int) -> np.ndarray | torch.Tensor:
        if isinstance(done, torch.Tensor):
            return done.to(dtype=torch.bool, device=self.device)
        if isinstance(done, np.ndarray):
            return done.astype(np.bool_, copy=False)
        if isinstance(done, (list, tuple)):
            return np.asarray(done, dtype=np.bool_)
        done_array = np.zeros(batch_size, dtype=np.bool_)
        done_array[:] = bool(done)
        return done_array

    def _ensure_info_list(self, info: Any, batch_size: int) -> List[Dict[str, Any]]:
        if isinstance(info, list):
            if len(info) == batch_size:
                return info
            if len(info) == 1:
                return info * batch_size
            padded = list(info)
            while len(padded) < batch_size:
                padded.append({})
            return padded[:batch_size]
        if isinstance(info, dict):
            return [dict(info) for _ in range(batch_size)]
        return [{} for _ in range(batch_size)]

    def _extract_info_for_env(self, info: Any, env_idx: int) -> Dict[str, Any]:
        if isinstance(info, list):
            if not info:
                return {}
            idx = min(max(env_idx, 0), len(info) - 1)
            item = info[idx]
            return item if isinstance(item, dict) else {}

        if not isinstance(info, dict):
            return {}

        extracted: Dict[str, Any] = {}
        for key, value in info.items():
            if isinstance(value, torch.Tensor):
                if value.ndim == 0:
                    extracted[key] = value.item()
                elif value.shape[0] > env_idx:
                    item = value[env_idx]
                    extracted[key] = item.item() if item.ndim == 0 else item.detach().cpu().tolist()
                else:
                    extracted[key] = value.detach().cpu().tolist()
                continue
            if isinstance(value, np.ndarray):
                if value.ndim == 0:
                    extracted[key] = value.item()
                elif value.shape[0] > env_idx:
                    item = value[env_idx]
                    extracted[key] = item.item() if np.ndim(item) == 0 else item.tolist()
                else:
                    extracted[key] = value.tolist()
                continue
            if isinstance(value, (list, tuple)):
                if value:
                    idx = min(max(env_idx, 0), len(value) - 1)
                    extracted[key] = value[idx]
                else:
                    extracted[key] = value
                continue
            extracted[key] = value
        return extracted

    def _extract_env_observation(self, observation: Dict[str, Any], env_idx: int) -> Dict[str, Any]:
        obs_array = self._obs_batch_array(observation)
        env_obs = obs_array[env_idx]

        agents_field = observation.get("agents", [])
        env_agents: Any
        if agents_field and isinstance(agents_field[0], dict):
            env_agents = agents_field
        elif agents_field and env_idx < len(agents_field):
            env_agents = agents_field[env_idx]
        else:
            env_agents = []

        done_field = observation.get("done")
        if isinstance(done_field, torch.Tensor):
            env_done = bool(done_field[env_idx].item())
        elif isinstance(done_field, (list, tuple, np.ndarray)):
            env_done = bool(done_field[env_idx])
        else:
            env_done = bool(done_field) if done_field is not None else False

        winner_field = observation.get("winner")
        if isinstance(winner_field, torch.Tensor):
            env_winner = winner_field[env_idx].item()
        elif isinstance(winner_field, (list, tuple, np.ndarray)):
            env_winner = winner_field[env_idx]
        else:
            env_winner = winner_field

        steps_field = observation.get("steps")
        if isinstance(steps_field, torch.Tensor):
            env_steps = int(steps_field[env_idx].item())
        elif isinstance(steps_field, (list, tuple, np.ndarray)):
            env_steps = int(steps_field[env_idx])
        else:
            env_steps = int(steps_field) if steps_field is not None else 0

        local_obs_field = observation.get("local_obs")
        if isinstance(local_obs_field, torch.Tensor):
            env_local_obs = local_obs_field[env_idx]
        elif isinstance(local_obs_field, np.ndarray):
            env_local_obs = local_obs_field[env_idx]
        elif isinstance(local_obs_field, (list, tuple)) and len(local_obs_field) > env_idx:
            env_local_obs = local_obs_field[env_idx]
        else:
            env_local_obs = None

        return {
            "obs": env_obs,
            "local_obs": env_local_obs,
            "agents": env_agents,
            "done": env_done,
            "winner": env_winner,
            "steps": env_steps,
        }

    def _reward_row_to_dict(self, reward_row: Any) -> Dict[int, float]:
        return {
            agent_id: float(reward_row[agent_id]) for agent_id in range(self.env.config.num_agents)
        }

    def _episode_harvested_tiles(self) -> float | None:
        def _harvested_tiles_for_env(env: SimEnv) -> float | None:
            farm_grid = getattr(env, "farm_grid", None)
            if isinstance(farm_grid, np.ndarray):
                return float(np.sum(farm_grid > 0))
            if isinstance(farm_grid, torch.Tensor):
                if farm_grid.ndim == 3:
                    env_totals = (farm_grid > 0).sum(dim=(1, 2)).to(dtype=torch.float32)
                    return float(env_totals.mean().item())
                return float((farm_grid > 0).to(dtype=torch.float32).sum().item())
            agents = getattr(env, "agents", None)
            if agents:
                return float(sum(getattr(agent, "harvested_tiles", 0) for agent in agents))
            return None

        envs = getattr(self.env, "envs", None)
        if isinstance(envs, list):
            totals: List[float] = []
            for sub_env in envs:
                value = _harvested_tiles_for_env(sub_env)
                if value is not None:
                    totals.append(value)
            if totals:
                return float(sum(totals) / len(totals))
            return None
        return _harvested_tiles_for_env(self.env)

    def _reward_to_tensor(self, reward: Any, batch_size: int) -> torch.Tensor:
        reward_array = self._reward_to_array(reward, batch_size)
        if isinstance(reward_array, torch.Tensor):
            return reward_array.to(device=self.device, dtype=torch.float32)
        return torch.as_tensor(reward_array, device=self.device, dtype=torch.float32)

    def _done_to_tensor(self, done: Any, batch_size: int) -> torch.Tensor:
        done_array = self._done_to_array(done, batch_size)
        if isinstance(done_array, torch.Tensor):
            return done_array.to(device=self.device, dtype=torch.bool)
        return torch.as_tensor(done_array, device=self.device, dtype=torch.bool)

    def _active_env_mask(self, observation: Dict[str, Any], batch_size: int) -> torch.Tensor:
        done_field = observation.get("done")
        if done_field is None:
            return torch.ones(batch_size, dtype=torch.bool, device=self.device)
        done_tensor = self._done_to_tensor(done_field, batch_size)
        return ~done_tensor

    def _rollout_horizon_steps(self) -> int:
        num_agents = max(len(getattr(self, "agents", [])), 1)
        per_agent_capacity = max(self.replay_buffer.max_size // num_agents, 1)
        return max(per_agent_capacity // max(self.env_batch_size, 1), 1)

    def _capture_rollout_bootstrap_values(self, observation: Dict[str, Any]) -> None:
        batch_size = self._batch_size_from_obs(observation)
        active_mask = self._active_env_mask(observation, batch_size)
        if not bool(active_mask.any().item()):
            zero_values = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            self._tensor_final_values = {
                agent.agent_id: zero_values.clone() for agent in self.agents
            }
            return

        obs_tensor = self._prepare_obs_tensor(observation)
        local_obs_tensor = self._prepare_local_obs_tensor(observation, batch_size)
        final_values: Dict[int, torch.Tensor] = {}

        with torch.no_grad():
            centralized_values: torch.Tensor | None = None
            if self.use_ctde:
                centralized_values = self._critic_value(obs_tensor).squeeze(-1).detach()

            for agent in self.agents:
                if self.use_ctde and centralized_values is not None:
                    final_values[agent.agent_id] = centralized_values.to(dtype=torch.float32)
                    continue

                actor_obs = local_obs_tensor[:, agent.agent_id] if self.use_ctde else obs_tensor
                with self._autocast_context():
                    _logits, value = agent.policy(actor_obs)
                final_values[agent.agent_id] = value.squeeze(-1).detach().to(dtype=torch.float32)

        self._tensor_final_values = final_values

    def _ensure_tensor_buffers(
        self,
        obs_shape: tuple[int, ...],
        global_obs_shape: tuple[int, ...] | None = None,
    ) -> None:
        if not self.agents:
            return
        num_agents = max(len(self.agents), 1)
        capacity = max(self.replay_buffer.max_size // num_agents, 1)
        needs_reset = (
            self._tensor_buffer_capacity != capacity
            or self._tensor_obs_shape != obs_shape
            or any(agent.agent_id not in self._tensor_buffers for agent in self.agents)
        )
        if not needs_reset:
            return

        self._tensor_buffers = {}
        self._tensor_buffer_sizes = {}
        self._tensor_buffer_ptrs = {}
        self._tensor_buffer_capacity = capacity
        self._tensor_obs_shape = obs_shape

        for agent in self.agents:
            agent_id = agent.agent_id
            self._tensor_buffers[agent_id] = {
                "obs": torch.empty(
                    (capacity, *obs_shape),
                    dtype=self.dtype,
                    device=self.device,
                ),
                "action": torch.empty((capacity,), dtype=torch.int64, device=self.device),
                "log_prob": torch.empty((capacity,), dtype=torch.float32, device=self.device),
                "value": torch.empty((capacity,), dtype=torch.float32, device=self.device),
                "reward": torch.empty((capacity,), dtype=torch.float32, device=self.device),
                "done": torch.empty((capacity,), dtype=torch.bool, device=self.device),
                "valid": torch.empty((capacity,), dtype=torch.bool, device=self.device),
            }
            if self.use_ctde:
                if global_obs_shape is None:
                    raise ValueError("CTDE requires global observation shape for tensor buffers")
                self._tensor_buffers[agent_id]["global_obs"] = torch.empty(
                    (capacity, *global_obs_shape),
                    dtype=self.dtype,
                    device=self.device,
                )
            self._tensor_buffer_sizes[agent_id] = 0
            self._tensor_buffer_ptrs[agent_id] = 0

    def _tensor_buffer_add(
        self,
        agent_id: int,
        *,
        obs: torch.Tensor,
        global_obs: torch.Tensor | None = None,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        buffer = self._tensor_buffers.get(agent_id)
        if buffer is None:
            return
        capacity = self._tensor_buffer_capacity
        batch_size = int(action.shape[0])
        if batch_size <= 0 or capacity <= 0:
            return

        if batch_size > capacity:
            start = batch_size - capacity
            obs = obs[start:]
            action = action[start:]
            log_prob = log_prob[start:]
            value = value[start:]
            reward = reward[start:]
            done = done[start:]
            if global_obs is not None:
                global_obs = global_obs[start:]
            batch_size = capacity

        ptr = self._tensor_buffer_ptrs[agent_id]
        indices = (torch.arange(batch_size, device=self.device) + ptr) % capacity
        buffer["obs"].index_copy_(0, indices, obs[:batch_size])
        buffer["action"].index_copy_(0, indices, action[:batch_size])
        buffer["log_prob"].index_copy_(0, indices, log_prob[:batch_size].to(dtype=torch.float32))
        buffer["value"].index_copy_(0, indices, value[:batch_size].to(dtype=torch.float32))
        buffer["reward"].index_copy_(0, indices, reward[:batch_size].to(dtype=torch.float32))
        buffer["done"].index_copy_(0, indices, done[:batch_size])
        buffer["valid"].index_copy_(0, indices, valid[:batch_size].to(dtype=torch.bool))
        if self.use_ctde:
            if global_obs is None:
                raise ValueError("CTDE requires global_obs when adding tensor buffer data")
            buffer["global_obs"].index_copy_(0, indices, global_obs[:batch_size])

        self._tensor_buffer_ptrs[agent_id] = (ptr + batch_size) % capacity
        self._tensor_buffer_sizes[agent_id] = min(
            capacity,
            self._tensor_buffer_sizes[agent_id] + batch_size,
        )

    def _reset_tensor_buffers(self) -> None:
        self._tensor_final_values = {}
        for agent in self.agents:
            agent_id = agent.agent_id
            if agent_id in self._tensor_buffer_sizes:
                self._tensor_buffer_sizes[agent_id] = 0
            if agent_id in self._tensor_buffer_ptrs:
                self._tensor_buffer_ptrs[agent_id] = 0

    def _update_agent_from_tensor_buffer(self, agent: SimAgent) -> None:
        if agent.policy is None:
            return
        agent.policy.train()
        if self.use_ctde and self.centralized_critic is not None:
            self.centralized_critic.train()

        buffer = self._tensor_buffers.get(agent.agent_id)
        if buffer is None:
            return
        current_size = self._tensor_buffer_sizes.get(agent.agent_id, 0)
        if current_size <= 0:
            return

        capacity = self._tensor_buffer_capacity
        ptr = self._tensor_buffer_ptrs.get(agent.agent_id, 0)
        start_idx = (ptr - current_size) % capacity
        if start_idx + current_size <= capacity:
            ordered_indices = torch.arange(start_idx, start_idx + current_size, device=self.device)
        else:
            first = torch.arange(start_idx, capacity, device=self.device)
            second = torch.arange(0, (start_idx + current_size) % capacity, device=self.device)
            ordered_indices = torch.cat((first, second), dim=0)

        env_count = max(self.env_batch_size, 1)
        usable_count = (int(ordered_indices.shape[0]) // env_count) * env_count
        if usable_count <= 0:
            return
        if usable_count != int(ordered_indices.shape[0]):
            ordered_indices = ordered_indices[-usable_count:]

        actions_all = buffer["action"].index_select(0, ordered_indices)
        old_log_probs_all = (
            buffer["log_prob"].index_select(0, ordered_indices).to(dtype=torch.float32)
        )
        sampled_values_all = (
            buffer["value"].index_select(0, ordered_indices).to(dtype=torch.float32)
        )
        rewards_all = buffer["reward"].index_select(0, ordered_indices).to(dtype=torch.float32)
        dones_all = buffer["done"].index_select(0, ordered_indices)
        valid_all = buffer["valid"].index_select(0, ordered_indices)
        final_values = self._tensor_final_values.get(
            agent.agent_id,
            torch.zeros(env_count, dtype=torch.float32, device=self.device),
        )

        advantages_all = self._compute_vectorized_gae(
            rewards=rewards_all,
            values=sampled_values_all,
            dones=dones_all,
            valid=valid_all,
            final_values=final_values,
            env_count=env_count,
        )
        returns_all = advantages_all + sampled_values_all
        valid_positions = torch.nonzero(valid_all, as_tuple=False).squeeze(-1)
        if valid_positions.numel() == 0:
            return

        policy_advantages_all = advantages_all.clone()
        valid_advantages = policy_advantages_all.index_select(0, valid_positions)
        if self.normalize_advantages and valid_advantages.numel() > 1:
            normalized_advantages = (valid_advantages - valid_advantages.mean()) / (
                valid_advantages.std() + 1e-8
            )
            policy_advantages_all.index_copy_(0, valid_positions, normalized_advantages)

        sample_count = int(valid_positions.numel())

        for epoch in range(self.training_epochs):
            permutation = torch.randperm(sample_count, device=self.device)
            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            updates = 0

            for start in range(0, sample_count, self.batch_size):
                mb_valid_positions = permutation[start : start + self.batch_size]
                if mb_valid_positions.numel() == 0:
                    continue

                mb_positions = valid_positions.index_select(0, mb_valid_positions)
                mb_buffer_indices = ordered_indices.index_select(0, mb_positions)
                observations = buffer["obs"].index_select(0, mb_buffer_indices)
                actions = actions_all.index_select(0, mb_positions)
                old_log_probs = old_log_probs_all.index_select(0, mb_positions)
                policy_advantages = policy_advantages_all.index_select(0, mb_positions)
                returns = returns_all.index_select(0, mb_positions)

                with self._autocast_context():
                    logits, value = agent.policy(observations)
                logits_f32 = logits.to(dtype=torch.float32)
                value_f32 = value.squeeze(-1).to(dtype=torch.float32)
                if not bool(torch.isfinite(logits_f32).all().item()):
                    raise RuntimeError("Non-finite logits detected during PPO update")
                dist = torch.distributions.Categorical(logits=logits_f32)
                log_prob = dist.log_prob(actions)
                ratio = torch.exp(log_prob.to(dtype=torch.float32) - old_log_probs)
                entropy = dist.entropy().mean()

                surr1 = ratio * policy_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    * policy_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()
                if self.use_ctde:
                    value_loss = torch.zeros((), dtype=torch.float32, device=self.device)
                    loss = policy_loss - self.entropy_coef * entropy
                else:
                    value_loss = 0.5 * (returns - value_f32).pow(2).mean()
                    loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy

                optimizer = self._get_optimizer(agent.agent_id)
                self._optimizer_step(optimizer, loss, agent.policy)

                if self.use_ctde:
                    sampled_global_obs = buffer["global_obs"].index_select(0, mb_buffer_indices)
                    if self.centralized_critic_optimizer is None:
                        raise RuntimeError("Missing centralized critic optimizer for CTDE")
                    with self._autocast_context():
                        predicted_values = self._critic_value(sampled_global_obs).squeeze(-1)
                    critic_loss = 0.5 * (returns.detach() - predicted_values).pow(2).mean()
                    self._optimizer_step(
                        self.centralized_critic_optimizer,
                        critic_loss,
                        self.centralized_critic,
                    )
                    value_loss = critic_loss.detach()

                epoch_policy_loss += float(policy_loss.item())
                epoch_value_loss += float(value_loss.item())
                updates += 1

            if updates == 0:
                continue

            avg_policy_loss = epoch_policy_loss / updates
            avg_value_loss = epoch_value_loss / updates
            training_logger.log_epoch(epoch, self.training_epochs, avg_policy_loss, avg_value_loss)
            self.stats.push_agent_losses(agent.agent_id, avg_policy_loss, avg_value_loss)
            self.stats.log_wandb(step=self.stats.steps)

    def _init_logging(self, title: str = "Training"):
        training_logger.header(title)
        if self.config:
            training_logger.config(self.config)

        if self.use_wandb and _WANDB_AVAILABLE:
            training_logger.info("Weights & Biases logging enabled")
            wandb.init(project=self.project_name, name=self.run_name, config=self.config)
            self._wandb_initialized = True
        elif self.use_wandb:
            imported_path = getattr(wandb, "__file__", None) if wandb is not None else None
            suffix = f" (imported module: {imported_path})" if imported_path else ""
            training_logger.warning(
                "Weights & Biases API unavailable. If installed, check for local module shadowing."
                f"{suffix}"
            )

    def _finish_logging(self):
        if self._wandb_initialized and self.use_wandb and _WANDB_AVAILABLE:
            wandb.finish()
            training_logger.success("Wandb run finished")

    # TODO: Looking suspicious, need to check if this is correct
    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        next_value: float,
        dones: List[bool],
    ) -> torch.Tensor:
        """
        Compute Generalized Advantage Estimation for a trajectory.

        Args:
            rewards: List of rewards for each step
            values: List of value estimates for each step
            next_value: Value estimate for the final next state (bootstrap)
            dones: List of done flags for each step

        Returns:
            Tensor of advantages for each step
        """
        gae = 0.0
        advantages = []
        n_steps = len(rewards)

        for step in reversed(range(n_steps)):
            if step == n_steps - 1:
                next_val = next_value
            else:
                next_val = values[step + 1]

            done_mask = 1.0 - float(dones[step])
            delta = rewards[step] + self.gamma * next_val * done_mask - values[step]
            gae = delta + self.gamma * self.gae_lambda * done_mask * gae
            advantages.insert(0, gae)

        return torch.tensor(advantages, dtype=torch.float32)

    def _compute_vectorized_gae(
        self,
        *,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        valid: torch.Tensor,
        final_values: torch.Tensor,
        env_count: int,
    ) -> torch.Tensor:
        total = int(rewards.shape[0])
        if total <= 0:
            return torch.empty(0, dtype=torch.float32, device=self.device)

        usable = (total // env_count) * env_count
        if usable <= 0:
            return torch.empty(0, dtype=torch.float32, device=self.device)

        if usable != total:
            start = total - usable
            rewards = rewards[start:]
            values = values[start:]
            dones = dones[start:]
            valid = valid[start:]

        rewards_seq = rewards.to(dtype=torch.float32).reshape(-1, env_count)
        values_seq = values.to(dtype=torch.float32).reshape(-1, env_count)
        dones_seq = dones.reshape(-1, env_count).to(dtype=torch.bool)
        valid_seq = valid.reshape(-1, env_count).to(dtype=torch.bool)
        bootstrap_values = final_values.to(device=self.device, dtype=torch.float32)
        if bootstrap_values.ndim == 0:
            bootstrap_values = bootstrap_values.repeat(env_count)
        if int(bootstrap_values.shape[0]) != env_count:
            raise ValueError(
                f"Expected {env_count} bootstrap values, received {int(bootstrap_values.shape[0])}"
            )

        next_values = torch.zeros_like(values_seq)
        if values_seq.shape[0] > 1:
            next_values[:-1] = torch.where(
                valid_seq[1:],
                values_seq[1:],
                torch.zeros_like(values_seq[:-1]),
            )
        next_values[-1] = bootstrap_values

        advantages = torch.zeros_like(values_seq, dtype=torch.float32)
        gae = torch.zeros(env_count, dtype=torch.float32, device=self.device)
        for step in range(values_seq.shape[0] - 1, -1, -1):
            valid_step = valid_seq[step].to(dtype=torch.float32)
            non_terminal = valid_step * (~dones_seq[step]).to(dtype=torch.float32)
            delta = (
                rewards_seq[step] + self.gamma * next_values[step] * non_terminal - values_seq[step]
            )
            gae = valid_step * delta + self.gamma * self.gae_lambda * non_terminal * gae
            advantages[step] = gae
        return advantages.reshape(-1)

    def _log_rollout_progress(
        self,
        *,
        step: int,
        batch_envs: int,
        competitive_zero_sum: bool,
        episode_reward: float,
        episode_agent_reward: np.ndarray,
        progress: RolloutProgress,
    ) -> None:
        if (step + 1) % 100 != 0 and step != self.env.config.max_steps - 1:
            return

        active_time = max(
            time.perf_counter() - progress.training_start - progress.paused_time,
            1e-8,
        )
        delta_steps = progress.total_agent_steps - progress.last_logged_steps
        delta_time = max(active_time - progress.last_active_time, 1e-8)
        steps_per_sec = delta_steps / delta_time
        progress.last_active_time = active_time
        progress.last_logged_steps = progress.total_agent_steps

        if competitive_zero_sum:
            reward_per_env = float(episode_agent_reward[0]) / max(batch_envs, 1)
        else:
            reward_per_env = episode_reward / max(batch_envs, 1)
        training_logger.log_step(
            step + 1,
            self.env.config.max_steps,
            {
                "rewards": reward_per_env,
                "steps_per_sec": round(steps_per_sec, 2),
            },
        )

    def _run_fastpath_episode(
        self,
        *,
        episode: int,
        obs: Dict[str, Any],
        record_env_idx: int | None,
        competitive_zero_sum: bool,
        progress: RolloutProgress,
        start_step: int = 0,
        max_segment_steps: int | None = None,
        active_mask: torch.Tensor | None = None,
    ) -> EpisodeSummary:
        episode_reward = torch.zeros((), dtype=torch.float32, device=self.device)
        episode_agent_reward = torch.zeros(
            (self.env.config.num_agents,),
            dtype=torch.float32,
            device=self.device,
        )
        episode_steps = 0
        all_done = False
        remaining_steps = max(self.env.config.max_steps - start_step, 0)
        rollout_steps = (
            remaining_steps
            if max_segment_steps is None
            else min(remaining_steps, max_segment_steps)
        )

        for local_step in range(rollout_steps):
            step = start_step + local_step
            obs_tensor = self._prepare_obs_tensor(obs)
            batch_envs = obs_tensor.shape[0]
            active_mask = self._active_env_mask(obs, batch_envs)
            local_obs_tensor = self._prepare_local_obs_tensor(obs, batch_envs)

            if self.use_ctde:
                self._ensure_tensor_buffers(
                    tuple(local_obs_tensor.shape[2:]),
                    global_obs_shape=tuple(obs_tensor.shape[1:]),
                )
            else:
                self._ensure_tensor_buffers(tuple(obs_tensor.shape[1:]))

            collected_agent_data: Dict[int, Dict[str, torch.Tensor]] = {}
            env_actions = torch.zeros(
                (batch_envs, self.env.config.num_agents),
                dtype=torch.int64,
                device=self.env.device,
            )

            for agent in self.agents:
                agent.policy.eval()

            with torch.no_grad():
                centralized_values: torch.Tensor | None = None
                if self.use_ctde:
                    centralized_values = self._critic_value(obs_tensor).squeeze(-1).detach()
                for agent in self.agents:
                    actor_obs = local_obs_tensor[:, agent.agent_id] if self.use_ctde else obs_tensor
                    with self._autocast_context():
                        logits, value = agent.policy(actor_obs)
                    logits_f32 = logits.to(dtype=torch.float32)
                    if not bool(torch.isfinite(logits_f32).all().item()):
                        raise RuntimeError("Non-finite logits detected during action sampling")
                    dist = torch.distributions.Categorical(logits=logits_f32)
                    action = dist.sample()
                    log_prob = dist.log_prob(action).to(dtype=torch.float32)

                    collected_agent_data[agent.agent_id] = {
                        "action": action.detach(),
                        "log_prob": log_prob.detach(),
                        "value": (
                            centralized_values.to(dtype=torch.float32)
                            if centralized_values is not None
                            else value.squeeze(-1).detach().to(dtype=torch.float32)
                        ),
                    }
                    env_actions[:, agent.agent_id] = action.detach()

            obs, reward, done, info = self.env.step(env_actions)
            reward_tensor = self._reward_to_tensor(reward, batch_envs)
            done_tensor = self._done_to_tensor(done, batch_envs)

            if record_env_idx is not None:
                env_to_record = min(record_env_idx, batch_envs - 1)
                if bool(active_mask[env_to_record].item()):
                    frame_actions = {
                        agent_id: int(agent_data["action"][env_to_record].item())
                        for agent_id, agent_data in collected_agent_data.items()
                    }
                    self._record_episode_step(
                        observation=obs,
                        actions=frame_actions,
                        reward_row=reward_tensor[env_to_record].detach().float().cpu().numpy(),
                        info=self._extract_info_for_env(info, env_to_record),
                        episode=episode + 1,
                        step=step + 1,
                        done=bool(done_tensor[env_to_record].item()),
                        env_idx=env_to_record,
                    )

            obs_batch = obs_tensor.detach()
            done_batch = done_tensor.detach()
            valid_batch = active_mask.detach()
            for agent_id, agent_data in collected_agent_data.items():
                actor_obs_batch = (
                    local_obs_tensor[:, agent_id].detach() if self.use_ctde else obs_batch
                )
                self._tensor_buffer_add(
                    agent_id,
                    obs=actor_obs_batch,
                    global_obs=obs_batch if self.use_ctde else None,
                    action=agent_data["action"],
                    log_prob=agent_data["log_prob"],
                    value=agent_data["value"],
                    reward=reward_tensor[:, agent_id].detach(),
                    done=done_batch,
                    valid=valid_batch,
                )
            active_envs = int(active_mask.sum().item())
            if active_envs > 0:
                self.stats.step(active_envs)

            episode_reward += reward_tensor.sum()
            episode_agent_reward += reward_tensor.sum(dim=0)
            episode_steps = local_step + 1
            progress.total_agent_steps += active_envs * max(len(self.agents), 1)

            self._log_rollout_progress(
                step=step,
                batch_envs=batch_envs,
                competitive_zero_sum=competitive_zero_sum,
                episode_reward=float(episode_reward.item()),
                episode_agent_reward=episode_agent_reward.detach().float().cpu().numpy(),
                progress=progress,
            )

            all_done = bool(done_tensor.all().item())
            if all_done:
                break

        self._capture_rollout_bootstrap_values(obs)
        return EpisodeSummary(
            total_reward=float(episode_reward.item()),
            agent_rewards=episode_agent_reward.detach().float().cpu().numpy(),
            steps=episode_steps,
            next_observation=obs,
            all_done=all_done,
            active_mask=~done_tensor if episode_steps > 0 else active_mask,
        )

    def _run_legacy_episode(
        self,
        *,
        episode: int,
        obs: Dict[str, Any],
        record_env_idx: int | None,
        competitive_zero_sum: bool,
        progress: RolloutProgress,
        start_step: int = 0,
        max_segment_steps: int | None = None,
        active_mask: torch.Tensor | None = None,
    ) -> EpisodeSummary:
        if self.use_ctde:
            raise RuntimeError(
                "CTDE currently requires torch fastpath. Set config['torch_fastpath']=True."
            )

        episode_reward = 0.0
        episode_agent_reward = np.zeros((self.env.config.num_agents,), dtype=np.float64)
        episode_steps = 0
        all_done = False
        remaining_steps = max(self.env.config.max_steps - start_step, 0)
        rollout_steps = (
            remaining_steps
            if max_segment_steps is None
            else min(remaining_steps, max_segment_steps)
        )
        current_active_mask = active_mask

        for local_step in range(rollout_steps):
            step = start_step + local_step
            obs_tensor = self._prepare_obs_tensor(obs)
            batch_envs = obs_tensor.shape[0]
            active_mask = (
                self._active_env_mask(obs, batch_envs)
                if current_active_mask is None
                else current_active_mask
            )
            self._ensure_tensor_buffers(tuple(obs_tensor.shape[1:]))

            actions_per_env: List[Dict[int, int]] | None = None
            action_tensors: List[torch.Tensor] = []
            if not isinstance(self.env, SimEnv):
                actions_per_env = [{} for _ in range(batch_envs)]
            collected_agent_data: Dict[int, Dict[str, torch.Tensor]] = {}

            for agent in self.agents:
                agent.policy.eval()
                with torch.no_grad():
                    with self._autocast_context():
                        logits, value = agent.policy(obs_tensor)
                    logits_f32 = logits.to(dtype=torch.float32)
                    if not bool(torch.isfinite(logits_f32).all().item()):
                        raise RuntimeError("Non-finite logits detected during action sampling")
                    dist = torch.distributions.Categorical(logits=logits_f32)
                    action = dist.sample()
                    log_prob = dist.log_prob(action).to(dtype=torch.float32)

                collected_agent_data[agent.agent_id] = {
                    "action": action,
                    "log_prob": log_prob,
                    "value": value.squeeze(-1).to(dtype=torch.float32),
                }

                if isinstance(self.env, SimEnv):
                    action_tensors.append(action)
                else:
                    action_cpu = action.detach().cpu()
                    for env_idx in range(batch_envs):
                        actions_per_env[env_idx][agent.agent_id] = int(action_cpu[env_idx].item())

            env_actions: Union[Sequence[Dict[int, int]], Dict[int, int], torch.Tensor]
            if isinstance(self.env, SimEnv):
                if action_tensors:
                    env_actions = torch.stack(action_tensors, dim=1)
                else:
                    env_actions = torch.zeros(
                        (batch_envs, self.env.config.num_agents),
                        dtype=torch.int64,
                        device=self.env.device,
                    )
            elif batch_envs == 1:
                env_actions = actions_per_env[0] if actions_per_env else {}
            else:
                env_actions = actions_per_env or [{} for _ in range(batch_envs)]

            obs, reward, done, info = self.env.step(env_actions)

            reward_array = self._reward_to_array(reward, batch_envs)
            done_array = self._done_to_array(done, batch_envs)
            if isinstance(reward_array, torch.Tensor):
                reward_array_cpu = reward_array.detach().cpu().numpy()
            else:
                reward_array_cpu = reward_array
            if isinstance(done_array, torch.Tensor):
                done_array_cpu = done_array.detach().cpu().numpy()
            else:
                done_array_cpu = done_array
            info_list = self._ensure_info_list(info, batch_envs)
            reward_tensor = self._reward_to_tensor(reward, batch_envs)
            done_tensor = self._done_to_tensor(done, batch_envs)

            if record_env_idx is not None:
                env_to_record = min(record_env_idx, batch_envs - 1)
                if bool(active_mask[env_to_record].item()):
                    if isinstance(self.env, SimEnv):
                        frame_actions = {
                            agent.agent_id: int(
                                collected_agent_data[agent.agent_id]["action"][env_to_record].item()
                            )
                            for agent in self.agents
                        }
                    else:
                        frame_actions = actions_per_env[env_to_record] if actions_per_env else {}
                    self._record_episode_step(
                        observation=obs,
                        actions=frame_actions,
                        reward_row=reward_array_cpu[env_to_record],
                        info=info_list[env_to_record],
                        episode=episode + 1,
                        step=step + 1,
                        done=bool(done_array_cpu[env_to_record]),
                        env_idx=env_to_record,
                    )

            obs_batch = obs_tensor.detach()
            valid_batch = active_mask.detach()
            for agent_id, agent_data in collected_agent_data.items():
                self._tensor_buffer_add(
                    agent_id,
                    obs=obs_batch,
                    action=agent_data["action"].detach(),
                    log_prob=agent_data["log_prob"].detach(),
                    value=agent_data["value"].detach(),
                    reward=reward_tensor[:, agent_id].detach(),
                    done=done_tensor.detach(),
                    valid=valid_batch,
                )
            active_envs = int(active_mask.sum().item())
            if active_envs > 0:
                self.stats.step(active_envs)

            episode_reward += float(np.sum(reward_array_cpu))
            episode_agent_reward += reward_array_cpu.sum(axis=0, dtype=np.float64)
            episode_steps = local_step + 1
            progress.total_agent_steps += active_envs * max(len(self.agents), 1)

            self._log_rollout_progress(
                step=step,
                batch_envs=batch_envs,
                competitive_zero_sum=competitive_zero_sum,
                episode_reward=episode_reward,
                episode_agent_reward=episode_agent_reward,
                progress=progress,
            )

            current_active_mask = ~done_tensor
            all_done = bool(np.all(done_array_cpu))
            if all_done:
                break

        self._capture_rollout_bootstrap_values(obs)
        return EpisodeSummary(
            total_reward=episode_reward,
            agent_rewards=episode_agent_reward,
            steps=episode_steps,
            next_observation=obs,
            all_done=all_done,
            active_mask=current_active_mask,
        )

    def _update_agent_from_replay_buffer(self, agent: SimAgent) -> None:
        agent.policy.train()

        for epoch in range(self.training_epochs):
            trajectory = self.replay_buffer.sample_for_agent(agent.agent_id, self.batch_size)
            if not trajectory:
                break

            observations = [exp.observation for exp in trajectory]
            rewards = [
                sum(exp.reward.values()) if isinstance(exp.reward, dict) else exp.reward
                for exp in trajectory
            ]
            values = [exp.value.squeeze().item() for exp in trajectory]
            dones = [
                exp.done if isinstance(exp.done, bool) else bool(exp.done) for exp in trajectory
            ]

            with torch.no_grad():
                _, next_value = agent.policy(observations[-1])
                next_value = next_value.squeeze().item()

            advantages = self.compute_gae(rewards, values, next_value, dones).to(self.device)
            returns = advantages + torch.tensor(values, dtype=torch.float32, device=self.device)
            policy_loss_value = 0.0
            value_loss_value = 0.0

            for i, exp in enumerate(trajectory):
                with self._autocast_context():
                    logits, value = agent.policy(exp.observation)
                logits_f32 = logits.to(dtype=torch.float32)
                if not bool(torch.isfinite(logits_f32).all().item()):
                    raise RuntimeError("Non-finite logits detected during PPO update")
                dist = torch.distributions.Categorical(logits=logits_f32)
                log_prob = dist.log_prob(exp.action)

                ratio = torch.exp(
                    log_prob.to(dtype=torch.float32) - exp.log_prob.to(dtype=torch.float32)
                )

                adv = advantages[i]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = (
                    0.5 * (returns[i] - value.squeeze().to(dtype=torch.float32)).pow(2).mean()
                )

                loss = policy_loss + 0.5 * value_loss
                optimizer = self._get_optimizer(agent.agent_id)
                self._optimizer_step(optimizer, loss, agent.policy)
                policy_loss_value = float(policy_loss.item())
                value_loss_value = float(value_loss.item())

            training_logger.log_epoch(
                epoch,
                self.training_epochs,
                policy_loss_value,
                value_loss_value,
            )
            self.stats.push_agent_losses(
                agent.agent_id,
                policy_loss_value,
                value_loss_value,
            )
            self.stats.log_wandb(step=self.stats.steps)

    def train(
        self,
        env: SimEnv,
        agents: List[SimAgent],
        title: str = "Training",
    ):
        self.env = env
        self.agents = agents
        self._env_metadata_cache = None
        self._move_policies_to_device()

        self._init_logging(title)
        training_logger.success("Environment and policies initialized")
        if isinstance(self.env, SimEnv):
            mode = "enabled" if self.enable_torch_fastpath else "disabled"
            training_logger.info(f"Torch fastpath {mode}")

        training_logger.start_training(self.episodes)
        progress = RolloutProgress(training_start=time.perf_counter())

        for episode in range(self.episodes):
            training_logger.start_episode(episode + 1)
            self.stats.reset_episode()
            # Keep PPO updates on-policy: each episode trains only on fresh rollout data.
            self.replay_buffer.clear()
            self._reset_tensor_buffers()

            obs = self.env.reset()
            self.env_batch_size = self._batch_size_from_obs(obs)
            self.stats.set_env_count(self.env_batch_size)
            record_env_idx = (
                random.randrange(max(self.env_batch_size, 1)) if self.episode_save_dir else None
            )
            use_torch_fastpath = isinstance(self.env, SimEnv) and self.enable_torch_fastpath
            competitive_zero_sum = hasattr(self.env.config, "score_delta_reward")
            rollout_horizon = self._rollout_horizon_steps()
            episode_total_reward = 0.0
            episode_agent_rewards = np.zeros((self.env.config.num_agents,), dtype=np.float64)
            episode_steps = 0
            all_done = False
            current_active_mask: torch.Tensor | None = None

            while episode_steps < self.env.config.max_steps and not all_done:
                self._reset_tensor_buffers()
                if use_torch_fastpath:
                    segment_summary = self._run_fastpath_episode(
                        episode=episode,
                        obs=obs,
                        record_env_idx=record_env_idx,
                        competitive_zero_sum=competitive_zero_sum,
                        progress=progress,
                        start_step=episode_steps,
                        max_segment_steps=rollout_horizon,
                    )
                else:
                    segment_summary = self._run_legacy_episode(
                        episode=episode,
                        obs=obs,
                        record_env_idx=record_env_idx,
                        competitive_zero_sum=competitive_zero_sum,
                        progress=progress,
                        start_step=episode_steps,
                        max_segment_steps=rollout_horizon,
                        active_mask=current_active_mask,
                    )

                if segment_summary.steps <= 0:
                    break

                # Clear the step progress line before training logs
                print()

                for agent in self.agents:
                    self._update_agent_from_tensor_buffer(agent)

                episode_total_reward += segment_summary.total_reward
                episode_agent_rewards += segment_summary.agent_rewards
                episode_steps += segment_summary.steps
                obs = segment_summary.next_observation or obs
                all_done = segment_summary.all_done
                current_active_mask = segment_summary.active_mask

            episode_summary = EpisodeSummary(
                total_reward=episode_total_reward,
                agent_rewards=episode_agent_rewards,
                steps=episode_steps,
                next_observation=obs,
                all_done=all_done,
                active_mask=current_active_mask,
            )

            env_count = max(self.env_batch_size, 1)
            if competitive_zero_sum:
                episode_reward_per_env = float(episode_summary.agent_rewards[0]) / env_count
            else:
                episode_reward_per_env = episode_summary.total_reward / env_count
            avg_reward = episode_reward_per_env / max(episode_summary.steps, 1)
            training_logger.end_episode(
                episode + 1,
                total_reward=episode_reward_per_env,
                avg_reward=avg_reward,
                steps=episode_summary.steps,
            )

            if competitive_zero_sum:
                self.stats.push_reward(
                    float(episode_summary.agent_rewards[0]),
                    env_count=self.env_batch_size,
                )
            else:
                self.stats.push_reward(episode_summary.total_reward, env_count=self.env_batch_size)
            self.stats.push_episode_metrics(
                steps=episode_summary.steps,
                harvested_tiles=self._episode_harvested_tiles(),
            )
            self.stats.log_wandb(step=self.stats.steps)

            pause_start = time.perf_counter()
            if self.episode_save_dir:
                serializable_config = {
                    key: (str(value) if isinstance(value, torch.dtype) else value)
                    for key, value in self.config.items()
                }
                metadata = {
                    "env_config": self._env_metadata(),
                    "training_config": serializable_config,
                }
                output_path = self.stats.dump_episode_recording(
                    self.episode_save_dir,
                    episode + 1,
                    metadata=metadata,
                )
                training_logger.info(f"Saved episode metrics to {output_path}")

            self.save_checkpoint(f"checkpoints/ppo_checkpoint_{episode}.pth")
            progress.paused_time += time.perf_counter() - pause_start

        training_logger.finish(
            {
                "avg_episode_reward": sum(self.stats.episode_rewards)
                / max(len(self.stats.episode_rewards), 1),
                "final_policy_loss": self.stats.policy_losses[-1]
                if self.stats.policy_losses
                else 0.0,
                "final_value_loss": self.stats.value_losses[-1] if self.stats.value_losses else 0.0,
            }
        )

        self._finish_logging()
