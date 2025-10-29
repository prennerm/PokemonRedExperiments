"""Shared-memory vectorized environment inspired by SB3's ShmemVecEnv."""

from __future__ import annotations

import multiprocessing as mp
import time
from multiprocessing import shared_memory
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import (
    CloudpickleWrapper,
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)
from stable_baselines3.common.vec_env.patch_gym import _patch_env

from utils import decode_packed_structure


def _flatten_space(space: spaces.Space) -> Iterable[Tuple[str, spaces.Space]]:
    if isinstance(space, spaces.Dict):
        return space.spaces.items()
    return [("observation", space)]


def _shared_worker(  # noqa: C901
    remote: mp.connection.Connection,
    parent_remote: mp.connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
) -> None:
    parent_remote.close()
    env = _patch_env(env_fn_wrapper.var())
    shared_arrays: Dict[str, np.ndarray] = {}
    shared_handles: Dict[str, shared_memory.SharedMemory] = {}
    terminal_arrays: Dict[str, np.ndarray] = {}
    terminal_handles: Dict[str, shared_memory.SharedMemory] = {}
    terminal_version: int = 0
    reset_info: Optional[dict[str, Any]] = {}
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "configure_shared_memory":
                shared_arrays.clear()
                shared_handles.clear()
                terminal_arrays.clear()
                terminal_handles.clear()
                terminal_version = 0
                specs = data if isinstance(data, dict) else {"obs": data, "terminal": []}
                for key, name, shape, dtype_str in specs.get("obs", []):
                    shm = shared_memory.SharedMemory(name=name)
                    shared_handles[key] = shm
                    shared_arrays[key] = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
                for key, name, shape, dtype_str in specs.get("terminal", []):
                    shm = shared_memory.SharedMemory(name=name)
                    terminal_handles[key] = shm
                    terminal_arrays[key] = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
                remote.send(None)
            elif cmd == "step":
                debug_env = getattr(env, "debug_reset_timing", False) and hasattr(env, "_log_reset_timing")
                step_start_ts = time.perf_counter() if debug_env else None
                observation, reward, terminated, truncated, info = env.step(data)
                if debug_env:
                    elapsed = time.perf_counter() - step_start_ts
                    if elapsed >= 0.2:
                        try:
                            env._log_reset_timing(  # type: ignore[attr-defined]
                                "worker_step_elapsed "
                                f"reset={getattr(env, 'reset_count', -1)} "
                                f"step_count={getattr(env, 'step_count', -1)} "
                                f"dt={elapsed:.6f} done={int(terminated or truncated)}"
                            )
                        except Exception:
                            pass
                info["TimeLimit.truncated"] = truncated and not terminated
                done = terminated or truncated
                _write_observation(observation, shared_arrays)
                if done:
                    terminal_version += 1
                    info["terminal_obs_ref"] = {
                        "version": terminal_version,
                    }
                    _write_observation(observation, terminal_arrays)
                    observation, reset_info = env.reset()
                    _write_observation(observation, shared_arrays)
                remote.send((reward, terminated, truncated, info, reset_info))
            elif cmd == "reset":
                maybe_options = {"options": data[1]} if data[1] else {}
                observation, reset_info = env.reset(seed=data[0], **maybe_options)
                _write_observation(observation, shared_arrays)
                remote.send((None, reset_info))
            elif cmd == "env_method":
                method = env.get_wrapper_attr(data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(env.get_wrapper_attr(data))
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))  # type: ignore[func-returns-value]
            elif cmd == "is_wrapped":
                from stable_baselines3.common.env_util import is_wrapped

                remote.send(is_wrapped(env, data))
            elif cmd == "close":
                env.close()
                for shm in shared_handles.values():
                    shm.close()
                for shm in terminal_handles.values():
                    shm.close()
                remote.close()
                break
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break
        except KeyboardInterrupt:
            break


def _write_observation(observation: Any, shared_arrays: Dict[str, np.ndarray]) -> None:
    if isinstance(observation, dict):
        for key, value in observation.items():
            np.copyto(shared_arrays[key], value)
    else:
        np.copyto(shared_arrays["observation"], observation)


class SharedMemoryVecEnv(VecEnv):
    """VecEnv implementation that uses shared memory buffers for observations."""

    def __init__(self, env_fns: list[Callable[[], gym.Env]], start_method: Optional[str] = None):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)

        if start_method is None:
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            process = ctx.Process(target=_shared_worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)), daemon=True)  # type: ignore[attr-defined]
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()
        super().__init__(len(env_fns), observation_space, action_space)

        self._buffers: list[Dict[str, Tuple[shared_memory.SharedMemory, np.ndarray]]] = []
        self._terminal_buffers: list[Dict[str, Tuple[shared_memory.SharedMemory, np.ndarray]]] = []
        self._terminal_versions: list[int] = []
        self._allocate_shared_buffers()

    def _allocate_shared_buffers(self) -> None:
        shared_specs_per_env = []
        terminal_specs_per_env = []
        for _ in range(self.num_envs):
            env_buffers: Dict[str, Tuple[shared_memory.SharedMemory, np.ndarray]] = {}
            terminal_buffers: Dict[str, Tuple[shared_memory.SharedMemory, np.ndarray]] = {}
            specs = []
            terminal_specs = []
            for key, space in _flatten_space(self.observation_space):
                dtype = space.dtype
                shape = space.shape
                size = int(np.prod(shape)) * np.dtype(dtype).itemsize
                shm = shared_memory.SharedMemory(create=True, size=size)
                array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
                env_buffers[key] = (shm, array)
                specs.append((key, shm.name, shape, dtype.str))
                term_shm = shared_memory.SharedMemory(create=True, size=size)
                term_array = np.ndarray(shape, dtype=dtype, buffer=term_shm.buf)
                terminal_buffers[key] = (term_shm, term_array)
                terminal_specs.append((key, term_shm.name, shape, dtype.str))
            self._buffers.append(env_buffers)
            self._terminal_buffers.append(terminal_buffers)
            shared_specs_per_env.append(specs)
            terminal_specs_per_env.append(terminal_specs)
            self._terminal_versions.append(0)
        for remote, specs, term_specs in zip(self.remotes, shared_specs_per_env, terminal_specs_per_env):
            remote.send(("configure_shared_memory", {"obs": specs, "terminal": term_specs}))
            remote.recv()

    def step_async(self, actions: np.ndarray) -> None:
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        rewards, terminated, truncated, infos, self.reset_infos = zip(*results)  # type: ignore[assignment]
        infos_list = list(infos)
        for idx, info in enumerate(infos_list):
            ref = info.pop("terminal_obs_ref", None)
            if ref:
                version = int(ref.get("version", 0))
                info["terminal_observation"] = self._gather_terminal_observation(idx, version)
        obs = self._gather_observations()
        dones = np.array(terminated, dtype=bool) | np.array(truncated, dtype=bool)
        rewards_arr = np.array(rewards)
        return _stack_obs(obs, self.observation_space), rewards_arr, dones, infos_list

    def reset(self) -> VecEnvObs:
        for env_idx, remote in enumerate(self.remotes):
            remote.send(("reset", (self._seeds[env_idx], self._options[env_idx])))
        results = [remote.recv() for remote in self.remotes]
        _, self.reset_infos = zip(*results)  # type: ignore[assignment]
        self._reset_seeds()
        self._reset_options()
        obs = self._gather_observations()
        return _stack_obs(obs, self.observation_space)

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        for env_buffers, term_buffers in zip(self._buffers, self._terminal_buffers):
            for shm, _ in env_buffers.values():
                shm.close()
                shm.unlink()
            for term_shm, _ in term_buffers.values():
                term_shm.close()
                term_shm.unlink()
        self.closed = True

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> list[Any]:
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("env_method", (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("get_attr", attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("set_attr", (attr_name, value)))
        for remote in target_remotes:
            remote.recv()

    def env_is_wrapped(self, wrapper_class: type[gym.Wrapper], indices: VecEnvIndices = None) -> list[bool]:
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("is_wrapped", wrapper_class))
        return [remote.recv() for remote in target_remotes]

    def _get_target_remotes(self, indices: VecEnvIndices) -> tuple[mp.connection.Connection, ...]:
        if indices is None:
            return self.remotes
        if isinstance(indices, (list, tuple)):
            return tuple(self.remotes[i] for i in indices)
        if isinstance(indices, np.ndarray):
            indices = indices.tolist()
            return tuple(self.remotes[i] for i in indices)
        return (self.remotes[indices],)

    def _gather_observations(self) -> list[Dict[str, np.ndarray]]:
        obs_batch = []
        for env_buffers in self._buffers:
            obs_dict: Dict[str, np.ndarray] = {}
            for key, (_, array) in env_buffers.items():
                obs_dict[key] = np.array(array, copy=True)
            obs_batch.append(obs_dict)
        return obs_batch

    def _gather_terminal_observation(self, env_idx: int, version: int) -> Dict[str, Any]:
        if version < self._terminal_versions[env_idx]:
            version = self._terminal_versions[env_idx]
        else:
            self._terminal_versions[env_idx] = version

        terminal_buffers = self._terminal_buffers[env_idx]
        terminal_obs: Dict[str, Any] = {}
        for key, (_, array) in terminal_buffers.items():
            terminal_obs[key] = np.array(array, copy=True)
        return decode_packed_structure(terminal_obs)

def _stack_obs(obs: list[Dict[str, np.ndarray]], space: spaces.Space) -> VecEnvObs:
    if isinstance(space, spaces.Dict):
        return {key: np.stack([o[key] for o in obs]) for key in space.spaces.keys()}
    return np.stack([o["observation"] for o in obs])
