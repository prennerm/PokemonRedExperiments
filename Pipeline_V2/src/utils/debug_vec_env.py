"""Debug versions of SubprocVecEnv with worker-level logging."""

from __future__ import annotations

import multiprocessing as mp
from typing import Any, Callable, Optional

import time
import gymnasium as gym
import numpy as np
import pickle
from gymnasium import spaces

from stable_baselines3.common.vec_env.base_vec_env import (
    CloudpickleWrapper,
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)
from stable_baselines3.common.vec_env.patch_gym import _patch_env


def _debug_worker(  # noqa: C901
    remote: mp.connection.Connection,
    parent_remote: mp.connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
) -> None:
    from stable_baselines3.common.env_util import is_wrapped

    parent_remote.close()
    env = _patch_env(env_fn_wrapper.var())
    reset_info: Optional[dict[str, Any]] = {}
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                observation, reward, terminated, truncated, info = env.step(data)
                done = terminated or truncated
                info["TimeLimit.truncated"] = truncated and not terminated

                if done:
                    info["terminal_observation"] = observation
                    observation, reset_info = env.reset()

                debug_env = getattr(env, "debug_reset_timing", False) and hasattr(env, "_log_reset_timing")
                if debug_env:
                    step_count = getattr(env, 'step_count', -1)
                    reset_id = getattr(env, 'reset_count', -1)
                else:
                    step_count = -1
                    reset_id = -1

                if debug_env:
                    env._log_reset_timing(  # type: ignore[attr-defined]
                        f"worker_send_start reset={reset_id} step_count={step_count} done={int(done)}"
                    )

                if debug_env and 0 <= step_count <= 512:
                    payload_bytes = len(pickle.dumps((observation, reward, done, info, reset_info), protocol=pickle.HIGHEST_PROTOCOL))
                    env._log_reset_timing(  # type: ignore[attr-defined]
                        f"worker_payload reset={reset_id} step_count={step_count} bytes={payload_bytes}"
                    )

                remote.send((observation, reward, done, info, reset_info))

                if debug_env:
                    env._log_reset_timing(  # type: ignore[attr-defined]
                        f"worker_send_end reset={reset_id} step_count={step_count}"
                    )
            elif cmd == "reset":
                maybe_options = {"options": data[1]} if data[1] else {}
                observation, reset_info = env.reset(seed=data[0], **maybe_options)
                remote.send((observation, reset_info))
            elif cmd == "render":
                remote.send(env.render())
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "env_method":
                method = env.get_wrapper_attr(data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(env.get_wrapper_attr(data))
            elif cmd == "has_attr":
                try:
                    env.get_wrapper_attr(data)
                    remote.send(True)
                except AttributeError:
                    remote.send(False)
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))  # type: ignore[func-returns-value]
            elif cmd == "is_wrapped":
                remote.send(is_wrapped(env, data))
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break
        except KeyboardInterrupt:
            break


class DebugSubprocVecEnv(VecEnv):
    """SubprocVecEnv variant that logs worker send timings when debug_reset_timing is enabled."""

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
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            process = ctx.Process(target=_debug_worker, args=args, daemon=True)  # type: ignore[attr-defined]
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()

        super().__init__(len(env_fns), observation_space, action_space)

        # Determine if underlying envs expose debug logging hook
        try:
            self.remotes[0].send(("get_attr", "debug_reset_timing"))
            self._debug_logging = bool(self.remotes[0].recv())
        except Exception:
            self._debug_logging = False

    def step_async(self, actions: np.ndarray) -> None:
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        results = []
        for idx, remote in enumerate(self.remotes):
            if self._debug_logging:
                print(f"[DebugSubprocVecEnv] recv_start idx={idx}")
            start_ts = time.perf_counter() if self._debug_logging else None
            result = remote.recv()
            if self._debug_logging and start_ts is not None:
                duration = time.perf_counter() - start_ts
                print(f"[DebugSubprocVecEnv] recv_end idx={idx} duration={duration:.6f}s")
            elif self._debug_logging:
                print(f"[DebugSubprocVecEnv] recv_end idx={idx}")
            results.append(result)
        self.waiting = False
        obs, rews, dones, infos, self.reset_infos = zip(*results)  # type: ignore[assignment]
        return _stack_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos  # type: ignore[return-value]

    def reset(self) -> VecEnvObs:
        for env_idx, remote in enumerate(self.remotes):
            remote.send(("reset", (self._seeds[env_idx], self._options[env_idx])))
        results = [remote.recv() for remote in self.remotes]
        obs, self.reset_infos = zip(*results)  # type: ignore[assignment]
        self._reset_seeds()
        self._reset_options()
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
        self.closed = True

    def get_images(self) -> list[Optional[np.ndarray]]:
        if self.render_mode != "rgb_array":
            return [None for _ in self.remotes]
        for pipe in self.remotes:
            pipe.send(("render", None))
        outputs = [pipe.recv() for pipe in self.remotes]
        return outputs

    def has_attr(self, attr_name: str) -> bool:
        target_remotes = self._get_target_remotes(indices=None)
        for remote in target_remotes:
            remote.send(("has_attr", attr_name))
        return all(remote.recv() for remote in target_remotes)

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

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> list[Any]:
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("env_method", (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

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


def _stack_obs(obs: tuple[Any, ...], space: spaces.Space) -> VecEnvObs:
    if isinstance(space, spaces.Dict):
        return {key: np.stack([o[key] for o in obs]) for key in space.spaces.keys()}
    return np.stack(obs)
