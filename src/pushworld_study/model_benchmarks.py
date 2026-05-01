from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Literal

import numpy as np
import torch
from torch import nn

from pushworld_study.envs import ObservationMode, make_pushworld_env
from pushworld_study.models import PushWorldCNN


BenchmarkMode = Literal["infer", "train"]


@dataclass(frozen=True)
class ModelBenchmarkResult:
    label: str
    mode: BenchmarkMode
    device: str
    batch_size: int
    iterations: int
    elapsed_seconds: float
    steps_per_second: float
    milliseconds_per_step: float


class PushWorldPolicyHead(nn.Module):
    def __init__(self, feature_extractor: PushWorldCNN, action_dim: int = 4) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.policy = nn.Linear(feature_extractor.features_dim, action_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(observations)
        return self.policy(features)


def _synchronize_if_needed(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _make_batch(
    puzzle_path: str | Path | None,
    observation_mode: ObservationMode,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    env = make_pushworld_env(
        puzzle_path=puzzle_path,
        observation_mode=observation_mode,
        channel_first=observation_mode == "rgb",
        uint8_observation=observation_mode == "rgb",
    )
    observations = []
    for seed in range(batch_size):
        observation, _ = env.reset(seed=seed)
        observations.append(observation)
    env.close()

    batch = torch.as_tensor(np.stack(observations, axis=0), device=device)
    if observation_mode == "rgb":
        return batch.float() / 255.0
    return batch.float()


def _build_model(
    puzzle_path: str | Path | None,
    observation_mode: ObservationMode,
    device: str,
    features_dim: int,
) -> nn.Module:
    env = make_pushworld_env(
        puzzle_path=puzzle_path,
        observation_mode=observation_mode,
        channel_first=observation_mode == "rgb",
        uint8_observation=observation_mode == "rgb",
    )
    try:
        model = PushWorldPolicyHead(
            PushWorldCNN(env.observation_space, features_dim=features_dim),
            action_dim=env.action_space.n,
        )
    finally:
        env.close()
    return model.to(device)


def _time_inference(
    model: nn.Module,
    batch: torch.Tensor,
    iterations: int,
    warmup: int,
    device: str,
) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(batch)
        _synchronize_if_needed(device)
        started_at = time.perf_counter()
        for _ in range(iterations):
            model(batch)
        _synchronize_if_needed(device)
    elapsed_seconds = time.perf_counter() - started_at
    return elapsed_seconds, iterations / elapsed_seconds


def _time_training(
    model: nn.Module,
    batch: torch.Tensor,
    iterations: int,
    warmup: int,
    device: str,
) -> tuple[float, float]:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    targets = torch.zeros(batch.shape[0], dtype=torch.long, device=device)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()
    _synchronize_if_needed(device)

    started_at = time.perf_counter()
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()
    _synchronize_if_needed(device)
    elapsed_seconds = time.perf_counter() - started_at
    return elapsed_seconds, iterations / elapsed_seconds


def benchmark_model_compile(
    puzzle_path: str | Path | None = None,
    observation_mode: ObservationMode = "planes",
    device: str = "cpu",
    batch_size: int = 256,
    iterations: int = 200,
    warmup: int = 30,
    features_dim: int = 256,
    compile_mode: str = "default",
    fullgraph: bool = False,
) -> dict[str, str | int | float]:
    batch = _make_batch(
        puzzle_path=puzzle_path,
        observation_mode=observation_mode,
        batch_size=batch_size,
        device=device,
    )

    base_model = _build_model(
        puzzle_path=puzzle_path,
        observation_mode=observation_mode,
        device=device,
        features_dim=features_dim,
    )
    compiled_model = _build_model(
        puzzle_path=puzzle_path,
        observation_mode=observation_mode,
        device=device,
        features_dim=features_dim,
    )
    compiled_model.load_state_dict(base_model.state_dict())

    infer_elapsed_eager, infer_sps_eager = _time_inference(
        base_model, batch, iterations, warmup, device
    )
    train_elapsed_eager, train_sps_eager = _time_training(
        base_model, batch, iterations, warmup, device
    )

    try:
        compiled_model = torch.compile(
            compiled_model,
            mode=compile_mode,
            fullgraph=fullgraph,
        )
        infer_elapsed_compiled, infer_sps_compiled = _time_inference(
            compiled_model, batch, iterations, warmup, device
        )
        train_elapsed_compiled, train_sps_compiled = _time_training(
            compiled_model, batch, iterations, warmup, device
        )
        compile_success = 1
        compile_error = ""
    except Exception as exc:
        infer_elapsed_compiled = None
        infer_sps_compiled = None
        train_elapsed_compiled = None
        train_sps_compiled = None
        compile_success = 0
        compile_error = f"{type(exc).__name__}: {exc}"

    return {
        "puzzle_path": str(puzzle_path) if puzzle_path is not None else "default",
        "observation_mode": observation_mode,
        "device": device,
        "batch_size": batch_size,
        "iterations": iterations,
        "warmup": warmup,
        "features_dim": features_dim,
        "compile_mode": compile_mode,
        "fullgraph": int(fullgraph),
        "compile_success": compile_success,
        "compile_error": compile_error,
        "observation_shape": tuple(batch.shape[1:]),
        "eager_infer_steps_per_second": infer_sps_eager,
        "compiled_infer_steps_per_second": infer_sps_compiled,
        "infer_speedup": None if infer_sps_compiled is None else infer_sps_compiled / infer_sps_eager,
        "eager_infer_ms_per_step": infer_elapsed_eager * 1000.0 / iterations,
        "compiled_infer_ms_per_step": None
        if infer_elapsed_compiled is None
        else infer_elapsed_compiled * 1000.0 / iterations,
        "eager_train_steps_per_second": train_sps_eager,
        "compiled_train_steps_per_second": train_sps_compiled,
        "train_speedup": None if train_sps_compiled is None else train_sps_compiled / train_sps_eager,
        "eager_train_ms_per_step": train_elapsed_eager * 1000.0 / iterations,
        "compiled_train_ms_per_step": None
        if train_elapsed_compiled is None
        else train_elapsed_compiled * 1000.0 / iterations,
    }
