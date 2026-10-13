"""PDE-independent runtime contract for fixed-budget sampling mutation."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
import math
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import torch


FIXED_PDE_LOCAL_FRACTION = 0.70
FIXED_PDE_GLOBAL_FRACTION = 0.30


def sampling_policy_fractions(
    parameters: Mapping[str, Any], *, default_local_fraction: float = FIXED_PDE_LOCAL_FRACTION
) -> tuple[float, float]:
    """Resolve LLM allocation and legacy aliases without imposing a ratio."""
    local = parameters.get("local_fraction", parameters.get("replacement_fraction"))
    global_ = parameters.get("global_fraction", parameters.get("exploration_fraction"))
    if isinstance(local, bool) or isinstance(global_, bool):
        raise ValueError("Sampling fractions must be numbers, not booleans")
    if local is None and global_ is None:
        local = default_local_fraction
    if local is None:
        local = float(Decimal(1) - Decimal(str(global_)))
    if global_ is None:
        global_ = float(Decimal(1) - Decimal(str(local)))
    local, global_ = float(local), float(global_)
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in (local, global_)):
        raise ValueError("Sampling fractions must be finite numbers in [0,1]")
    if not math.isclose(local + global_, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("Local and global sampling fractions must sum to 1")
    for alias, expected in (("replacement_fraction", local), ("exploration_fraction", global_)):
        if alias in parameters and not math.isclose(float(parameters[alias]), expected, abs_tol=1e-6):
            raise ValueError(f"Conflicting sampling fraction alias: {alias}")
    return local, float(Decimal(1) - Decimal(str(local)))


def fixed_pde_cloud_counts(
    point_count: int, local_fraction: float = FIXED_PDE_LOCAL_FRACTION
) -> tuple[int, int]:
    """Keep the point budget exact under the chosen allocation; round local down."""

    count = int(point_count)
    if count <= 0:
        raise ValueError("PDE point count must be positive")
    fraction = float(local_fraction)
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("Local sampling fraction must be in [0,1]")
    local_count = int(Decimal(str(fraction)) * count)
    return local_count, count - local_count


@dataclass(frozen=True)
class SamplingBatch:
    sampler_id: str
    points: Any = field(repr=False)
    point_metadata: tuple[Mapping[str, Any], ...] = ()
    state_version: int = 0

    @property
    def point_count(self) -> int:
        shape = getattr(self.points, "shape", ())
        return int(shape[0]) if shape else 0


@dataclass(frozen=True)
class PointScoreResult:
    sampler_id: str
    scores: torch.Tensor = field(repr=False)
    component_scores: Mapping[str, torch.Tensor] = field(default_factory=dict, repr=False)
    statistics: Mapping[str, float] = field(default_factory=dict)
    scoring_component_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SamplingMutationResult:
    sampler_id: str
    point_count_before: int
    point_count_after: int
    points_replaced: int
    exploration_points: int
    state_version_before: int
    state_version_after: int


@runtime_checkable
class SamplingRuntimeProtocol(Protocol):
    def get_training_points(self, sampler_id: str) -> SamplingBatch: ...

    def score_points(
        self,
        model: torch.nn.Module,
        component_ids: tuple[str, ...],
        points: SamplingBatch,
        context: Any,
    ) -> PointScoreResult: ...

    def generate_candidate_points(
        self, sampler_id: str, count: int, rng_state: Any | None = None
    ) -> SamplingBatch: ...

    def replace_subset(
        self,
        sampler_id: str,
        retained_indices: Sequence[int],
        replacement_points: SamplingBatch,
        **context: Any,
    ) -> SamplingMutationResult: ...

    def snapshot_state(self) -> Mapping[str, Any]: ...

    def restore_state(self, state: Mapping[str, Any]) -> None: ...
