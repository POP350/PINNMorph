"""Reference-free model checkpoint selected by a configured control metric."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any

import torch


class BestTrainLossCheckpointManager:
    """Track the best label-free training or independent-physics score."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = dict(config or {})
        self.enabled = bool(config.get("enabled", False))
        self.final_model_policy = str(config.get("final_model_policy", "last"))
        self.selection_metric = str(
            config.get(
                "selection_metric",
                "physics_audit_score"
                if self.final_model_policy == "best_physics_validation"
                else "training_loss",
            )
        )
        self.evaluation_interval = max(1, int(config.get("evaluation_interval", 100)))
        self.best_score: float | None = None
        self.best_model_state: dict[str, torch.Tensor] | None = None
        self.best_architecture_state: list[dict[str, Any]] | None = None
        self.best_metadata: dict[str, Any] | None = None

    def consider(
        self,
        *,
        model: torch.nn.Module,
        training_loss: float | None = None,
        physics_validation_score: float | None = None,
        iteration: int,
        stage_id: str,
        algorithm_spec_id: str,
        directory: str | Path | None = None,
        score_context: str | None = None,
    ) -> bool:
        """Update on every strict decrease of the configured label-free metric."""

        if self.final_model_policy == "best_train_loss":
            raw_score = training_loss
            expected_metric = "training_loss"
        elif self.final_model_policy == "best_physics_validation":
            raw_score = physics_validation_score
            expected_metric = "physics_audit_score"
        else:
            return False
        if raw_score is None or self.selection_metric != expected_metric:
            return False
        selection_score = float(raw_score)
        context_changed = (score_context is not None and
                           (self.best_metadata or {}).get("score_context") != score_context)
        if (
            not self.enabled
            or not math.isfinite(selection_score)
            or (not context_changed and self.best_score is not None and selection_score >= self.best_score)
        ):
            return False
        self.best_score = selection_score
        self.best_model_state = {
            name: value.detach().clone().cpu()
            for name, value in model.state_dict().items()
        }
        architecture_state = getattr(model, "architecture_state", None)
        self.best_architecture_state = (
            deepcopy(list(architecture_state()))
            if callable(architecture_state)
            else None
        )
        self.best_metadata = {
            "iteration": int(iteration),
            "stage": str(stage_id),
            "selection_metric": self.selection_metric,
            "selection_score": selection_score,
            "score_context": score_context,
            "training_loss": (
                float(training_loss) if training_loss is not None else None
            ),
            "physics_validation_score": (
                float(physics_validation_score)
                if physics_validation_score is not None
                else None
            ),
            "algorithm_spec_id": str(algorithm_spec_id),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if directory is not None:
            self.save_checkpoint(directory)
        return True

    def save_checkpoint(self, directory: str | Path) -> Path | None:
        """Persist the currently accepted in-memory checkpoint state.

        This deliberately writes from the manager rather than from a live
        model.  A physics guard can therefore restore an earlier manager
        snapshot and atomically replace a checkpoint that was recorded during
        a subsequently rejected optimizer excursion.
        """

        if self.best_model_state is None or self.best_metadata is None:
            return None
        path = Path(directory) / f"best_{self.selection_metric}_checkpoint.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.best_model_state,
                "online_morph_architecture": self.best_architecture_state,
                "metadata": self.best_metadata,
            },
            path,
        )
        return path

    def restore_best_model(self, model: torch.nn.Module) -> bool:
        if self.best_model_state is None:
            return False
        restore_architecture = getattr(model, "restore_architecture", None)
        if self.best_architecture_state is not None:
            if not callable(restore_architecture):
                raise RuntimeError(
                    "Best checkpoint contains dynamic architecture state but the model "
                    "cannot restore it"
                )
            restore_architecture(self.best_architecture_state)
        device = next(model.parameters()).device
        model.load_state_dict(
            {name: value.to(device) for name, value in self.best_model_state.items()}
        )
        return True

    def state_dict(self) -> dict[str, Any]:
        return {
            "best_score": self.best_score,
            "best_model_state": deepcopy(self.best_model_state),
            "best_architecture_state": deepcopy(self.best_architecture_state),
            "best_metadata": deepcopy(self.best_metadata),
        }

    def load_state_dict(self, state: dict[str, Any] | None) -> None:
        state = dict(state or {})
        score = state.get("best_score")
        self.best_score = float(score) if score is not None else None
        self.best_model_state = deepcopy(state.get("best_model_state"))
        self.best_architecture_state = deepcopy(
            state.get("best_architecture_state")
        )
        self.best_metadata = deepcopy(state.get("best_metadata"))
