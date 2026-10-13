"""Identity-safe delayed physics feedback for online PINN interventions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from .adaptation_program import AdaptationProgram, SAMPLING_ACTIONS
from .rollback_whitelist import is_rollback_whitelisted


FEEDBACK_HORIZONS = {"short": 100, "medium": 500, "persistent": 1000}


@dataclass
class InterventionRecord:
    intervention_id: str
    policy_id: str
    policy_version: int
    state_id: str
    program: Mapping[str, Any]
    start_step: int
    short_due_step: int | None
    medium_due_step: int | None
    persistent_due_step: int | None
    short_feedback: Mapping[str, Any] | None = None
    medium_feedback: Mapping[str, Any] | None = None
    persistent_feedback: Mapping[str, Any] | None = None
    feedback_exempt: bool = False
    rollback_allowed: bool = True
    executed_primitives: tuple[str, ...] = ()

    def feedback_complete(self) -> bool:
        return self.feedback_exempt or self.persistent_feedback is not None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InterventionTracker:
    """Bind every delayed result to an explicit intervention identity.

    Direct credit is returned only with the selected policy id.  The complete
    state/program/outcome event is also exposed as shared evidence.
    """

    horizons: Mapping[str, int] = field(default_factory=lambda: dict(FEEDBACK_HORIZONS))
    records: dict[str, InterventionRecord] = field(default_factory=dict)
    shared_history: list[dict[str, Any]] = field(default_factory=list)
    _sequence: int = 0

    def register(
        self,
        *,
        policy_id: str,
        policy_version: int,
        state_id: str,
        program: AdaptationProgram,
        start_step: int,
        state_before: Mapping[str, Any],
        executed_primitives: tuple[str, ...] = (),
    ) -> InterventionRecord:
        if not program.has_intervention():
            raise ValueError("KEEP programs do not create interventions")
        self._sequence += 1
        intervention_id = f"intervention-{int(start_step):08d}-{self._sequence:05d}"
        feedback_exempt = program.action in SAMPLING_ACTIONS
        record = InterventionRecord(
            intervention_id=intervention_id,
            policy_id=str(policy_id),
            policy_version=int(policy_version),
            state_id=str(state_id),
            program=program.to_dict(),
            start_step=int(start_step),
            short_due_step=None if feedback_exempt else int(start_step) + int(self.horizons["short"]),
            medium_due_step=None if feedback_exempt else int(start_step) + int(self.horizons["medium"]),
            persistent_due_step=None if feedback_exempt else int(start_step) + int(self.horizons["persistent"]),
            feedback_exempt=feedback_exempt,
            rollback_allowed=not is_rollback_whitelisted({
                "program": program.to_dict(), "executed_primitives": executed_primitives,
            }),
            executed_primitives=executed_primitives,
        )
        self.records[intervention_id] = record
        self.shared_history.append(
            {
                "intervention_id": intervention_id,
                "policy_id": str(policy_id),
                "policy_version": int(policy_version),
                "state_id": str(state_id),
                "state_before": deepcopy(dict(state_before)),
                "program": program.to_dict(),
                "start_step": int(start_step),
                "feedback": {},
                "feedback_exempt": feedback_exempt,
                "rollback_allowed": record.rollback_allowed,
            }
        )
        return record

    def due(self, step: int) -> list[tuple[str, str]]:
        result: list[tuple[str, str]] = []
        current = int(step)
        for record in self.records.values():
            if record.feedback_exempt:
                continue
            for horizon in ("short", "medium", "persistent"):
                if getattr(record, f"{horizon}_feedback") is not None:
                    continue
                if current >= int(getattr(record, f"{horizon}_due_step")):
                    result.append((record.intervention_id, horizon))
        return result

    def record_feedback(
        self,
        *,
        intervention_id: str,
        horizon: str,
        feedback: Mapping[str, Any],
    ) -> dict[str, Any]:
        if intervention_id not in self.records:
            raise KeyError(f"Unknown intervention_id: {intervention_id}")
        if horizon not in FEEDBACK_HORIZONS:
            raise ValueError(f"Unknown feedback horizon: {horizon}")
        record = self.records[intervention_id]
        if record.feedback_exempt:
            return {
                "intervention_id": intervention_id,
                "policy_id": record.policy_id,
                "policy_version": record.policy_version,
                "state_id": record.state_id,
                "horizon": horizon,
                "feedback": {"feedback_status": "whitelisted", "credit_eligible": False},
                "recorded": False,
                "feedback_exempt": True,
            }
        attribute = f"{horizon}_feedback"
        existing = getattr(record, attribute)
        if existing is not None:
            return {
                "intervention_id": intervention_id,
                "policy_id": record.policy_id,
                "policy_version": record.policy_version,
                "state_id": record.state_id,
                "horizon": horizon,
                "feedback": deepcopy(dict(existing)),
                "recorded": False,
                "duplicate_ignored": True,
            }
        payload = deepcopy(dict(feedback))
        setattr(record, attribute, payload)
        shared = next(
            item for item in self.shared_history
            if item["intervention_id"] == intervention_id
        )
        shared["feedback"][horizon] = deepcopy(payload)
        return {
            "intervention_id": intervention_id,
            "policy_id": record.policy_id,
            "policy_version": record.policy_version,
            "state_id": record.state_id,
            "horizon": horizon,
            "feedback": payload,
            "recorded": True,
            "duplicate_ignored": False,
        }

    def evolution_ready(self, last_consumed_sequence: int) -> bool:
        """Evolution is unlocked only by new +500/+1000 real evidence."""

        for index, item in enumerate(self.shared_history, start=1):
            if index <= int(last_consumed_sequence):
                continue
            if item.get("feedback_exempt", False):
                continue
            feedback = dict(item.get("feedback") or {})
            if "medium" in feedback or "persistent" in feedback:
                return True
        return False

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "horizons": dict(self.horizons),
            "records": {key: item.to_dict() for key, item in self.records.items()},
            "shared_history": deepcopy(self.shared_history),
            "sequence": int(self._sequence),
        }

    def load_state_dict(self, payload: Mapping[str, Any] | None) -> None:
        if not payload:
            return
        self.horizons = {
            key: int(value)
            for key, value in dict(payload.get("horizons") or FEEDBACK_HORIZONS).items()
        }
        self.records = {
            str(key): InterventionRecord(**dict(value))
            for key, value in dict(payload.get("records") or {}).items()
        }
        self.shared_history = deepcopy(list(payload.get("shared_history") or []))
        for record in self.records.values():
            record.rollback_allowed = not is_rollback_whitelisted({
                "program": record.program, "executed_primitives": record.executed_primitives,
            })
            # Previously saved sampling trials must not resume delayed checks.
            if AdaptationProgram.from_dict(record.program).action in SAMPLING_ACTIONS:
                record.feedback_exempt = True
                record.short_due_step = None
                record.medium_due_step = None
                record.persistent_due_step = None
        for item in self.shared_history:
            record = self.records.get(str(item.get("intervention_id") or ""))
            if record is not None and record.feedback_exempt:
                item["feedback_exempt"] = True
            if record is not None:
                item["rollback_allowed"] = record.rollback_allowed
        self._sequence = int(payload.get("sequence", len(self.records)))


__all__ = ["FEEDBACK_HORIZONS", "InterventionRecord", "InterventionTracker"]
