"""Separate measured safety, observed improvement, and causal policy credit."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PhysicalObservation:
    complete: bool
    safe: bool
    improved: bool
    maximum_degradation: float


def physical_observation(feedback, maximum_degradation=0.1):
    probes = [feedback.get(k) for k in
              ("raw_before_after_telemetry", "raw_audit_before_after_telemetry")]
    complete = all(isinstance(p, dict) and
                   math.isfinite(float(p.get("max_component_degradation", math.inf))) and
                   math.isfinite(float(p.get("aggregate_relative_improvement", -math.inf)))
                   for p in probes)
    if not complete:
        return PhysicalObservation(False, False, False, math.inf)
    worst = max(float(p["max_component_degradation"]) for p in probes)
    safe = worst <= maximum_degradation
    return PhysicalObservation(True, safe, safe and all(
        float(p["aggregate_relative_improvement"]) >= 0.01 for p in probes), worst)


def expiry_decision(feedback, *, reversible, extension_used, remaining_steps, config):
    """Uncertain observations never create credit or increase action influence."""
    observation = physical_observation(feedback, config.get("safety_degradation", 0.1))
    if not observation.safe or feedback.get("confounded"):
        return "rollback"
    if feedback.get("credit_eligible") and feedback.get("accepted"):
        return "confirm"
    extension = int(config.get("extension_steps", 400))
    retirement = int(config.get("retirement_steps", 100))
    if (not feedback.get("credit_eligible") and observation.improved and
            not extension_used and remaining_steps >= extension + retirement):
        return "extend"
    if reversible and remaining_steps >= retirement:
        return "retire"
    return "rollback"
