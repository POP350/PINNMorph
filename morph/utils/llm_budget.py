"""Shared, resumable per-run call/time limits for both online LLM roles."""
import math
import time


class LLMBudgetExceeded(RuntimeError):
    pass


class BudgetedProvider:
    def __init__(self, provider, state, config):
        self.provider, self.state, self.config = provider, state, config

    def ensure_available(self):
        reason = budget_reason(self.state, self.config)
        if reason:
            raise LLMBudgetExceeded(reason)

    def complete_text(self, *args, **kwargs):
        self.ensure_available()
        self.state["calls"] = self.state.get("calls", 0) + 1
        started = time.perf_counter()
        try:
            return self.provider.complete_text(*args, **kwargs)
        finally:
            self.state["wall_seconds"] = self.state.get("wall_seconds", 0.0) + time.perf_counter() - started


def budget_reason(state, config):
    if not config.get("enabled"):
        return None
    call_limit = int(config.get("maximum_calls", 0))
    if call_limit > 0 and state.get("calls", 0) >= call_limit:
        return "llm_call_budget_exhausted"
    wall_limit = float(config.get("maximum_wall_seconds", 900))
    if wall_limit > 0 and state.get("wall_seconds", 0.0) >= wall_limit:
        return "llm_time_budget_exhausted"
    return None


def decision_signature(state, generation):
    """Changes in bottleneck evidence or population invalidate a previous decision."""
    return (generation, tuple(state.get("online_adaptation", {}).get("supported_hypotheses", ())),
            tuple(bool(state.get(group, {}).get(key)) for group, key in (
                ("gradient", "persistent_gradient_conflict"),
                ("physics", "initial_condition_dominant"),
                ("physics", "boundary_condition_dominant"),
                ("spatial", "localized_residual_concentration"))),
            state.get("pde", {}).get("online_time_progress", {}).get("level"))


def unchanged_decision(state, generation, previous, config):
    if not config.get("enabled") or not previous:
        return False
    if list(decision_signature(state, generation)) != list(previous.get("signature", [])):
        # JSON / checkpoints may contain lists instead of tuples.
        import json
        if json.dumps(decision_signature(state, generation)) != json.dumps(previous.get("signature")):
            return False
    if state["step"] - previous["step"] >= config.get("refresh_steps", 600):
        return False
    old = previous.get("loss")
    new = state.get("optimization", {}).get("current_total_loss")
    if old is None or new is None or not math.isfinite(new):
        return False
    return abs(new - old) / max(abs(old), 1e-12) < config.get("relative_change", 0.2)
