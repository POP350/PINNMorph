"""Trainer integration for progressive trials and residual-driven time coverage."""
from copy import deepcopy

import torch

from .adaptive_strategy import advance_time_frontier, blend_control, control_snapshot, trial_transition
from .trial_control import expiry_decision, physical_observation
from .rollback_whitelist import is_rollback_whitelisted


class AdaptiveOnlineMixin:
    def _adaptive_config(self):
        controller = self.online_morph_controller
        return dict(controller.config.get("adaptive_strategy") or {}) if controller else {}

    def _adaptive_trial_pending(self):
        controller = self.online_morph_controller
        return bool(controller and
                    (self._adaptive_config().get("progressive_enabled") or
                     controller.config.get("block_new_morph_until_persistent_feedback", False)) and
                    (controller.active_trial or controller.pending_persistent_feedback))

    def _initialize_adaptive_trial(self, applied):
        if is_rollback_whitelisted(applied):
            return
        if not self._adaptive_config().get("progressive_enabled"):
            return
        trial = self.online_morph_controller.active_trial
        snapshot = self._pending_online_morph_snapshot
        if not trial or not snapshot:
            raise RuntimeError("Progressive trial requires its pre-intervention snapshot")
        baseline = control_snapshot(snapshot["loss_state"])
        target = control_snapshot(self.built.loss_function.state_dict())
        # Bind every additive module in a composed program to the same intervention.
        old_count = len(snapshot.get("online_morph_architecture") or [])
        self.morphable_model.bind_trial_modules(trial["intervention_id"], old_count)
        trial["adaptive_trial"] = {"influence": 0.25, "target_influence": 0.25, "status": "probation",
            "before": baseline, "target": target, "ramp_start_step": int(self._actual_iterations),
            "ramp_start_influence": 0.25, "ramp_steps": 100 if trial["operator"] not in {"OBJECTIVE_REBALANCING", "GRADIENT_SURGERY", "GRADIENT_CONFLICT_MITIGATION", "REGION_AWARE_REBALANCING"} else 50}
        self._ramp_adaptive_trial(trial, immediate=True)
        applied["adaptive_trial"] = {"status": "probation", "initial_influence": 0.25,
            "confirmation_horizon": "persistent", "checkpoint_eligible": False}
        self._append_morph_event_record({"event": "ADAPTIVE_TRIAL_STARTED", "step": self._actual_iterations,
            "intervention_id": trial["intervention_id"], **applied["adaptive_trial"]})

    def _ramp_adaptive_trial(self, trial, immediate=False):
        if is_rollback_whitelisted(trial):
            return
        state = trial.get("adaptive_trial")
        if not state:
            return
        elapsed = max(0, self._actual_iterations - state["ramp_start_step"])
        fraction = 1.0 if immediate else min(1.0, elapsed / state["ramp_steps"])
        influence = state["ramp_start_influence"] + fraction * (state["target_influence"] - state["ramp_start_influence"])
        if not immediate and abs(influence - state["influence"]) < 1e-12:
            return
        state["influence"] = influence
        self.morphable_model.set_intervention_influence(trial["intervention_id"], influence)
        loss = self.built.loss_function
        current = loss.state_dict()
        current.update(blend_control(state["before"], state["target"], influence))
        loss.load_state_dict(current)

    def _update_adaptive_trial(self, trial, feedback, horizon):
        if is_rollback_whitelisted(trial):
            return None
        state = trial.get("adaptive_trial")
        if not state:
            return None
        target, status = trial_transition(state["influence"], feedback, horizon)
        layered = self._adaptive_config().get("trial_control", {})
        if layered.get("enabled") and horizon != "persistent" and target == 0.0 and physical_observation(feedback, layered.get("safety_degradation", 0.1)).safe:
            target, status = max(0.0625, state["influence"] * 0.5), "attenuated_pending_physical_confirmation"
        if horizon == "persistent" and not (feedback.get("credit_eligible") and feedback.get("accepted")):
            target, status = 0.0, "expired_without_long_term_confirmation"
        state.update(target_influence=target, status=status, ramp_start_step=int(self._actual_iterations),
                     ramp_start_influence=state["influence"])
        event = {"event": "ADAPTIVE_TRIAL_TRANSITION", "intervention_id": trial["intervention_id"],
            "step": self._actual_iterations, "horizon": horizon, "status": status,
            "influence_before": state["influence"], "target_influence": target}
        self._append_morph_event_record(event)
        return event

    def _adaptive_expiry_decision(self, trial, feedback):
        if is_rollback_whitelisted(trial):
            return "confirm"
        config = self._adaptive_config().get("trial_control", {})
        if not config.get("enabled"):
            return "confirm" if feedback.get("credit_eligible") and feedback.get("accepted") else "rollback"
        model_reversible = self.morphable_model.trial_is_reversible(trial["intervention_id"])
        # Point replacement, optimizer phase changes and direct topology cannot be blended away.
        program = trial.get("adaptation_program", {})
        excluded = {"ADAPT_SAMPLING_DISTRIBUTION", "ENTER_LBFGS_PHASE", "WIDEN", "ADD_LAYER", "PRUNE", "REGROW", "REDISTRIBUTE", "LOCAL_REINIT"}
        names = {op.get("primitive") for op in program.get("operations", [])}
        reversible = model_reversible and not names.intersection(excluded) and program.get("action") not in {"SAMPLING_ADAPTATION", "OPTIMIZER_PHASE_ADAPTATION"}
        remaining = int(self._adaptive_config().get("freeze_step", 14000)) - self._actual_iterations
        return expiry_decision(feedback, reversible=reversible,
            extension_used=trial["adaptive_trial"].get("extension_used", False), remaining_steps=remaining, config=config)

    def _finish_adaptive_trial(self, trial, feedback, persistent_event):
        if persistent_event.get("duplicate_ignored"):
            return
        if is_rollback_whitelisted(trial):
            trial.pop("adaptive_trial", None)
            persistent_event["rollback_allowed"] = False
            persistent_event["adaptive_trial"] = {
                "status": "retained_by_whitelist",
                "step": self._actual_iterations,
                "intervention_id": trial.get("intervention_id"),
            }
            return True
        if persistent_event.get("confounded_by_intervention_ids"):
            raise RuntimeError("Progressive trials require isolated persistent feedback")
        state = trial["adaptive_trial"]
        config = self._adaptive_config().get("trial_control", {})
        if state.get("status") == "retiring":
            outcome = "retired" if physical_observation(feedback, config.get("safety_degradation", 0.1)).safe else "rollback"
        else:
            outcome = self._adaptive_expiry_decision(trial, feedback)
        if outcome in {"extend", "retire"}:
            duration = int(config.get("extension_steps", 400) if outcome == "extend" else config.get("retirement_steps", 100))
            state.update(status="extended_observation" if outcome == "extend" else "retiring",
                extension_used=True, observation_end_step=int(self._actual_iterations) + duration,
                target_influence=state["influence"] if outcome == "extend" else 0.0,
                ramp_start_step=int(self._actual_iterations), ramp_start_influence=state["influence"], ramp_steps=duration)
            persistent_event["adaptive_trial"] = {"status": state["status"], "observation_end_step": state["observation_end_step"],
                "policy_credit_assigned": False, "influence": state["influence"]}
            self._append_morph_event_record({"event": "ADAPTIVE_OBSERVATION_EXTENDED" if outcome == "extend" else "ADAPTIVE_RETIREMENT_STARTED",
                "step": self._actual_iterations, "intervention_id": trial["intervention_id"], **persistent_event["adaptive_trial"]})
            return False
        transition = {"status": "confirmed" if outcome == "confirm" else "retired_without_rewind" if outcome == "retired" else "rollback",
            "step": self._actual_iterations, "intervention_id": trial["intervention_id"], "target_influence": state["influence"] if outcome == "confirm" else 0.0}
        if outcome == "confirm":
            # Keep the successfully tested influence. Future trials may build on it;
            # do not introduce a previously untested jump at confirmation.
            state = trial["adaptive_trial"]
            state["target_influence"] = state["influence"]
            transition["target_influence"] = state["influence"]
            self._ramp_adaptive_trial(trial, immediate=True)
        elif outcome == "retired":
            state["target_influence"] = 0.0
            self._ramp_adaptive_trial(trial, immediate=True)
            self.morphable_model.retire_trial_modules(trial["intervention_id"])
            self.adaptive_control_summary["online_morph_soft_retirements"] = int(self.adaptive_control_summary.get("online_morph_soft_retirements", 0)) + 1
        else:
            snapshot = self._pending_online_morph_snapshot
            if snapshot is None:
                raise RuntimeError("Unconfirmed trial lost its rollback snapshot")
            # Consumed optimizer steps remain consumed; only the training state is restored.
            self._restore_online_morph_snapshot(snapshot, restore_controller=False)
            persistent_event["adaptive_rollback"] = True
            self.adaptive_control_summary["online_morph_trial_rollbacks"] = int(
                self.adaptive_control_summary.get("online_morph_trial_rollbacks", 0)) + 1
            self.adaptive_control_summary["rollbacks"] = int(
                self.adaptive_control_summary.get("rollbacks", 0)) + 1
            self.adaptive_compute_budget_ledger.rollback_count += 1
        persistent_event["adaptive_trial"] = transition
        for event in reversed(self.online_morph_controller.events):
            if event.get("event") == "HEURISTIC_PERSISTENT_FEEDBACK" and event.get("intervention_id") == trial["intervention_id"]:
                event.update(deepcopy(persistent_event))
                break
        self._pending_online_morph_snapshot = None
        self._active_morph_result = None
        return True

    def _guard_adaptive_influence(self):
        """Bound the actual output, gradient and governing-residual increment of a branch."""
        config = self._adaptive_config().get("influence_guard", {})
        controller = self.online_morph_controller
        if not config.get("enabled") or not controller or not self.morphable_model:
            return
        trials = ([controller.active_trial] if controller.active_trial else []) + controller.pending_persistent_feedback
        points = self.validation_probe_manager.probe_set.batch.domain_samples
        if points is None:
            return
        points = points[:int(config.get("points", 32))]
        loss = self.built.loss_function
        for trial in trials:
            if is_rollback_whitelisted(trial):
                continue
            state = trial.get("adaptive_trial")
            if not state or state.get("influence", 0) == 0 or not self.morphable_model.trial_is_reversible(trial["intervention_id"]):
                continue
            architecture = self.morphable_model.architecture_state()
            if not any(s.get("intervention_id") == trial["intervention_id"] and not s.get("direct_topology") for s in architecture):
                continue
            def evaluate(alpha):
                self.morphable_model.set_intervention_influence(trial["intervention_id"], alpha)
                x = points.detach().clone().requires_grad_(True)
                y = self.built.model(x)
                gradient = torch.autograd.grad(y.sum(), x)[0].detach()
                residual = loss.evaluate_registered_pde_residuals(self.built.model, points)
                return {"output": y.detach().cpu(), "gradient": gradient.cpu(), **{f"pde:{k}": v for k, v in residual.items()}}
            alpha = state["influence"]
            was_training = self.built.model.training
            self.built.model.eval()
            evaluations = 0
            try:
                off = evaluate(0.0)
                evaluations += 1
                cap = alpha
                for attempt in range(7):
                    on = evaluate(cap)
                    evaluations += 1
                    ratios = {}
                    for key in on:
                        delta = float((on[key] - off[key]).square().mean().sqrt())
                        scale = max(float(off[key].square().mean().sqrt()), float(config.get("scale_floor", 0.1)))
                        limit = config.get("maximum_gradient_ratio", 0.5) if key == "gradient" else config.get("maximum_output_ratio", 0.25) if key == "output" else config.get("maximum_residual_ratio", 0.5)
                        ratios[key] = delta / (scale * limit)
                    finite = bool(torch.isfinite(torch.tensor(list(ratios.values()))).all())
                    if finite and max(ratios.values()) <= 1.0:
                        break
                    # Explicitly test the reduced influence for nonlinear PDEs.
                    cap = 0.0 if not finite or attempt == 5 else cap * min(0.5, 1.0 / max(ratios.values()))
                if cap < alpha:
                    state.update(influence=cap, target_influence=min(cap, state["target_influence"]),
                        ramp_start_influence=cap, ramp_start_step=int(self._actual_iterations))
                    self._ramp_adaptive_trial(trial, immediate=True)
                    self._append_morph_event_record({"event": "ADAPTIVE_INFLUENCE_CAPPED", "step": self._actual_iterations,
                        "intervention_id": trial["intervention_id"], "before": alpha, "after": cap, "physical_increment_ratios": ratios})
            finally:
                self.morphable_model.set_intervention_influence(trial["intervention_id"], state["influence"])
                self.built.model.train(was_training)
            self.adaptive_compute_budget_ledger.physics_probe_evaluations += evaluations
            self.adaptive_compute_budget_ledger.physics_probe_points_evaluated += evaluations * len(points)

    def _initial_health(self, state, problem, batch):
        samples = batch.constraint_samples
        if not samples:
            state["initial_errors"] = {}
            return
        residuals = problem.compute_constraint_residuals(self.built.model, samples, create_graph=False)
        errors, boundaries = {}, []
        for name, sample in samples.items():
            if name not in residuals:
                continue
            mse = float(residuals[name].detach().square().mean().cpu())
            if sample.category == "initial":
                target = sample.pointset_values
                # Nonzero known targets define their own scale; zero velocity uses
                # a declared physical RMS tolerance, never a near-zero denominator.
                scale = max(float(target.square().mean().sqrt().cpu()) if target is not None else 0.0, 0.1)
                errors[name] = (mse ** 0.5) / scale
            elif sample.category == "boundary":
                boundaries.append(mse)
        state.update(initial_errors=errors, boundary_scores=boundaries)
        if not boundaries:  # Exact periodic representation has no separate spatial BC loss.
            state["boundary_scores"] = [0.0]
        clear = getattr(problem, "clear_autodiff_cache", None)
        if callable(clear):
            clear()

    def _observe_directional_frequency(self):
        controller = self.online_morph_controller
        if not controller or not controller.config.get("fourier_axis_frequencies"):
            return
        from .frequency_observer import residual_slice_spectra
        spectra, count = residual_slice_spectra(self.built.loss_function, self.built.model)
        self.adaptive_control_summary["residual_frequency_axes"] = spectra
        self.adaptive_compute_budget_ledger.physics_probe_evaluations += 1
        self.adaptive_compute_budget_ledger.physics_probe_points_evaluated += count

    def _observe_adaptive_time_progress(self):
        config = dict(self._adaptive_config().get("temporal") or {})
        if not config.get("enabled"):
            return
        loss = self.built.loss_function
        problem = loss.problem
        spec = problem.get_spec()
        if spec.problem_id not in config.get("problem_ids", ["kuramoto_sivashinsky_1d", "wave_2d_heterogeneous"]):
            return
        if "t" not in spec.input_variables:
            raise ValueError("Online temporal control requires an explicit time axis")
        manager = self.validation_probe_manager
        if manager is None or manager.probe_set is None:
            return
        points = manager.probe_set.batch.domain_samples
        if points is None or len(points) == 0:
            return
        state = loss._online_time_progress
        state.setdefault("axis", list(spec.input_variables).index("t"))
        state.setdefault("bounds", list(spec.domain.bounds["t"]))
        state.setdefault("bins", int(config.get("bins", 4)))
        state.setdefault("future_weight", float(config.get("future_weight", 0.2)))
        state.setdefault("level", 1)
        state.setdefault("last_advance", 0)
        component_ids = tuple(c.component_id for c in self.registry.loss_components if c.category == "equation_residual")
        scores = loss.compute_pointwise_component_scores(self.built.model, component_ids, points)
        if not scores:
            return
        values = torch.stack([scores[k].detach() for k in sorted(scores)]).mean(dim=0)
        start, end = state["bounds"]
        index = (((points[:, state["axis"]] - start) / (end - start)) * state["bins"]).long().clamp(0, state["bins"] - 1)
        bins = [float(values[index == i].mean().cpu()) if (index == i).any() else None for i in range(state["bins"])]
        if config.get("health_gate"):
            self._initial_health(state, problem, manager.probe_set.batch)
            frontier = int(state["level"]) / state["bins"]
            unit = (points[:, state["axis"]] - start) / (end - start)
            junction = (unit - frontier).abs() <= 0.5 / state["bins"]
            state["junction_score"] = float(values[junction].mean().cpu()) if junction.any() else None
        event = advance_time_frontier(state, bins, int(self._actual_iterations), blocked=self._adaptive_trial_pending(), config=config)
        self.adaptive_compute_budget_ledger.physics_probe_evaluations += 1
        self.adaptive_compute_budget_ledger.physics_probe_points_evaluated += len(points)
        if event:
            self._append_morph_event_record(event)
        self.adaptive_control_summary["online_time_progress"] = deepcopy(state)
