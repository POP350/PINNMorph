"""The single AlgorithmSpec transport contract used by the default pipeline.

The top-level structure is fixed. Independent legal field choices and their
parameter spaces live in :mod:`morph.pipeline.d_algorithm_generation.specs.option_registry`; complete
algorithm combinations are composed dynamically and are never pre-enumerated.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


REQUIRED_TOP_LEVEL_FIELDS = (
    "spec_version",
    "network",
    "sampling",
    "loss",
    "optimization",
    "training",
    "generation_metadata",
)


def algorithm_spec_template() -> dict[str, Any]:
    """Return a fresh complete example/template, never a shared mutable value."""

    return deepcopy(
        {
            "spec_version": "1.1",
            "network": {
                "architecture": "mlp",
                "hidden_layers": [128, 128, 128, 128],
                "activation": {"name": "tanh", "parameters": {}},
                "output_activation": {"name": "identity", "parameters": {}},
                "initialization": {"name": "xavier_normal", "parameters": {}},
                "input_transform": {"name": "none", "parameters": {}},
                "residual_connections": {"enabled": False, "parameters": {}},
                "extra_parameters": {},
            },
            "sampling": {
                "interior": {"strategy": "uniform", "parameters": {"n_points": 5000}},
                "boundary": {"strategy": "uniform", "parameters": {"n_points": 256}},
                "initial": {"strategy": "uniform", "parameters": {"n_points": 256}},
                "adaptive_refinement": {
                    "enabled": False,
                    "strategy": "residual_adaptive",
                    "parameters": {},
                },
            },
            "constraint_enforcement": {
                "method": "soft_penalty",
                "transform_id": "none",
                "parameters": {},
            },
            "loss": {
                "terms": [
                    {
                        "name": "pde_residual",
                        "loss_function": "mse",
                        "weight": 1.0,
                        "parameters": {},
                    },
                    {
                        "name": "boundary_condition",
                        "loss_function": "mse",
                        "weight": 10.0,
                        "parameters": {},
                    },
                    {
                        "name": "initial_condition",
                        "loss_function": "mse",
                        "weight": 10.0,
                        "parameters": {},
                    },
                ],
                "weighting_strategy": {"name": "fixed", "parameters": {}},
                "aggregation": {"name": "weighted_sum", "parameters": {}},
            },
            "optimization": {
                "phases": [
                    {
                        "optimizer": "adam",
                        "iterations": 3000,
                        "parameters": {"learning_rate": 0.001},
                        "scheduler": {"name": "none", "parameters": {}},
                    },
                    {
                        "optimizer": "lbfgs",
                        "iterations": 500,
                        "parameters": {
                            "learning_rate": 1.0,
                            "history_size": 100,
                            "line_search_fn": "strong_wolfe",
                        },
                        "scheduler": {"name": "none", "parameters": {}},
                    },
                ]
            },
            "training": {
                "batch_mode": "full_batch",
                "batch_size": None,
                "gradient_clipping": {"enabled": False, "parameters": {}},
                "early_stopping": {"enabled": False, "parameters": {}},
                "time_strategy": {"name": "none", "parameters": {}},
                "initial_state_pretraining": {
                    "enabled": False,
                    "parameters": {},
                },
                "extra_parameters": {
                    "adaptive_control": {
                        "enabled": False,
                        "freeze_lbfgs_objective": True,
                        "lbfgs_stall_recovery": {"enabled": False, "action": "disabled"},
                        "optimizer_phase_switch": {
                            "enabled": True,
                            "latest_lbfgs_entry_step": 14000,
                        },
                        "sampling_control": {"enabled": False, "strategy": "none"},
                        "curriculum_control": {"enabled": False, "strategy": "none"},
                        "online_morphing": {
                            "enabled": False,
                            "decision_mode": "deterministic",
                            "policy_set": {
                                "enabled": True,
                                "population_size": 5,
                                "scoring": {
                                    "temperature": 0.0,
                                    "keep_margin": 0.05,
                                },
                                "evolution": {
                                    "enabled": False,
                                    "offspring_per_generation": 5,
                                    "maximum_attempts_per_mode": 5,
                                    "maximum_transient_attempts_per_mode": 2,
                                    "transient_retry_initial_seconds": 1.0,
                                    "transient_retry_max_seconds": 4.0,
                                    "minimum_valid_offspring": 5,
                                },
                            },
                            "allowed_primitives": [
                                "WIDEN",
                                "ADD_LAYER",
                                "PRUNE",
                                "REGROW",
                                "REDISTRIBUTE",
                                "LOCAL_REINIT",
                                "ADD_RESIDUAL_BRANCH",
                                "ADD_CORRECTION_BRANCH",
                                "ADD_FOURIER_BRANCH",
                                "ADD_GATED_BRANCH",
                                "ADD_LOCALIZED_EXPERT",
                                "REBALANCE_LOSS_WEIGHTS",
                                "EMPHASIZE_CONSTRAINTS",
                                "MITIGATE_GRADIENT_CONFLICT",
                                "ENTER_LBFGS_PHASE",
                            ],
                            "adaptation_freeze_step": 14000,
                            "block_new_morph_until_persistent_feedback": True,
                            "architecture_executor": {
                                "recovery_window_steps": {
                                    "mild": 100,
                                    "moderate": 250,
                                    "strong": 500,
                                }
                            },
                            "physics_weight_growth_factor": 1.0,
                            "preservation": {
                                "probe_size": 16,
                                "function_tolerance": 1e-7,
                                "first_derivative_tolerance": 1e-6,
                                "second_derivative_tolerance": 1e-6,
                                "pde_residual_tolerance": 1e-6,
                            },
                            "utility": {
                                "enabled": True,
                                "mode": "component_relative",
                                "minimum_relative_improvement": 0.01,
                                "max_component_degradation": 0.10,
                                "continuation_history_points": 7,
                                "continuation_minimum_confidence": 0.35,
                                "continuation_maximum_extrapolation_factor": 4.0,
                                "continuation_reject_clipped_forecast": True,
                                "minimum_trigger_observations": 3,
                                "short_horizon_steps": 100,
                                "medium_horizon_steps": 500,
                                "persistent_horizon_steps": 1000,
                                "retire_failed_additive_on_persistent": True,
                                "weights": {
                                    "pde": 1.0,
                                    "boundary": 1.0,
                                    "initial": 1.0,
                                    "observation": 1.0,
                                },
                            },
                        },
                        "rollback": {"enabled": False},
                    },
                    "physics_validation": {"enabled": False, "strategy": "disabled"},
                    "best_checkpoint": {"enabled": False, "final_model_policy": "last"},
                },
            },
            "generation_metadata": {
                "proposal_strategy": "knowledge_guided",
                "design_hypothesis": "",
                "parent_spec_ids": [],
                "changed_modules": [],
                "expected_advantages": [],
                "possible_risks": [],
                "macro_physics_principles_applied": [],
                "architecture_design_principles_applied": [],
                "posterior_evidence_applied": [],
                "principle_tradeoffs": [],
                "online_method": None,
                "role_design_objective": "",
                "role_requirements_applied": [],
                "role_input_focus": [],
                "role_modifiable_modules": [],
                "primary_diversity_axis": "",
                "diversity_hypothesis": "",
                "major_executable_differences_expected": [],
                "iteration_budget_rationale": "",
            },
        }
    )
