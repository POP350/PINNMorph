"""Check reported method settings against the supplied manuscript, not results."""
from morph.pipeline.g_training_evaluation.training.adaptation_program import PRIMITIVE_LIBRARY


def validate_paper_protocol(config):
    """Reject accidental default-method changes; smoke runs may use fewer seeds."""
    errors = []

    def require(condition, message):
        if not condition:
            errors.append(message)

    require(config.get("paper_protocol") == "provided_iclr2027_submission", "paper protocol identity")
    require(config["budget"]["total_optimizer_steps"] == 20000, "20,000 optimizer steps (B)")
    algorithm = config["algorithm_overrides"]
    network = algorithm["network"]
    require(network["architecture"] == "mlp" and network["hidden_layers"] == [128]*4,
            "four hidden layers of width 128 (B.1)")
    require(network["activation"]["name"] == "tanh", "tanh activation (B.1)")
    transform = "periodic_positional_encoding" if config["pde"] == "burgers_2d" else "none"
    require(network["input_transform"]["name"] == transform, "reported positional encoding (B.1)")
    phases = algorithm["optimization"]["phases"]
    require([p["optimizer"] for p in phases] == ["adam", "lbfgs"], "Adam then L-BFGS (B.1)")
    require([p["parameters"]["learning_rate"] for p in phases] == [0.001, 1.0], "reported learning rates (B.1)")
    require(all(p["scheduler"]["name"] == "none" for p in phases), "no learning-rate scheduler (B.1)")
    require(sum(p["iterations"] for p in phases) == 20000, "phase budget equals 20,000 (B)")
    training = algorithm["training"]
    require(training["time_strategy"]["name"] == "none", "no unreported time curriculum")
    require(not training["initial_state_pretraining"]["enabled"], "no unreported IC pretraining")
    physics = training["extra_parameters"]["physics_validation"]
    for field, value in (("points_per_component", 128), ("maximum_total_points", 4096),
                         ("audit_points_per_component", 2048), ("audit_maximum_total_points", 32768)):
        require(physics[field] == value, f"held-out probe budget {field} (B.3)")
    require(physics["audit_enabled"], "independent audit set (B.3)")
    initial_guard = training["extra_parameters"]["best_checkpoint"].get("initial_condition_guard", {})
    require(initial_guard.get("enabled") is True
            and initial_guard.get("maximum_normalized_error") == 0.1,
            "best-checkpoint IC eligibility is matched across variants")
    online = config["online_morphing"]
    expected_sampling_step = 200 if config.get("ablation_mode", "none") == "none" else 0
    require(online.get("forced_initial_sampling_step", 200) == expected_sampling_step,
            "mandatory step-200 sampling is Full-only")
    require(config.get("capability_ablation") == "none", "complete action library (C.1)")
    require(tuple(online["allowed_primitives"]) == PRIMITIVE_LIBRARY, "all 23 primitives (C.1)")
    require("fourier_axis_frequencies" not in online, "no PDE-specific Fourier bank override (C.1)")
    adaptive = online.get("adaptive_strategy") or {}
    require(not adaptive.get("bottleneck_enabled") and not adaptive.get("progressive_enabled"),
            "no extra candidate-family restrictions or influence ramps")
    for field in ("temporal", "trial_control", "influence_guard", "llm_efficiency"):
        require(not (adaptive.get(field) or {}).get("enabled"), f"unreported extension {field} disabled")
    for field, value in (("diagnostic_interval", 400), ("minimum_iterations_before_morph", 400),
                         ("adaptation_freeze_step", 14000), ("activation_only_iterations", 10),
                         ("new_parameters_only_iterations", 50)):
        require(online[field] == value, f"schedule {field} (B.2)")
    require(online["block_new_morph_until_persistent_feedback"], "isolated persistent horizon (B.2)")
    require(online["optimizer_state_after_morph"]["mode"] == "preserve", "optimizer state transfer (C.3)")
    require(online["joint_learning_rate_scale"] == 1.0, "reported learning rates retained after warm-up (B.1)")
    policy = online["policy_set"]
    require(policy["population_size"] == 5, "five policies (3.2)")
    require(policy["scoring"]["temperature"] == 0.0 and policy["evolution"]["temperature"] == 0.7,
            "LLM temperatures (B.5)")
    utility = online["utility"]
    for field, value in (("continuation_history_points", 7), ("continuation_minimum_confidence", 0.35),
                         ("continuation_maximum_extrapolation_factor", 4.0),
                         ("minimum_relative_improvement", 0.01), ("max_component_degradation", 0.10),
                         ("short_horizon_steps", 100), ("medium_horizon_steps", 500),
                         ("persistent_horizon_steps", 1000)):
        require(utility[field] == value, f"continuation/feedback {field} (D)")
    require(utility["continuation_reject_clipped_forecast"], "clipped forecasts withhold credit (D)")
    if errors:
        raise ValueError("Paper protocol mismatch: " + "; ".join(errors))
