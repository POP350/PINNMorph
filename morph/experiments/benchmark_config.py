"""The supported PINNMorph benchmark scope and matched experiment profiles."""
from pathlib import Path

from .experiment_config import apply_ablation_mode, load_experiment_config, validate_experiment_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "configs"
SUITE_PDES = (
    "burgers_1d", "burgers_2d", "gray_scott_2d",
    "heat_2d_complex_geometry", "heat_2d_multiscale", "heat_2d_varying_coefficient",
    "kuramoto_sivashinsky_1d", "navier_stokes_2d_backstep",
    "poisson_2d_classic", "poisson_boltzmann_2d", "poisson_2d_many_area",
    "poisson_3d_complex_geometry", "poisson_5d",
)
VARIANTS = ("no_morph_matched", "random_selection", "random_evolution", "full")


def profile(pde, variant, seeds):
    if pde not in SUITE_PDES:
        raise ValueError(f"Unsupported benchmark PDE: {pde}")
    if variant not in VARIANTS:
        raise ValueError(f"Unsupported benchmark variant: {variant}")
    config = load_experiment_config(CONFIG_ROOT / f"{pde}.yaml")
    config = {k: v for k, v in config.items() if not k.startswith("_")}
    config.update(seeds=list(seeds), diagnostic_variant=variant, no_morph_optimizer_schedule="matched")
    modes = {"no_morph_matched": "no_morph", "random_selection": "random_selection",
             "random_evolution": "random_evolution", "full": "none"}
    config = apply_ablation_mode(config, modes[variant])
    if variant == "no_morph_matched":
        # Fixed scheduled Adam/L-BFGS phases remain; no online phase request.
        config["algorithm_overrides"]["training"]["extra_parameters"]["adaptive_control"]["optimizer_phase_switch"]["enabled"] = False
    validate_experiment_config(config)
    from .paper_contract import validate_paper_protocol
    validate_paper_protocol(config)
    return config
