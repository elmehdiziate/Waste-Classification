import os
import copy
import yaml

# Recursively merge two dictionaries (override takes precedence)
def deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key == "base":
            continue
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# Load experiment config, merge with base.yaml, apply CLI overrides
def load_config(experiment_path: str, cli_overrides: dict = None) -> dict:
    with open(experiment_path) as f:
        experiment = yaml.safe_load(f)

    # Merge with base config if specified
    if "base" in experiment:
        with open(experiment["base"]) as f:
            base = yaml.safe_load(f)
        config = deep_merge(base, experiment)
    else:
        config = experiment

    # Apply CLI overrides (training‑level only)
    if cli_overrides:
        for key, value in cli_overrides.items():
            if value is not None:
                config["training"][key] = value

    return config


# Save resolved config to the run directory
def save_config(config: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "config.yaml")
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"[Config] Saved to {path}")
