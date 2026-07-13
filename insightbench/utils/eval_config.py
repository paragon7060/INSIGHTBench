"""Dependency-light validation for public evaluation configs."""

from __future__ import annotations

from omegaconf import OmegaConf


def load_eval_config(config_path: str, overrides: list[str]):
    cfg = OmegaConf.load(config_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


def is_missing_config_value(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == "" or value.strip().lower() in {"none", "null"}
    return False


def validate_required_eval_inputs(cfg, config_path: str) -> None:
    """Fail before Isaac launch when a policy's required inputs are absent."""
    policy_cfg = OmegaConf.select(cfg, "policy")
    if policy_cfg is None:
        raise SystemExit(f"ERROR: {config_path} must define a policy section.")

    policy_type = OmegaConf.select(policy_cfg, "type") or "<unknown>"
    checkpoint = OmegaConf.select(policy_cfg, "checkpoint")
    if is_missing_config_value(checkpoint):
        raise SystemExit(
            "ERROR: policy.checkpoint is required before launching Isaac Sim.\n"
            f"Config: {config_path}\n"
            f"Policy: {policy_type}\n"
            "Override it with a verified HuggingFace Hub repo ID or a local pretrained_model path, for example:\n"
            "  policy.checkpoint=your-hf-user-or-org/policy-repo\n"
            "or:\n"
            "  policy.checkpoint=local/pretrained_model"
        )

    if policy_type != "smolvla":
        return

    stats_repo = OmegaConf.select(policy_cfg, "dataset_stats_repo")
    stats_root = OmegaConf.select(policy_cfg, "dataset_stats_root")
    if not is_missing_config_value(stats_repo) or not is_missing_config_value(stats_root):
        return

    raise SystemExit(
        "ERROR: SmolVLA requires dataset statistics before launching Isaac Sim.\n"
        f"Config: {config_path}\n"
        "Provide one verified source:\n"
        "  policy.dataset_stats_repo=your-hf-user-or-org/dataset-repo\n"
        "or:\n"
        "  policy.dataset_stats_root=path/to/lerobot-dataset"
    )
