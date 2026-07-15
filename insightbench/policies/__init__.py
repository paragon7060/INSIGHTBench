from insightbench.policies.base import PolicyBase


def load_policy(policy_cfg, device):
    """Factory: instantiate the correct policy wrapper from config."""
    _registry = {
        "pi0":             "insightbench.policies.pi0:Pi0Wrapper",
        "diffusion":       "insightbench.policies.diffusion:DiffusionWrapper",
        "groot":           "insightbench.policies.groot:GrootWrapper",
        "groot060_client": "insightbench.policies.groot060_client:Groot060ClientWrapper",
        "groot060_insight_client": "insightbench.policies.groot060_client:Groot060ClientWrapper",
        "smolvla":         "insightbench.policies.smolvla:SmolVLAWrapper",
        "instruction_gpt": "insightbench.policies.instruction_gpt:InstructionGPTWrapper",
    }
    entry = _registry.get(policy_cfg.type)
    if entry is None:
        raise ValueError(
            f"Unknown policy type '{policy_cfg.type}'. "
            f"Choose from: {list(_registry.keys())}"
        )
    module_path, cls_name = entry.split(":")
    import importlib
    cls = getattr(importlib.import_module(module_path), cls_name)
    policy = cls(policy_cfg, device)
    policy.load()
    return policy


__all__ = ["PolicyBase", "load_policy"]
