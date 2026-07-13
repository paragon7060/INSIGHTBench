"""Pi0 policy wrapper."""

from __future__ import annotations

import json
import os
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

import torch

from insightbench.policies.base import PolicyBase

try:
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    from lerobot.constants import OBS_STATE, ACTION
except ImportError as exc:  # LeRobot is optional for lightweight unit tests.
    PI0Policy = None
    OBS_STATE = "observation.state"
    ACTION = "action"
    _LEROBOT_IMPORT_ERROR = exc
else:
    _LEROBOT_IMPORT_ERROR = None


_DIRECT_PALIGEMMA_REPLACEMENTS = (
    (
        ".paligemma_with_expert.paligemma.model.vision_tower",
        ".paligemma_with_expert.paligemma.vision_tower",
    ),
    (
        ".paligemma_with_expert.paligemma.model.multi_modal_projector",
        ".paligemma_with_expert.paligemma.multi_modal_projector",
    ),
    (
        ".paligemma_with_expert.paligemma.model.language_model",
        ".paligemma_with_expert.paligemma.language_model.model",
    ),
    (
        ".paligemma_with_expert.paligemma.lm_head",
        ".paligemma_with_expert.paligemma.language_model.lm_head",
    ),
)

_TIED_WEIGHT_PAIRS = (
    (
        ".paligemma_with_expert.paligemma.lm_head.weight",
        ".paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight",
    ),
    (
        ".paligemma_with_expert.paligemma.language_model.lm_head.weight",
        ".paligemma_with_expert.paligemma.language_model.model.embed_tokens.weight",
    ),
)


def _rewrite_key(key: str, replacements: tuple[tuple[str, str], ...]) -> str:
    for old, new in replacements:
        key = key.replace(old, new)
    return key


def _uses_nested_paligemma(model) -> bool:
    return any(
        ".paligemma_with_expert.paligemma.model.vision_tower" in key
        or ".paligemma_with_expert.paligemma.model.language_model" in key
        for key in model.state_dict()
    )


def _ensure_tied_weight_aliases(
    state_dict: dict[str, torch.Tensor],
    target_keys: set[str],
) -> None:
    for lm_head_suffix, embed_suffix in _TIED_WEIGHT_PAIRS:
        lm_head_key = next((key for key in state_dict if key.endswith(lm_head_suffix)), None)
        embed_key = next((key for key in state_dict if key.endswith(embed_suffix)), None)

        if lm_head_key is not None and embed_key is None:
            candidate = lm_head_key.removesuffix(lm_head_suffix) + embed_suffix
            if candidate in target_keys:
                state_dict[candidate] = state_dict[lm_head_key]
        elif embed_key is not None and lm_head_key is None:
            candidate = embed_key.removesuffix(embed_suffix) + lm_head_suffix
            if candidate in target_keys:
                state_dict[candidate] = state_dict[embed_key]


def _adapt_pi0_state_dict_to_model(
    state_dict: dict[str, torch.Tensor],
    model,
) -> dict[str, torch.Tensor]:
    """Adapt PaliGemma checkpoint keys to the installed Transformers layout."""
    target_keys = set(model.state_dict())

    if _uses_nested_paligemma(model):
        adapted = PI0Policy._insightbench_original_transform_state_dict_keys(state_dict)
    else:
        adapted = {
            _rewrite_key(key, _DIRECT_PALIGEMMA_REPLACEMENTS): value
            for key, value in state_dict.items()
        }

    _ensure_tied_weight_aliases(adapted, target_keys)
    return adapted


def _paligemma_vision_tower(paligemma):
    if hasattr(paligemma, "vision_tower"):
        return paligemma.vision_tower
    return paligemma.model.vision_tower


def _paligemma_language_backbone(paligemma):
    language_model = getattr(paligemma, "language_model", None)
    if language_model is None:
        language_model = paligemma.model.language_model
    if hasattr(language_model, "layers"):
        return language_model
    if hasattr(language_model, "model") and hasattr(language_model.model, "layers"):
        return language_model.model
    raise AttributeError("Could not locate PaliGemma language backbone layers.")


def _install_pi0_transformers_compat() -> None:
    """Patch the InsightBench LeRobot fork across PaliGemma layout variants."""
    if PI0Policy is None or getattr(PI0Policy, "_insightbench_compat_installed", False):
        return

    from safetensors.torch import load_file

    from lerobot.policies.pi0.paligemma_with_expert import PaliGemmaWithExpertModel
    from lerobot.policies.utils import log_model_loading_keys
    from lerobot.utils.utils import init_logging

    if not hasattr(PI0Policy, "_insightbench_original_transform_state_dict_keys"):
        PI0Policy._insightbench_original_transform_state_dict_keys = PI0Policy._transform_state_dict_keys

    @classmethod
    def _load_as_safetensor(cls, model, model_file: str, map_location: str, strict: bool):
        init_logging()
        state_dict = load_file(model_file, device=map_location)
        adapted_state_dict = _adapt_pi0_state_dict_to_model(state_dict, model)
        msg = model.load_state_dict(adapted_state_dict, strict=strict)
        log_model_loading_keys(msg.missing_keys, msg.unexpected_keys)
        return model

    def set_requires_grad(self) -> None:
        if self.config.freeze_vision_encoder:
            vision_tower = _paligemma_vision_tower(self.paligemma)
            vision_tower.eval()
            for params in vision_tower.parameters():
                params.requires_grad = False

        if self.config.train_expert_only:
            self.paligemma.eval()
            for params in self.paligemma.parameters():
                params.requires_grad = False

    def train(self, mode: bool = True):
        super(PaliGemmaWithExpertModel, self).train(mode)

        if self.config.freeze_vision_encoder:
            _paligemma_vision_tower(self.paligemma).eval()

        if self.config.train_expert_only:
            self.paligemma.eval()

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.get_input_embeddings()(tokens)

    original_forward = PaliGemmaWithExpertModel.forward

    def forward(self, *args: Any, **kwargs: Any):
        original_language_model = getattr(self.paligemma, "language_model", None)
        if original_language_model is None or not hasattr(original_language_model, "layers"):
            try:
                self.paligemma.language_model = _paligemma_language_backbone(self.paligemma)
                return original_forward(self, *args, **kwargs)
            finally:
                if original_language_model is None:
                    del self.paligemma.language_model
                else:
                    self.paligemma.language_model = original_language_model
        return original_forward(self, *args, **kwargs)

    PI0Policy._load_as_safetensor = _load_as_safetensor
    PaliGemmaWithExpertModel.set_requires_grad = set_requires_grad
    PaliGemmaWithExpertModel.train = train
    PaliGemmaWithExpertModel.embed_language_tokens = embed_language_tokens
    PaliGemmaWithExpertModel.forward = forward
    PI0Policy._insightbench_compat_installed = True


class Pi0Wrapper(PolicyBase):
    """Wraps PI0Policy for InsightBench evaluation."""

    def load(self) -> None:
        if PI0Policy is None:
            raise ImportError("Pi0Wrapper requires LeRobot to load PI0Policy.") from _LEROBOT_IMPORT_ERROR
        _install_pi0_transformers_compat()

        cfg = self.cfg
        stats = self._load_stats(cfg)
        self._trim_stats(stats, cfg)

        checkpoint = cfg.checkpoint
        subfolder = getattr(cfg, "checkpoint_subfolder", None)
        if subfolder and not os.path.isdir(checkpoint):
            from huggingface_hub import snapshot_download
            local_dir = snapshot_download(
                repo_id=checkpoint,
                allow_patterns=[f"{subfolder}/**"],
            )
            checkpoint = os.path.join(local_dir, subfolder)

        self.policy = PI0Policy.from_pretrained(checkpoint, dataset_stats=stats)
        self.policy.to(self.device)
        self.policy.eval()

    @staticmethod
    def _load_stats(cfg) -> dict:
        """Load normalisation stats from local stats.json or HuggingFace Hub."""
        local_root = getattr(cfg, "dataset_stats_root", None)
        if local_root:
            stats_path = Path(local_root) / "meta" / "stats.json"
        else:
            from huggingface_hub import hf_hub_download
            stats_path = hf_hub_download(
                repo_id=cfg.dataset_stats_repo,
                filename="meta/stats.json",
                repo_type="dataset",
            )
        with open(stats_path) as f:
            raw = json.load(f)
        # Convert lists → float32 tensors
        return {
            key: {k: torch.tensor(v, dtype=torch.float32) for k, v in sub.items() if k != "count"}
            for key, sub in raw.items()
        }

    @staticmethod
    def _trim_stats(stats: MutableMapping[str, MutableMapping[str, torch.Tensor]], cfg) -> None:
        """Slice stats mapping entries to the dims actually used.

        ``stats`` must be the dataset stats mapping itself (for example
        ``LeRobotDatasetMetadata.stats``), not a metadata object.
        """
        s_idx, s_dim = cfg.state_start_idx, cfg.state_dim
        a_idx, a_dim = 0, cfg.action_dim

        for key, idx, dim in [(OBS_STATE, s_idx, s_dim), (ACTION, a_idx, a_dim)]:
            if key not in stats:
                continue
            for stat in ("mean", "std", "min", "max"):
                if stat in stats[key]:
                    stats[key][stat] = stats[key][stat][..., idx: idx + dim]

    def select_action(self, obs_state, obs_imgs, task_prompts):
        input_batch = {OBS_STATE: obs_state, "task": task_prompts, **obs_imgs}
        with torch.no_grad():
            self.policy.config.action_feature.shape = (self.cfg.action_dim,)
            action = self.policy.select_action(input_batch)
        return action
