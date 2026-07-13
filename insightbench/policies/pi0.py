"""Pi0 policy wrapper."""

from __future__ import annotations

import json
import os
from collections.abc import MutableMapping
from copy import deepcopy
from functools import wraps
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch
from packaging.version import Version

from insightbench.policies.base import PolicyBase
from insightbench.utils.pi0_siglip_compat import install_pi0_siglip_compat

try:
    import lerobot
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi0.configuration_pi0 import PI0Config
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    from lerobot.utils.constants import ACTION, OBS_STATE
except ImportError as exc:  # LeRobot is optional for lightweight unit tests.
    _LEROBOT_VERSION = None
    PreTrainedConfig = None
    PI0Config = None
    PI0Policy = None
    OBS_STATE = "observation.state"
    ACTION = "action"
    _LEROBOT_IMPORT_ERROR = exc
else:
    _LEROBOT_VERSION = Version(lerobot.__version__)
    _LEROBOT_IMPORT_ERROR = None

try:
    from lerobot.policies.pi0.processor_pi0 import make_pi0_pre_post_processors
except ImportError:  # LeRobot 0.3.x keeps normalization inside PI0Policy.
    make_pi0_pre_post_processors = None


class Pi0CheckpointConfigCompatibilityError(ValueError):
    """Raised when a legacy Pi0 config cannot be proven compatible with 0.4.1."""


def _uses_pi0_processor_pipeline() -> bool:
    """Return whether the documented 0.4.x Pi0 processor API is required."""
    if _LEROBOT_VERSION is None:
        raise ImportError("Pi0Wrapper requires the documented LeRobot checkout.") from _LEROBOT_IMPORT_ERROR
    if _LEROBOT_VERSION < Version("0.4.0"):
        return False
    if _LEROBOT_VERSION >= Version("0.5.0"):
        raise RuntimeError(
            f"Pi0Wrapper supports LeRobot 0.3.x or the pinned 0.4.x baseline, not {_LEROBOT_VERSION}."
        )
    if make_pi0_pre_post_processors is None:
        raise RuntimeError(
            "LeRobot 0.4.x Pi0 processor API is unavailable. Use the documented LeRobot 0.4.1 checkout."
        )
    return True


def _require_legacy_value(config: dict[str, Any], field: str, expected: Any, reason: str) -> None:
    """Remove a legacy field only when its value matches the 0.4.1 equivalent."""
    if field not in config:
        return
    value = config.pop(field)
    matches = value is expected if isinstance(expected, bool) else value == expected
    if not matches:
        raise Pi0CheckpointConfigCompatibilityError(
            f"Pi0 checkpoint config field `{field}={value!r}` is unsupported by the "
            f"LeRobot 0.4.1 baseline: {reason}"
        )


def _normalize_pi0_checkpoint_config(raw_config: dict[str, Any]) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Translate only the proven 0.3.4 Pi0 config subset to the 0.4.1 schema.

    The supplied InsightBench Pi0 checkpoints use the 0.3.4 defaults.  Values
    outside that exact contract are rejected instead of being silently dropped.
    """
    if PI0Config is None:
        raise ImportError("Pi0 checkpoint migration requires the documented LeRobot checkout.") from _LEROBOT_IMPORT_ERROR

    config = deepcopy(raw_config)
    if config.get("type") != "pi0":
        raise Pi0CheckpointConfigCompatibilityError(
            f"Expected a Pi0 checkpoint config, got type={config.get('type')!r}."
        )

    migrated: list[str] = []

    if "resize_imgs_with_padding" in config:
        image_resolution = config.pop("resize_imgs_with_padding")
        if (
            not isinstance(image_resolution, (list, tuple))
            or len(image_resolution) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in image_resolution)
            or "image_resolution" in config
        ):
            raise Pi0CheckpointConfigCompatibilityError(
                "`resize_imgs_with_padding` can only migrate to a positive two-value "
                "`image_resolution` when no target value is already present."
            )
        config["image_resolution"] = list(image_resolution)
        migrated.append("resize_imgs_with_padding->image_resolution")

    if "num_steps" in config:
        num_steps = config.pop("num_steps")
        if (
            isinstance(num_steps, bool)
            or not isinstance(num_steps, int)
            or num_steps <= 0
            or "num_inference_steps" in config
        ):
            raise Pi0CheckpointConfigCompatibilityError(
                "`num_steps` can only migrate to a positive `num_inference_steps` when no target value is present."
            )
        config["num_inference_steps"] = num_steps
        migrated.append("num_steps->num_inference_steps")

    if "proj_width" in config:
        proj_width = config.pop("proj_width")
        action_expert_variant = config.get("action_expert_variant", "gemma_300m")
        if proj_width != 1024 or action_expert_variant != "gemma_300m":
            raise Pi0CheckpointConfigCompatibilityError(
                "`proj_width` is only compatible when it is 1024, matching LeRobot "
                "0.4.1 action_expert_variant='gemma_300m'."
            )
        config["action_expert_variant"] = "gemma_300m"
        migrated.append("proj_width=1024->action_expert_variant=gemma_300m")

    _require_legacy_value(
        config,
        "adapt_to_pi_aloha",
        False,
        "0.4.1 has no Aloha state/action coordinate transform.",
    )
    if "adapt_to_pi_aloha" in raw_config:
        migrated.append("adapt_to_pi_aloha=false")
    _require_legacy_value(
        config,
        "use_delta_joint_actions_aloha",
        False,
        "0.3.4 itself marks delta Aloha actions as unported, and 0.4.1 has no equivalent.",
    )
    if "use_delta_joint_actions_aloha" in raw_config:
        migrated.append("use_delta_joint_actions_aloha=false")
    _require_legacy_value(
        config,
        "use_cache",
        True,
        "0.4.1 uses its fixed cache schedule during Pi0 inference.",
    )
    if "use_cache" in raw_config:
        migrated.append("use_cache=true")
    _require_legacy_value(
        config,
        "attention_implementation",
        "eager",
        "0.4.1 Pi0 calls the eager attention implementation directly.",
    )
    if "attention_implementation" in raw_config:
        migrated.append("attention_implementation=eager")

    training_only_fields = (
        "freeze_vision_encoder",
        "train_expert_only",
        "train_state_proj",
    )
    for field in training_only_fields:
        if field not in config:
            continue
        value = config.pop(field)
        if not isinstance(value, bool):
            raise Pi0CheckpointConfigCompatibilityError(
                f"Pi0 checkpoint config field `{field}` must be a boolean to be ignored for evaluation."
            )
    if any(field in raw_config for field in training_only_fields):
        migrated.append("dropped training-only freeze/train flags")

    target_fields = set(PI0Config.__dataclass_fields__)
    unknown_fields = sorted(set(config) - target_fields - {"type"})
    if unknown_fields:
        raise Pi0CheckpointConfigCompatibilityError(
            "Pi0 checkpoint config contains unclassified field(s) that would be silently ignored: "
            + ", ".join(unknown_fields)
        )

    return config, tuple(migrated)


def _decode_pi0_checkpoint_config(config: dict[str, Any]):
    """Decode normalized config through LeRobot without changing the checkpoint on disk."""
    if PreTrainedConfig is None or PI0Config is None:
        raise ImportError("Pi0 checkpoint migration requires the documented LeRobot checkout.") from _LEROBOT_IMPORT_ERROR

    with TemporaryDirectory(prefix="insightbench-pi0-config-") as directory:
        config_path = Path(directory) / "config.json"
        with config_path.open("w") as file:
            json.dump(config, file)
        decoded = PreTrainedConfig.from_pretrained(directory)

    if not isinstance(decoded, PI0Config):
        raise Pi0CheckpointConfigCompatibilityError(
            f"Expected the normalized checkpoint to decode as PI0Config, got {type(decoded).__name__}."
        )
    return decoded


def _load_pi0_checkpoint_config(checkpoint: str | Path):
    """Load and normalize a local or Hub Pi0 config while preserving the source file."""
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.is_dir():
        config_path = checkpoint_path / "config.json"
    else:
        from huggingface_hub import hf_hub_download

        config_path = Path(hf_hub_download(repo_id=str(checkpoint), filename="config.json"))

    with config_path.open() as file:
        raw_config = json.load(file)

    normalized_config, migrated = _normalize_pi0_checkpoint_config(raw_config)
    decoded_config = _decode_pi0_checkpoint_config(normalized_config)
    if migrated:
        print(
            "[Pi0] migrated legacy checkpoint config in memory: "
            + ", ".join(migrated)
            + ". Source config.json was not modified."
        )
    return decoded_config


def _paligemma_language_backbone(paligemma):
    language_model = getattr(paligemma, "language_model", None)
    if language_model is None:
        language_model = paligemma.model.language_model
    if hasattr(language_model, "layers"):
        return language_model
    if hasattr(language_model, "model") and hasattr(language_model.model, "layers"):
        return language_model.model
    raise AttributeError("Could not locate PaliGemma language backbone layers.")


def _restore_pi0_prefix_cache(past_key_values: Any, prefix_length: int) -> None:
    """Restore the immutable prefix cache expected by Pi0 denoising.

    In Transformers 4.57, a supplied DynamicCache is updated in place even when
    Gemma receives ``use_cache=False``. Pi0 reuses only the image/language
    prefix cache across denoising steps, so the action suffix must be removed.
    """
    get_seq_length = getattr(past_key_values, "get_seq_length", None)
    crop = getattr(past_key_values, "crop", None)
    if not callable(get_seq_length) or not callable(crop):
        raise RuntimeError(
            "Pi0 on the documented Transformers 4.57.1 baseline requires a crop-capable KV cache."
        )

    cache_length = get_seq_length()
    if cache_length < prefix_length:
        raise RuntimeError(
            "Pi0 KV cache is shorter than its image/language prefix: "
            f"cache={cache_length}, prefix={prefix_length}."
        )
    if cache_length == prefix_length:
        return

    crop(prefix_length)
    restored_length = get_seq_length()
    if restored_length != prefix_length:
        raise RuntimeError(
            "Pi0 could not restore the KV cache to its image/language prefix: "
            f"expected={prefix_length}, got={restored_length}."
        )


def _install_pi0_transformers_compat() -> None:
    """Bridge LeRobot 0.4.1's embedded Pi0 class to Transformers 4.57.1."""
    if PI0Policy is None:
        return

    from lerobot.policies.pi0.modeling_pi0 import PI0Pytorch, PaliGemmaWithExpertModel

    if not getattr(PI0Policy, "_insightbench_compat_installed", False):
        if not hasattr(PI0Policy, "_load_as_safetensor"):
            raise RuntimeError(
                "The pinned LeRobot 0.4.1 PI0Policy checkpoint loader is unavailable. "
                "Use the documented LeRobot checkout and commit."
            )

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

        # PI0Policy's 0.4.1 loader already owns checkpoint key handling. Replacing
        # it with a fork-specific transformer would corrupt this baseline contract.
        PaliGemmaWithExpertModel.embed_language_tokens = embed_language_tokens
        PaliGemmaWithExpertModel.forward = forward
        PI0Policy._insightbench_compat_installed = True

    if not getattr(PI0Pytorch, "_insightbench_prefix_cache_compat_installed", False):
        original_denoise_step = PI0Pytorch.denoise_step

        @wraps(original_denoise_step)
        def denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
            prefix_length = prefix_pad_masks.shape[1]
            get_seq_length = getattr(past_key_values, "get_seq_length", None)
            if not callable(get_seq_length):
                raise RuntimeError(
                    "Pi0 on the documented Transformers 4.57.1 baseline requires a queryable KV cache."
                )
            if get_seq_length() != prefix_length:
                raise RuntimeError(
                    "Pi0 KV cache no longer matches the image/language prefix before denoising: "
                    f"cache={get_seq_length()}, prefix={prefix_length}."
                )

            try:
                return original_denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep)
            finally:
                # Legacy Pi0 used fill_kv_cache=False for suffix denoising. Keep
                # the same prefix-only cache contract with 4.57's DynamicCache.
                _restore_pi0_prefix_cache(past_key_values, prefix_length)

        PI0Pytorch.denoise_step = denoise_step
        PI0Pytorch._insightbench_prefix_cache_compat_installed = True



class Pi0Wrapper(PolicyBase):
    """Wraps PI0Policy for InsightBench evaluation."""

    def __init__(self, policy_cfg, device: torch.device):
        super().__init__(policy_cfg, device)
        self._preprocessor = None
        self._postprocessor = None
        self._uses_processor_pipeline = False

    def load(self) -> None:
        if PI0Policy is None:
            raise ImportError("Pi0Wrapper requires LeRobot to load PI0Policy.") from _LEROBOT_IMPORT_ERROR
        self._uses_processor_pipeline = _uses_pi0_processor_pipeline()
        if self._uses_processor_pipeline:
            install_pi0_siglip_compat()
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

        if self._uses_processor_pipeline:
            checkpoint_config = _load_pi0_checkpoint_config(checkpoint)
            # LeRobot 0.4.x moved normalization out of PI0Policy. Pass stats
            # exclusively to its official pre/post processor factory.
            self.policy = PI0Policy.from_pretrained(checkpoint, config=checkpoint_config)
            self._preprocessor, self._postprocessor = make_pi0_pre_post_processors(
                config=self.policy.config,
                dataset_stats=stats,
            )
        else:
            # The 0.3.x policy keeps normalization inside PI0Policy.
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
        if self._uses_processor_pipeline:
            if self._preprocessor is None or self._postprocessor is None:
                raise RuntimeError("Pi0 0.4.x processor pipeline was not initialized during policy loading.")
            batch = self._preprocessor(deepcopy(input_batch))
            with torch.no_grad():
                action = self.policy.select_action(batch)
            return self._postprocessor(action)

        with torch.no_grad():
            self.policy.config.action_feature.shape = (self.cfg.action_dim,)
            action = self.policy.select_action(input_batch)
        return action
