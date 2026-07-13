"""Regression coverage for Pi0's 0.3.4 cache contract on Transformers 4.57."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
FIXTURE = Path(__file__).parent / "fixtures" / "pi0_034_checkpoint_config.json"


def _require_unified_pi0_runtime():
    lerobot = pytest.importorskip("lerobot")
    if lerobot.__version__ != "0.4.1" or transformers.__version__ != "4.57.1":
        pytest.skip("requires the documented LeRobot 0.4.1 / Transformers 4.57.1 baseline")
    return importlib.import_module("insightbench.policies.pi0")


def _legacy_sequence_lengths(pi0) -> tuple[int, int]:
    with FIXTURE.open() as file:
        raw_config = json.load(file)
    normalized_config, _ = pi0._normalize_pi0_checkpoint_config(raw_config)
    config = pi0._decode_pi0_checkpoint_config(normalized_config)

    # The fixture's three 224x224 Pi0 cameras yield 16x16=256 SigLIP tokens
    # apiece. The suffix is one state token plus one action token per chunk step.
    image_feature_count = len(config.image_features)
    prefix_length = image_feature_count * 16 * 16 + config.tokenizer_max_length
    suffix_length = 1 + config.chunk_size
    assert (image_feature_count, prefix_length, suffix_length) == (3, 816, 51)
    return prefix_length, suffix_length


def test_pi0_denoise_cache_restores_legacy_attention_position_contract() -> None:
    """Use Gemma's real 4.57 cache API with the migrated Pi0 sequence sizes."""
    pi0 = _require_unified_pi0_runtime()
    from transformers import GemmaConfig
    from transformers.cache_utils import DynamicCache
    from transformers.models.gemma.modeling_gemma import GemmaModel
    from lerobot.policies.pi0.modeling_pi0 import PI0Pytorch

    pi0._install_pi0_transformers_compat()
    assert getattr(PI0Pytorch, "_insightbench_prefix_cache_compat_installed")
    prefix_length, suffix_length = _legacy_sequence_lengths(pi0)
    config = GemmaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=2048,
    )
    model = GemmaModel(config).eval()
    cache = DynamicCache()

    prefix = torch.randn(1, prefix_length, config.hidden_size)
    prefix_mask = torch.zeros(1, 1, prefix_length, prefix_length)
    prefix_positions = torch.arange(prefix_length).unsqueeze(0)
    model(
        inputs_embeds=prefix,
        attention_mask=prefix_mask,
        position_ids=prefix_positions,
        past_key_values=cache,
        use_cache=True,
    )
    assert cache.get_seq_length() == prefix_length

    suffix = torch.randn(1, suffix_length, config.hidden_size)
    full_mask = torch.zeros(1, 1, suffix_length, prefix_length + suffix_length)
    suffix_positions = torch.arange(prefix_length, prefix_length + suffix_length).unsqueeze(0)
    assert suffix_positions.shape[-1] == suffix_length
    assert cache.get_seq_length() + suffix_length == full_mask.shape[-1]

    # This reflects Pi0's denoise call. DynamicCache grows despite use_cache=False.
    model(
        inputs_embeds=suffix,
        attention_mask=full_mask,
        position_ids=suffix_positions,
        past_key_values=cache,
        use_cache=False,
    )
    assert cache.get_seq_length() == full_mask.shape[-1]
    assert cache.get_seq_length() + suffix_length == 918

    # Restore the legacy fill_kv_cache=False behavior before the next step.
    pi0._restore_pi0_prefix_cache(cache, prefix_length)
    assert cache.get_seq_length() == prefix_length
    assert cache.get_seq_length() + suffix_length == full_mask.shape[-1]

    # The next denoise step now receives matching attention-mask and KV lengths.
    model(
        inputs_embeds=suffix,
        attention_mask=full_mask,
        position_ids=suffix_positions,
        past_key_values=cache,
        use_cache=False,
    )
    assert cache.get_seq_length() == full_mask.shape[-1]
