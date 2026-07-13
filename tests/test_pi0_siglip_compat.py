"""Unified Pi0/GR00T Transformers compatibility coverage."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def siglip_compat():
    if transformers.__version__ != "4.57.1":
        pytest.skip("requires the documented unified transformers==4.57.1 baseline")

    from insightbench.utils import pi0_siglip_compat

    pi0_siglip_compat.install_pi0_siglip_compat()
    return pi0_siglip_compat


def test_pi0_siglip_shim_installs_check_only_with_patched_forward(siglip_compat) -> None:
    from transformers.models.siglip import check, modeling_siglip

    assert check.check_whether_transformers_replace_is_installed_correctly()
    vision_transformer = modeling_siglip.SiglipVisionTransformer
    assert getattr(vision_transformer, "_insightbench_pi0_siglip_compat_installed")
    assert getattr(vision_transformer.forward, "_insightbench_pi0_siglip_compat_forward")

    # The LeRobot-facing check is tied to the replacement forward, not a
    # stand-alone unconditional success flag.
    delattr(vision_transformer.forward, "_insightbench_pi0_siglip_compat_forward")
    assert not check.check_whether_transformers_replace_is_installed_correctly()
    siglip_compat.install_pi0_siglip_compat()
    assert check.check_whether_transformers_replace_is_installed_correctly()


def test_pi0_siglip_shim_casts_embeddings_for_bfloat16_encoder(siglip_compat) -> None:
    """Exercise the OpenPI bfloat16 semantic without constructing a vision model."""
    from transformers.models.siglip import modeling_siglip

    class Embeddings(torch.nn.Module):
        def forward(self, pixel_values, *, interpolate_pos_encoding=False):
            return torch.ones((1, 1, 4), dtype=torch.float32)

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [
                SimpleNamespace(
                    self_attn=SimpleNamespace(
                        q_proj=SimpleNamespace(weight=torch.empty(1, dtype=torch.bfloat16))
                    )
                )
            ]
            self.input_dtype = None

        def forward(self, *, inputs_embeds, output_attentions, output_hidden_states):
            self.input_dtype = inputs_embeds.dtype
            return SimpleNamespace(last_hidden_state=inputs_embeds, hidden_states=None, attentions=None)

    class VisionStub:
        config = SimpleNamespace(
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        embeddings = Embeddings()
        encoder = Encoder()
        post_layernorm = torch.nn.Identity()
        head = None
        use_head = False

    vision = VisionStub()
    output = modeling_siglip.SiglipVisionTransformer.forward(vision, torch.zeros(1, 3, 2, 2))

    assert vision.encoder.input_dtype == torch.bfloat16
    assert output.last_hidden_state.dtype == torch.bfloat16


def test_unified_transformers_keeps_groot_eagle_api_available(siglip_compat) -> None:
    pytest.importorskip("lerobot")

    from lerobot.policies.groot.eagle2_hg_model.modeling_eagle2_5_vl import (
        Eagle25VLForConditionalGeneration,
    )
    from transformers.models.siglip import check

    assert Eagle25VLForConditionalGeneration.__name__ == "Eagle25VLForConditionalGeneration"
    assert check.check_whether_transformers_replace_is_installed_correctly()
