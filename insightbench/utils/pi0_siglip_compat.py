"""OpenPI SigLIP compatibility for the pinned unified Transformers runtime."""

from __future__ import annotations

import sys
import types

import torch


TRANSFORMERS_VERSION = "4.57.1"
_PATCH_ATTRIBUTE = "_insightbench_pi0_siglip_compat_installed"
_FORWARD_PATCH_ATTRIBUTE = "_insightbench_pi0_siglip_compat_forward"
_CHECK_MODULE = "transformers.models.siglip.check"


def _is_installed(siglip_modeling, transformers_version: str) -> bool:
    return (
        transformers_version == TRANSFORMERS_VERSION
        and getattr(siglip_modeling.SiglipVisionTransformer, _PATCH_ATTRIBUTE, False)
        and getattr(
            siglip_modeling.SiglipVisionTransformer.forward,
            _FORWARD_PATCH_ATTRIBUTE,
            False,
        )
    )


def _install_check_module(siglip_package, siglip_modeling, transformers_version: str) -> None:
    check_module = types.ModuleType(_CHECK_MODULE)
    check_module.check_whether_transformers_replace_is_installed_correctly = lambda: _is_installed(
        siglip_modeling, transformers_version
    )
    sys.modules[_CHECK_MODULE] = check_module
    setattr(siglip_package, "check", check_module)


def install_pi0_siglip_compat() -> None:
    """Install OpenPI's two SigLIP compatibility changes on Transformers 4.57.1.

    LeRobot 0.4.1 Pi0 checks for an OpenPI SigLIP replacement. Relative to the
    branch's Transformers 4.53.3 base, that replacement adds a self-check module
    and casts vision embeddings to bfloat16 before a bfloat16 encoder. This
    targeted patch preserves GR00T's required 4.57.1 Eagle APIs.
    """
    import transformers
    from transformers.models import siglip as siglip_package
    from transformers.models.siglip import modeling_siglip

    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError(
            "Pi0/GR00T unified compatibility is pinned to "
            f"transformers=={TRANSFORMERS_VERSION}, got {transformers.__version__}."
        )

    vision_transformer = modeling_siglip.SiglipVisionTransformer
    if not _is_installed(modeling_siglip, transformers.__version__):

        @modeling_siglip.can_return_tuple
        def pi0_compatible_forward(
            self,
            pixel_values,
            output_attentions=None,
            output_hidden_states=None,
            interpolate_pos_encoding=False,
        ):
            output_attentions = (
                output_attentions if output_attentions is not None else self.config.output_attentions
            )
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )

            hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
            if (
                len(self.encoder.layers) > 0
                and self.encoder.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16
            ):
                hidden_states = hidden_states.to(torch.bfloat16)

            encoder_outputs = self.encoder(
                inputs_embeds=hidden_states,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            last_hidden_state = self.post_layernorm(encoder_outputs.last_hidden_state)
            pooler_output = self.head(last_hidden_state) if self.use_head else None

            return modeling_siglip.BaseModelOutputWithPooling(
                last_hidden_state=last_hidden_state,
                pooler_output=pooler_output,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
            )

        setattr(pi0_compatible_forward, _FORWARD_PATCH_ATTRIBUTE, True)
        vision_transformer.forward = pi0_compatible_forward
        setattr(vision_transformer, _PATCH_ATTRIBUTE, True)

    _install_check_module(siglip_package, modeling_siglip, transformers.__version__)
