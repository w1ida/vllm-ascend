from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.patch.worker.patch_qwen3_next import \
    AscendQwen3Next_GatedDeltaNet


def _build_decode_only_metadata():
    return SimpleNamespace(
        has_initial_state=torch.tensor([True]),
        spec_query_start_loc=None,
        non_spec_query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        spec_sequence_masks=None,
        spec_token_indx=None,
        non_spec_token_indx=None,
        spec_state_indices_tensor=None,
        non_spec_state_indices_tensor=torch.tensor([0], dtype=torch.int32),
        num_actual_tokens=1,
        num_accepted_tokens=torch.tensor([1], dtype=torch.int32),
        num_prefills=0,
        num_decodes=1,
    )


def _build_module():
    module = AscendQwen3Next_GatedDeltaNet.__new__(AscendQwen3Next_GatedDeltaNet)
    module.prefix = "layer"
    module.conv1d = SimpleNamespace(
        weight=torch.ones(2, 1, 2),
        bias=torch.zeros(2),
    )
    module.activation = "silu"
    module.A_log = torch.ones(2)
    module.dt_bias = torch.zeros(2)
    module.kv_cache = [(
        torch.zeros(1, 2, 2),
        torch.zeros(1, 2, 2, 3),
    )]
    module.rearrange_mixed_qkv = MagicMock(
        return_value=(
            torch.randn(1, 1, 1, 2),
            torch.randn(1, 1, 1, 2),
            torch.randn(1, 1, 2, 3),
        ))
    return module


def test_qwen3_next_decode_uses_fused_sigmoid_fast_path():
    module = _build_module()
    forward_context = SimpleNamespace(
        attn_metadata={"layer": _build_decode_only_metadata()},
        virtual_engine=0,
    )
    mixed_qkv = torch.randn(1, 2)
    b = torch.randn(1, 2)
    a = torch.randn(1, 2)
    core_attn_out = torch.zeros(1, 2, 3)
    fused_out = torch.ones(1, 1, 2, 3)

    with patch(
        "vllm_ascend.patch.worker.patch_qwen3_next.get_forward_context",
        return_value=forward_context,
    ), patch(
        "vllm_ascend.patch.worker.patch_qwen3_next.enable_sp",
        return_value=False,
    ), patch(
        "vllm_ascend.patch.worker.patch_qwen3_next.causal_conv1d_update",
        side_effect=lambda *args, **kwargs: args[0],
    ), patch(
        "vllm_ascend.patch.worker.patch_qwen3_next.fused_sigmoid_gating_delta_rule_update",
        return_value=fused_out,
    ) as fused_sigmoid_mock, patch(
        "vllm_ascend.patch.worker.patch_qwen3_next.fused_gdn_gating_patch",
    ) as fused_gdn_mock, patch(
        "vllm_ascend.patch.worker.patch_qwen3_next.torch_npu.npu_recurrent_gated_delta_rule",
    ) as recurrent_mock:
        module._forward_core(mixed_qkv, b, a, core_attn_out)

    fused_sigmoid_mock.assert_called_once()
    fused_gdn_mock.assert_not_called()
    recurrent_mock.assert_not_called()
    torch.testing.assert_close(core_attn_out, fused_out.squeeze(0))


def test_qwen3_next_spec_decode_keeps_original_path():
    module = _build_module()
    metadata = _build_decode_only_metadata()
    metadata.spec_sequence_masks = torch.ones(1, dtype=torch.bool)
    metadata.num_spec_decodes = 1
    metadata.spec_query_start_loc = torch.tensor([0, 1], dtype=torch.int32)
    metadata.spec_state_indices_tensor = torch.tensor([[0]], dtype=torch.int32)
    metadata.spec_token_indx = torch.tensor([0], dtype=torch.int64)
    metadata.non_spec_token_indx = torch.tensor([], dtype=torch.int64)
    forward_context = SimpleNamespace(
        attn_metadata={"layer": metadata},
        virtual_engine=0,
    )
    mixed_qkv = torch.randn(1, 2)
    b = torch.randn(1, 2)
    a = torch.randn(1, 2)
    core_attn_out = torch.zeros(1, 2, 3)

    with patch(
        "vllm_ascend.patch.worker.patch_qwen3_next.get_forward_context",
        return_value=forward_context,
    ), patch(
        "vllm_ascend.patch.worker.patch_qwen3_next.enable_sp",
        return_value=False,
    ), patch(
        "vllm_ascend.patch.worker.patch_qwen3_next.causal_conv1d_update",
        side_effect=lambda *args, **kwargs: args[0],
    ), patch(
        "vllm_ascend.patch.worker.patch_qwen3_next.fused_sigmoid_gating_delta_rule_update",
    ) as fused_sigmoid_mock, patch(
        "vllm_ascend.patch.worker.patch_qwen3_next.fused_gdn_gating_patch",
        return_value=(torch.ones(1, 1), torch.ones(1, 1)),
    ) as fused_gdn_mock, patch(
        "vllm_ascend.patch.worker.patch_qwen3_next.l2norm_fwd",
        side_effect=lambda x: x,
    ), patch(
        "vllm_ascend.patch.worker.patch_qwen3_next.torch_npu.npu_recurrent_gated_delta_rule",
        return_value=torch.ones(1, 2, 3),
    ) as recurrent_mock:
        module._forward_core(mixed_qkv, b, a, core_attn_out)

    fused_sigmoid_mock.assert_not_called()
    fused_gdn_mock.assert_called_once()
    recurrent_mock.assert_called()
