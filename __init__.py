import torch

from comfy.ldm.modules import attention as comfy_attention

mha_fwd = None
_AITER_IMPORT_ERROR = None
if torch.version.hip is not None:
    try:
        from aiter.ops.mha import mha_fwd
    except (ImportError, OSError) as exc:
        _AITER_IMPORT_ERROR = exc


def fallback_attention(q, k, v, *positional, **kwargs):
    mask = kwargs.get("mask")
    if kwargs.get("heads") is None and positional:
        positional_mask = positional[1:]
    else:
        positional_mask = positional
    if mask is None and positional_mask:
        mask = positional_mask[0]
    attention = comfy_attention.optimized_attention_for_device(q.device, mask=mask is not None)
    return attention(q, k, v, *positional, **kwargs)


def aiter_attention(q, k, v, *positional, **kwargs):
    heads = kwargs.get("heads")
    remaining = positional
    if heads is None and remaining:
        heads, *remaining = remaining

    reason = bool(remaining)
    reason |= kwargs.get("mask") is not None
    reason |= kwargs.get("attn_precision") is not None
    reason |= q.dtype not in (torch.float16, torch.bfloat16)
    reason |= q.device.type != "cuda"
    reason |= torch.version.hip is None
    reason |= k.device != q.device or v.device != q.device
    reason |= k.dtype != q.dtype or v.dtype != q.dtype
    reason |= not isinstance(heads, int) or heads <= 0
    reason |= kwargs.get("enable_gqa", False)
    reason |= kwargs.get("is_causal", False)
    reason |= bool(set(kwargs) - {
        "heads", "mask", "attn_precision", "skip_reshape", "skip_output_reshape",
        "scale", "enable_gqa", "is_causal", "_inside_attn_wrapper", "transformer_options",
    })
    if reason:
        return fallback_attention(q, k, v, *positional, **kwargs)

    skip_reshape = bool(kwargs.get("skip_reshape", False))
    skip_output_reshape = bool(kwargs.get("skip_output_reshape", False))
    if skip_reshape:
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or any(t.shape[1] != heads for t in (q, k, v)):
            return fallback_attention(q, k, v, *positional, **kwargs)
        q_bshd, k_bshd, v_bshd = (t.transpose(1, 2).contiguous() for t in (q, k, v))
    else:
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3 or any(t.shape[-1] % heads for t in (q, k, v)):
            return fallback_attention(q, k, v, *positional, **kwargs)
        q_bshd, k_bshd, v_bshd = (
            t.reshape(t.shape[0], t.shape[1], heads, t.shape[-1] // heads).contiguous()
            for t in (q, k, v)
        )

    if (
        q_bshd.shape[0] != k_bshd.shape[0] or q_bshd.shape[0] != v_bshd.shape[0]
        or q_bshd.shape[2] != k_bshd.shape[2] or q_bshd.shape[2] != v_bshd.shape[2]
        or k_bshd.shape[1] != v_bshd.shape[1]
        or q_bshd.shape[3] != k_bshd.shape[3] or q_bshd.shape[3] != v_bshd.shape[3]
    ):
        return fallback_attention(q, k, v, *positional, **kwargs)

    dim_head = q_bshd.shape[-1]
    scale = kwargs.get("scale")
    softmax_scale = float(scale) if scale is not None else dim_head ** -0.5
    output = mha_fwd(
        q_bshd, k_bshd, v_bshd,
        0.0, softmax_scale, False, -1, -1, 0, False, False,
    )[0]
    if skip_output_reshape:
        return output.transpose(1, 2)
    return output.reshape(output.shape[0], output.shape[1], -1)


class AiterAttentionPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model/patching"
    EXPERIMENTAL = True

    def patch(self, model):
        if mha_fwd is None:
            raise RuntimeError(
                "Aiter Attention Patch requires ROCm/HIP PyTorch and a native "
                "Aiter FMHA build with Windows FMHA/JIT support (ROCm/aiter#4340 "
                "or a later release). Install the runtime, then restart ComfyUI."
            ) from _AITER_IMPORT_ERROR
        patched = model.clone()
        patched.set_model_optimized_attention(aiter_attention)
        return (patched,)


NODE_CLASS_MAPPINGS = {"AiterAttentionPatch": AiterAttentionPatch}
NODE_DISPLAY_NAME_MAPPINGS = {"AiterAttentionPatch": "Aiter Attention Patch"}
