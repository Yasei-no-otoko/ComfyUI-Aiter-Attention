# Aiter Attention Patch

An experimental MODEL patch node that sends supported ComfyUI attention calls to Aiter's `mha_fwd`. Connect it on the model path before the sampler. It affects only that patched model and does not change ComfyUI's global attention choice.

## Requirements

Tested on Windows with Python 3.13, AMD ROCm PyTorch 2.15.0a0+rocm10.2.0a20260918, HIP 7.16, and a Radeon 8060S (`gfx1151`). The Windows Aiter FMHA/JIT support is from [ROCm/aiter#4340](https://github.com/ROCm/aiter/pull/4340), which is still open; use that build or a later release that includes its support. Install Aiter and its runtime in the same Python/Torch environment as ComfyUI before using the node. This extension does not install or upgrade Aiter, PyTorch, or ROCm packages. After installing a compatible native Aiter build, clone this repository into `ComfyUI/custom_nodes/ComfyUI-Aiter-Attention` and restart ComfyUI. Connect `MODEL → Aiter Attention Patch → sampler`.

## Routing

Aiter is attempted only for compatible unmasked FP16/BF16 calls without causal attention, GQA, or an attention-precision override, with matching Q/K/V device and dtype and compatible head layout. Other calls use ComfyUI's per-device attention selection; GPU calls follow the selected backend, while CPU calls use ComfyUI's CPU attention path. For example, on GPU with `--use-pytorch-cross-attention`, fallback uses PyTorch SDPA. ComfyUI's Anima attention calls PyTorch SDPA directly and bypasses `optimized_attention`, so this patch does not affect Anima's direct-SDPA path.

The patch replaces any existing per-model `optimized_attention_override`; if multiple model patch nodes set one, the last connected patch wins and the earlier override is not chained. `--enable-triton-backend` enables comfy-kitchen's Triton backend; it does not select Aiter attention.

## SDXL validation

The run used ComfyUI 0.38.0 and Triton 3.7.1. The Aiter source checkout was [`bf4064642df4277054ebf46832490cd61c0a12c7`](https://github.com/ROCm/aiter/commit/bf4064642df4277054ebf46832490cd61c0a12c7); its prebuilt native FP16 MHA JIT extension was reused from the existing JIT cache, so this test did not freshly rebuild the extension from that checkout.

Five paired 20-step runs used `waiIllustriousSDXL_v160.safetensors`, 1024×1024, batch 1, DPM++ 2M Karras, CFG 7, and seed 20261008. The Aiter patch routed 2,800 attention calls through native Aiter in each timed run, with zero fallbacks. The baseline used `--use-flash-attention --disable-xformers --enable-triton-backend --fast autotune`; `flash_attn_2_cuda` was unavailable, so ComfyUI's Flash Attention route used its Triton implementation.

| Median timing | ComfyUI baseline | Aiter patch | Baseline / Aiter |
| --- | ---: | ---: | ---: |
| Sampler wall time | 15.62130 s | 14.29597 s | 1.09271× |
| Full generation wall time | 17.33664 s | 16.04979 s | 1.08018× |

Full-generation timing starts after device synchronization and includes sampling, VAE decode, image save, finite-value checks, and queue checks. It excludes checkpoint loading and text conditioning, which were reused. The median full-generation wall time was 7.42% lower; the five paired reductions ranged from 6.923% to 8.084% (median 7.928%). Queue monitoring recorded 243 polls and no overlap.

Both seeds produced finite outputs with matching latent and image shapes and dtypes, but outputs were not bit-identical. Visual review found similar compositions with small detail changes. On seed 20261008, decoded-image mean absolute difference was 0.0162 (RMSE 0.0382); on seed 20261009 it was 0.0257 (RMSE 0.0572). Direct attention checks for both observed SDXL Q shapes were within the configured tolerance.

These results cover this Windows gfx1151 SDXL workflow only; they do not establish a speedup for other models or attention paths.
