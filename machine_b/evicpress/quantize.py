"""
Machine B-side quantization for KV-cache blocks.

Machine A always ships / receives fp16 torch.save() bytes. Machine B is free
to store the block at a lower precision internally (int8 / int4) as decided
by the placement band. On Retrieve, Machine B dequantizes back to an fp16
tensor so the wire format Machine A sees never changes.

Payload formats (all torch.save() tuples):
  fp16  : the raw bytes from Machine A (passthrough)
  int8  : (q_tensor[int8], scale[float32], orig_dtype, orig_shape)
  int4  : (packed[uint8], scale[float32], orig_dtype, orig_shape, orig_numel)

Quantization is symmetric per-tensor. Int4 uses two 4-bit values packed per
byte. Both are lossy — int8 keeps ~7 bits of magnitude, int4 ~3 bits. This is
intentional: the utility function decides when a block is cold enough to
accept that loss.
"""

import io

import torch


_VALID_LEVELS = ("fp16", "int8", "int4")


def _load(data: bytes):
    return torch.load(io.BytesIO(data), weights_only=True)


def _save(obj) -> bytes:
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def quantize(data: bytes, level: str) -> bytes:
    """Encode a torch-saved fp16 tensor into the given quant level."""
    if level == "fp16":
        return data
    if level not in _VALID_LEVELS:
        raise ValueError(f"unknown quant level {level!r}")

    tensor = _load(data).detach().cpu()
    orig_dtype = tensor.dtype
    orig_shape = tuple(tensor.shape)
    flat = tensor.flatten().to(torch.float32)

    amax = flat.abs().amax().item()
    if level == "int8":
        scale = amax / 127.0 if amax > 0 else 1.0
        q = (flat / scale).round().clamp(-127, 127).to(torch.int8)
        return _save(("int8", q, float(scale), str(orig_dtype), orig_shape))

    # int4 — pack two 4-bit signed values into each uint8 byte.
    scale = amax / 7.0 if amax > 0 else 1.0
    q = (flat / scale).round().clamp(-7, 7).to(torch.int8)
    numel = q.numel()
    if numel % 2:
        q = torch.cat([q, torch.zeros(1, dtype=torch.int8)])
    lo = (q[0::2] & 0x0F).to(torch.uint8)
    hi = ((q[1::2] & 0x0F) << 4).to(torch.uint8)
    packed = lo | hi
    return _save(("int4", packed, float(scale), str(orig_dtype), orig_shape, numel))


def dequantize(data: bytes, level: str) -> bytes:
    """Decode quantized bytes back into a torch-saved fp16 tensor."""
    if level == "fp16":
        return data
    if level not in _VALID_LEVELS:
        raise ValueError(f"unknown quant level {level!r}")

    obj = _load(data)
    tag = obj[0]
    if tag != level:
        raise ValueError(f"payload tag {tag!r} doesn't match requested level {level!r}")

    if level == "int8":
        _, q, scale, orig_dtype_str, orig_shape = obj
        dtype = _resolve_dtype(orig_dtype_str)
        t = q.to(torch.float32) * scale
        return _save(t.reshape(orig_shape).to(dtype))

    # int4
    _, packed, scale, orig_dtype_str, orig_shape, numel = obj
    dtype = _resolve_dtype(orig_dtype_str)
    lo = (packed & 0x0F).to(torch.int8)
    hi = ((packed >> 4) & 0x0F).to(torch.int8)
    # sign-extend 4-bit → 8-bit
    lo = torch.where(lo >= 8, lo - 16, lo)
    hi = torch.where(hi >= 8, hi - 16, hi)
    flat = torch.empty(lo.numel() + hi.numel(), dtype=torch.int8)
    flat[0::2] = lo
    flat[1::2] = hi
    flat = flat[:numel].to(torch.float32) * scale
    return _save(flat.reshape(orig_shape).to(dtype))


def _resolve_dtype(name: str) -> torch.dtype:
    """Map the stringified original dtype ('torch.float16') back to a dtype."""
    if name.startswith("torch."):
        name = name[len("torch."):]
    return getattr(torch, name)
