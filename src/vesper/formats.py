"""NumPy specification oracles. These are not the official TileLang kernels.

E2M1 and E4M3 encodings follow their finite floating-point formats. The block
quantizer follows DeepSeek's pinned kernel.py formulas; see THIRD_PARTY_NOTICES.
"""

from __future__ import annotations

import numpy as np

FP4_MAGNITUDES = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)


def fp4_values(codes: np.ndarray) -> np.ndarray:
    codes = np.asarray(codes, dtype=np.uint8)
    if np.any(codes > 15):
        raise ValueError("An E2M1 code occupies four bits")
    values = FP4_MAGNITUDES[codes & 7].copy()
    bits = values.view(np.uint32)
    bits |= (codes.astype(np.uint32) & 8) << 28
    return values


def e8m0_values(codes: np.ndarray) -> np.ndarray:
    codes = np.asarray(codes, dtype=np.uint8)
    bits = codes.astype(np.uint32) << 23
    bits = np.where(codes == 0, np.uint32(0x00400000), bits).astype(np.uint32)
    bits = np.where(codes == 255, np.uint32(0x7FC00000), bits).astype(np.uint32)
    return bits.view(np.float32)


def unpack_fp4(packed: np.ndarray, scales: np.ndarray, block: int = 32) -> np.ndarray:
    packed = np.asarray(packed, dtype=np.uint8)
    if block != 32 or packed.shape[-1] * 2 % block:
        raise ValueError("Native expert blocks contain 32 values")
    codes = np.empty((*packed.shape[:-1], packed.shape[-1] * 2), dtype=np.uint8)
    codes[..., 0::2], codes[..., 1::2] = packed & 15, packed >> 4
    scales = np.asarray(scales, dtype=np.uint8)
    if scales.shape != (*codes.shape[:-1], codes.shape[-1] // block):
        raise ValueError("Scale shape does not match packed FP4 weights")
    with np.errstate(over="ignore", invalid="ignore"):
        return fp4_values(codes) * np.repeat(e8m0_values(scales), block, axis=-1)


def e4m3_values(codes: np.ndarray) -> np.ndarray:
    codes = np.asarray(codes, dtype=np.uint8)
    exp, mantissa = (codes >> 3) & 15, codes & 7
    values = np.where(exp == 0, mantissa.astype(np.float32) / 512,
                      np.ldexp(1 + mantissa.astype(np.float32) / 8, exp.astype(np.int32) - 7)).astype(np.float32)
    values[(exp == 15) & (mantissa == 7)] = np.nan
    values.view(np.uint32)[:] |= (codes.astype(np.uint32) & 128) << 24
    return values


def encode_e4m3(values: np.ndarray) -> np.ndarray:
    """Independent nearest-table encoding, round-to-nearest-even, finite saturation."""
    values = np.asarray(values, dtype=np.float32)
    levels = e4m3_values(np.arange(127, dtype=np.uint8))
    magnitude = np.minimum(np.abs(values), np.float32(448))
    upper = np.minimum(np.searchsorted(levels, magnitude), 126)
    lower = np.maximum(upper - 1, 0)
    dl, du = magnitude - levels[lower], levels[upper] - magnitude
    use_upper = (du < dl) | ((du == dl) & ((upper & 1) == 0))
    result = np.where(use_upper, upper, lower).astype(np.uint8)
    result |= np.where(np.signbit(values), 128, 0).astype(np.uint8)
    return np.where(np.isnan(values), 127, result).astype(np.uint8)


def bf16_bits(values: np.ndarray) -> np.ndarray:
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    nan = ((bits & 0x7F800000) == 0x7F800000) & ((bits & 0x7FFFFF) != 0)
    return np.where(nan, (bits >> 16) | 0x40, rounded >> 16).astype(np.uint16)


def bf16_values(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def quantize_activation(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The reference's BF16 input, per-32 amax floor, E8M0 scales and E4M3 cast."""
    x = bf16_values(bf16_bits(values))
    if x.shape[-1] % 32 or not np.isfinite(x).all():
        raise ValueError("Activation must be finite with a width divisible by 32")
    grouped = x.reshape(*x.shape[:-1], -1, 32)
    amax = np.maximum(np.abs(grouped).max(axis=-1), np.float32(1e-4))
    scaled_max = amax * np.float32(1 / 448)
    bits = scaled_max.view(np.uint32)
    exponents = ((bits >> 23) & 255).astype(np.int32) - 127 + ((bits & 0x7FFFFF) != 0)
    if np.any((exponents < -126) | (exponents > 127)):
        raise ValueError("Activation scale outside this probe's finite normal range")
    scales = (exponents + 127).astype(np.uint8)
    normalized = grouped / e8m0_values(scales)[..., None]
    return encode_e4m3(np.clip(normalized, -448, 448)).reshape(x.shape), scales


def fp4_gemv_spec(activation: np.ndarray, a_scales: np.ndarray,
                  packed_weight: np.ndarray, w_scales: np.ndarray) -> np.ndarray:
    """Block dots in float64, rounded/scaled/accumulated in FP32, final BF16.

    The high-precision inner dot is an independent oracle, not a claim about
    Tensor Core reduction order. Official-kernel parity is a separate gate.
    """
    a = e4m3_values(np.asarray(activation, dtype=np.uint8)).reshape(-1, 32)
    weight = np.asarray(packed_weight, dtype=np.uint8)
    n, packed_k = weight.shape
    k = packed_k * 2
    if a.size != k or np.shape(w_scales) != (n, k // 32) or np.size(a_scales) != k // 32:
        raise ValueError("Mismatched GEMV dimensions")
    codes = np.empty((n, k), dtype=np.uint8)
    codes[:, 0::2], codes[:, 1::2] = weight & 15, weight >> 4
    w = fp4_values(codes).reshape(n, -1, 32)
    dots = np.einsum("bg,nbg->nb", a.astype(np.float64), w.astype(np.float64)).astype(np.float32)
    sa, sw = e8m0_values(a_scales).reshape(-1), e8m0_values(w_scales)
    result = np.zeros(n, dtype=np.float32)
    for group in range(k // 32):
        result = result + (dots[:, group] * sa[group]) * sw[:, group]
    return bf16_bits(result)
