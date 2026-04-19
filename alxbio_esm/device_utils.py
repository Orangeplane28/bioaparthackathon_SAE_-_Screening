"""Device detection and precision helpers for ESM embedding scripts."""

from __future__ import annotations

import torch


def get_device(requested: str = "auto") -> str:
    """Resolve a device string, auto-detecting the best available backend.

    Priority: cuda → mps (Apple Silicon) → cpu
    """
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def clear_cache(device: str) -> None:
    """Free the accelerator memory cache for the active device."""
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def coerce_precision(precision: str, device: str) -> str:
    """Downgrade precision when the device does not support it.

    MPS does not support bfloat16; CPU is stable only on fp32.
    """
    if device == "mps" and precision == "bf16":
        print(f"[device] MPS does not support bf16 — falling back to fp16.")
        return "fp16"
    if device == "cpu" and precision in ("fp16", "bf16"):
        print(f"[device] CPU does not support {precision} — falling back to fp32.")
        return "fp32"
    return precision
