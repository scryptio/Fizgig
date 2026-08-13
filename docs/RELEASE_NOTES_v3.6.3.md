# Fizgig v3.6.3 — MiniMax 4-bit training starts

## If MiniMax crashed on the first step, this is the fix

Training MiniMax H3 with the base loaded **4-bit** stopped on step one with an `AssertionError`
from bitsandbytes, after a `FP4 quantization state not initialized` warning. The AdaLN layers were
being set up as 4-bit and then loaded as full precision, and nothing noticed until the first
forward pass.

This is what a **16–24 GB card** gets when **Base Precision** is on Auto, so if that is your card,
this is the release to have. Larger cards load int8 and were never affected.

Nothing to change at your end — update and start the run. Your cached latents and captions are
still valid.

## Upgrading

Nothing to do. Your model paths, datasets, caches and presets are untouched.
