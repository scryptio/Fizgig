# Fizgig v3.6.2 — pick your GPU, and MiniMax caching on 16 GB

## Choose which graphics card Fizgig uses

If your machine has more than one GPU, **Preferences** now has a **Graphics Card** card listing
each one by name and size. Pick one and everything follows it — training, caching, samples, the
workbench tabs and the VRAM gauge in the status bar.

Single-GPU machines see nothing new.

The VRAM gauge also reads the right card now if you were already setting `CUDA_VISIBLE_DEVICES`
yourself: it used to report card 0 whatever you had chosen. Your own environment variable still
wins over the preference, so an existing setup keeps working exactly as it did.

Reported by **@llefort001** (#60).

## MiniMax text caching fits a 16 GB card

Caching could sit on `Encoding dataset [0]` for a very long time on a 16 GB card, with no error
and nothing to explain it. The text encoder was a little too large for the card, and Windows
handles that by quietly moving the overflow into system RAM rather than stopping — so instead of
failing, the pass crawled.

It now needs **1.5 GB less VRAM**, which brings it inside what a 16 GB card can give it. Nothing
about your captions changes: the cached values are bit-for-bit what they were, so existing caches
stay valid and nothing needs re-running.

The pass also reports what it found before it starts, and says so plainly when there genuinely
is not enough room:

```
[vram] text encoder resident 12.8 GB, 2.3 GB of 15.9 GB free
```

If you are close to the line, close anything else using the GPU first — ComfyUI, and browsers,
which can hold a couple of GB on hardware acceleration.

## LoRA Royale: crossfade settings sit with the crossfade

**Size** and **Max renders** have moved from Setup down into the Crossfade card, which is the only
thing they affect. The seed row in Setup now says which features share it.

## Upgrading

Nothing to do. Your model paths, datasets, caches and presets are untouched.
