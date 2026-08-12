# Fizgig v3.6.1 — MiniMax H3 fixes

A small release on top of v3.6.0, mostly from contributors.

## 24 GB cards: MiniMax could run out of memory on the first step

If you train MiniMax on a 24 GB card with **Base Precision** on Auto, this is the one to have.

The planner decides how to load the base model and how much of it to keep on the GPU. It was
making the right decision and then the loader was quietly making a different one — planning for a
4-bit base of around 11 GB while actually loading the 21 GB one, with nothing moved to system RAM
to make room. The run then failed on step one with a CUDA out-of-memory error, having printed both
decisions a few lines apart in the log.

The two now agree. On a 24 GB card that means 4-bit, nothing swapped, and a run several times
faster than the workaround of pinning Base Precision by hand.

It also means **the precision recorded in your LoRA is now the one that was used** — affected runs
saved a file whose metadata said 4-bit while it had actually trained on int8. The LoRAs themselves
are fine; only the label was wrong.

Found, diagnosed and fixed by **@ioritree**.

## Setting Blocks Swap by hand now says what it did to your precision

A numeric **Blocks Swap** skips the planner entirely, so Base Precision on Auto falls back to
matching the checkpoint rather than your free VRAM — which on a pre-quantized file always means
int8. It works, it just runs several times slower, and nothing said why. Now the console does:

```
[vram] base precision: int8 — chosen from the checkpoint, not from free VRAM, because Blocks Swap
is set to 4 rather than Auto.
```

Leave both on Auto and they get planned together.

## MiniMax H3 is no longer labelled experimental

It trains well enough that the label was arguing with the advice. Load the **✨ MiniMax H3 Fast**
preset and press Start.

The workbench tabs — Repair Studio, LoRA the Explorer, LoRA Royale, Profiler and Extract — are
still Klein and Krea 2 only. They are planned for H3 rather than ruled out, and it will take a
while: they all render full images on demand, which is a different problem on a 33B video model.

## Also in this release

- **H3's own markup tokens are understood.** A prompt containing dialogue markup was being split
  into fragments instead of read as the tokens H3 was trained on. Plain captions are unaffected,
  so no re-caching is needed. Thanks **@johndpope**.
- **The H3 base can be loaded straight from a directory of Hub shards**, rather than needing them
  merged into one 66 GB file first. Thanks **@johndpope**.
- **The video VAE can encode a clip, not just a still.** Groundwork — training still works from
  still images — but the encoder no longer refuses. Thanks **@johndpope**.

## Upgrading

Nothing to do. Your model paths, datasets, caches and presets are untouched.
