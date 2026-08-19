# Fizgig v4.1.1 — MiniMax training that actually fits a 16 GB card

A maintenance release, and an important one if you're training MiniMax H3 on 16 GB:
a real-world 16 GB machine went from crashing before the first step to training cleanly.
Everything here came out of one afternoon of live debugging on a 16 GB 4090 — thank you
to everyone who has reported from smaller cards.

## The planner now respects your system RAM, not just your VRAM

The int8 training plan streams the blocks that don't fit VRAM from system RAM — ~12 GB of
staging on a 16 GB card. On Windows the GPU itself is backed by system memory too, so on a
32 GB-RAM machine that combination could exhaust RAM and kill the run with a misleading
"CUDA out of memory" while the card had space free.

The trainer now checks available system RAM before choosing that plan. When it doesn't
genuinely fit, it picks the 4-bit base instead: ~10.5 GB, everything stays on the card,
nothing staged, and steps run faster because nothing crosses PCIe. The console states which
plan it chose and why.

## Previews sized for 16 GB cards

On 16 GB cards, MiniMax previews now cap at **768×640 and 22 frames** — with sound kept.
Longer or larger picks in the Samples menu still work; they simply clamp with a console
note. A full 768×768 clip ran a 16 GB card to its last 100 MB, and the bigger clip lengths
were the trigger for the crash spiral above. There's also a new **22 frames with sound**
option in the Sample length dropdown — the sweet spot for smaller cards.

## Sturdier under pressure everywhere

If the OS can't page-lock the streaming buffers, training now continues with ordinary RAM
staging (slower copies, working run) instead of dying at startup. Driver-level out-of-memory
errors during previews now walk the same shorter-clip-then-lower-resolution retry ladder as
ordinary ones instead of skipping the preview.

## Upgrading

Nothing to do. Your model paths, datasets, caches and presets are untouched.
