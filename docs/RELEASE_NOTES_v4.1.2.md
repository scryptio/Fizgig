# Fizgig v4.1.2 — the post-preview slowdown on 16 GB cards, fixed

A maintenance release finishing the 16 GB work from v4.1.1. If you train MiniMax H3 on a
16 GB card with previews on, this one matters: the slow-motion training that set in after
the first preview is gone.

## Previews now give the memory back

After the first preview of a run, some machines saw system RAM lock at its ceiling and
training steps run several times slower for the rest of the run. The cause was memory that
previews borrowed but could never truly return — every preview left more behind, and the
training loop paid for it on every step after.

That memory is now genuinely released. Steps return to full speed within seconds of each
preview, RAM stays level across the whole run, and the effect holds no matter how many
epochs and previews a run goes through. Verified end-to-end on a real 16 GB 4090.

Two smaller pieces of the same fix: the video decoder now loads once per run instead of
once per preview (a steady RAM climb on 32 GB machines), and previews on tight cards now
park only as much of the model as the decode actually needs.

## int8 on 16 GB machines now works — but 4-bit stays the default

With the fixes above, forcing **Base Precision: int8** on a 16 GB card with 32 GB of RAM
now runs where it previously crashed. It is still much slower than the 4-bit base — the
streaming that makes it fit is the cost — so the planner's automatic choice is unchanged.
Pick int8 manually if you want the checkpoint's exact base and can live with the speed.

## Clearer console during previews

Previews on smaller cards now log a one-line memory accounting at each phase, so if
anything is ever tight you can see exactly where. Deeper diagnostics only appear if
something is actually wrong.

## Upgrading

Nothing to do. Your model paths, datasets, caches and presets are untouched.
