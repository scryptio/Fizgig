# Fizgig v3.7.0 — MiniMax H3 LoRAs that work without the Turbo LoRA

## The strength problem, and what was actually causing it

Some users reported the same thing: a MiniMax H3 LoRA works great with Turbo, then you unload it,
run the stock 20-step workflow, and it goes soft or distorts. The working advice was to drop the
strength to 0.4 or 0.5.

That advice is no longer needed, because the cause has been found and removed. It was a setting
called **mid-concentrated**.

Mid-concentrated clustered training around the middle of the noise range and thinned both ends.
The end it quietly removed was the noisiest one — where pose, framing and the shape of a face are
decided. It left about 0.7% of a run up there. A 20-step render spends most of its time in exactly
that territory, so the LoRA was being asked to hold a structure it had barely been trained on.
Under Turbo's four steps that never showed. Without it, it did.

It has been removed rather than switched off. Across five datasets, LoRAs trained without it run
**at full strength, without Turbo, with no distortion** — and likeness did not suffer in the
process, so it was not buying anything it was supposed to be buying either.

**If you have a preset that used it**, it will simply train without it from now on. Older LoRAs are
unchanged and may still want the lower strength; anything you train from here should not.

## Training Structure

The setting behind all of this used to be a raw percentage in a collapsed section. It is now a
named control near the top of the Training tab, with two settings and a Custom box:

| Setting | What it's for |
|---|---|
| **Likeness and Style** *(default)* | Stills. Skin, hair, identity — and style, which is a surface property too. |
| **Model default, movement** | Weighted the way H3's own schedule is: movement and composition over fine detail. |
| **Custom** | Type your own share. |

There is deliberately no separate "style" setting. Style lives at the same clean end as likeness —
brushwork, palette and grain are surface properties, not compositional ones. What makes a style
LoRA different is often rank and LR, not the noise schedule.

## Medium to High LR

A new box beside it, and **best left at 100 unless you're experimenting.**

It controls what the noisier steps — where pose, framing and face shape are settled — do to the
learning rate. Across five datasets, at both structures, lowering it never improved anything and
100 held face shape visibly better every time.

It's worth having anyway, because it is the *safe* way to push a run toward surface detail.
Mid-concentrated did that by changing which noise levels were trained on at all, which is what
broke LoRAs outside Turbo. This leaves the schedule alone and only changes how much is learned from
the noisy end — so it doesn't distort at any setting. **Skin and texture LoRAs are the obvious use
case**, and there are likely others worth finding.

For a likeness LoRA, leave it at 100.

## Also in this release

**A substantial security audit by [@FNGarvin](https://github.com/FNGarvin)**. He went through the
dependency stack and the container properly, tracing each finding against what the code actually
calls rather than against version numbers, which is what made it worth acting on. We then worked
through the results together. Out of it came a **Pillow 12.3.0** bump (his PR), a documented
reason for the one `shell=True` call in the Krea 2 build path, and a write-up of the pod's
security posture in `docker/README.md` — what listens, what authenticates, and why the container
runs as root. A few findings were deliberately left as they are, with the reasoning recorded
rather than left implicit.

**Installing from a ZIP no longer leaves you stuck.** `update_fizgig.bat` needs git, so a ZIP
download fails to update with `fatal: not a git repository`. The README now carries the six
commands that convert the folder you already have into a proper checkout, keeping your models,
LoRAs, caches and presets exactly where they are — rather than pointing at an issue thread.

**An install shortcut at the top of the README**, since the install section sits a long way down.

## Upgrading

Nothing to do. Your model paths, datasets, caches and presets are untouched.
