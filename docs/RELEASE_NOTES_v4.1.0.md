# Fizgig v4.1.0 — choose your H3 base, and Gizmo stops cutting you off

## Train on the MiniMax model you deploy on

A new **Training Base** dropdown sits right under the Base Model selector at the top of the
Training tab.
**First/last frame (fl2va)** — the standard model most workflows run — stays the default;
pick **Reference (ref2va)** if your LoRA's home is the Reference-to-Video workflow. A LoRA is
most faithful on the base it trained against, so now you train on the one you'll actually use.

Presets deliberately never change this — it's a deployment choice, not a recipe ingredient —
but **Load Settings From Last Train** and queued runs remember it. If you pick ref2va without
the reference model set up, the launch check tells you exactly what to do: tick *Include the
reference DiT* on the Preferences tab and hit *⬇ Download models for me*.

## Gizmo records for as long as you talk

Recordings were being trimmed to the length of the take slot — hold the key for a minute,
keep five seconds. Long holds are now kept in full, so you can record a couple of minutes of
speech and carve out the takes you want in the editor.

There's also a new **Free speech** toggle on the record card: with it on, nothing is trimmed
at all — speak as long as you like, drop the recording on the timeline, cut it up from there.

## Upgrading

Nothing to do. Your model paths, datasets, caches and presets are untouched.
