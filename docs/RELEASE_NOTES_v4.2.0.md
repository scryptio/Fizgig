# Fizgig v4.2.0 — the workbench opens to MiniMax H3, and what it found ships as features

Every post-training tool now speaks H3: the Profiler, Extract, the Repair Studio, LoRA the
Explorer and LoRA Royale. Then we used those tools on real H3 LoRAs, mapped which of the 50
blocks do what — and the biggest finding became a training feature that's on by default.

## How do I…

- **…train a person/likeness LoRA?** Nothing new to do — Optimised Likeness Learning is on by
  default and makes photo training sharper, more prompt-responsive, better-sounding and faster.
- **…train a style?** Load the new **✨ MiniMax H3 Style** preset. It picks the right blocks and
  a gentler learning rate for you.
- **…fix an overbaked H3 LoRA?** Repair Studio → pick MiniMax → load it, drag sliders, watch the
  preview. Same workflow as Klein and Krea 2.
- **…shrink an H3 LoRA?** Extract tab → MiniMax → Fast SVD. No GPU needed.

## Optimised Likeness Learning

The headline feature, and it's a checkbox that's already ticked. Photo steps now train on blocks 20-49, where I've found the best likeness and convergence speed while keeping the results from the model sharp and flexible; video and audio clips still train the full model.
Measured against full-model photo training on the same dataset: sharper output, much better
prompt following, better sound, fewer epochs — and better training previews.

Why it works: photos carry little signal for the early blocks, which hold composition, anatomy and
rendering — so training there mostly damaged the base model's skills. Freeze them
and your LoRA's whole capacity lands on the subject.

Untick it for style or scene training (the Style preset does). While it's on, Blocks to Train
is disabled with a note.

## The MiniMax H3 block map

The numbers behind the feature, measured with Fizgig's own tools and yours to use in the Blocks
to Train box:

- **Likeness: `20-49`** — identity, the subject's pose and presentation, and voice.
- **Style: `0-3, 6-47`** — style lives almost everywhere; only 4-5 and 48-49 are droppable.
- **Voice: core `38-48`**, generous `34-49`. (Block 5 is the audio *embedder* — essential in
  the base model, but measured to barely adapt under voice training: audio flows through the
  front, voices are learned in the back.)
- Blocks **0-19** are the fragile shared machinery: likeness training should never touch them
  except maybe at a very low LR. Style training uses them at the normal rate without trouble.

## ✨ MiniMax H3 Style preset

The Fast recipe on the measured style blocks, `0-3, 6-47` — style lives almost everywhere in
H3 except the few blocks that only do identity and voice. Optimised Likeness Learning is off
in this preset by design.

*Updated after release:* the preset originally shipped at a halved 1e-4 with a
lower-it-further tip, but real style runs found the standard **2e-4** to be what style
actually needs — the gentler rates just train slower for no measured benefit. Update Fizgig
and reload the preset to pick up the new rate.

## The workbench tabs, for H3

- **Repair Studio** — 50 block sliders + 2 refiners, live side-by-side preview. H3 previews
  render a 22-frame clip (the model's native regime) judged by its middle frame — Turbo 6-step,
  ~12 s once warm on a big card.
- **Profiler** — weight-profile HTML report for any H3 LoRA, no GPU needed, with the JSON sidecar
  the Repair Studio picks up to show your LoRA's hottest blocks inline.
- **Extract** — rank-reduce H3 LoRAs (Fast SVD, no GPU). LyCORIS sources supported.
- **LoRA the Explorer** — evolutionary block discovery, now on H3's 50 blocks.
- **LoRA Royale** — epoch-by-epoch battles for H3 checkpoint folders, ArcFace likeness scored on
  the clip's middle frame.
- Prompts are encoded once and cached to disk — the 32B text encoder's couple-of-minutes load
  happens on a prompt's first-ever use, then never again, across sessions.
- Base precision is planned from free VRAM: big cards load the checkpoint's exact int8 base,
  smaller ones the 4-bit — no configuration needed.

## Repair Studio, sturdier everywhere

- LoRA and pipeline loads run off the UI thread — no more frozen window on big models — with a
  green progress bar while they work.
- Unloading models now genuinely returns the VRAM, on every family.
- **Reset All Sliders** button beside Start; user presets are now saved per model family.
- The seed ↻ button (and Enter in the seed box) regenerates the preview immediately on a live
  session; the Res dropdown already did.
- LyCORIS H3 LoRAs (LoKR — the trainer's own optional format) now load correctly everywhere.

## Append Transcription — a clip's speech, straight into its caption

Open a training video in the caption editor (Captions tab → click a clip) and, when it carries
sound, a new **🎤 Append Transcription** button appears. One click transcribes the clip with
Whisper and appends the words as `saying "…"` — the same caption grammar Gizmo writes. Muted
(`_mute`) clips never show it.

Speech the model can hear but the caption doesn't mention is something it must explain away —
this is the one-click fix for clips that never went through Gizmo. It uses Gizmo's language
setting (English if you've never picked one) and leaves the words in the box for editing.

## Whisper joins the model downloader

Whisper (~300 MB) powers Gizmo's Transcribe and the new Append Transcription button. Every "⬇ Download
models for me" button in Preferences now fetches it up front, and once it's on disk it loads
locally and never reaches for the internet — transcription works fully offline. Nothing to do if
you'd rather not: first use still downloads it on demand, exactly as before.

## Also in this release

- 16 GB cards: the live sample-override box now obeys the same preview cap (768×640 / 640×768)
  as the Samples tab — it was a way to request a resolution the card can't survive.
- Blocks to Train hint now carries the measured recipes instead of "no recommended answer yet".
- The Repair Studio's second DiT selector no longer shows a stray empty option under MiniMax.
- Sturdier error handling when swapping LoRAs mid-session in the Repair Studio.

## Upgrading

Nothing to do. Your model paths, datasets, caches and presets are untouched. Existing runs and
settings files load as before — Optimised Likeness Learning simply appears ticked on MiniMax
training, and you can untick it.
