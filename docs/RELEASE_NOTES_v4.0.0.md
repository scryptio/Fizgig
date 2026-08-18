# Fizgig v4.0.0 — video, sound, training samples and 16 GB fixes

MiniMax H3 is an omni model — it generates video and audio together — and until now Fizgig
could only feed it photographs. This release completes the picture. Train on **short video
clips**, on **their sound**, on **voice recordings alone** (plain audio files — wav, mp3,
flac, m4a) — and crucially, **all of it at once: photos, clips and voice recordings in one
folder train one LoRA in one run.**

The strict part — clips and voice segments have to be exactly on H3's spec — is handled by
**Gizmo**, a new prep tool that ships with Fizgig. It cuts clips from any footage, cuts voice
segments from any recording, and includes a push-to-record studio that builds a voice dataset
from nothing but a microphone and ten minutes of reading.

There's a second release hiding underneath: **16 GB and 24 GB cards now train H3 on the
accurate int8 base instead of 4-bit**, thanks to a redesign of block swap contributed by
[@rintic-13](https://github.com/rintic-13) that makes it several times faster.

## How do I…

**…train on video clips?** Cut them with **Gizmo** (launch it from the Image Prep tab, or the
*Launch Gizmo* .bat in Fizgig's folder) — it exports clips already on H3's spec — then drop
them into the training folder next to your images, and caption them on Fizgig's
**Captions tab** exactly like a photo (each clip shows a frame from its middle there).
**Photos, clips and voice recordings all train together in the same folder** — no settings, no
separate runs. (Clips from elsewhere work too, if they're exactly on spec: 24 fps, one of
eight lengths, /32 dimensions, 32 kHz audio.)

**…make clips from my footage?** Open Gizmo (from the Image Prep tab, or the *Launch Gizmo*
.bat), drop a video on it, scrub to a moment, pick a length, *Add to queue* — repeat for every
section you want, then *Export queue*. Crop to the subject so every token goes on what you
actually want learned.

**…chop a long video automatically?** Gizmo's **✂ Auto-chop** scans the whole source for scene
cuts and offers every segment as a thumbnail — click to keep or skip, double-click to inspect,
and the keepers join the export queue. No clip ever straddles a cut.

**…train a voice from a recording?** Gizmo's **Voice** tab: open any audio file — or a video,
to use just its soundtrack — mark segments on the waveform, caption them (describe only the
sound; **Transcribe** appends the spoken words), export. Segments come out sample-exact with
their caption `.txt` beside them, ready to drop into the training folder. Voice captions are
written here in Gizmo, not on the Captions tab — describing a voice means hearing it, and the
Captions tab points you back here when it sees audio files.

**…record a voice dataset from scratch?** Voice tab → **🎙 Record**. Gizmo prompts a sentence
to read and a delivery style — rotating both take by take, so tonal range arrives on its own —
you hold the button (or the **R** key) while you speak, and every release lands as a take with
its caption already written, loaded into the editor ready to queue.

**…keep a clip's sound out of training?** Mute it — a per-clip toggle in Gizmo that adds
`_mute` to the filename, so you can also change your mind later by renaming. A muted clip
trains its video normally.

**…train photos, clips and a voice into one LoRA?** Put them all in the same folder — one
trigger word, one run, any mix. If one side of the dataset is much smaller than the other, the
new **Finish one category early** row on the Training tab lets voice (or photos & clips)
finish at a chosen epoch while the rest trains on.

**…get fast previews while training?** Set the **Turbo LoRA** (~780 MB, its own Preferences
row with a download link — you may already have it in ComfyUI's loras folder): previews then
render in **6 steps** with the Turbo applied at **75%** on top of your training LoRA, the
same pairing fast ComfyUI inference uses. Previews only — your saved LoRA never contains it.
Steps and strength are adjustable on the Samples tab.

**…hear what it's generating while training?** Pick a **"with sound"** Sample length on the
Samples tab (56 or 124 frames). Each preview denoises its soundtrack along with the picture,
and the gallery opens it as a **real playable clip** — click play, never autoplay. This is
how you hear a voice LoRA converge without leaving Fizgig.

**…set it up?** One extra model file for sound: the **audio VAE** (~605 MB), on its own
Preferences row with a download link. Leave it blank and clips simply train silent; it's
required only once the folder contains voice recordings. If your H3 paths are already set,
Fizgig points out the new files (audio VAE, Turbo LoRA) once at startup — download or
dismiss.

## Gizmo — the clip and voice prep tool

Fizgig refuses off-spec media rather than quietly transcoding it, because silent fixes make
two identical-looking datasets train differently. Gizmo is the other half of that deal: a
separate app (it opens in under a second — no torch, no CUDA) that turns whatever you have
into files Fizgig accepts.

<p align="center"><img src="https://raw.githubusercontent.com/shootthesound/Fizgig/master/assets/gizmo_video.png" alt="Gizmo — Find the moment: first/last frame previews with frame-accurate stepping" width="720"></p>

**Video clips:** open any footage — any format, frame rate or size — and cut to-spec clips
from it. A **▶ Play** button plays the marked clip exactly as it will export — same span,
same pace, slow motion included, sound and crop overlay along for the ride. First-and-last-
frame previews before you commit, frame-accurate stepping, crop to the subject with shape
locks (1:1, 16:9, 9:16…), per-clip sound-or-mute, and a mark-everything-then-export-once
queue. The preview follows the playhead live while you drag, and Gizmo tells you which clip
lengths your card can actually train, at which megapixels, before you cut anything.

<p align="center"><img src="https://raw.githubusercontent.com/shootthesound/Fizgig/master/assets/gizmo_voice.png" alt="Gizmo — Voice tab: waveform with a marked segment, trigger word, transcribed caption and grid lengths" width="720"></p>

**Voice segments:** a waveform editor for cutting training segments out of any recording.
Zoom rides the mouse wheel, segments snap to H3's allowed lengths, space and J-K-L work like
an edit suite, and Whisper transcription (one-time ~150 MB download) appends the spoken words
to your caption. The timeline always extends a little past the audio, so a segment that fills
the whole file keeps its edge grips reachable. Exports are sample-exact — the trainer's
strict duration check always passes. The trigger word and both save folders are remembered
between sessions.

**The recorder:** push-to-record (hold the button or the **R** key), in a card on the Voice
tab — your takes survive leaving and re-entering record mode. Every take is cleaned up for
you: the silence at both ends and the key click itself are trimmed off, and the cut is
widened to exactly the grid length that holds your words, using the real room tone around
them — so the auto-picked length always fits what you said. Prompted sentences rotate
through every length from a quick interjection to a line that fills the 5.2 s slot, and
across five tonal flavours; the delivery style (cheerfully, wearily, quietly…) rolls at
random after every take, visible in a dropdown you can override — or switch off in ⚙
settings, along with the trimming. The mic rolls continuously, so push-to-record clips
nothing.

## 16 GB and 24 GB cards: int8, streamed

Block swap used to round-trip every swapped block between GPU and CPU. But the base model is
*frozen* during LoRA training — nothing about it changes — so the return leg was pure waste.
[@rintic-13](https://github.com/rintic-13) proposed and prototyped a one-way design: blocks
stream host-to-GPU into a small ring of buffers, the next block prefetching while the current
one computes. As promised on [#73](https://github.com/shootthesound/Fizgig/issues/73): this is
their speedup, and 16 GB users get the biggest share of it.

Measured at the same swap depth on the same card, the streamed path is **6.4× faster** than
the old round-trip (rintic-13 measured 2.7× on a 5060 Ti with their prototype). That changes
what the Auto planner picks: swap is now cheap enough that **16 GB and 24 GB cards get the
int8 base** — the checkpoint's own storage, ~0.17% error — where they previously fell back to
4-bit (~9% error the LoRA had to spend capacity correcting):

| Free VRAM | What Auto does now |
|---|---|
| ~30 GB | int8, no block swap |
| ~22 GB | int8, ~14 blocks streamed |
| ~15 GB | int8, ~36 blocks streamed |
| ≤12 GB | 4-bit, as before |

Alongside it, **text-encoder caching now genuinely fits a 16 GB card** — the nvfp4 encoder's
dequantization was rebuilt to run in bounded chunks, removing the out-of-memory (and the
"caching appears frozen" symptom) that 16 GB users hit at the start of a run. Both paths were
validated end-to-end on a hard-capped 16 GB budget, through full runs with previews cycling.

## Turbo previews

Set the **Turbo LoRA** in Preferences (~780 MB) and in-training previews render in **6 steps
instead of 20** — the Turbo is applied at 75% on top of your training LoRA for the render
only, then removed before the next training step. Your saved LoRA never contains it, training
math never sees it, and a failed load falls back to the standard 20-step pass with a console
note rather than rendering 6-step mush. Steps and strength are adjustable on the Samples tab;
6 at 75% is the tested recommendation. If your H3 paths are set but the Turbo (or the audio
VAE) isn't, Fizgig mentions it once at startup.

The application matches what the community's dedicated Turbo loader node does, including the
part that keeps few-step **audio** intact: the Turbo's AdaLN correction cannot live as
weights on the pruned base, so it is re-injected at render time through the full model's
silu(t\_emb) grid (bundled, Apache-2.0, credit @larryvrh) — and removed with everything else
before training resumes.

## Previews with sound

Pick a **"with sound"** Sample length and each clip preview carries the model's **generated
audio**: the soundtrack it denoises alongside the picture is decoded (the audio VAE's decoder
half, newly ported from the reference) and muxed with every frame into a playable mp4. In the
gallery those samples wear a 🎬 badge and open as a **real video player** — click play, never
autoplay; scrub-only clips keep the frame slider as before. Pairs with the Turbo for 6-step
picture-and-sound previews, which is how a voice LoRA gets judged without leaving Fizgig.
Needs the audio VAE in Preferences — the same file voice training uses.

Previews default to **768×768, 56-frame clips with sound** — the regime H3 was trained in
(stills and other lengths stay in the dropdown; no audio VAE just means silent clips). A
preview that outgrows VRAM steps itself down a ladder rather than dying — a shorter clip
first (56 → 22 frames, since a shorter clip is still a clip), then resolution (768×640,
640×640, on to a 512×512 floor) — triggered by a hard OOM *or* by detecting the Windows
paging crawl, which never raises an error on its own — with a note that below 768 the model
is outside its training canvas, and the size that fit is saved as the new default so the
next run starts there.

## Finish one category early

Mixed datasets are rarely balanced — thirty photos and two hundred voice takes, or the
reverse. The **Finish one category early** row (Training tab, visible when the dataset is
mixed) takes a category, an epoch, and a mode: **anchor at 10% LR** keeps the finished
category gently in the loop so the shared adapter doesn't drift away from it (recommended), or
**stop completely** skips its steps for speed. Recorded in the LoRA's metadata.

## Voices train best at Likeness and Style

Tested head-to-head on the same voice dataset: **Likeness and Style** converges much faster
and sounds better than *Model default* — identity lives at the clean end of the noise
schedule for voices just as it does for faces. Fizgig now says so on the Training tab whenever
it sees voice recordings in the dataset, and the sample gallery header shows when a dataset is
audio-only.

## Also in this release

- **Resuming a paused H3 run now rebuilds the checkpoint's own network** — rank, alpha and
  LoRA-vs-LoKR are read from the saved state and override whatever the Training tab shows,
  with a console note. Settings that moved on since the pause used to crash the relaunch.
- **The built-in H3 presets are now two genuinely different recipes**: Defaults runs rank 16
  at a flat 1e-4, Fast runs rank 8 at a flat 2e-4 — Adapter-relative LR ships off in both
  and stays available in Other Options.
- **The Captions tab no longer freezes on a folder with clips** — clip thumbnails come from
  a single seeked frame (cached) instead of decoding every frame of every clip on the way in.
- **Clips with sound cache correctly at every length** — 5-, 56- and 107-frame clips were
  refused by an audio rounding mismatch; the fit is now sample-exact for all eight lengths.
- **LoRA names that can't become filenames are refused before the run starts**, with a plain
  message, instead of failing at the first save — reported by
  [@ioritree](https://github.com/ioritree) ([#70](https://github.com/shootthesound/Fizgig/issues/70)).
- **The VRAM warning names the teacher when the teacher is the cause** — diagnosis by
  [@volnodumcev](https://github.com/volnodumcev) ([#71](https://github.com/shootthesound/Fizgig/issues/71)).
- **A clip's cache is never a deleted file's leftovers** — stale cache entries from removed or
  renamed clips are detected and skipped, per file, with a console note.
- **Preferences model-path sections fold up**, and the missing-paths badge says which model
  family it's talking about.
- **Widescreen previews no longer crop in the gallery grid** — cards letterbox the whole
  frame, so composition reads honestly at a glance.
- **A preview still crawling at the ladder floor gets plain advice** — set the Turbo LoRA,
  or switch to a still — instead of a mystery slowdown. (Above the floor, the crawl triggers
  the ladder itself; see Previews with sound.)
- **Using the Turbo LoRA in ComfyUI? Skip its custom sampler** — current ComfyUI samples H3
  audio cleanly with stock Euler, and the community consensus is 8 steps. Details in the
  README.

## Upgrading

Nothing to do — model paths, datasets, caches and presets are untouched. If you want sound:
add the audio VAE (~605 MB) on its new Preferences row. If you want fast previews: the Turbo
LoRA (~780 MB) likewise — Fizgig will point both out once at startup if your H3 paths are
set. Everything else is optional too; stills-only training works exactly as before.
