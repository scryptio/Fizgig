# Fizgig v3.6.0 — MiniMax H3: have the cake and eat it

**This is the release where MiniMax stops asking you to choose between quality and speed.** The
new **Fast** preset reaches full likeness in a few hundred steps, and the lower rank it uses
tends to come out *more* flexible rather than less. On top of that, a single LoRA can now hold
**two subjects**, each with its own folder and its own trigger word.

## The short version: load Fast, press Start

MiniMax has a reputation for being awkward to train, so it is worth saying plainly that it
does not need much from you any more.

Load the **✨ MiniMax H3 Fast** preset, point the Start tab at a folder of images, give them
captions with your trigger word, and press Start. The preset carries the rank, the epoch count
and the learning rate, and the memory settings work themselves out from your card. **You should
not need to change anything else**, and the defaults are where we would start ourselves.

**How many images?** Thirty-five to forty-five is plenty. For a character the easiest route
there is 20–25 full-length and mid-length shots at the highest resolution you have, then
**Resize + face close-ups** on the Image Prep tab — it keeps each photo and saves a zoomed face
crop beside it, which lands you at the right number without going hunting for more pictures.

If you would rather not use it, make sure you include some tight shots yourself. A set of nothing
but full-length photos does not give the model enough of the face to work with.

Every epoch saves a file, so you are choosing a favourite from the run rather than hoping the
last one landed — and likeness usually arrives early.

## Judging quality: use Pause, not the previews

Read this bit even if you skip the rest.

**Previews are image-based by default, and images are not what H3 is.** They will sometimes show
wild distortion that simply is not there when you load the same checkpoint in ComfyUI. That is a
property of asking a video model for a single frame — it is not your LoRA going wrong, and it is
not worth reacting to.

**Video previews are available** from the **Sample length** dropdown and are closer to the real
thing, but they slow the app down considerably and still aren't a guarantee of what ComfyUI will
give you.

**So use previews to watch likeness arrive, and judge quality somewhere else.** Every epoch
saves a `.safetensors`, and **Pause** frees the GPU — so the reliable loop is: pause the run,
load an epoch in ComfyUI, close ComfyUI, resume. It costs a couple of minutes and it is the only
read you should trust.

## Multi Concept — two subjects, one LoRA

Tick **Multi Concept** on the Training tab and a second folder picker appears. Each subject gets
its own folder, its own trigger word, and its own dataset entry.

That last part is what makes it work. In identity-learn mode every image is marked against
*other photos of the same person*, and that pairing runs per folder — so subject A is only ever
compared against A. Put two people in one folder and the pairing crosses them, which blends them
together rather than keeping them apart.

Ticking the mode also sets the settings that suit it — identity-learn on, 4 references, a short
identity-first phase and caption dropout at `0.10` — and tells you in the console what it
changed. **Nothing is locked**; they're starting points. It leaves the learning rate alone:
which LR strategy you want is the preset's business, not this box's.

Two things it expects of you: **caption both folders yourself**, each with its own unique trigger
word in *every* caption (that word is the only thing telling the two apart), and note the second
folder is training-only — Image Prep, Captions and the Look filter still follow the Start folder.

### Identity-first

Part of identity-learn, and a good deal of why two subjects stay apart. It trains the first
stretch against the **teacher only** — the model as it behaves when shown a reference photo —
then drops the teacher entirely and trains on the **photographs only**.

Photo training then starts from an adapter that already knows who each trigger word means,
instead of discovering the identities and the detail at the same time. Phase 1 runs at a third
of the Learning Rate box, since it is placing identity rather than reproducing detail, and phase
2 skips the teacher pass altogether so it costs about half as much per step.

**Auto** sizes the first phase from your dataset — enough steps for the teacher side to
converge, which is roughly the same number of steps whatever your image count and therefore
rather more epochs on a small set. Or pick a fixed number of epochs; **Off** keeps the blended
loss throughout.

**Where else it might help.** Two subjects is where this has been tested, but nothing about it
is specific to them. The same argument applies to a single character whose likeness is competing
with strong backgrounds, to a subject you have few photos of, or to any run where identity
arrives late and the rest arrives early. That is untested rather than known to be wrong — worth
trying if the ordinary route is not getting you there, but not something we are recommending
yet. It does need the reference DiT either way.

## Adapter-relative LR

**On by default in the Defaults preset** (Fast runs flat instead).

A LoRA starts at zero, so a rate that's safe at epoch 1 is far too slow by epoch 50 — and one
that's right later wrecks a fresh adapter. There's no single number that works for both.

**Adapter-relative LR** turns the Learning Rate box into a **ceiling** rather than a setting. The
run starts below it and climbs, keeping every step a fixed fraction of the adapter's current
size:

| Setting | |
|---|---|
| `0.003` | slow build — **the shipped default** |
| `0.005` | climbs faster |
| `0.01` | fast build |

Set the Learning Rate to where you want to *end up*. Every epoch the console reports the
adapter's size, its growth rate, and how much of the ceiling is in use. Off gives a flat run at
the box value.

## Two presets, both standard LoRA

| | |
|---|---|
| **✨ MiniMax H3 Defaults** | dim/alpha 16, 60 epochs, Adapter-relative LR at `0.003` |
| **✨ MiniMax H3 Fast** | dim/alpha **8**, **40 epochs**, flat 2e-4 with no Adapter-relative LR |

Fast reaches likeness in a few hundred steps, and the lower rank tends to come out **more
flexible** — it hasn't room to memorise your backgrounds and framing, so it encodes the subject
instead.

Both ship **standard LoRA**, not LoKR. LoKR moves considerably further per unit of learning rate,
which meant the same Learning Rate box behaved very differently depending on which Network Type
sat above it. LoKR is still a dropdown away.

## A simpler Training tab

- **Per-step movement clip** — retired
- **LR warmup** — retired
- **Weight averaging (EMA)** — still there, now **off by default**
- **Caption dropout** — now has a control (Off / `0.05` / `0.10`). It was fixed at `0.05` with no
  way to change it, and it turns out to matter: it's doing real work on this family, so it stays
  on by default.

Existing presets and saved configs still load; the retired settings load as off.

## Also in this release

- **Re-launching the same dataset no longer re-encodes every image.** The VAE pass is skipped when
  the cache already matches — and it still re-encodes everything if you change Target Megapixels,
  because the check compares the cached latent to the current bucket rather than trusting the
  filename.
- **Identity-learn reports where the learning is coming from.** Each epoch prints the teacher and
  photo errors and what share of the loss real pixels actually carry — which is usually rather
  more than the weight alone suggests, since matching a real photograph is harder than matching
  the model's own output. The teacher weight now also offers `0.4` and `0.5`.
- **Gradient accumulation now works on MiniMax.** The field was on the tab but never reached the
  trainer, so it silently did nothing on this family.
- **Previews can be clips.** A **Sample length** dropdown renders each sample as a short video you
  can scrub in the gallery — closer to what H3 actually is, at the cost of slowing the app down
  noticeably. Stills at 1024×1024 remain the default: seconds rather than minutes, and neither
  option replaces checking an epoch in ComfyUI. **640** added to the sample resolution list.
- **A preview that runs out of memory retries shorter** — 141 → 56 → 22 frames — instead of
  dropping the whole run to single frames. And a preview that's crawling now says so, and points
  you at Pause → check an epoch in ComfyUI → Resume.
- **Re-caching with fewer references no longer leaves the old ones behind**, where they could
  still be picked up and train against a pairing from a previous configuration.
- **Identity mode tells you when the model it needs is missing.** It runs on the reference
  (ref2va) DiT, and Fizgig used to let you set the whole run up and only refuse at Start — with
  a 21 GB download as the remedy. It now says so the moment you switch the mode on. **Download
  models for me** can fetch that file too: tick *Include the reference DiT* beside the button.
  It stays off by default, since it's 21 GB that ordinary training never touches.
- **The Samples tab no longer describes other models** when MiniMax is selected — the controls
  that don't apply to it are gone rather than greyed.
- **The Training tab is sorted into its sections.** Base Precision moved to *Memory & Precision*;
  Weight averaging, Adapter-relative LR, Caption dropout, Blocks to Train and the low-noise share
  moved to *Other Options*. Training Parameters keeps the settings you actually reach for. Purely
  a move — no value, default or preset changed.

## Known issue

**Resume uses whatever the Training tab currently shows, not the settings the run was launched
with.** If you restart Fizgig and hit Resume without loading the run's settings first, it will
resume with different ones — and if the network type differs, the run fails partway through with
an unhelpful error. Worse, the failed attempt overwrites the last-train snapshot, so *Load
Settings From Last Train* then returns the broken config. Load the run's settings before
resuming. A proper fix is coming.

## Upgrading

Nothing to do. Your model paths, datasets and caches are untouched. If you've saved MiniMax
presets, they'll load with the retired controls off.
