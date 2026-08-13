<h1 align="center">Fizgig — Klein 9B, Krea 2 & MiniMax H3 LoRA Studio</h1>

<p align="center">
  <strong>Fix broken LoRAs without retraining. Remix any LoRA into new variations in seconds.</strong><br>
  A train · repair · explore workbench built end-to-end for <strong>Flux 2 Klein 9B</strong> and <strong>Krea 2</strong>.
</p>

<p align="center">
  <a href="https://console.runpod.io/deploy?type=GPU&gpu=RTX+5090&count=1&template=faoq8ed6um&ref=vkb387ep"><img src="https://img.shields.io/badge/⚡%20Deploy%20on%20RunPod-673AB7?style=for-the-badge&logoColor=white" alt="Deploy Fizgig on RunPod"></a>
  <a href="https://buymeacoffee.com/lorasandlenses"><img src="https://img.shields.io/badge/Buy%20me%20a%20coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black" alt="Buy Me A Coffee"></a>
</p>
<p align="center">
  <sub>No GPU, or want a bigger one? Fizgig runs on rented hardware — one click, nothing to install.<br>
  Deploying through that link supports Fizgig's development at no extra cost to you.</sub>
</p>

<p align="center">
  <a href="https://www.youtube.com/watch?v=yrz0l6URGGk"><img src="assets/hero.png" alt="Fizgig LoRA Studio — watch the full video tutorial" width="600"></a>
</p>

<p align="center">
  <a href="https://www.youtube.com/watch?v=yrz0l6URGGk"><img src="https://img.shields.io/badge/▶%20Watch%20the%20full%20video%20tutorial-FF0000?style=for-the-badge&logo=youtube&logoColor=white" alt="Watch the full video tutorial on YouTube"></a><br>
  <sub>Start-to-finish walkthrough — install, prep, caption, train, and the workbench tools</sub>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/models-Klein%209B%20%2B%20Krea%202%20%2B%20MiniMax%20H3-blue?style=for-the-badge" alt="Klein 9B + Krea 2 + MiniMax H3">
</p>

> ### 📰 Latest news
> - **MiniMax H3 stops asking you to choose between quality and speed** (3.6.0) — the new **✨ MiniMax H3 Fast** preset reaches full likeness in a few hundred steps at dim/alpha 8, and the lower rank tends to come out *more* flexible rather than less. Load it, caption a folder of 35–45 images, press Start — you shouldn't need to touch anything else. **Multi Concept** is new too: two subjects in one LoRA, each with its own folder and its own trigger word. One thing worth knowing — judge quality by **pausing** and loading an epoch in ComfyUI; the image previews are for watching likeness arrive, and can show distortion that simply isn't there in a real render. [Details ↓](#minimax-h3--third-model-family) · [Release notes](docs/RELEASE_NOTES_v3.6.0.md)
> - **MiniMax H3 now hits real likeness.** H3 LoRAs used to nail pose and framing while staying soft on the face; they don't any more. The change that did it was the **optimizer** — MiniMax trains with `adamw` instead of `adamw8bit`, and it's the new default. Alongside it: a single **% of training in low noise** box replacing Detail Focus, a **mid-concentrated** shape toggle, and new defaults (LoKR 8, dim/alpha 16, 60 epochs, 0.5 MP, 60% low noise). [Details ↓](#minimax-h3--third-model-family)
> - **MiniMax on 24 GB just got ~4× faster** (3.4.1) — the trainer now picks base precision *and* block swap together instead of swap alone. A 24 GB card used to park 38 of 50 blocks on CPU; it now loads the same file 4-bit and swaps nothing. New **Base Precision** dropdown on the Training tab, Auto by default. [Details ↓](#minimax-h3--third-model-family)
> - **LoRA training for MiniMax H3** — Fizgig trains LoRAs for MiniMax's ~33B H3 video model, from ordinary still-image datasets, on a single consumer GPU. It's the biggest model Fizgig supports, with in-training previews, LoKR, Adaptive LR and Pause/Resume all wired. Pick **MiniMax H3** from the Base Model selector; the model files it needs have download links on their rows in **Preferences**. [Details ↓](#minimax-h3--third-model-family)
> - **Fizgig 3.0 trains LoKR.** LoKR (LyCORIS Kronecker) covers the whole weight matrix with structure instead of a thin low-rank slice — in our validation runs it produced the **highest likeness we have ever measured** with this app's own scorer, with noticeably more natural skin and light than standard LoRA on the same dataset and settings. Pick it from the **Network Type** dropdown on the Krea 2 Training tab: one **Factor** dial replaces rank and alpha, output loads straight into ComfyUI, and Repair Studio / LoRA the Explorer edit and save LoKR **natively — lossless, no conversion**. The short version of the trade: LoKR is higher quality, standard LoRA trains ~20% faster — and keep the factor at 8 (or below), where LoKR earns its cost. [Training ↓](#training)
> - **One-click cloud training on RunPod** — no GPU, or want a 5090 for the afternoon? The official Fizgig template deploys the full app to a rented GPU in your browser: nothing to install, your files persist until you terminate the pod, and the in-app RunPod panel can even **auto-stop the pod when your run finishes** so an idle GPU never bills overnight. [**⚡ Deploy →**](https://console.runpod.io/deploy?type=GPU&gpu=RTX+5090&count=1&template=faoq8ed6um&ref=vkb387ep) · [Guide](docker/README.md)
> - **Krea 2 trains on 8 GB** — confirmed by users running nothing but the stock preset defaults at batch size 1, with everything left on Auto. 10–12 GB cards do the same with headroom to spare. [VRAM guidance ↓](#vram-guidance)
> - **Never lose a run** (v2.11) — training state saves at every checkpoint *and* at run end, so a crash costs nothing and a **finished LoRA can be trained further**: raise the epoch count, resume, and it carries on with optimizer and learning-rate history intact.
> - **Caption with the model that trains your LoRA** (v2.10) — the Captions tab can use **Qwen3-VL**, the same vision-language model that conditions Krea 2 training. Every task is an **editable preset** — including a style-LoRA preset validated on real training runs. The slot takes fp8_scaled builds and community fine-tunes; since that model writes your captions, swapping it changes how your dataset gets described.

> **Two model families, one workbench.** Everything here works with both **Flux 2 Klein 9B** and **Krea 2 (12.9B)** — Repair Studio, Explorer, Royale, Profiler, Extract, plus Context LoRA, Adaptive LR and Pause/Resume. [Krea 2 details ↓](#krea-2--second-model-family)

> **Two things worth reading about before you start.** The Krea 2 trainer **curates your dataset while it trains** — detecting problem images from their loss alone, throttling them, having the text encoder *look at* the stuck ones and rewrite their captions, and telling you the best epoch when the run plateaus. [Details ↓](#the-trainer-curates-your-dataset-while-it-trains-krea-2-experimental) And the **sample gallery is an instrument**, not a contact sheet: it scores every sample's likeness against your own photos live during training, with a Royale-style **Training Run Visualiser** to scrub and export the run. [Details ↓](#the-sample-gallery-is-an-instrument-both-families)

---

## What Fizgig is

Every trainer makes LoRAs. Fizgig is built around what you do with them **afterwards** — and that's the part nobody else has.

- **Fix** a baked LoRA block-by-block, no retraining — overbaked identity, crushed style, drag a slider, save a new `.safetensors`.
- **Explore** new variations like a game — the app proposes mutations, you pick favourites, the LoRA evolves through selection.
- **Find** the best LoRA by eye — **LoRA Royale** renders every epoch of a run (or any folder of LoRAs) on one seed; crossfade to the one that *feels* right.
- **Share** what you made — LoRA Royale exports the epoch morph, or travels a single LoRA through seeds, prompts, or strength, as a looping MP4/GIF made to share.
- **Profile** exactly which blocks carry identity, style, and detail — so you know what to touch before you touch it.

Under that workbench sits a fast, light trainer tuned for its models — and tuned to **fit your GPU**, not a datacenter's. Because everything is built natively for Klein 9B and Krea 2 instead of bolted onto a dozen models, Fizgig can do things the generalists can't:

- **Big models on modest cards.** A full **Klein 9B** LoRA trains on a **16 GB card** — and the 12.9B **Krea 2** trains on **8 GB**, confirmed by users running nothing but the stock preset defaults at batch size 1 with everything on Auto. That's the 4-bit (NF4) base doing the work (~8 GB resident, QLoRA-style: the base is 4-bit, your LoRA still trains in bf16 on top). Block swap, quantisation and previews **size themselves to your VRAM automatically** — nothing to configure — and if a preview can't fit, it steps aside so **training keeps running and saving**. You don't need a 4090 to train on the newest 12.9B model.
- **A workbench nobody else has.** Repair broken LoRAs block-by-block, evolve new ones like a game, and crossfade every epoch of a run to find the sweet spot — then the tools **read each other's output** (profile → repair → explore → compare, one closed loop).
- **It just works on your files.** Loads kohya / PEFT / OneTrainer / AI-Toolkit / LyCORIS, auto-converted; saves kohya `.safetensors` that drop straight into ComfyUI.

**📣 Help map Krea 2 — [open an issue](https://github.com/shootthesound/Fizgig/issues).** Krea 2's per-block roles — which blocks carry **identity, style, and detail** — aren't charted yet, which is why the colour-coded sliders and layer-targeting presets are Klein-only for now. The **Profiler** is the instrument for finding them: spot a pattern, share it in **[GitHub Issues](https://github.com/shootthesound/Fizgig/issues)**, and it directly drives the colour-coding and finer layer targeting coming to Krea 2's presets and Repair Studio.

**Free and open source.** A good first run is the **✨ Old Reliable** preset on the Training tab — then try **✨ Old Reliable · Flavour 8** (rank 8). Much of the old rank-16 instinct predates models this size; on Klein 9B, rank 8 is often plenty.

---

## The workbench

The reason to use Fizgig. Each tool works on a trained run's output **or any Klein LoRA you've downloaded** — and they hand off to each other.

### Repair Studio
Thirty-two live sliders — one per transformer block — with a side-by-side Distilled preview that updates instantly. **Turbo Preview** — activation caching for a live LoRA-tweaking UI, which no other LoRA tool does — caches per-block outputs and prompt encodings across the denoising steps, so late-block edits redraw up to **97% faster**; the baked save is always exact, Turbo or not. Quick-set buttons on every slider (`[0]` `[1]` `[±]` `[⚖]`); **Balance** holds the combined primary + donor weight at 1.0 per block, ideal for cross-fading two LoRAs. Optional donor-LoRA blending mixes blocks from a second LoRA via rank concatenation. Previews can be conditioned on a **reference image** (Klein is an edit model), so you see how your LoRA edits a real photo. Click a preview to pop it into a resizable window. Browse a new LoRA and it auto-swaps — no manual reset. Saves a baked `.safetensors` that works in ComfyUI at strength 1.0.

### LoRA the Explorer
Evolutionary discovery. The app mutates blocks and shows four variants — pick a favourite and it becomes the new baseline. **Freeze Tweaked Blocks** locks what you like so future mutations only touch the rest. A **Structure** slider sets how far the composition anchor drifts each round; seed cycling checks variants across seeds. Found a direction you love? **Refine this baseline in Repair Studio** sends all 32 slider values straight over — and Repair Studio sends state back the same way. Discover → refine → discover, in a loop.

### LoRA Royale
Find the best LoRA the human way — then turn the winner into share-ready clips. Point it at a training output folder and it renders **every epoch on one fixed seed** (Distilled 4-step), with a **crossfade slider** that blends smoothly between consecutive epochs — drag until it looks best and stop. A thumbnail grid sits below; click any epoch to jump there. Drop in a **reference image** (Klein is an edit model) and every epoch edits the same photo. An optional **likeness score** (InsightFace ArcFace, CPU — no extra VRAM) rates each epoch against a training shot — **close-up headshots included** (it pads-and-retries so a face that fills the frame still detects) — and flags the closest in gold with one-click **Jump to best**. **Export likeness clip** turns it into a share-ready, side-by-side *subject vs each epoch* comparison with the score burned in, morphing epoch by epoch. **Promote** copies the winner to a clean `.safetensors`. Not a training run? Point it at **any folder of LoRAs** and it compares them by name — or flip to **Single-LoRA mode** to run everything below on one downloaded LoRA, no folder required.

**Comparison sheet** builds the share image people actually post for a new LoRA — one row per prompt, columns for *without LoRA* vs *with LoRA* (or one column per epoch), the same seed across a row so only the LoRA changes, headers and captions drawn in, saved as a single PNG. The no-LoRA column also strips your trigger word, so the baseline isn't fed a token the base model has never seen.

Because the morph *is* the magic, the payoff is four **travel** tools that each render a sequence you **scrub to review and only save if you like it** — as a looping MP4 or GIF, re-saveable in either format without re-rendering, with an optional **deflicker** pass (the timelapse trick DaVinci uses) for flicker-free clips. **Export the morph** saves the whole epoch sweep, a face resolving epoch by epoch — or **Save all stills** dumps every rendered epoch to a folder as full-res PNGs (the renders otherwise live only in memory). **Seed travel** slerps through a journey of seeds to show the LoRA's range. **Prompt travel** interpolates the text embedding through waypoints — Time of day, Season, Age, Era, or your own words — so one subject flows through the change; pick a **Preset + Subject** and it writes the prompt for you. And **LoRA strength travel** ramps the LoRA from 0 (base model) to full and beyond, so you literally *watch the effect fade in*. Every travel can be anchored to a reference to hold the subject steady, with interpolation and seed-drift knobs for a smooth, brightness-even result. (The epoch morph shows the LoRA *learning*; the travels show what it can *do*.)

### Profiler
A per-block activation profile with a colour-coded, five-bucket HTML report — which blocks carry style, identity, and detail signal, and where they overlap. Writes a JSON sidecar that Repair Studio reads automatically, showing the findings inline when you load the same LoRA. Krea 2 LoRAs get a weight-only per-block report (no models loaded) — the instrument for the community block-mapping effort below.

### Extract
Distil any Klein or Krea 2 LoRA to a lower rank — Klein with block and timestep targeting, Krea 2 via weight-only SVD. Fast presets run pure weight SVD with no GPU models loaded; activation-weighted presets (Klein) use forward passes for better accuracy. Supports PEFT and LyCORIS (LoKR / LoHa) sources. Expect roughly **5 minutes for a full-model Klein LoRA and ~25 for Krea 2** (its 264 modules are 6144-wide) — a long quiet stretch mid-extract is normal, not a hang.

---

## Krea 2 — second model family

Krea 2 is a from-scratch **native** port — no external tooling at runtime: a 12.9B single-stream MMDiT, the Qwen-Image VAE, and a Qwen3-VL-4B text encoder. **Train on the RAW model** (fp8, ~14 GB resident); in-training previews render **on the training model itself** with the official **Turbo LoRA** applied (8-step, CFG-free, same settings as the Turbo model — the LoRA auto-downloads, ~470 MB, and switches on only while a preview renders). Your live LoRA — and Context LoRA if set — stay active in every preview, exactly as they'd be deployed. Pick Krea 2 from the **Base Model selector** at the top of the Training tab.

Everything works on Krea 2: **all five workbench tools** (Profiler, Extract, Repair Studio, Explorer, Royale), plus **Pause/Resume** (full state), **Context LoRA**, **Adaptive LR**, **reference images** (through the text encoder's vision path — "prompt from a picture"), and the live sample override. A few Training-tab controls are hidden for Krea 2 for now (not removed): per-block Model-Area targeting (no Krea 2 block map yet), the Timestep section, and the FP8-Scaled / FP8-TE / Gradient-Checkpointing toggles.

> **📣 Help map Krea 2's blocks — [open an issue](https://github.com/shootthesound/Fizgig/issues).** The colour-coded sliders, Model-Area targeting, and block-aware presets are Klein-only right now because Krea 2's per-block roles (which blocks carry **identity**, **style**, and **detail**) aren't mapped yet. The **Profiler**'s weight-only report is the instrument for discovering them. If you find patterns — a block that clearly drives identity, a range that governs style — please share your findings in the **[GitHub Issues](https://github.com/shootthesound/Fizgig/issues)**. Community block-discovery is what will drive the colour-coding and finer layer targeting coming to the Krea 2 presets and Repair Studio.

**Runs on smaller cards, and adapts to yours.** Krea 2 is a bigger model than Klein, but the low-VRAM paths are wired — and with everything on **Auto**, Fizgig plans the whole run for you.

> **8 GB is enough.** Users have trained full Krea 2 LoRAs on **8 GB** cards with everything left on **Auto**, batch size **1**, and the stock defaults of any of the Krea 2 presets — no hand-tuning. **10–12 GB** cards do the same with more headroom to spare, so there's room to raise batch size or resolution before anything gets tight. On 8 GB, leave batch size at 1 and let Auto do its thing.

- **Auto memory strategy** — leave Blocks Swap and 4-bit Base on **Auto** and Fizgig picks the best of **INT8 W8A8** (fastest, near-exact — the default wherever it fits), **NF4 4-bit**, or fp8 from your *free* VRAM — budgeted for your actual run shape (batch size is the big cost: ~2.4 GB per extra image, measured). The console explains what it chose and why.
- **torch.compile speedup** — on longer runs the transformer blocks compile automatically (needs the MSVC C++ Build Tools on Windows; triton installs with the requirements). Roughly 2× faster steady-state steps on the INT8 path after a one-off warm-up — putting per-step speed **approximately on a par with OneTrainer**. Combine that with the real-time dataset intelligence below (which showed faster likeness and a higher ceiling in matched-epoch A/Bs) and time-to-a-*good*-LoRA should now favour Fizgig: same step speed, smarter steps.
- **4-bit (NF4) base** — the base trains frozen at ~5.6 GB (base + LoRA ~8.3 GB), so a full Krea 2 LoRA fits a **10–12 GB card with no block swap at all**, and an **8 GB** card with the swap Auto sizes for it — QLoRA-style, the LoRA still trains in bf16 on top. Auto picks it when it's the right call; the *4-bit Base* dropdown forces it On/Off.
- **Previews with no model swapping** — the default **RAW + Turbo LoRA** preview engine renders samples on the model that's already training, so nothing big is loaded or moved between epochs. The Samples tab keeps the classic mode (load the fp8 Turbo checkpoint per preview) if you prefer it; the workbench tools (Repair Studio / Explorer / Royale) still use the Turbo checkpoint, auto-sizing its block swap to your GPU.
- **Previews never crash a run** — if a preview can't fit, previews auto-disable and **training keeps going and saving**; evaluate the LoRA in ComfyUI.

Krea 2 trains real, ComfyUI-compatible LoRAs, and its training recipe is verified against the reference implementation — same noised/target flow-matching, `krea2_shift` timestep sampling, and gradient clipping.

**Two built-in presets:** **✨ Krea 2 Defaults** (rank 32, 30 epochs — the standard pick, applied automatically when you switch to Krea 2) and **✨ Krea 2 Ultra Fast** (rank 8, Adaptive LR at an aggressive 2e-4 floor, 20 epochs — fewer epochs to a usable LoRA when you want a quick result or a fast test of a dataset).

### The trainer curates your dataset while it trains (Krea 2, experimental)

Four Training-tab toggles turn a run into a live dataset curator — no other trainer does any of this:

- **Detect problem images** — every image's loss is tracked across epochs, normalized for the random noise level each step draws (raw per-step loss mostly ranks the dice roll, not the image). Images that stay hard **without improving** get flagged in the console and in the live **Problem Images window** (thumbnails, verdicts, per-image trends, auto-refreshing every epoch). In real runs the top flags were all caption/image mismatches — e.g. from-behind shots whose captions never said so. The detector finds them from the loss trajectory alone.
- **Per-image adaptive LR** — flagged images are throttled (suspects ×0.7 from ~epoch 3, confirmed-stuck ×0.5 escalating toward ×0.1) so one bad caption can't keep yanking the weights all run, while fully-mined images ease off to prevent overbake and consistently-healthy learned images get a gentle ×1.1 boost. In matched-epoch A/Bs this gave faster likeness *and* a higher final ceiling — with real skin texture where the untreated run went plastic.
- **Auto-recaption stuck images** — the same Qwen3-VL that conditions training *looks at* each confirmed-stuck image between epochs, rewrites its caption from what's actually visible (appending your trigger word if set), re-encodes it, and gives the image a fresh start. A second attempt goes exhaustive-detail; still stuck after two means the image is **excluded** for the rest of the run — so the loss average stops carrying its permanent error term — and the exclusion is remembered per-dataset (`fizgig_excluded.json`, travels with your images). Fix the caption and it's automatically re-admitted.
- **Warm up look outliers** — real-but-unusual images (tight angles, profiles, occlusion) that the Look Consistency Filter scored as outliers keep their unique information but **ease in at ×0.4 LR**, ramping to full over the first ~4 epochs — refining the identity instead of fighting it while it forms — and release to full LR early the moment they start improving. Prior-then-evidence: the face-embedding score covers exactly the epochs before the loss watch has a trend to act on.

You can also edit any caption yourself mid-run from the Problem Images window — the trainer re-encodes it at the next epoch boundary, no restart. Once nothing is improving any more, the watch tells you you're **done**: a plateau banner with a best-checkpoint estimate and a suggested epoch window to scrub in LoRA Royale — and it's honest about certainty, distinguishing a *provisional* plateau (images still being adjudicated that may give the run a second wind) from a *confirmed* one. And pausing or restarting loses nothing: a **resumed run replays its own loss log** to restore every verdict, trend, and exclusion exactly where they were.

---

## MiniMax H3 — third model family

Fizgig now trains LoRAs for **MiniMax H3**, MiniMax's open-weight ~33B video model — the biggest model Fizgig supports — from **ordinary still-image datasets**, on a single consumer GPU. It's a native port, and the output is a standard `.safetensors` that loads straight into ComfyUI's H3 workflows, including the pruned inference builds.

**Training only, for now.** MiniMax H3 trains, previews and pauses/resumes like the other two
families, but the workbench tabs — Repair Studio, LoRA the Explorer, LoRA Royale, Profiler and
Extract — are Klein and Krea 2 only. They are planned for H3 rather than ruled out, and it is
honest to say it will take a while: every one of them renders full images on demand, and doing
that on a 33B video model is a different problem from doing it on a 9B image one. Nothing stops
you using an H3 LoRA in ComfyUI in the meantime.

**How it works:** pick **MiniMax H3** from the Base Model selector at the top of the Training tab, and the usual flow applies — Start-tab folder, Captions, Samples, Training. The base trains frozen and quantized with your LoRA in bf16 on top, so a **32 GB card trains comfortably with everything on defaults**. Leave **Blocks Swap** and **Base Precision** on Auto: at launch the trainer reads your **free** VRAM — not your card's size, so close ComfyUI first — and picks the base precision and the block-swap count *together*. Measured, with the shipped defaults:

| Free VRAM | What Auto does | Peak |
|---|---|---|
| ~30 GB | **int8**, no block swap, up to 1 MP | ~24.5 GB |
| ~22 GB | **4-bit**, no block swap | ~18 GB |
| ~14 GB | **4-bit** with block swap | ~13.5 GB |

int8 is the checkpoint's own storage and the most accurate base (~0.17% error); 4-bit loads the *same file* at ~10.5 GB instead of ~21, at ~9% error in the frozen base. Auto only reaches for 4-bit when the alternative is most of the model crossing PCIe every step — which is several times slower — and the console tells you when it does. Pin either one from the dropdown and the swap plan is built around your choice. Mileage varies with your dataset and bucket sizes; if you hit an out-of-memory, set **Blocks Swap** to a number to override the planner outright. **Adaptive LR**, **LoKR**, **Pause/Resume** and **in-training previews** are all wired.

**Quality:** H3 LoRAs used to get pose, hair and framing right while staying soft on the face. That's fixed — the fix was the optimizer, and MiniMax now trains with `adamw` rather than `adamw8bit` by default. If you're loading an older preset or your own saved settings, set **Optimizer Type** to `adamw` on the Training tab; it costs about 1.9 GB more VRAM and nothing else.

Two built-in presets ship. **Defaults** is applied the moment you pick the family:

| Preset | Settings |
|---|---|
| **✨ MiniMax H3 Defaults** | LoRA dim/alpha 16, 60 epochs, **0.25 MP**, 60% low noise + mid-concentrated, `adamw`, LR ceiling 2e-4 with **Adapter-relative LR** on at `0.003` |
| **✨ MiniMax H3 Fast** | The same, at **rank 8 for 40 epochs** with a **flat 2e-4** (no Adapter-relative LR). Reaches likeness in a few hundred steps, and the lower rank tends to come out more flexible — it has no room to memorise your backgrounds and framing, so it encodes the subject instead. |

**0.25 MP is the default, and it holds up.** It's four times cheaper per step than 1 MP and the extra resolution has not paid for itself in testing — including on wider-framed sets, where you might expect it to. H3's canvas is 768 on the short edge, but that governs what it *renders*, not what it can be *trained* on. Raise it if a specific dataset asks for it; don't assume it needs raising.

**Previews are 1024×1024 stills by default**, which render in seconds. H3 is a video model, so the **Sample length** dropdown can also give you a short clip you can scrub in the gallery — useful when motion is what you need to check, but a clip costs minutes rather than seconds, so set **Generate every N epochs** to match if you switch.

**What it needs** — three files (plus one optional), each with a **Download link on its row in Preferences** (the *Model Paths (MiniMax H3)* card at the bottom):

| Model | Size | Notes |
|---|---|---|
| DiT — pruned int8 | ~21 GB | The training base. Use `minimax_h3_fl2va_pruned_int8_convrot.safetensors` — it's the file ComfyUI runs, so your LoRA trains against the weights it'll be deployed on, and it keeps its int8 weights rather than being re-quantized. The ~66 GB bf16 file also works (NF4 at load, ~11 GB resident) |
| Qwen3-VL-32B text encoder | ~15.7 GB | The **nvfp4** file — the same one ComfyUI uses, so you may already have it. The bf16 file (~51.5 GB) also works; the *int8_convrot* variant is not supported |
| Video VAE | ~4.9 GB | Used while caching your images, and to decode previews |
| DiT — reference *(optional)* | ~21 GB | Only for reference distillation. `minimax_h3_ref2va_pruned_int8_convrot.safetensors` — a different fine-tune, and the only H3 build that accepts reference images. You may already have it if you use ComfyUI's r2v workflow. **Download models for me** leaves it out unless you tick **Include the reference DiT** beside the button. Leave blank for ordinary training |

The text encoder is loaded for the one-time caching pass then freed — it never shares VRAM with training.

### The MiniMax Training tab, control by control

Every control below also has a short hint in the app; this is the full version.

**Where they live.** *Training Parameters* keeps the settings you reach for on a normal run.
*Memory & Precision* holds **Blocks Swap** and **Base Precision**. *Other Options* holds the
rest — **Weight averaging (EMA)**, **Adapter-relative LR**, **Caption dropout**, **Blocks to
Train**, and the **low-noise share** with its mid-concentrated tick. Reference distillation and
**Multi Concept** stay in *Training Parameters*, next to the settings they change.

**Low-noise training** (default 60%, **mid-concentrated** ticked) — how much of the run trains on nearly-clean images (noise below the halfway point) instead of heavily noised ones. Low noise is where fine detail and likeness are learned; high noise is where pose, framing and composition live. MiniMax's own default works out at about 8% — tuned for video, it leaves almost nothing for detail on stills. There's no cap and no table to cross-reference: type whatever share you want. Higher means more detail training, but each of those steps also lands harder, so if a high value overbakes, drop the Learning Rate before you drop the number. **mid-concentrated** changes the *shape* without changing the percentage: the same share of low-noise steps, but the mass bunched around the middle of the range (where identity is resolved) instead of spread evenly to both extremes — Krea 2 and Klein both train that way. The setting is recorded in the LoRA's metadata and shown on the training queue, so runs stay comparable.

**Blocks to Train** (default all 50) — an experiment with no recommended answer yet. H3 is 50 identical blocks and nobody has published what each one does. A block that doesn't carry identity still gets trained on *something* — your backgrounds, framing and lighting — so leaving blocks out may give a **cleaner** likeness with less memorised set, not just a faster run. The offered ranges are scaled from Klein's block map (a different architecture), so treat them as starting guesses: train one against a full-model run on the same dataset and judge the pair. You can type your own ranges and single blocks, comma-separated (`3-12, 14-15, 22, 31-33`; blocks are numbered 0–49). Fewer blocks also means faster steps, a smaller file, and less capacity overall — give a narrow range a few more epochs before calling it. Recorded as `ss_train_blocks` in the LoRA's metadata.

**Learn identity from my dataset — reference distillation** (experimental, off by default) — H3 already renders a person well when it's *shown* a photo of them; this teaches your LoRA to do the same from the trigger word alone, so you don't need a reference at generation time. Your dataset is used exactly as normal — same folder, same captions, every image still trained on. What changes is what each answer is *marked against*: normally it's the photograph itself (which is why a LoRA also learns your backgrounds and framing); with this on, most of the marking comes from what the model produces when shown *other* photos of the same person from your folder — identity without the scenery. Every image takes a turn as a reference, and no image is ever its own. **Teacher 0.8** means 80% of that and 20% still the real photo, which keeps genuine skin and texture; 1.0 is pure and caps the LoRA at what reference mode can already do. Needs the **ref2va** model in Preferences — tick **Include the reference DiT** beside **Download models for me** if you don't have it, and Fizgig will say so anyway the moment you switch this on. Caching takes longer and uses more disk. The whole run trains on ref2va — it is the only build that accepts references — but the LoRA it produces works fine on the ordinary fl2va model you already generate with, so there is nothing to change at deployment. **Aimed at Multi Concept.** That is where it has been tested and where it demonstrably helps — holding two people apart. On a single character it is unproven: not known to be worse, simply not tested, so treat it as an experiment there rather than a recommendation.

**Identity-first** (part of reference distillation, default **Auto**) — train the first stretch against the **teacher only**, then drop the teacher entirely and train on the **photographs only**. Photo training then starts from an adapter that already knows who the trigger word means, instead of discovering the identity and the detail at the same time. Phase 1 runs at a third of the Learning Rate box, since it is placing the identity rather than reproducing detail, and phase 2 skips the teacher pass altogether so it costs about half as much per step. **Auto** sizes the first phase from your dataset — enough steps for the teacher side to converge, which is roughly the same number of steps whatever your image count and therefore rather more epochs on a small set. Or pick a fixed number of epochs; **Off** keeps the blended loss and hands the teacher-weight box back.

**Adapter-relative LR** (**on at `0.003`** — also `0.005` and `0.01`) — on in the **Defaults** preset, off in **Fast**. Worth leaving on unless you are deliberately testing a flat rate. A LoRA starts at zero, so a rate that's safe at epoch 1 is far too slow by epoch 50, and one that's right later wrecks a fresh adapter. With this on, the Learning Rate box becomes a **ceiling**: the run starts below it and climbs, keeping each step a fixed fraction of the adapter's current size. Set the LR to where you want to *end up*. Raise the fraction to climb faster; Off gives a flat run at the box value. The console reports the adapter size, growth rate and how much of the ceiling is in use each epoch.

**Multi Concept** (default off) — hold **two subjects in one LoRA**. Tick it and a second folder picker appears; each subject gets its own folder, its own trigger word, and its own dataset entry. That last part is what makes it work: in identity-learn mode every image is marked against *other photos of the same person*, and that pairing runs per folder — so subject A is only ever compared against A. Put two people in one folder and the pairing crosses them, which blends them rather than keeping them apart. Ticking it also applies the settings that suit it (identity-learn on, 4 references, a short identity-first phase, caption dropout `0.10`) and says so in the console — **nothing is locked**, they are starting points. Two things it expects of you: caption both folders yourself, each with its own unique trigger word in *every* caption, and note the second folder is training-only — Image Prep, Captions and the Look filter still follow the Start folder.

**Caption dropout** (default `0.05` — also Off and `0.10`) — trains a few percent of steps with no caption at all, so the LoRA does not lean entirely on the trigger word. It was fixed at `0.05` with no way to change it until now, and it earns its place: turning it off measurably hurt quality in testing, so leave it on unless you have a reason.

**Base Precision** (default Auto) — Auto reads your **free** VRAM at launch and picks the base precision and block swap together. int8 is the checkpoint's own storage and the most accurate base (~0.17% error) — it needs about 30 GB free to run without block swap. 4-bit loads the *same file* at ~11 GB instead of ~21, so it fits smaller cards with no swap, at ~9% error in the frozen base — the LoRA then spends some capacity correcting error that won't exist at inference. Auto only reaches for 4-bit when the alternative is most of the model crossing PCIe every step. Pin either one and the swap plan is built around your choice.

**Weight averaging (EMA)** (default Off — `0.98`, `0.99`, `0.995`) — saves and previews a smoothed running average of the weights instead of the raw values. Big steps zigzag around the good solution, and the average is the centre of the zigzag, so checkpoints come out crisper. Training itself always runs on the raw weights, so it costs no speed. Worth switching on when you're pushing the learning rate hard.

The learning rate is yours to set — there's no automatic throttle on top of it. Either judge it from your samples, or switch on **Adapter-relative LR** above and let the run work its way up to it.

**When do changes apply?** Settings are read when a run launches — changing them mid-run does nothing. Pause → Resume relaunches with your current settings, so these *can* be changed at a pause. Dataset and caption changes need a fresh run (Resume says no re-caching).

**Current caveats — this is deliberately minimal for now:**
- **Image datasets only.** H3 is an omni audio + video model, but Fizgig trains it from still images only (each treated as a single video frame) — **no video-clip training and no audio training yet**. The LoRAs you get are learned from stills; you can still deploy them in ComfyUI's video workflows.
- **Audio isn't rendered.** Preview clips are silent — H3 generates a soundtrack alongside the video, and Fizgig denoises it so the picture is conditioned correctly, but only the frames are saved.
- **Training only.** The workbench tools (Repair Studio, Explorer, Royale, Profiler, Extract) don't support H3 yet, and Context LoRA isn't wired for it. Pause/Resume and resumable state saving work as on the other families.
- **Batch size 1**, and the Training tab hides the controls that don't apply — what you see is what's wired.

---

## Training

The foundation: fast, light, and tuned for one model.

- **LoKR training (Krea 2)** — the LyCORIS Kronecker parametrization the community rates highest for likeness, trainable with the loss watch, adaptive LR and auto-recaption attached. Pick it from the **Network Type** dropdown: one **Factor** dial replaces rank/alpha (lower factor ≈ more capacity, bigger file: factor 8 ≈ 400 MB, 16 ≈ 100 MB). Output is standard LyCORIS format — drops straight into ComfyUI — and Repair Studio / Explorer edit and re-save it natively, lossless. Our testing in short: LoKR is higher quality at factor 8 or below; standard LoRA trains ~20% faster. Raising the factor above 8 keeps LoKR's speed cost while losing its quality edge — at that point pick LoRA instead.
- **Proven presets** for single subject through multi-character — or roll your own.
- **Context LoRA** — load an existing LoRA as a frozen *active* layer so the new one learns to coexist. Train a face on top of a style and they stop fighting at inference; train an outfit on top of a character and the clothes drape correctly. No other trainer does this.
- **Multiple sample prompts** — on the Samples tab, each LINE of the prompt box is its own prompt, and every line renders its own sample per epoch. Keep one prompt per line (long ones wrap).
- **Distilled training samples** — 4-step previews that match ComfyUI output closely (a separate Distilled DiT, ComfyUI Euler Simple schedule). On by default; toggle on the Samples tab. On tight cards the sample model auto-swaps its own blocks by VRAM so 4-step previews keep working on 16 GB. On 24 GB+ it stays resident and is cached in system RAM between epochs (RAM-checked, saves ~3–4 s/epoch).
- **Reference-conditioned samples** — Klein is an edit model, so previews can *edit* a reference photo instead of generating from scratch. Auto-resized to ~0.20 MP so it can't OOM; works on Base and Distilled samples.
- **Adaptive LR** — a bi-directional plateau tracker that probes up on steady loss descent and pulls down (with optional weight rollback) on plateau, heavy gradient clipping, or weight-norm runaway. Two knobs, not three: you set the **Min/Max window** and the run starts at its geometric midpoint — the Learning Rate box greys out and is ignored while adaptive is on.
- **fp8 Base training** — the fp8 Base DiT stays resident at ~9.6 GB instead of dequantising to ~18 GB, so a full 9B LoRA trains in ~14 GB and fits a 16 GB card — lossless, no quality cost. Automatic (Fizgig detects the pre-quantised file), no flag.
- **Gradient checkpointing toggle** — on by default (it's what fits a 9B LoRA on 16 GB). Turn it **off** on a 24 GB+ card for meaningfully faster steps. A VRAM-aware warning fires if you switch it off on a card that can't spare the activation memory.
- **Pause / Resume** — graceful epoch-boundary pause that frees your GPU mid-run and resumes with full optimizer state and no quality regression. Fire up Rocket League, come back, carry on. Bonus: Resume relaunches with your *current* settings, so a pause is also the moment to change them mid-run (dataset/caption changes still need a fresh run).
- **Model Area targeting** — train only Identity, Style, or Detail blocks, or the full model.
- **Auto VRAM management** — block swap auto-detects from GPU VRAM; OOM detection tells you exactly what to change. Supports bf16 and fp8 Base DiT, with block swap.
- **Per-dataset caches, cross-checked** — every dataset gets its own cache folder, and the trainer verifies each cached item against the images actually in your folder before training. Deleted an image? It's gone from the run. Switched datasets? The old one can never leak in.
- **Diffusers LoRA support** — OneTrainer LoRAs with split Q/K/V keys are auto-fused on load.

> **A note on Base previews:** the default Distilled 4-step previews track ComfyUI closely, including with a Context LoRA active. Only **Base multi-step** previews (Distilled toggled off) can look softer than the deployed LoRA — they come from a mid-training fp8 checkpoint, so colours and detail can be slightly off even when the LoRA is excellent. Judging from Base previews? Confirm final quality in ComfyUI.

### Live status bar
A bottom bar with stacked **VRAM and system-RAM gauges** (smooth gradient fills, plus a per-run peak marker so you can see how high a run pushed memory). VRAM is read at the device level, so it catches other apps holding the GPU too. A top-right **IDLE / BUSY** light shows at a glance whether the app is working. Hide or show the whole bar with one click; it remembers.

Beside it sits a **live sample override** — tick it to set a prompt, seed, width/height, and optional reference image for the *next* samples, mid-run, no restart. The text encoder only re-runs when the prompt text changes, so seed / resolution / reference tweaks are instant.

### The sample gallery is an instrument (both families)

The browser gallery of training samples now *measures* the run instead of just showing it — on Klein and Krea 2 alike:

- **Live likeness scoring** — pick the **3 dataset photos** that best nail the look, and every sample gets a colour-coded likeness badge (ArcFace face embeddings averaged across all three baselines — one photo would bias every score with its own angle and lighting). Scoring runs on **CPU with zero impact on training speed**, newest samples first, and keeps up live as each epoch's previews land. A **trend chart** plots per-epoch average likeness for the current run with the best epoch highlighted — an objective likeness-vs-epoch curve, *while the run is still going*. (It measures identity likeness only — overbake and skin texture still need your eyes.)
- **Training Run Visualiser** — scrub the current run epoch by epoch, Royale-style: a slider carousel per sample prompt, play/pause with ping-pong looping, likeness score inline, and share-ready export — a WebM clip with the epoch ticker and Fizgig tag burned in, or full-res PNG frames. It's a taste of the **LoRA Royale** tab, right in the browser.

### Dataset prep
- **AI captioning, with the captioner that trains your model** — if you have Krea 2's Qwen3-VL text encoder, the Captions tab uses it by default. It's the same vision-language model that conditions training, so it captions to the doctrine that actually matters for LoRAs: name the camera viewpoint, say whether the face is visible, describe what's there instead of hedging. Five tasks — and **every one of them is an editable preset, not a fixed mode**.

  Four are for training a subject: **training caption** (the default — viewpoint-aware), **short**, **detailed** and **exhaustive**. The fifth, **Style**, is for training a *look*, and it works the opposite way round: it describes the contents of each image in detail and never the style itself, so your trigger word is the only thing every caption has in common and the look binds to it. Set a trigger word, pick **Style**, caption the folder, train. Tested on real runs across Klein and Krea 2 — the style comes through fast.
 Open the instruction the model is actually given, rewrite it in plain English, and save it; each preset keeps its own wording, your edits persist between sessions, and a Reset button restores the built-in whenever you want it back. Tune one preset for products and another for portraits and just switch between them. The trainer's mid-run auto-recaption picks up your edited wording too, so a caption style you settle on is the style the run keeps writing. **Florence-2** remains the zero-setup option, downloading itself on first use. Either way it's bulk-generate in one click, with your trigger word prepended.

  The text encoder slot is **open**: bf16, the smaller **fp8_scaled** (recommended — 4.9 GB resident vs 8.3 GB, output we couldn't tell apart), or a **community fine-tune / abliterated build**. Because the same model writes your captions, swapping it changes how your dataset gets described — useful when a stock instruct model hedges on or refuses your subject matter.
- **Bilingual captions** — optionally append Chinese via Helsinki-NLP. Klein's Qwen3 text encoder has deep Chinese training, so bilingual captions act as text-level data augmentation, improving visual quality without changing loss. In a controlled A/B (same data, seed, and hyperparameters — captions the only change) the loss curves stayed within ±0.001/epoch, yet the bilingual run produced visibly more skin detail and faster visual convergence.
- **Image Prep** — batch resize, PNG conversion, and InsightFace face-crop derivatives, with optional **gender targeting** (largest male/female face) so it locks onto your subject in group shots. Pairing a tight crop with a full shot adds a lot to a character dataset — at the default 0.25 MP a face in a wide shot reaches the model as only a handful of latent pixels, while a crop of the same face gets the whole frame (~40× the face area), which is what keeps likeness sharp without raising the resolution. Training defaults to ~512² (0.25 MP) and resizes in-cache, so any resolution or aspect ratio just works — nothing has to be square or pre-sized.
- **Look Consistency Filter** — the final prep stage, built for **synthetic-heavy datasets**: the subtly off-look near-misses that drag a likeness down are *easy* for the model to reconstruct, so a loss curve never sees them — but face-embedding distance does. Pick the **3 images that best nail the look** and every image is scored against all three, averaged (close-up faces included — detection pads-and-retries). Worst matches surface first with colour-coded verdicts; mark drifters by click or let **Auto-Suggest** flag the statistical outliers, then move them out of the dataset in one go (to a subfolder — nothing is deleted, and moving them back re-admits them). The scores save with your dataset and drive the trainer's **look-outlier warm-up**.

### Compatibility
Loads kohya, PEFT, OneTrainer (OMI + legacy), AI-Toolkit, and LyCORIS (LoKR / LoHa) — all auto-converted on load. LoKR and LoHa run **natively at inference** — no pre-conversion — anywhere in the app: as a primary or donor in Repair Studio, in the Profiler, in Extract, even as a Context LoRA. Fizgig also **trains** LoKR on Krea 2, and Repair Studio / Explorer **save LoKR as LoKR** — edits bake losslessly into the Kronecker factors; SVD conversion only happens for donor-blended blocks, where it's mathematically unavoidable. Output is `.safetensors` that drops straight into ComfyUI. Every tab links to the relevant section of the walkthrough video.

---

## No GPU? Rent one

Fizgig ships as a ready-made cloud image, so you can train on a card far bigger than the one in
your machine — or keep your own GPU free while a run goes on somewhere else.

**It's the whole app, not a cut-down web version.** Training, Repair Studio, LoRA the Explorer,
LoRA Royale, Profiler, Extract and the sample gallery, in a browser tab. Drag datasets in and
finished LoRAs out with a built-in file manager, download models in one click, and optionally have
the pod **shut itself down when training finishes** so an overnight run doesn't bill until morning.

Your models and datasets live on persistent storage, so you download them once and every future
session picks up where you left off.

**[⚡ Deploy on RunPod →](https://console.runpod.io/deploy?type=GPU&gpu=RTX+5090&count=1&template=faoq8ed6um&ref=vkb387ep)**  ·  [Read the guide first](docker/README.md)

---

## Requirements

- **GPU** — NVIDIA RTX 30 / 40 / 50-series, or **AMD Radeon** with ROCm (RDNA1 through RDNA4, Strix Point / Halo, Instinct MI300+). **16 GB+ VRAM** recommended (24 GB+ comfortable), but the floor is lower than that suggests: **Klein 9B** needs 16 GB, while **Krea 2** trains on **8 GB** with everything on Auto and batch size 1 — see [VRAM guidance](#vram-guidance). The fp8 Base's VRAM savings apply on NVIDIA Ada+; on AMD, NF4 and INT8 are the primary quant paths.
- **NVIDIA driver** — 555+ on Windows, 550+ on Linux (for the CUDA 12.8 PyTorch wheels).
- **AMD ROCm** — **Windows:** `install_fizgig_rocm.bat` (supported path). **Linux:** `./install_fizgig_rocm.sh` — **highly experimental** (newer gfx like RDNA4, desktop compositor + training on the same GPU, and driver resets are common; use Windows ROCm or NVIDIA Linux for production training). Optional system `amdrocm-amdsmi` for accurate status-bar VRAM via `amd-smi`.
- **OS** — Windows 10 / 11 or Linux. macOS handles captioning and image prep, but training needs CUDA or ROCm.
- **Python** — 3.10, 3.11, 3.12, or 3.13.
- **Disk** — ~10 GB for the venv, plus ~40 GB for model files.
- **Visual Studio Build Tools** (Windows only) — needed to compile InsightFace, and for the **torch.compile training speedup**. Direct installer (no hunting on the MS site): **[aka.ms/vs/17/release/vs_BuildTools.exe](https://aka.ms/vs/17/release/vs_BuildTools.exe)** — tick the **"Desktop development with C++"** workload. The installer and `update_fizgig.bat` detect it and print this link if it's missing; without it everything still works, you just skip the compile speedup. (triton, compile's other dependency, installs automatically with the requirements.)

---

## Install

Clone the repo (or download the ZIP via the green **Code** button and extract):

```bash
git clone https://github.com/shootthesound/Fizgig.git
cd Fizgig
```

**Windows (NVIDIA, one-click)** — double-click `install_fizgig.bat`. It creates a venv, installs CUDA 12.8 PyTorch and all dependencies, pre-downloads the InsightFace models, and verifies CUDA is visible to PyTorch. Launch with `run_fizgig.bat`; update later with `update_fizgig.bat`.

**Windows (AMD ROCm)** — double-click `install_fizgig_rocm.bat` (NVIDIA users never run this). It picks **Python 3.12** via `py -3.12` / `python3.12` (not whatever `python` defaults to — e.g. 3.14), or downloads portable 3.12.10 if none is found (same approach as comfyui-rocm). GPU detection follows, then pinned multi-arch wheels from **AMD ROCm nightlies** (`https://rocm.nightlies.amd.com/whl-multi-arch/` — not built by Fizgig):

- `torch==2.12.0+rocm7.15.0a20260728`
- `torchvision==0.27.0+rocm7.15.0a20260728`
- `rocm-sdk-devel==7.15.0a20260728`

Override with `TORCH_PIN` / `TORCHVISION_PIN` / `ROCM_SDK_DEVEL_PIN` if needed. **bitsandbytes** is a pinned community Windows ROCm wheel from [0xDELUXA/bitsandbytes_win_rocm](https://github.com/0xDELUXA/bitsandbytes_win_rocm) — built by neither AMD nor Fizgig. Shared deps come from `requirements.txt` with CUDA `torch`/`bitsandbytes` and NVIDIA-only `nvidia-ml-py` filtered out (`filter_requirements_rocm.py`). Launch with `run_fizgig_rocm.bat`.

**Linux (AMD ROCm — highly experimental)** — expect crashes, GPU resets, and incomplete model support on many setups. Best-effort only; Windows ROCm or NVIDIA Linux are the supported training paths. Prerequisites: amdgpu driver loaded (`/dev/kfd`), user in `render`/`video` groups. See [Install ROCm](https://rocm.docs.amd.com/en/latest/install/rocm.html) and [PyTorch for ROCm](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html). Then:

```bash
chmod +x install_fizgig_rocm.sh
./install_fizgig_rocm.sh
./run_fizgig_rocm.sh
```

The script detects your gfx target (`detect_gpu_linux.py`). **Stable** (default) installs **ROCm 7.14** PyTorch from `repo.amd.com`, pinned to **`2.12.0+rocm7.14.0`** (covers cp310–cp314; override with `TORCH_PIN=…`). **Nightly** follows [TheRock multi-arch RELEASES.md](https://github.com/ROCm/TheRock/blob/main/RELEASES.md): one index URL plus a `[device-gfx*]` extra for your GPU (e.g. `gfx1201` → `device-gfx1201`):

```bash
ROCM_CHANNEL=nightly ./install_fizgig_rocm.sh
```

Nightly pulls pinned **ROCm 7.14** (not latest 7.16+) for `libbitsandbytes_rocm714.so`: `rocm[libraries,devel,device-${ARCH}]==…` + `torch[device-${ARCH}]==…` + `torchvision[device-${ARCH}]` from the nightly index, defaulting to the latest **2.12** + **7.14.0a*** build. Override with `TORCH_PIN=…`, `ROCM_META_PIN=…`, or `TORCH_NIGHTLY_MINOR=…`.

Then shared deps from `requirements.txt` (filtered) and `bitsandbytes>=0.50.0` for ROCm.

**Linux / macOS (NVIDIA CUDA path)** — `install_fizgig.py` is CUDA-only (captioning / image prep on macOS; training needs a CUDA or ROCm GPU). On AMD-only Linux hosts it prints a hand-off to the ROCm installer and exits:

```bash
python install_fizgig.py
chmod +x run_fizgig.sh
./run_fizgig.sh
```

**VRAM status bar on AMD:** the existing NVIDIA `pynvml` / `nvidia-smi` path is unchanged; AMD readers (`vram_monitor.read_amd_gpu_vram`) run only as a fallback. Windows ROCm uses `typeperf`; Linux ROCm uses the **`amd-smi`** CLI when available ([AMD SMI / ROCm Core SDK](https://rocm.docs.amd.com/projects/amdsmi/en/latest/install/install.html), e.g. `sudo apt install amdrocm-amdsmi`). Fizgig picks the GPU with the largest VRAM total (skips empty iGPU entries). Legacy `rocm-smi` is a fallback. Do not `pip install amdsmi` — the PyPI package is outdated.

Three small models auto-download on first use: InsightFace `buffalo_l` (~300 MB, during install), Florence-2 (~500 MB–1.5 GB, first AI caption), and Helsinki-NLP `opus-mt-en-zh` (~300 MB, first bilingual translation).

---

## Model downloads (you provide)

Fizgig doesn't bundle weights — they're large and licensing varies. You only need the family you're using.

> **Or let Fizgig fetch them.** Preferences has a **⬇ Download models for me** button under each model card: it downloads that family's files, verifies them, and fills in the paths for you. **Krea 2 needs no HuggingFace account** — those files aren't gated. Klein does (Black Forest Labs require you to accept their licence), so Fizgig asks for a free read token and tells you which pages to accept on. Interrupted downloads resume rather than restart. **MiniMax has one extra tick** — *Include the reference DiT* — off by default because it's another 21 GB used only by identity mode. There's a CLI too:
>
> ```bash
> python -m fizgig.scripts.fetch_models --family krea2   # ~32 GB, no account needed
> python -m fizgig.scripts.fetch_models --family klein   # ~34 GB, needs a token
> python -m fizgig.scripts.fetch_models --family tools   # Florence-2, face model, translator
> ```

Prefer to do it by hand? Each row in **Preferences** also has a **Download** link to the right HuggingFace page.

### Klein 9B

| Model | File | Size | Source |
|---|---|---|---|
| **Base DiT (fp8) — recommended** | `flux-2-klein-base-9b-fp8.safetensors` | ~9.5 GB fp8 | [black-forest-labs/FLUX.2-klein-base-9b-fp8](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9b-fp8) |
| Base DiT (bf16) | `flux-2-klein-base-9b.safetensors` | ~17 GB bf16 | [black-forest-labs/FLUX.2-klein-base-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) |
| Distilled DiT | `flux-2-klein-9b-fp8.safetensors` | ~9 GB fp8 | [black-forest-labs/FLUX.2-klein-9b-fp8](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8) |
| VAE / AE | `ae.safetensors` | ~320 MB | [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev/blob/main/ae.safetensors) (from root, **not** the `vae/` subfolder) |
| Text Encoder | `qwen_3_8b.safetensors` | ~15 GB | [Comfy-Org/vae-text-encorder-for-flux-klein-9b](https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/blob/main/split_files/text_encoders/qwen_3_8b.safetensors) |

Training runs on the **Base DiT**, and the **fp8 version is recommended on every GPU**: same training quality at roughly half the VRAM (resident at ~9.6 GB, so a 9B LoRA trains in ~14 GB and fits a 16 GB card).

The VRAM savings and quality are the same across all supported cards (RTX 30 / 40 / 50-series) — fp8 Base is worth it on every GPU.

It's all automatic — Fizgig detects pre-quantised files and the right path for your GPU, so you never need to touch the "FP8 Base" checkbox (the bf16 version works too if you prefer). The **Distilled DiT** powers the fast 4-step previews — on by default during training, and always used in the Profiler, Repair Studio, and Explorer — so grab both if you'll use the workbench.

### Krea 2

All four files live in the one [**Comfy-Org/Krea-2**](https://huggingface.co/Comfy-Org/Krea-2) repo.

| Model | File | Size | Source |
|---|---|---|---|
| **RAW DiT (bf16) — training** | `krea2_raw_bf16.safetensors` | ~26 GB bf16 | [Comfy-Org/Krea-2 → diffusion_models](https://huggingface.co/Comfy-Org/Krea-2/blob/main/diffusion_models/krea2_raw_bf16.safetensors) |
| **Turbo DiT (fp8) — workbench / classic previews** | `krea2_turbo_fp8_scaled.safetensors` | ~13 GB fp8 | [Comfy-Org/Krea-2 → diffusion_models](https://huggingface.co/Comfy-Org/Krea-2/blob/main/diffusion_models/krea2_turbo_fp8_scaled.safetensors) |
| **Turbo LoRA — in-training previews** (auto-downloads) | `krea2_turbo_lora_rank_64_bf16.safetensors` | ~470 MB bf16 | [Comfy-Org/Krea-2 → loras](https://huggingface.co/Comfy-Org/Krea-2/blob/main/loras/krea2_turbo_lora_rank_64_bf16.safetensors) |
| Qwen-Image VAE | `qwen_image_vae.safetensors` | ~250 MB | [Comfy-Org/Krea-2 → vae](https://huggingface.co/Comfy-Org/Krea-2/blob/main/vae/qwen_image_vae.safetensors) |
| **Text Encoder — recommended** | `qwen3vl_4b_fp8_scaled.safetensors` | ~5.2 GB fp8 | [Comfy-Org/Krea-2 → text_encoders](https://huggingface.co/Comfy-Org/Krea-2/blob/main/text_encoders/qwen3vl_4b_fp8_scaled.safetensors) |
| Text Encoder — full precision | `qwen3vl_4b_bf16.safetensors` | ~8.9 GB bf16 | [Comfy-Org/Krea-2 → text_encoders](https://huggingface.co/Comfy-Org/Krea-2/blob/main/text_encoders/qwen3vl_4b_bf16.safetensors) |

Training and its previews run on the **RAW DiT** (the Turbo LoRA auto-downloads — no need to grab it by hand); the **fp8 Turbo checkpoint** powers the workbench tools (Repair Studio / Explorer / Royale) and the classic preview mode, so grab it if you'll use those. On smaller cards, the **4-bit (NF4)** toggle shrinks the RAW base to ~5.6 GB — which is how Krea 2 fits 10–12 GB GPUs, and 8 GB ones at batch size 1. Leave it on Auto and Fizgig decides.

**The text encoder slot is open.** Fizgig loads any Qwen3-VL-4B checkpoint you point it at, so you have real choice here:

- **fp8_scaled — recommended.** Measured **4.9 GB resident vs 8.3 GB** for bf16, with captions we couldn't tell apart. ComfyUI's fp8 conversion quantises only the language layers and ships the **full bf16 vision tower**, so reference images and AI captioning are unaffected. Fizgig keeps the weights in fp8 and dequantises per matmul, so the saving is real memory, not just a smaller download.
- **bf16** — the full-precision original, if you'd rather not quantise at all.
- **Community fine-tunes and abliterated builds work too.** Any Qwen3-VL-4B variant in the ComfyUI layout loads, in bf16 or fp8_scaled. Since the same model writes your captions, swapping it changes *how your dataset gets described* — an uncensored or domain-tuned build will caption subjects a stock instruct model hedges on or refuses. Verified against a third-party abliterated fp8 build alongside the official files.

Nothing else about Krea 2 is needed to use it as a captioner — Klein-only datasets benefit just as much, since captions are plain `.txt`.

---

## VRAM guidance

### Klein 9B

**Inference tools** (Profiler / Repair Studio / Explorer / Extract) on Distilled 4-step:

| Block Swap | Min VRAM | Notes |
|---|---|---|
| 0 | 24 GB+ | No swap — fastest |
| 4 | 20 GB | Light swap |
| 8 | 16 GB | Moderate swap |
| 12 | 14 GB | Aggressive swap |
| 16 | 12 GB | Maximum swap — slower, but fits |

**Training** — the fp8 Base DiT stays resident at ~9.6 GB (not dequantised to bf16), so a 9B LoRA fits comfortably in **16 GB** — around 14 GB observed at block-swap 0 with a Context LoRA active, a little less without. VRAM scales with resolution and batch size; raise block swap to fit smaller cards.

**Smaller cards — 4-bit (NF4) base.** fp8 training needs ~14 GB: it fits a 16 GB card with no swap, but a **10–12 GB card has to block-swap**, paying a PCIe-transfer penalty every step. The opt-in **4-bit (NF4) base** mode (the *4-bit Base* toggle in Memory & FP8 / FP4) quantizes the frozen base to 4-bit — halving DiT VRAM to ~5.6 GB so a full 9B LoRA trains in **~7.5 GB**, which fits 10–12 GB cards with **no swap at all** (and so beats fp8-with-swap on those cards). The LoRA still trains in bf16 on top, QLoRA-style, and the base loads layer-by-layer so the card never holds the whole model. It's a lower-precision base, so it's a slight quality trade — always check the output in ComfyUI — and **16 GB+ cards should stick with fp8** (same quality, no swap).

**DiT Block Swap (inference)** in Preferences applies only to the workbench tools. Training has its own separate block-swap setting, and its Distilled samples auto-swap by VRAM — so this preference never touches a training run. On first launch Fizgig auto-detects your VRAM and picks a sensible default; once you choose a value, your choice sticks.

### Krea 2

Krea 2 is a bigger model, so the numbers differ — but Fizgig **auto-sizes block swap and quantisation to your card** for both training and the workbench, so there's nothing to tune:

| Your card | What to do | What to expect |
|---|---|---|
| **8 GB** | Everything on **Auto**, **batch size 1**, stock preset defaults | Trains full Krea 2 LoRAs — reported working by users on real runs. Keep batch size at 1; that's the one setting worth leaving alone |
| **10–12 GB** | Same — everything on Auto | Same, with headroom to spare. Room to raise batch size or resolution before it gets tight |
| **16 GB+** | Same | Comfortable; Auto will usually pick the faster INT8 path over 4-bit |

- **Training** runs on the RAW fp8 base (~14 GB resident); block swap auto-detects from VRAM (32 GB → none, scaling up to maximum on sub-16 GB cards). The **4-bit (NF4)** toggle drops the base to ~5.6 GB (base + LoRA ~8.3 GB), fitting a **10–12 GB card with no swap** and an 8 GB card with the swap Auto sizes for it. You don't need to pick: Auto budgets from your *free* VRAM and the console explains its choice.
- **In-training previews** default to the **RAW + Turbo LoRA** engine: they render on the model already training, so previews add almost nothing on top of the training footprint — no checkpoint load, no CPU↔GPU shuffling between epochs. The **workbench** (Repair Studio / Explorer / Royale) and the classic preview mode run on the fp8 Turbo checkpoint, which peaks ~22.6 GB unswapped — Fizgig auto-swaps it to fit your GPU (≈17 GB at swap 12; 16 GB cards swap enough to fit). If a preview still can't fit, it **auto-disables and training keeps running and saving** — evaluate the LoRA in ComfyUI.

### Desktop feels juddery while training? (Windows)

If your mouse or video playback stutters during a run — even with CPU, RAM and VRAM all showing plenty free — turn off **Hardware-accelerated GPU scheduling**: Windows Settings → System → Display → Graphics → *Default graphics settings*, then reboot. Training and your desktop share the same GPU, and with that setting on, Windows can't prioritise between them; with it off, Fizgig runs training at low priority so your desktop stays smooth and training speed is unaffected.

---

## INT8 fast inference (on by default)

Previews and the whole workbench (Repair Studio / Explorer / LoRA Royale, plus in-training previews) run an **INT8 (W8A8)** matmul instead of fp8 — faster, at **near-identical quality**, on **both Klein and Krea 2**. Key points:

- It **only affects previews** — your **saved LoRA is always exact**, INT8 or not. It changes what you *see* while working, never what you *ship*.
- It's a **speed** knob, not a memory one: int8 is 8-bit like fp8, so **same VRAM**. It also **stacks with block swap**, so it helps small cards too.
- The win **varies by GPU**: **biggest on RTX 30-series** (which have no fast fp8 tensor cores), modest on 40/50-series where fp8 is already fast. Measured so far: ~1.19× vs fp8 on a 5090, larger on a 3090.

It's **on by default**; flip **INT8 fast inference** off in **Preferences → Inference Performance** to fall back to fp8.

---

## Getting started

Launch Fizgig and work left-to-right through the numbered tabs:

1. **Start** — set your training image folder. If model paths aren't configured, a prompt points you to Preferences.
2. **Image Prep** (optional) — resize, PNG-convert, or face-crop your images; finish with the **Look Consistency Filter** to weed out off-look images before they train.
3. **Captions** — write trigger-word captions or generate them with AI (Qwen3-VL or Florence-2); optionally translate to bilingual English + Chinese.
4. **Samples** — configure the preview prompts that render during training (Distilled 4-step on by default).
5. **Training** — pick a preset, tune, click **Start Training**.

The unnumbered tabs are the post-training workbench — and work on any Klein LoRA you've downloaded: **Profiler**, **Repair Studio**, **LoRA the Explorer**, **LoRA Royale**, **Extract**, and **Preferences** (model paths, output directories, inference block-swap preset, default Browse folders).

---

## Headless / CLI training

Everything the trainer does is also available from the command line — the GUI is a front-end over the scripts in `src/fizgig/scripts/`, so the CLI is always feature-complete: adaptive LR, the full per-image loss watch, auto-recaptioning, Context LoRA, pause/resume. Works on Windows and Linux alike, including display-less boxes. See **[docs/CLI.md](docs/CLI.md)** for the pipeline, dataset config format, and worked examples for both Klein 9B and Krea 2.

---

## Support the project

If Fizgig saves you time or helps you make better LoRAs, consider supporting development:

<a href="https://buymeacoffee.com/lorasandlenses"><img src="https://img.shields.io/badge/Buy%20me%20a%20coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black" alt="Buy Me A Coffee"></a>

---

## License

Fizgig is open source under the **[Apache License 2.0](LICENSE)** — free to use, modify, and redistribute, including commercially, with attribution and no warranty. It includes third-party components under their respective licenses (musubi-tuner — Apache-2.0; ai-toolkit — MIT; Diffusers / FLUX — Apache-2.0; comfyui-rocm `detect_gpu.py` — GPL-3.0); see **[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)**.

Copyright © 2026 Peter Neill.

Model weights are **not** covered by this license — each model carries its own terms from its publisher (see the Download links in Preferences).
