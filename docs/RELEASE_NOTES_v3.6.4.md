# Fizgig v3.6.4 — the workbench checks your LoRA before it loads 9 GB

A community release: everything below came from **@FNGarvin**.

## Pick the wrong LoRA and you'll know instantly

Load a Krea 2 LoRA with the family selector on Klein 9B — in **LoRA Royale**, **LoRA the Explorer**
or **Repair Studio** — and Fizgig used to load the entire wrong pipeline first, spend about 25
seconds on it, print a few hundred warnings and finish with a traceback.

It now reads which family a LoRA was trained for straight from the file, in a few milliseconds,
before anything loads. If the selector is on the other family it simply switches to match and
carries on. Pick a folder of epochs from two different families by mistake and it says so instead
of quietly using one of them.

MiniMax H3 LoRAs are recognised too, and those tabs will tell you plainly that they don't support
H3 yet rather than half-loading something.

## Faster installs, and less disk used

The installer and updater asked uv to **copy** packages out of its cache rather than link them,
which cost disk space and time for everyone whose cache and Fizgig folder are on the same drive.
They now link where linking is possible, and say so in one calm line where it isn't.

Dependency installs are also more precise: PyTorch's own wheel index is now consulted **only** for
torch and torchvision, where before it could influence every other package in the file.

Nothing to do — this applies from your next update onwards.

## Upgrading

Nothing to do. Your model paths, datasets, caches and presets are untouched.

Thanks to **@FNGarvin** for all of the above.
