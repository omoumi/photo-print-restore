# photo-print-restore

Tools for restoring old photo prints from flatbed scans, stage by stage.

This is not a one-click restorer. Each stage measures first; where it makes
decisions -- where a print lies, which way is up, which prints belong
together, colour or B&W -- it shows them to you as previews, a review
window or a plan with contact sheets, and you can correct them. Every
stage writes its results into a folder of its own.

---

## The stages

| Stage | What it does | Main scripts |
|---|---|---|
| 1 | Repairs the thin vertical seams a CIS flatbed scanner leaves in every scan | `calibrate_scanner.py`, `detect_seams.py`, `repair_seams.py` |
| 2 | Finds each print on the scan, straightens it, crops it and turns it the right way up | `detect_borders.py`, `suggest_orientation.py`, `review.py`, `deskew_crop.py` |
| 3 | Gives the prints new numbers so that prints that belong together sit side by side | `renumber_by_similarity.py` |
| 4 | Corrects the colour: a faded colour print gets its colour back, a B&W print becomes neutral grey | VueScan, then `restore_colors.py` |
| 5 | Colorizes the B&W prints with a model that runs on your own computer, keeping each print's own lightness and detail | `colorize_model.py`, `transfer_color.py` |

Run the stages in order: each one works on the folder the previous one wrote.

---

## Where each stage writes

Keep your raw scans: everything else can be made again from them.

- Stage 1 writes the repaired scans to a new folder.
- Stage 2 writes the straightened, cropped prints to a new folder.
- Stage 3 links (or copies) the prints into a new folder under their new
  names.
- Stage 4 writes the colour-corrected prints to a new folder.
- Stage 5 writes the colorized prints to a new folder.

Stages 1 and 2 add or move pixels, so they have a `--verify` option: it
rebuilds every output from its input, with code of its own where it can, and
compares the two. Stages 3 and 4 work by plan: they analyse first, you look
at the plan and the contact sheets, then `--apply` writes the result and
`--undo` removes it again. Stage 5 first has the model make a colour guide
for each B&W print; then it writes small previews to look at, and `--apply`
the full-size results (its `run_pipeline.bat` goes straight to these).

---

## Where this came from

A personal project, built to digitise a family collection of around 700
prints. It is published because the problems it solves are not specific to
that collection.

The equipment it was developed with:

| | |
|---|---|
| Scanner | HP LaserJet Pro M1132 MFP (flatbed, CIS sensor) |
| Scanning software | VueScan 9.8.33 |
| Output | 16-bit TIFF, 600 dpi |
| Platform | Windows |

The scanner is ordinary consumer hardware. Its sensor is built from several
chips laid end to end, and at each joint it drops one column of the image.
That is why Stage 1 exists.

If your scanner is different, the code still works, but its numbers are not
yours. For example, the seam spacing found here, 344 px at 600 dpi, belongs
to this one scanner. Stage 1 measures your own scanner for you; do not copy
the number.

---

## Layout

```
photo-print-restore/
    stage-1-seam-repair/
    stage-2-print-extraction/
    stage-3-renumbering/
    stage-4-color-correction/
    stage-5-colorization/
    LICENSE
```

Each stage folder stands on its own: its own README, its own scripts and its
own copy of the shared code (`common.py`, `parallel.py`). No stage reads
another stage's files, so any stage can be copied out and run alone.

The shared part of `common.py` and all of `parallel.py` are the same in every
stage; only the header and each stage's own section differ. Every step
prints the same way: a heading, then one `[phase n of m]` heading for each
part of its work, and at the end what it wrote and the command of the next
step.

---

## Getting started

1. Install Python 3 on Windows (tick "Add python.exe to PATH").
2. Open the folder of the stage you need and read its README. It lists the
   packages to install with `pip`.
3. Double-click that stage's `run_pipeline.bat`. It asks where your files
   are, checks that Python and the packages are there, and walks you
   through the steps.

Every step can also be run on its own from a command prompt; each stage's
README shows how. Stage 4 begins in VueScan, with the settings its README
lists; Stage 5 needs ComfyUI, set up as its README describes.

---

## Licence

GNU General Public License v3.0 or later. Copyright (C) 2026 omoumi.
See [`LICENSE`](LICENSE).
