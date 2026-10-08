# Stage 1 — repair the CIS scanner seams

Turns a folder of raw flatbed scans into scans with the scanner's dropped
columns put back. The repaired scans go to a folder of their own, and are
the input of Stage 2.

A contact-image-sensor flatbed is built from several chips laid end to end,
and at each junction it drops one column of the image: a faint,
one-pixel-wide, full-height step. The junctions are physical, so the seams
repeat at a fixed **pitch** and sit at a fixed **phase** in every scan the
machine makes. Everything here is built on that one fact.

Three steps. Each takes one optional argument, the project root (the folder
holding the scans), and finds everything else by fixed names inside it.

```
python calibrate_scanner.py     step 1   measure pitch and phase
python detect_seams.py          step 2   locate the seams in each scan
python repair_seams.py          step 3   put the dropped column back
```

On Windows, `run_pipeline.bat` runs all three in order.

Not steps:

| File | Role |
|---|---|
| `common.py` | shared code; must sit beside the steps |
| `parallel.py` | worker-count policy; without it every step runs in one process and `--workers` is ignored |
| `pixel_ruler.py` | helper for step 2: read a seam's column by eye |

Every step prints the same way: a heading (`STAGE 1 step 2  -  ...`), then
one `[phase n of m]` heading for each part of its work, and at the end what
it wrote and the command of the next step.

---

## Install

```
pip install numpy tifffile pillow imagecodecs psutil matplotlib
```

- `numpy` and `tifffile` are needed by every step.
- `pillow` is needed only for scans that are not TIFF and for
  `detect_seams.py --preview`.
- `imagecodecs` is needed only for compressed TIFFs.
- `matplotlib` is needed only for `pixel_ruler.py`.
- `psutil` is optional; it sizes the worker pool from free RAM.

Python 3.10 or newer. On Windows use `python`, not `python3`.

---

## The project folder

```
D:\archive\scans\
    raw0001.tif ...                     the scans
    json files\
        boxes_calibrate_scanner.json    written by step 1 (pitch and phase)
        boxes_detect_seams.json         written by step 2 (the seam columns)
        boxes_repair_seams.json         written by step 3 (the manifest)
    seam_previews\                      step-2 previews, only with --preview
    seam-repaired\                      step-3 output: the input of Stage 2
```

The ROOT argument can be the scans folder itself, the `json files` folder, a
file inside it, or a single scan; all lead to the same place.

---

## Which steps change pixels

| Step | Changes pixels? |
|---|---|
| 1 calibrate_scanner | No. Writes two numbers. |
| 2 detect_seams | No. Writes column numbers. |
| 3 repair_seams | Adds one column per seam. Every scanned sample is copied through as it is. |

Take the inserted columns out of a repaired scan and you have the raw scan
back, bit for bit; step 3 checks this on every scan before writing it.
Output keeps the bit depth of the scan: 16-bit in, 16-bit out.

---

## Step 1 — calibrate_scanner.py

Measures the pitch and phase of the scanner once, from many scans.

```
python calibrate_scanner.py
python calibrate_scanner.py --max-images 200
```

| Option | What it does |
|---|---|
| `--max-images N` | how many scans to measure (default 40; more is better) |
| `--min-pitch N`, `--max-pitch N` | pitch search range, px (default 80..1200) |
| `--exts .tif,...` | which files count as scans |
| `--workers N` | parallel workers (default: auto; `1` = no pool) |

How it works:

- **Many scans, not one.** In one photo a seam is a few grey levels, buried
  under the photo's own texture. The texture differs from photo to photo
  while the seams never move, so the average of many scans shows the grid.
- **The fundamental.** A grid of pitch P also scores well at 2P, 3P...,
  because every other line still lands on a seam, so the smallest pitch
  scoring near the best is chosen.
- **It refuses to guess.** A wrong pitch would mis-grid every scan in
  silence, so a result that is not clearly ahead writes nothing and says
  what to do.

---

## Step 2 — detect_seams.py

Writes the seam columns of every scan.

```
python detect_seams.py
python detect_seams.py --preview           also write a marked PNG per scan
python detect_seams.py --pitch 344         skip step 1 (use your own pitch)
```

| Option | What it does |
|---|---|
| `--pitch PX` | seam pitch (default: from step 1) |
| `--phase COL` | use this seam column as the phase |
| `--preview` | write `seam_previews\<n>_seam.png` to check by eye |
| `--no-consensus` | let every scan keep the phase it locked alone |
| `--workers N` | parallel workers (default: auto; `1` = no pool) |
| `--bands`, `--lock` | detection tuning |

Notes:

- **The grid is rigid.** Every seam is marked at the exact pitch, including
  the ones on smooth or dark content that cannot be seen. No single line is
  moved onto nearby texture.
- **The phase** from step 1 or `--phase` is used exactly as given. Without
  one, each scan locks its own; once the folder agrees on a phase, every
  scan is put on it exactly.
- **Every seam is marked,** up to the edges of the scan.
- **HIGH, MEDIUM, LOW** say how many seams are clearly *visible* in that
  photo (two or more, one, none), not whether their positions are right.
  Faint seams are normal, so LOW is only warned about for a scan whose
  phase rests on itself alone.

---

## Step 3 — repair_seams.py

Inserts one column at every seam and writes the result to `seam-repaired\`,
or to the folder `--out` names.

```
python repair_seams.py
python repair_seams.py --verify
```

| Option | Default | What it does |
|---|---|---|
| `--method NAME` | directional | `directional` or `linear` |
| `--overwrite` | off | replace repairs that already exist |
| `--out DIR` | `seam-repaired` | where the repairs go; never the scans folder itself |
| `--verify` | off | afterwards, rebuild every repair and compare (see below) |
| `--workers N` | auto | parallel workers; `1` = no pool |

How it works:

- **directional** blends the horizontal average of the two neighbours with
  the two diagonal averages, each weighted by how well its direction matches
  across the gap, so a slanted edge is bridged instead of smeared. It is
  computed in float64 and rounded once when written.
- **linear** is the plain average of the two neighbours, rounded half up
  for integer data.
- **The check.** Each repair is checked before it is written: taking the new
  columns out must give the scan back exactly. A repair that fails is not
  written.
- A scan that is not TIFF is written as TIFF. Two scans that would get the
  same output name (`raw0001.tif` and `raw0001.png`) stop the step before
  anything is written.

---

## pixel_ruler.py — read a seam's column by eye

Opens one scan at its true size and shows the column under the pointer, so
a seam can be found or checked without counting pixels. The number it reads
off is what `detect_seams.py --phase` expects.

```
python pixel_ruler.py raw0001.tif
```

| Key or mouse | What it does |
|---|---|
| move the mouse | the readout under the image shows the column and row |
| left click | mark that column and print it |
| magnifier / pan | drag to zoom or pan; the tool releases itself |
| `z` / `x` | zoom in / out around the pointer |
| `c` / `u` | clear all marks / undo the last one |
| `s` | save the window as `<scan>_marked.png`, beside the scan |

The steps do not take a `_marked.png` for a scan.

---

## Running everything: run_pipeline.bat

Keep it beside the five scripts and double-click it, or run it from a
command prompt. The scans can be in any folder. It asks:

1. **Where the scans are.** Enter = the folder the file is in. A pasted path
   with quotes or a trailing backslash is fine.
2. **Step 1** runs only if `boxes_calibrate_scanner.json` does not exist
   yet: the pitch and phase belong to the machine, not to the batch. It
   measures up to 200 scans. Delete that file to measure again.
3. **Step 2** runs.
4. **Where to create the output folder, and its name.** Enter = the scans
   folder and `seam-repaired`. The scans folder itself is refused.
5. **Verify?** Enter = no. Then step 3 runs.

Once it has the scans folder, it checks that Python, numpy and tifffile are
installed. A failed step stops the run. At the end it reports whether
verification failed (exit code 3), a scan could not be repaired (2), or
another error occurred (1). When every scan is repaired, it names the
folder to give Stage 2: the output folder chosen above.

---

## --verify

Rebuilds every repair listed in the manifest from its raw scan. It checks
that the raw scan and the repair, with its inserted columns taken out, both
still hold the original samples the manifest recorded (SHA-256), and that
the inserted columns are what the recorded method gives. It also checks bit
depth, colour model, ICC profile, resolution and that no TIFF tag was lost.
Exit code 3 on failure.

It can be run on its own at any time; if every repair already exists,
nothing is remade and only the check runs.

How independent it is:

- **Rebuilt with separate code:** the TIFF reader, which columns were
  inserted, every metadata check.
- **Shared with step 3:** the fill of the new columns (`repair_image`), and
  Pillow for a scan that is not TIFF. A bug there would not be caught.

The directional fill is repeatable on one installation. If the repairs were
made with a different numpy version, that is reported; the original samples
are compared all the same.

---

## Metadata

Copied to every repair: bit depth, photometric interpretation, planar
configuration, extra samples, compression, resolution and its unit, ICC
profile, description, date, software, make, model, artist, copyright, host
computer, page name, and every other tag of the scan, `Orientation` and XMP
included.

Not copied: the Exif and GPS blocks, which are stored elsewhere in the file.
A lossy TIFF codec (JPEG, WEBP, JPEG XL) is written as deflate.

---

## Running again

Every step is safe to run again.

- Steps 1 and 2 overwrite only their own file.
- Step 3 skips repairs that already exist (`--overwrite` remakes them).
  Repairs left in place keep their manifest entries, so `--verify` still
  checks them. If step 2 has changed the seams of one since, step 3 names it.

---

## Messages

| Message | Meaning |
|---|---|
| `not writing a profile: this result is not clear enough` | step 1 could not separate the pitch from its rivals; measure more scans, or give `--pitch` to step 2 |
| `no pitch given and no boxes_calibrate_scanner.json` | run step 1, or pass `--pitch` |
| `N scans locked their phase alone on little seam evidence` | no phase was given and the folder agreed on none; check those scans with `--preview`, or give `--phase` |
| `folder phase 271 (28 of 36 scans agree)` | the phase the folder agreed on |
| `N outputs left in place were repaired with other seams or another method` | step 2 changed; `--overwrite` redoes them |
| `FAILED THE CHECK, not written` | a repair did not give its scan back; nothing was written |
| `these scans would write the same output name` | rename one scan of each group, then run step 2 again |
| `the raw scan has changed since it was repaired` | verification failed: the raw scan is no longer the one the manifest recorded |
| `PIXELS CHANGED: N samples differ` | verification failed; do not use that file |

---

## The manifest

`json files\boxes_repair_seams.json` records where the repairs are (relative
to the scans folder when they are inside it, so the archive can be moved or
copied whole) and, per scan: the seam columns used, the method, the size
before and after, the dtype, the numpy version, and the SHA-256 of the
original samples. That is everything needed to rebuild any repair from its
raw scan, which is what `--verify` does; the hash lets it check that the
samples inside a repaired file are those of the raw scan.

---

## Licence

GNU General Public License v3.0 or later. Copyright (C) 2026 omoumi.
See `LICENSE` in the repository root.
