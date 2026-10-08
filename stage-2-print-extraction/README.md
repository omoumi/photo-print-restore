# Stage 2 — find, straighten and crop the prints

Turns a folder of flatbed scans (Stage 1's repaired scans) into straightened,
cropped image files. The crops go to a folder of their own, and are the input
of Stage 3.

Four steps. Each takes one optional argument, the project root (the folder
holding the scans), and finds everything else by fixed names inside it.

```
python detect_borders.py        step 1   find the print
python suggest_orientation.py   step 2   suggest the way up
python review.py                step 3   approve the box, turn and way up
python deskew_crop.py           step 4   straighten, crop and orient
```

On Windows, `run_pipeline.bat` runs all four in order. Step 3 opens a window
and waits for you, so this is not an unattended run.

Not steps:

| File | Role |
|---|---|
| `common.py` | shared code; must sit beside the steps |
| `parallel.py` | worker-count policy; without it every step runs in one process and `--workers` is ignored |
| `anchor_report.py` | optional diagnostic for step 1 (see below) |

Every step prints the same way: a heading (`STAGE 2 step 1  -  ...`), then
one `[phase n of m]` heading for each part of its work, and at the end what
it wrote and the command of the next step.

---

## Install

```
pip install numpy tifffile opencv-python imagecodecs pyqtgraph PyQt5 onnxruntime psutil
```

- `numpy`, `tifffile` and `opencv-python` are needed by every step.
- `pyqtgraph` and `PyQt5` are needed only for step 3.
- `onnxruntime` is needed only for step 2.
- `imagecodecs` is needed only for compressed TIFFs.
- `psutil` is optional; it sizes the worker pool from free RAM.

Python 3.10 or newer. On Windows use `python`, not `python3`.

---

## The project folder

```
D:\archive\scans\
    raw0001.tif ...                     the scans
    json files\
        boxes_detect_borders.json       written by step 1
        orientation_suggestions.json    written by step 2 (suggestions only)
        boxes_review.json               written by step 3 (your decisions)
        boxes_deskew_crop.json          written by step 4 (the manifest)
    detect_review\                      step-1 previews, only with --review
    deskewed-cropped\                   step-4 output: the input of Stage 3
```

The ROOT argument can be the scans folder itself, the `json files` folder, a
file inside it, or a single scan; all lead to the same place. Only step 4's
output folder can be moved elsewhere, with `--out`.

Steps 2 to 4 read `boxes_review.json` if it exists, otherwise
`boxes_detect_borders.json`.

---

## Which steps change pixels

| Step | Changes pixels? |
|---|---|
| 1 detect_borders | No. Writes coordinates only. |
| 2 suggest_orientation | No. Writes suggestions only. |
| 3 review | No. You edit numbers, not images. |
| 4 deskew_crop | Only for a print tilted by 0.2° or more. |

A print tilted less than 0.2° (`--min-angle`) is cut out without rotating, so
every sample value stays exactly as scanned. Output keeps the bit depth of
the scan: 16-bit in, 16-bit out.

---

## Step 1 — detect_borders.py

Finds each print on each scan and writes its four corners.

```
python detect_borders.py
python detect_borders.py --review          also write preview JPEGs
```

| Option | What it does |
|---|---|
| `--review` | write a `_marked.jpg` preview per scan into `detect_review\` |
| `--back` | the scans show card backs: frame the card edge |
| `--ring` | find the print by its edge ring instead: one print per scan, framed red (blue on a back) in the previews; not with `--back` |
| `--new-only` | measure only the scans not measured yet; keep every other record |
| `--workers N` | parallel workers (default: auto; `1` = no pool) |

Notes:

- `--ring` is off unless you ask for it. It measures each scan shrunk to
  1600 px on the long side, so its corners are less exact than the default
  search; `anchor_report.py` does not apply to it.
- Side lines are limited to 4° of tilt. A print tilted more than that needs
  correcting by hand in step 3.
- JPG, PNG, BMP and WEBP files are measured, but step 4 is TIFF-only and
  skips them. Step 1 lists them.
- Running it again measures every scan again and overwrites
  `boxes_detect_borders.json`. Your decisions in `boxes_review.json` are not
  touched; R in step 3 brings back a scan's new detection.
- `--new-only` measures only the scans the file does not hold yet and keeps
  every other record as it is. `run_pipeline.bat` runs step 1 this way, so
  scans added to the folder later are picked up.
- A scan step 1 measures after the review began is added to steps 2 to 4
  unconfirmed, and they name it.
- When a fitted frame fails its final plausibility check, the rough box is
  used instead, and every side is reported as `rough box (...)`.

### Why is a side where it is? anchor_report.py

```
python anchor_report.py raw0016.tif
python anchor_report.py raw0016.tif raw0020.tif --draw
```

For each print on each scan it prints, side by side: how many scan lines
found the print's outer edge (anchors) and how far those scatter, how many
found the picture's edge, whether a straight side could be fitted and if not
why, the margin between the two and whether it was accepted, and finally
which pass placed each side of the final frame. It runs the same code as
step 1, so it always looks at exactly what step 1 looked at.

| Option | What it does |
|---|---|
| `--back` | the scans show card backs, as `detect_borders.py --back` |
| `--draw` | also write `detect_review\<name>_points.jpg`: the frame with the points each side was fitted on |

In the `--draw` picture a **circle** is a point the final side was fitted on
and a **cross** is a point on the print's outer edge. The red frame must pass
through the circles. A side with no circles was not measured: it was placed
from its outer edge and the margin of the other sides. On a back there are
no circles; the frame is the outer envelope of the crosses.

Neither applies to `--ring`.

---

## Step 2 — suggest_orientation.py

Looks for faces in each print, in all four quarter turns, and suggests the
turn in which the faces are upright. Nothing is applied: step 3 shows the
suggestion and you confirm it.

```
python suggest_orientation.py
```

- **high** confidence: the scan opens in step 3 already turned.
- **medium**: step 3 only names the suggestion in the status bar.
- A print without a face gets no suggestion; set its way up by hand.

The face model (SCRFD-10G from InsightFace, about 17 MB) is downloaded once
into `models\` beside the scripts. InsightFace allows non-commercial research
use only.

---

## Step 3 — review.py

A window for approving or correcting what step 1 found.

```
python review.py
python review.py --flip                    the whole archive is upside down
python review.py --no-suggestions          ignore step 2 this time
```

`--flip` sets the archive-wide flip; it does not toggle it. Without the flag
the flip is off. `V` flips a single scan on top of it.

### Keys

| Key | Action |
|---|---|
| Enter / Space | confirm this scan and go to the next |
| Ctrl + Enter | confirm this and every later undecided scan as it would open, then close (asks first) |
| N / P | next / previous without confirming |
| arrows | move the box by 1 px (Shift = 10) |
| Ctrl + arrows | move the right/bottom edge by 1 px (Shift = 10) |
| Q / E (or { / }) | turn the box by 0.1° (Ctrl = 1°) |
| 0 | set the box square to the frame again |
| = | move the edge under the pointer until the box is square |
| V | flip this scan upside down |
| . / , | turn 90° clockwise / anticlockwise |
| A / D | add a box / delete the selected box |
| Tab | select the next box |
| R | reset this scan to what step 1 recorded |
| S | skip this scan |
| F / Z | fit to window / zoom to the selected box |
| Ctrl + S | save |
| Esc | save and quit |

Mouse: the wheel zooms, dragging the middle moves the box, the edge-midpoint
handles resize, the corner handles turn the box about its centre. Hold Ctrl
while turning to snap to 15°.

`=` is for prints that really are square, such as a contact print from a 6×6
negative. On a hand-trimmed print it would cut off a real edge.

Good to know:

- A box you did not touch is saved exactly as step 1 measured it.
- Turns and flips are stored as decisions; box coordinates always refer to
  the scan as it sits on disk.
- The status bar warns when a print is tilted 0.2° or more, i.e. when step 4
  will rotate it.
- Work is saved on every confirm, skip and move between scans. You can quit
  at any time; the next run opens at the first unconfirmed scan.
- A scan that cannot be read (deleted, moved or damaged since step 1) is
  shown empty with a note in the status bar; Enter or S skips it, and its
  boxes are left as they are.

---

## Step 4 — deskew_crop.py

For each print: measures the tilt from the corners approved in step 3,
rotates the scan by that tilt (if it is 0.2° or more), cuts out the print,
applies the approved flip and quarter turns, and writes the result.

```
python deskew_crop.py
python deskew_crop.py --verify
python deskew_crop.py D:\archive\scans --out D:\archive\crops --verify
```

| Option | Default | What it does |
|---|---|---|
| `--min-angle DEG` | 0.2 | below this tilt the print is cut out without rotating |
| `--compress NAME` | keep | `keep`, `none`, `lzw` or `deflate`; all lossless |
| `--all` | off | also crop scans never confirmed in step 3 |
| `--overwrite` | off | replace crops that already exist |
| `--out DIR` | `deskewed-cropped` | where the crops go; never the scans folder itself |
| `--verify` | off | afterwards, rebuild every crop and compare (see below) |
| `--workers N` | auto | parallel workers; `1` = no pool |

How it works:

- **The tilt** comes only from the four approved corners: the angle that
  keeps the largest upright rectangle of the print. It is at most ±45°; the
  way up is decided only by the flip and turns you approved.
- **Rotation** uses Lanczos interpolation (8 × 8 samples) at the exact
  position of each pixel, in float64, rounded once when written.
- **The crop** is the full bounding box of the straightened print, so no
  emulsion is lost; a sliver of album page may show in the corners.
- **Off the glass:** where part of a print lay outside the scan, those pixels
  are written black and counted in the manifest.
- A scan with two prints gives `NAME_01.tif` and `NAME_02.tif`; with one,
  `NAME.tif`.

---

## Running everything: run_pipeline.bat

Keep it beside the six scripts and double-click it, or run it from a command
prompt. The scans can be in any folder. It asks:

1. **Where the scans are.** Enter = the folder the file is in. A pasted path
   with quotes or a trailing backslash is fine.
2. **Step 1** measures only the scans not measured yet, so scans added to
   the folder later are picked up. Delete `boxes_detect_borders.json` to
   measure every scan again.
3. **Suggest the way up?** Enter = yes. If suggestions already exist: keep
   them, make them again, or do not use them this time. Keeping them does
   not need onnxruntime.
4. **Is the whole archive upside down?** Enter = no. This sets the flip, so
   answering *no* turns it off even if an earlier run turned it on.
5. The review window opens.
6. **Where to create the output folder, and its name.** Enter = the scans
   folder and `deskewed-cropped`. The scans folder itself is refused.
7. **Verify?** Enter = no.

Once it has the scans folder, it checks that Python and the packages the
steps need are installed. A scan that step 1 or step 2 cannot process is
named and left out, and the run goes on. If step 2 fails altogether, it continues without suggestions;
any other failed step stops the run. At the end it reports whether
verification failed (exit code 3), a scan could not be cropped (2), or
another error occurred (1).

---

## --verify

Rebuilds every crop listed in the manifest from its raw scan and checks that
it is identical pixel for pixel (SHA-256). It also checks bit depth, colour
model, ICC profile, resolution (swapped after a quarter turn) and that no
TIFF tag was lost. Exit code 3 on failure.

It can be run on its own at any time with the same `--out`; if every crop
already exists, nothing is remade and only the check runs.

How independent it is:

- **Rebuilt with separate code:** the TIFF reader, the rotation matrix and
  canvas size, the flip and quarter turns, the no-data count.
- **Shared with step 4:** the Lanczos resampling itself. A bug there would
  not be caught.

The resampling is repeatable on one installation. If the crops were made
with a different numpy version, that is reported. A print that was only cut
out (not rotated) is not affected by the numpy version.

---

## Metadata

Copied to every crop: photometric interpretation, planar configuration,
extra samples, bit depth, resolution and its unit, ICC profile, description,
date, software, make, model, artist, copyright, host computer, page name,
colormap. `--verify` checks each of these that the scan has. Other tags of
the scan (XMP, document name and the like) are not copied.

`Orientation` (tag 274) is **not** copied: the flip and turns are already
applied to the pixels, and the tag would make viewers turn the image again.

---

## Running again

Every step is safe to run again.

- Steps 1 and 2 overwrite only their own file.
- Step 4 skips crops that already exist (`--overwrite` remakes them). Crops
  left in place keep their manifest entries, so `--verify` still checks them.
- If a scan's number of boxes changes, its old crops stay behind (for
  example `NAME.tif` next to the new `NAME_01.tif`). Step 4 lists them and
  leaves them alone; delete them yourself.

---

## Messages

| Message | Meaning |
|---|---|
| `nothing confirmed yet; run review.py first, or pass --all` | no scan was confirmed in step 3 |
| `nothing to do: N crops already in ...` | every crop exists; not an error |
| `the scan is WxH but review.py recorded WxH - rerun review.py` | the scan file was replaced after review |
| `... has a box without its four corners` | the steps cannot use that project file; delete it and run the steps again from step 1 |
| `--out must not be the scans folder itself` | the crops are named after the scans, so they would replace them |
| `N files in ... belong to no box of the current plan` | leftover crops (see *Running again*) |
| `N crops left in place have no earlier manifest entry` | those crops cannot be verified; use `--overwrite` |
| `PIXELS CHANGED: N samples differ` | verification failed; do not use that file |

---

## The manifest

`json files\boxes_deskew_crop.json` records, per scan, its size, flip and
turn; per crop, whether it was rotated or only cut out, the tilt, the angle
applied, the fraction of the print kept, the canvas size, the crop box, the
exact inverse matrix, the number of no-data pixels, the output size and
where the print's outline lies inside the crop and the numpy version that
computed it; and for the run, the method, the minimum angle and the output
folder (relative to the scans folder when inside it, so the archive can be
moved or copied whole).

That is everything needed to rebuild any crop from its raw scan, which is
what `--verify` does.

---

## Licence

GNU General Public License v3.0 or later. Copyright (C) 2026 omoumi.
See `LICENSE` in the repository root.
