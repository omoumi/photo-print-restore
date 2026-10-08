# Stage 3 — renumber the prints so related ones sit side by side

Gives the prints Stage 2 cropped new consecutive numbers, so that prints which
belong together sit side by side: B&W first, then colour, and inside each,
group after group. The crops keep their own names; the prints go, under
their new names, into a folder of their own.

One step. It takes one optional argument, the project root (the folder
holding Stage 2's crops), and finds everything else by fixed names inside it.

```
python renumber_by_similarity.py           analyse and write the plan
python renumber_by_similarity.py --apply   link the prints under their new names
python renumber_by_similarity.py --undo    remove what --apply made
```

On Windows, `run_pipeline.bat` analyses, waits while you look at the plan,
then asks where to put the result and applies it.

Not steps:

| File | Role |
|---|---|
| `common.py` | shared code; must sit beside the step |
| `parallel.py` | worker-count policy; without it the step runs in one process and `--workers` is ignored |

The step prints the same way as every other: a heading
(`STAGE 3 step 1  -  ...`), then one `[phase n of m]` heading for each part
of its work, and at the end what it wrote and the command of the next step.

---

## Install

```
pip install opencv-python numpy scipy tifffile torch transformers imagecodecs psutil
```

- `torch` and `transformers` run the AI model. Without them the prints are
  grouped by size, cast and corners only. A GPU (CUDA build of `torch`) makes
  the first run much faster; later runs read the embeddings from the cache.
- The model, `facebook/dinov2-with-registers-large` (about 1.2 GB), is
  downloaded once by `transformers` into its own cache; the first run needs
  the internet. If it cannot be loaded, the analysis goes on without the AI
  and says so. Other models: see *Other AI models* below.
- `imagecodecs` is needed only for compressed TIFFs.
- `psutil` is optional; it sizes the worker pool from free RAM.

Python 3.10 or newer. On Windows use `python`, not `python3`.

---

## The project folder

```
D:\archive\scans\deskewed-cropped\
    raw0001.tif ...                     Stage 2's crops
    organize_overrides.txt              your corrections (you write it)
    organize_review\
        organize_plan.csv               the plan; --apply does exactly this
        organize_summary.txt            the groups as a tree, with notes
        sheet_NN.jpg                    contact sheets of the groups
        organize_log.csv                written by --apply, read by --undo
        _cache\                         measurements, thumbnails, embeddings
    organized\                          the prints under their new names
```

`organized\` can be anywhere else, with `--out`. It can never be the crops
folder itself.

**Names.** A crop is named as Stage 2 wrote it: `raw0137.tif`, or
`raw0137_02.tif` when one scan held several prints. Its new name keeps the
prefix and takes the new number: `raw0137_02.tif` may become `raw0005.tif`.
A TIFF without a number in its name is listed and left alone.

---

## What the analysis does

Five phases, each printed as it runs.

**1. Reading the prints.** Size (in mm from the DPI; in pixels when the
crops carry none), bare paper, colour, cast, corners and a thumbnail. The
slow part; cached, so a rerun only reads new or changed files.

**2. Fronts and backs.** A back looks like paper: its lightest tone is light,
and most of it lies close to that tone, as on aged, foxed paper and under
writing, stamps and stains, but never on a photo, however dark or flat.
Whether such a print is a back, the AI decides, from the sure backs and
fronts most like it. Then the archive rule: **a back is scanned right after
its front, its number one higher**, so no two prints in a row are both backs.
A back is filed right after its front; one without a front before it goes to
the end.

**3. Colour or B&W.** A B&W print, toned or not, keeps one hue per lightness.
A print is colour when at least 7 % of its picture strays off that in more
than one hue (blue-violet ink, as from stamps, left out), or when it has a
strong cast on the red-green axis: a colour print faded to one hue, most
often red, magenta or orange, though a strong green cast counts too. It is
colour too when at least 5 % of its picture strays off the ramp and it has
a yellow cast of 12 or more: a colour print faded to yellow or orange (a
yellowed B&W print has the cast, but on its ramp, not off it), or when at
least 4 % strays off in many hues (a pale colour print). Colour and B&W
prints never share a group. Prints close to a limit are listed and
noted `check it`.

**4. Grouping.** Inside each class, one clustering on the picture content
(by default DINOv2 with registers, large, on 448 px views, each print
compared in all four quarter turns, the distances re-ranked by shared
neighbours), plus three small, capped penalties: print size (free up to
3 mm), cast, and rounded against sharp corners. A penalty only makes two
prints less alike; it never keeps apart two whose pictures are close. The
split is chosen automatically (best silhouette score). The merge tree never
undoes a merge, so every print is then checked against the groups that
formed: one nearer on average to another group moves to it, until nothing
moves. A print the cut leaves alone stays single: the group it is nearest
to on average is not evidence that it belongs there. One far from every
group pairs with its nearest single print, when the two are near enough and
that print is nearer to it than to a group: two prints alike that no group
fits, such as two taken at one place and time. Groups and single prints,
and the prints inside each group, are ordered so that neighbours look
alike: a single print sits between the groups, beside the one most like
it.

**5. The review files.** The plan, the tree and the contact sheets are
written, with a summary of the groups.

The new numbers run: all B&W (its groups and single prints), then all
colour, then unreadable prints and backs without a front.

---

## Options

| Option | Default | What it does |
|---|---|---|
| `--out DIR` | `ROOT\organized` | with `--apply`: where the prints go |
| `--colour-thr F` | 0.07 | share of the picture off the grey ramp that makes a print colour |
| `--size-free F` | 3 | sides this many mm apart cost nothing |
| `--size-w F` | 0.15 | largest size penalty; `0` ignores the size |
| `--cast-w F` | 0.05 | largest cast penalty; `0` ignores the cast |
| `--corner-w F` | 0.15 | rounded against sharp corners; `0` ignores them |
| `--ai-thr F` | auto | the split; larger = fewer, bigger groups |
| `--pair-max F` | 1.4 | a single print far from every group pairs with another only within F × the split |
| `--no-grey-assist` | off | colour embedding only (twice as fast); `--no-gray-assist` works too |
| `--model NAME` | `facebook/dinov2-with-registers-large` | another Hugging Face image model (see below) |
| `--device D` | auto | `auto`, `cpu` or `cuda` |
| `--fresh` | off | measure everything again, ignoring the cache |
| `--workers N` | auto | parallel workers; `1` = no pool |

### Other AI models (`--model`)

`--model` takes any Hugging Face image model that `transformers` can load
with `AutoModel`, for example:

```
python renumber_by_similarity.py --model facebook/dinov2-base
```

Each model is downloaded once, the first time it is used. Its embeddings are
cached under its own name, so going back to an earlier model costs nothing.
With `--ai-thr auto` the split is found again for each model; compare the
sheets to choose.

| Model | Size | Notes |
|---|---|---|
| `facebook/dinov2-small` | ~90 MB | fastest; coarser groups |
| `facebook/dinov2-base` | ~350 MB | several times faster than the default |
| `facebook/dinov2-large` | ~1.2 GB | finer |
| `facebook/dinov2-giant` | ~4.5 GB | finest; a GPU is advisable |
| `facebook/dinov2-with-registers-base` | ~350 MB | DINOv2 with register tokens: cleaner detail |
| `facebook/dinov2-with-registers-large` | ~1.2 GB | the default: the same, larger |
| `openai/clip-vit-base-patch32` | ~600 MB | CLIP: groups more by subject than by look |
| `openai/clip-vit-base-patch16` | ~600 MB | CLIP, finer patches |
| `openai/clip-vit-large-patch14` | ~1.7 GB | CLIP, larger |
| `google/siglip-base-patch16-224` | ~800 MB | SigLIP, CLIP-like |

The DINOv2 models (every name containing `dinov2`) look at 448 px views;
every other model gets 224 px views as they are, without resizing, so it
must be one made for 224 px input. A model made for another size, such as
`google/siglip-so400m-patch14-384`, does not fit. The sizes are those of the
downloads and approximate.

---

## Looking at the plan

- **`sheet_NN.jpg`**: every group, and every single print, as a block of
  thumbnails under the new names, each with its own name below in grey;
  `+back` marks a print whose back follows it.
- **`organize_summary.txt`**: the same as a tree, every file with its new
  name and its notes.
- **`organize_plan.csv`**: one row per file, with every measurement.

Notes worth a look:

| Note | Meaning |
|---|---|
| `back (bare paper …; AI distance …)` | made a back by the AI, not a sure case |
| `front, though flat (…)` | flat like a back, but judged a front |
| `front: a print next to it is the stronger back` | the archive rule chose its neighbour |
| `… : check it` | the colour verdict is close to a limit |
| `moved: nearer to this group (…)` | the cut put it in another group; it fits this one better |
| `near no group; paired with … (distance …)` | two single prints made a group of their own |
| `single: near no group (…)` | the cut left it alone; it stays single, beside the group most like it |
| `size differs from its front` | a back that may not belong to its front |

---

## Corrections: organize_overrides.txt

Beside the crops, one print per line, then analyse again (fast: the
measurements and embeddings are cached). Name each print by its own name,
the grey one on the sheets (`original_name` in the plan), not by its new
number, which can change from one analysis to the next:

```
raw0012.tif back
raw0013_02.tif front
raw0020.tif bw
raw0021.tif colour      # a comment
```

The extension may be left out; `color`, `b&w`, `grey`, `gray` and `mono` are
understood too. A line that cannot be read is named and ignored; of two
lines that contradict each other (`front` and `back` for one print), the
later one is used and the clash is named. Every applied override is noted
in the plan.

---

## --apply and --undo

`--apply` checks first that every crop is still the file that was analysed,
then **links** each one into the output folder under its new name. Where a
link is impossible (another drive, a file system without links) it copies.
A linked print and its crop are one file on disk: edit neither in place.

`--undo` removes what the last `--apply` made, each file only while it is
still the file that was written; a file changed since is named and left.

A plan can be applied only once; `--undo` first to apply a new one. The log
records the output folder relative to the crops folder when it lies inside
it, so the archive can be moved or copied whole and `--undo` still works.
If a file cannot be written (a full disk, say), `--apply` stops there; the
files written so far are in the log, so `--undo` removes them.

---

## Running everything: run_pipeline.bat

Keep it beside the three scripts and double-click it, or run it from a
command prompt. The crops can be in any folder. It asks:

1. **Where Stage 2's crops are.** Enter = the folder the file is in. A pasted
   path with quotes or a trailing backslash is fine. It then checks that
   Python and the packages the step needs are installed.
2. If `torch` or `transformers` is missing: **continue without the AI?**
   Enter = no.
3. The analysis runs and the plan is written.
4. **What next?** Enter = apply; `R` = analyse again after editing
   `organize_overrides.txt`; `O` = open the plan's folder; `Q` = stop, the
   plan stays for later.
5. If an earlier plan was applied: **remove its files first?** Enter = yes.
6. **Where to create the output folder, and its name.** Enter = the crops
   folder and `organized`.

Unreadable prints are named, filed at the end, and the run goes on; when
none can be read, or any other error occurs, it stops.

---

## Running again

Every part is safe to run again.

- Measurements and thumbnails are cached per file; a changed file is
  measured again. `--fresh` measures everything again.
- Embeddings are cached per model and view size, so trying another
  `--model` and coming back costs nothing the second time.
- Changing a threshold (`--ai-thr`, `--colour-thr`, the penalties) or the
  overrides only redoes the fast part.

---

## Messages

| Message | Meaning |
|---|---|
| `no prints in …` | the folder given is not the one holding Stage 2's crops |
| `no numbered prints in …` | the TIFFs there have no number in their names |
| `no print could be read; no plan written` | every crop failed to read |
| `the model … could not be loaded` | no internet on the first run, or a wrong `--model`; grouped without the AI |
| `torch or transformers is not installed: no AI grouping` | install them for grouping by content |
| `N prints to check on the sheets` | colour verdicts close to a limit |
| `organize_overrides.txt line N ignored` | a correction that could not be read |
| `organize_overrides.txt line N contradicts an earlier line` | the later line is used |
| `the plan was applied already` | run `--undo` first |
| `nothing written: the folder no longer matches the plan` | a crop changed or went missing; analyse again |
| `the output folder cannot be the prints folder itself` | choose another `--out` |
| `stopped: could not write …` | `--apply` stopped part way; `--undo` removes what it wrote |
| `N files changed since --apply and left in place` | `--undo` kept files you edited |

---

## Licence

GNU General Public License v3.0 or later. Copyright (C) 2026 omoumi.
See `LICENSE` in the repository root.
