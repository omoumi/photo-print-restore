# Stage 4 — colour and B&W correction

Corrects the colour of the prints: a faded colour print gets its colour back,
a B&W print (toned, yellowed or not) becomes neutral grey, and the back of a
print is copied exactly as it is. The corrected prints go to a folder of
their own.

Two steps:

| Step | What | How |
|---|---|---|
| 1 of 2 | levels and curves | VueScan, with the settings below |
| 2 of 2 | colour and B&W correction | `restore_colors.py`, on the TIFFs VueScan wrote |

Step 2 takes one optional argument, the project root (the folder of the
TIFFs VueScan wrote), and finds everything else by fixed names inside it. It
reads nothing of any other stage or step.

```
python restore_colors.py           analyse and write the plan
python restore_colors.py --apply   write the corrected prints
python restore_colors.py --undo    remove what --apply wrote
```

On Windows, `run_pipeline.bat` analyses, waits while you look at the plan,
then asks where to put the result and applies it.

Not steps:

| File | Role |
|---|---|
| `common.py` | shared code; must sit beside the step |
| `parallel.py` | worker-count policy; without it the step runs in one process and `--workers` is ignored |

Step 2 prints the same way as every other step: a heading
(`STAGE 4 step 2  -  ...`), then one `[phase n of m]` heading for each part
of its work, and at the end what it wrote and the command of the next step.

---

## Step 1: VueScan

VueScan 9 (x64), reading the prints as files and writing new TIFFs. Given
the first print, it goes through the whole numbered series. Every tab, as
set:

**Input**

| Setting | Value |
|---|---|
| Options | Professional |
| Task | Scan to file |
| Source | File |
| Files | the prints (the first of them; batch scan takes them all) |
| Media | Color |
| Bits per pixel | 48 bit RGB |
| Batch scan | On |
| Frame number | 1 up to the number of files |
| Scan resolution | the prints' own resolution: the DPI they were scanned at |
| Auto rotate, Auto flip, Auto skew, Mirror | off |
| Rotation | None |
| Auto save | Scan |
| Auto print | None |

**Crop**

| Setting | Value |
|---|---|
| Crop size | Maximum |
| Multi crop | Off |
| Lock aspect ratio | Off |
| Border (%) | 0 |
| Buffer (%) | 8 |
| Default options | off |

**Filter**

| Setting | Value |
|---|---|
| Restore colors | off |
| Restore fading | Off |
| Grain reduction | None |
| Sharpen | None |
| Descreen | off |
| Flatten | Auto |
| All frames | off |

**Color**

| Setting | Value |
|---|---|
| Color balance | Neutral |
| Black point (%) | 1 |
| White point (%) | 1 |
| Curve low | 0.35 |
| Curve high | 0.85 |
| Brightness, Brightness red / green / blue | 1 |
| Scanner color space | Default |
| Printer color space | sRGB |
| Show IT8 outline | off |
| Output color space | sRGB |
| Monitor color space | sRGB |
| View color | RGB |
| Pixel colors | off |

**Output**

| Setting | Value |
|---|---|
| Default folder | the folder step 2 will read, not the prints' own folder |
| TIFF file | on |
| TIFF file name | the name of the first print, with `+` before the extension (`raw0005+.tif`) |
| TIFF size reduction | 1 |
| TIFF multi page | Off |
| TIFF file type | 48 bit RGB |
| TIFF compression | Off |
| TIFF DNG format | off |
| TIFF profile | off |
| JPEG file, PDF file, OCR text file, Index file | off |
| Printed size | Scan size |
| Magnification (%) | 100 |

Notes on these settings:

- **Scan resolution.** With a file as the source, VueScan resamples it to
  the resolution set here. Set it to the resolution the prints already have
  (their DPI, in the file's properties), so no detail is lost and nothing is
  enlarged.
- **Files and names.** VueScan reads the prints as one numbered series,
  starting from the file chosen under `Files`, and writes one TIFF per
  print. The `+` in `TIFF file name` is not written into the name: it tells
  VueScan to count up from that number. So every TIFF keeps the name of the
  print it came from (the TIFF made from `raw0005.tif` is `raw0005.tif` in
  VueScan's folder), and the names in step 2's plan are the prints' own.
  The prints must be numbered without gaps, as Stage 3 writes them.
- **Crop.** `Maximum` keeps the whole print. `Buffer (%)` does not crop:
  it only keeps the outer 8 % of the picture out of VueScan's colour
  measurement (the colour balance and the black and white points), so an
  edge or a corner of album page does not sway them. `Border (%)` 0 adds
  nothing around the print. `Lock aspect ratio` only matters when a crop box
  is dragged with the mouse.

The TIFFs are sRGB, 16 bits per channel; step 2 reads their colours as sRGB.

---

## Install (step 2)

```
pip install numpy scipy tifffile opencv-python torch transformers imagecodecs psutil
```

- `numpy`, `scipy`, `tifffile` and `opencv-python` are needed.
- `torch` and `transformers` run the AI model that tells backs from fronts.
  Without them backs are told by the pixel test only, which misses some.
- The model, `facebook/dinov2-base` (about 350 MB), is downloaded once by
  `transformers` into its own cache; the first run needs the internet. If it
  cannot be loaded, the analysis goes on with the pixel test and says so.
- `imagecodecs` is needed only for compressed TIFFs.
- `psutil` is optional; it sizes the worker pool from free RAM.

Python 3.10 or newer. On Windows use `python`, not `python3`.

---

## The project folder

```
D:\archive\vuescan\
    raw0001.tif ...                     VueScan's TIFFs
    restore_overrides.txt               your corrections (you write it)
    restore_review\
        restore_plan.csv                the plan; --apply does exactly this
        restore_sheet_NN.jpg            every print before and after
        restore_log.csv                 written by --apply, read by --undo
        _cache\                         measurements, look-up tables, thumbnails,
                                        AI embeddings
    restored\                           the corrected prints, under their own names
```

`restored\` can be anywhere else, with `--out`. It can never be the prints
folder itself.

---

## What the analysis does

Four phases, each printed as it runs.

**1. The prints.** Every TIFF in the folder, and the corrections file.

**2. Measuring.** Each print is read once, at 1200 px on its long side. The
whole picture is measured, apart from pixels that are 0 in every channel (no
data). Cached, so a rerun only reads new or changed files.

**3. Fronts and backs.** The prints that look like paper are compared with
the sure backs and fronts by the AI model (see below).

**4. The plan.** Every print's class (back, colour or B&W), its
correction, and the contact sheets.

### Front or back

The back of a print -- bare paper, writing, a stamp -- is copied exactly as
VueScan wrote it, byte for byte: it keeps its colour, whatever its front is.
A back looks like paper: its paper white reaches L* 75 and at least 45 % of
it lies within 10 L* of that white, as on aged, foxed paper and under writing
and stamps, but never on a photo, however dark or flat.

Whether such a print is a back, an AI image model decides (by default
DINOv2, base): a back when its five nearest sure backs are on average nearer
than its five nearest sure fronts. The sure backs look like paper, are at
least 60 % bare paper, and are flat and light; the sure fronts do not look
like paper. The pixels alone cannot tell: levels and curves make the
writing on a back dark and its contrast high. Each print is compared in all
four quarter turns, since a back may lie any way up.

Without `torch` and `transformers`, a print that looks like paper and is
flat and light for its own white, or has no mid-tones at all, is a back.

A back is scanned right after its front, one number higher, so no two
prints in a row are both backs: of a run of them, the ones with the
strongest case (the AI's margin, or the bare paper without it), no two
adjacent, stay backs, and the others are noted `check it`.

### Colour or B&W

One test, on the fronts, on the picture without 10 % of it along every side
(a border, a stamp in a corner):

- **The ramp test**. A B&W print, toned or not, keeps one hue per
  lightness (its grey ramp); a colour print varies in hue at one and the same
  lightness. A print is colour when at least 7 % of its picture strays off the
  ramp in more than one hue (blue-violet ink, as from stamps, left out), or
  when its least colourful pixels carry a strong cast on the red-green axis
  (a colour print faded to one hue: red, magenta or orange as a rule, green
  now and then), or when at least 5 % of it strays off the ramp and its
  least colourful pixels carry a yellow cast of 12 or more (a colour print
  faded to yellow or orange; a yellowed B&W print has the cast too, but on
  its ramp, not off it). Strays all of one hue, without that cast, are
  toning that varies across a B&W print.

A print whose verdict is close to a limit is noted `check it`. A print
stored with one channel is B&W without a test.

### A colour print

Two corrections, joined into one look-up table per channel:

- **Levels.** A black and a white point per channel, which takes most of the
  cast away. Where the tones go is one stretch for all three channels (their
  mean, its darkest `--black` % to black and lightest `--white` % to white).
  The balance at each end comes from the print's greyest surfaces there: the
  25 % least colourful of its 5 % darkest or lightest pixels, which every
  channel maps to the same tone. So a coloured lightest or darkest thing (a
  sunlit wall, a blue shadow) does not tint a white or a black that is grey.
  An end with no grey surfaces takes each channel's own darkest or lightest
  pixels instead. Beyond the points, a soft toe and shoulder (`--soft`)
  instead of a hard clip: no channel is cut off on its own, which would tint
  the lightest and darkest spots and lose their detail.
- **Curves.** The picture in `--bins` bands of lightness, each holding as many
  pixels as the next. In every band the least colourful quarter of the pixels
  stands for grey, and a monotone curve per channel takes `--neutral` of
  their remaining cast away. Faded prints have a different cast in shadows,
  mid-tones and highlights.

Written as RGB, 16 bits, with the source's colour profile and resolution.

### A B&W print

Its lightness (CIELAB L*) is stretched between its darkest `--black` % and
lightest `--white` %, and its tint is taken away. Written as one grey channel,
16 bits, with the source's resolution and without a colour profile (one made
for colour does not fit it).

---

## Options

| Option | Default | What it does |
|---|---|---|
| `--out DIR` | `ROOT\restored` | with `--apply`: where the corrected prints go |
| `--compress C` | `keep` | with `--apply`: `keep`, `none`, `lzw` or `deflate` |
| `--black F` | 0.3 | percentile set to black |
| `--white F` | 99.7 | percentile set to white |
| `--soft F` | 0.05 | share of the tone range each soft end takes, 0 to 0.5; `0` = a hard clip |
| `--neutral F` | 0.9 | share of the remaining cast the curves take away |
| `--bins N` | 6 | bands of lightness for the curves |
| `--model NAME` | `facebook/dinov2-base` | the Hugging Face image model that tells backs |
| `--device D` | auto | `auto`, `cpu` or `cuda`: where the model runs |
| `--fresh` | off | measure everything again, ignoring the cache (the AI embeddings too) |
| `--workers N` | auto | parallel workers; `1` = no pool |

The thresholds of the colour-or-B&W tests are in the settings at the top of
`restore_colors.py`.

---

## Looking at the plan

- **`restore_sheet_NN.jpg`**: every print as two pictures, before and after,
  with its name and class (`COLOUR`, `B&W` or `BACK`). The name is red when
  the plan notes `check it`.
- **`restore_plan.csv`**: one row per print: its class (`back`, `colour` or
  `bw`), whether the tests or you set it, the numbers of the tests (the
  colour test; the bare paper and paper white of the back test; for a print
  that looks like paper, its AI distance to the sure backs and fronts), and
  the notes.

Notes worth a look:

| Note | Meaning |
|---|---|
| `strong cast but little colour (a faded colour print?): check it` | B&W, though its cast is close to the colour limit |
| `colour off the grey ramp, but all of one hue (toning?): B&W: check it` | B&W, though coloured in places |
| `verdict close to its limit: check it` | the ramp test was nearly the other way |
| `too little mid-tone area to test: B&W: check it` | a very dark or very light print |
| `colour faded to yellow (b* cast …)` | colour by its yellow cast and the colour off its ramp |
| `back: copied as it is (AI distance …)` | a back, found by the AI (without the distances: by the pixel test) |
| `looks like a back, but a print next to it is the stronger back: check it` | two would-be backs in a row; this one was judged a front |
| `B&W in the corrections`, `COLOUR in the corrections`, `back in the corrections` | set in `restore_overrides.txt` |
| `skipped in the corrections` | not written |
| `one channel: cannot be a colour print, B&W used` | a one-channel print set to colour |

---

## Corrections: restore_overrides.txt

Beside the prints, one print per line, then analyse again (fast: the
measurements are cached):

```
raw0012 bw           a B&W print, whatever the tests say
raw0013 colour       a colour print, whatever the tests say
raw0014 back         a back: copied as it is
raw0015 front        not a back: the test says colour or B&W
raw0020 skip         not written
```

The extension may be left out; `color` is understood too; `#` starts a
comment. A line that cannot be read, or that names a print not in the folder,
is named and ignored. Of two lines that set one print both `bw` and
`colour`, or both `back` and `front`, the later one is used and the clash
is named. `bw` or `colour` makes a print a front too.

---

## --apply and --undo

`--apply` checks first that every print is still the file that was analysed,
then writes each corrected print into the output folder under its own name;
a back is copied as it is.

`--undo` removes what the last `--apply` wrote, each file only while it is
still the file that was written; a file changed since is named and left.

Each print is written under a temporary name first, so a crash never leaves
a half-written file behind. The log records the output folder relative to
the prints folder when it lies inside it, so the archive can be moved or
copied whole and `--undo` still works.

A plan can be applied only once; `--undo` first to apply a new one.

---

## Running step 2: run_pipeline.bat

Keep it beside the three scripts and double-click it, or run it from a
command prompt. The TIFFs can be in any folder. It asks:

1. **Where VueScan's TIFFs are.** Enter = the folder the file is in. A pasted
   path with quotes or a trailing backslash is fine.
2. If Python or a package is missing, it says what to install and stops. If
   only `torch` or `transformers` is missing: **continue without the AI?**
   Enter = no.
3. The analysis runs and the plan is written.
4. **What next?** Enter = apply; `R` = analyse again after editing
   `restore_overrides.txt`; `O` = open the plan's folder; `Q` = stop, the
   plan stays for later.
5. If an earlier plan was applied: **remove its files first?** Enter = yes.
6. **Where to create the output folder, and its name.** Enter = the prints
   folder and `restored`.

Prints that cannot be read, or written, are named and the run goes on;
when none can be read, or any other error occurs, it stops.

---

## Running again

Every part is safe to run again.

- Measurements are cached per file and per setting; a changed file, or a
  changed `--black`, `--white`, `--soft`, `--neutral` or `--bins`, is
  measured again. `--fresh` measures everything again.
- The AI embeddings are cached per file and per model.
- Changing the corrections only redoes the plan.
- The cache keeps only what the last analysis used: entries of files that
  have gone or changed, or of other settings, are dropped.

---

## Messages

| Message | Meaning |
|---|---|
| `no prints in …` | the folder given holds no TIFF |
| `restore_overrides.txt: N lines not understood, ignored` | a correction that could not be read |
| `restore_overrides.txt names prints that are not in the folder` | a correction for a missing print |
| `restore_overrides.txt: lines that contradict an earlier line` | one print set both `bw` and `colour`, or `back` and `front`; the later line is used |
| `N prints noted "check it"` | classes to check on the sheets |
| `torch or transformers is not installed: backs by the pixel test only` | install them to tell backs by the AI |
| `the model … could not be loaded` | no internet on the first run, or a wrong `--model`; backs by the pixel test |
| `too few sure backs or fronts to compare with` | backs by the pixel test |
| `N prints could not be measured; left out of the plan` | unreadable files |
| `no print could be measured; no plan written` | no file in the folder could be read |
| `the plan was applied already` | run `--undo` first |
| `nothing written: the folder no longer matches the plan` | a print changed or went missing; analyse again |
| `the output folder cannot be the prints folder itself` | choose another `--out` |
| `its measurement is gone from the cache: run the analysis again` | `_cache\` was emptied after the analysis |
| `N files changed since --apply and left in place` | `--undo` kept files you edited |

---

## Licence

GNU General Public License v3.0 or later. Copyright (C) 2026 omoumi.
See `LICENSE` in the repository root.
