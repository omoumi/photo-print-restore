# Stage 5 — colorizing the B&W prints

Colorizes the B&W prints: a model paints each of them, and only the model's
colour is kept, laid on the print's own lightness. Detail, grain and faces
are the print's own, whatever the model redrew. Colour prints and the backs
of prints are left out. The results go to a folder of their own.

Two steps:

| Step | What | How |
|---|---|---|
| 1 of 2 | colour from the model | `colorize_model.py`: Qwen-Image-Edit-2511 in ComfyUI paints every B&W print, at 2 megapixels |
| 2 of 2 | the colour on the print | `transfer_color.py`: the model's colour, fitted to the print and set on its own lightness, at full size |

Each step takes one optional argument, the project root (the folder of the
prints, Stage 4's output say), and finds everything else by fixed names
inside it. Stage 5 reads nothing of any other stage.

```
python colorize_model.py                 step 1: the guides (ComfyUI running)
python transfer_color.py                 step 2: small previews to look at
python transfer_color.py --apply         step 2: the full-size results
python colorize_model.py --apply         step 1, each print's result written
                                         as soon as its guide is made
```

On Windows, `run_pipeline.bat` does all of it unattended: it starts
ComfyUI, makes the guides, writes the results, and goes on where it stopped
when run again.

Not steps:

| File | Role |
|---|---|
| `calibrate_color.py` | measures step 2's colour numbers on colour prints made B&W |
| `common.py` | shared code; must sit beside the steps |
| `parallel.py` | worker-count policy; optional: without it step 2 runs in one process and `--workers` is ignored |
| `colorize_by_hand.json` | the same graph as step 1, for one print by hand in ComfyUI's page |

Every step prints the same way: a heading (`STAGE 5 step 1  -  ...`), then
one `[phase n of m]` heading for each part of its work, and at the end what
it wrote and, where there is one, the command of the next step.

---

## Install

### The steps

```
pip install numpy tifffile opencv-python imagecodecs psutil
```

- `imagecodecs` is needed only for compressed TIFFs.
- `psutil` is optional; it sizes step 2's worker pool from free RAM.

Python 3.10 or newer. On Windows use `python`, not `python3`.

### ComfyUI

Step 1 talks to ComfyUI on this computer; ComfyUI does the model's work.
Set up once (Windows, an NVIDIA card; paths as in `run_pipeline.bat`):

1. **torch and torchvision with CUDA**, in the main Python. On a GTX
   10-series card torch must be a CUDA 12.6 build (`+cu126`): the CUDA 12.8
   and later builds no longer run on those cards. Check, and note the two
   versions it prints:

   ```
   python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
   ```

2. **ComfyUI**, in a venv of its own that uses that torch:

   ```
   python -m venv C:\venvs\comfy --system-site-packages
   C:\venvs\comfy\Scripts\activate
   cd C:\tools
   git clone --depth 1 https://github.com/comfyanonymous/ComfyUI
   cd ComfyUI
   pip install -r requirements.txt "torch==T" "torchvision==V"
   ```

   `T` and `V` are the two versions step 1 printed, without the `+cu...`
   part (`2.13.0+cu126` is `2.13.0`). Pinning them keeps pip from
   installing another torch over the main Python's. `C:\venvs\comfy\pyvenv.cfg`
   must say `include-system-site-packages = true`.

3. **The GGUF loader** (ComfyUI-GGUF), inside ComfyUI:

   ```
   cd custom_nodes
   git clone --depth 1 https://github.com/city96/ComfyUI-GGUF
   pip install -r ComfyUI-GGUF\requirements.txt "torch==T"
   cd ..
   ```

4. **The model files**, about 23 GB, from the ComfyUI folder (`-C -` goes
   on where a broken download stopped):

   ```
   curl -f -L -C - -o models\vae\qwen_image_vae.safetensors https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors
   curl -f -L -C - -o models\loras\Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors
   curl -f -L -C - -o models\text_encoders\qwen_2.5_vl_7b_fp8_scaled.safetensors https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors
   curl -f -L -C - -o models\diffusion_models\qwen-image-edit-2511-Q4_K_M.gguf https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF/resolve/main/qwen-image-edit-2511-Q4_K_M.gguf
   ```

   All four are GGUF or safetensors: numbers only, no code that runs when
   they load.

5. **Start it** (`run_pipeline.bat` does this by itself):

   ```
   C:\venvs\comfy\Scripts\activate
   cd C:\tools\ComfyUI
   python main.py
   ```

   It serves only this computer, at `http://127.0.0.1:8188`; no print
   leaves it. This address is set in two places, `COMFY_URL` in
   `colorize_model.py` and in `run_pipeline.bat`: change both together.
   Step 1 needs a ComfyUI recent enough that its `ImageScaleToTotalPixels`
   node has the `resolution_steps` input; it says so when it does not.

The run can be stopped and taken up again as often as wanted.

---

## The project folder

```
D:\archive\restored\
    raw0001.tif ...                     the prints
    colorize_hints.txt                  a sentence for a print (optional; you write it)
    colorize_guides\
        raw0001_seed<N>.png ...         the model's colour pictures (step 1)
    colorize_preview\
        raw0001.jpg ...                 B&W beside colour, small (step 2)
    colorized\
        raw0001.tif ...                 the results, under their own names (step 2, --apply)
```

`colorized\` can be anywhere else, with `--out`. It can never be the prints
folder itself.

---

## Step 1: colorize_model.py

Three phases, each printed as it runs.

**1. The prints.** Every TIFF in the folder. A print stored with one
picture channel is B&W; a print with three is left out and counted, however
grey it looks. Stage 4 writes every B&W print as one grey channel and every
colour print and back of a print in RGB, so in its output no colour print
and no back is ever colorized. A back Stage 4 took for a B&W print is
corrected there (`back` in its `restore_overrides.txt`). A print whose
guide is made already -- there, and newer than the print -- is left out
too, so a run that stopped goes on where it stopped; a print written again
since (Stage 4 run again, say) gets a new guide. Two prints whose names
differ only in the extension or the case (`raw0001.tif`, `raw0001.TIFF`)
would share one guide: both are left out and named.

A guide is not made again when the prompt, a hint or a setting changes;
`--redo` does that, with `--only` for single prints. The step names the
prints with a hint whose guide is older than `colorize_hints.txt`.

**2. ComfyUI.** It must answer at `COMFY_URL` and have every node, input
and model file the graph uses; what is missing is named. Only one run of
this step may use a ComfyUI at a time, as every print goes up under one
name: a second run stops at once and says so. The run then takes off
ComfyUI's queue whatever an earlier run left there -- a run whose window
was closed leaves its print running in ComfyUI.

**3. The guides.** Every B&W print goes to ComfyUI as an 8-bit grey
picture at its own size, with the prompt, and ComfyUI brings it to 2
megapixels (`MODEL_MP`, Lanczos), whatever the print's size. The model's
colour picture comes back and is written as
`colorize_guides\<name>_seed<N>.png`. Every print goes up to ComfyUI under
one name, so its input folder does not fill up; ComfyUI keeps its own
copies of the guides in its temp folder, which it empties each time it
starts.

The graph: Qwen-Image-Edit-2511 (GGUF, Q4_K_M), the 8-step Lightning LoRA,
the print as the reference at the output's own size (so the model neither
crops nor shifts the picture), euler, 8 steps, CFG 1, denoise 1.

**The prompt** asks for natural, slightly muted colours, and neutral white
or grey where a colour cannot be told from the photograph. `--context`
adds one sentence for every print of the run; `colorize_hints.txt` adds one
for a single print:

```
raw0012: The car is dark red.
raw0047: The walls of the house are white.
```

One print a line; `#` starts a comment; the extension may be left out. A
line that cannot be read is named and ignored.

**The seed.** The same print, prompt and seed always give the same guide;
another seed gives other colours. No seed is known to be better than
another for this model; the default, 220476035819595 (`SEED` in
`common.py`, shared by both steps; `run_pipeline.bat` has its own copy), is
one that gave good colours on a test print. `--seed` sets another; step 2
must then be given the same `--seed`.

**The results as they come (`--apply`).** Each guide, once made, goes at
once to step 2, which writes the print's full-size result into
`colorized\` (or the folder `--out` names) in one background process, with one thread, while the model
works on the next print. So the results appear one by one, and a run that
is stopped keeps the result of every guide it made. A result that fails is
named; `transfer_color.py --apply` writes it later. This needs
`transfer_color.py` beside the step; `run_pipeline.bat` runs step 1 this
way.

**Too slow, or gone.** A question to ComfyUI that gets no answer within
`CALL_SEC` is asked again, `CALL_TRIES` times in all; a print whose answer
does not come fails, and the run goes on with the next. Only a ComfyUI
that does not answer at all stops the run.

### Options

| Option | Default | What it does |
|---|---|---|
| `--seed N` | 220476035819595 | the seed of the guides |
| `--only NAME ...` | all | only these prints; the extension may be left out |
| `--context TEXT` | none | a sentence for every print of this run, e.g. `"It was taken in the 1960s at the seaside."` |
| `--redo` | off | make the guides again, even those already there |
| `--apply` | off | write each print's full-size result (step 2) as soon as its guide is made |
| `--out DIR` | `ROOT\colorized` | with `--apply`: where the results go; never the prints folder itself |
| `--count` | off | only count the guides still to make: exit code 0 when there are some, 3 when there are none |

The model, the files, the prompt, `MODEL_MP`, `COMFY_URL`, `TIMEOUT_MIN`,
`CALL_SEC` and `CALL_TRIES` are in the settings at the top of
`colorize_model.py`; `SEED` is in `common.py`.

---

## Step 2: transfer_color.py

Step 2 colours the same prints as step 1: a B&W print, one picture channel,
with a guide newer than the print. A guide beside a print in RGB is never
used; a guide older than its print waits for step 1 to make it again.

The model redraws the whole picture, at low resolution, and may change a
face or a detail. So step 2 takes only its colour, and lays it on the
print, which keeps its own lightness and detail. The fit is measured at the
smaller of the two sizes -- the guide brought down to the print's size when
the print is smaller (a small print, whose guide the model drew larger),
the print's lightness brought down to the guide's otherwise. The choice
goes by the print's full size, for a preview too, so a preview has exactly
the result's fit:

1. **The whole guide onto the print**: an affine fit (ECC) of the guide's
   lightness to the print's -- a shift, a scale, a slight turn. A fit that
   scales outside `WARP_SCALE` (0.9 to 1.1), turns more than `WARP_TURN`
   (3 degrees) or shifts more than `WARP_SHIFT` (5 % of the side) has found
   something else than the guide's place; it is not used, and the print's
   line says `whole fit not used`.
2. **Every part of it onto the print**: dense optical flow (DIS), for what
   the model moved here and there, each move held to `FLOW_MAX` guide
   pixels.
3. **The colour onto the print's edges**: the colour, at the print's own
   size, through a guided filter with the print's lightness as the
   guide, so it changes where the print has an edge and stays even inside.

Then the colour is corrected:

- **The model's cast.** The model tints what should be white or grey
  yellow, by about `CAST_AB`. Nearly all of it is taken off faint colour,
  less off strong colour: the share is exp(-(C/`CAST_C`)²) at chroma C,
  about 0.9 at the cast's own chroma and 0.37 at `CAST_C`. So walls and
  white shirts turn white while skin and flowers keep their colour.
- **How strong each colour is.** The model makes the colours it gets right
  stronger than they were. Each hue family -- red, orange (skin), yellow,
  green, cyan, blue, purple -- has a gain in `HUE_GAINS`, and every colour
  is multiplied by `SATURATION`. A colour between two families takes a gain
  between theirs.
- **Light and dark.** The model most often colours what is white or near
  it -- walls, white shirts, sockets, tiles -- and tints deep shadow blue.
  So, by the print's own lightness, the colour fades out above
  `LIGHT_FADE` (L* 72 to 92) and down to 0.3 of itself below `DARK_FADE`
  (L* 35 to 15); the mid-tones keep it.

`CAST_AB` comes from `calibrate_color.py`; `CAST_C` is set by hand.
`HUE_GAINS` started there too; the yellow, cyan and blue gains were then
lowered by eye, with `LIGHT_FADE` and `DARK_FADE`, so that whites the model
tinted cream or turquoise turn white -- at the cost of some real yellow and
blue (a pool's water, a pale sky).

Every result keeps the print's lightness (CIE L*), to within the rounding
of its bits: a colour sRGB cannot show at a pixel's lightness is made
fainter until it can, rather than clipped (clipping would change the
lightness). Lab is computed exactly (CIE, D65), not with OpenCV's
approximation. Written as RGB, 16 bits when the print is 16-bit, with the
print's resolution and without a colour profile (a grey one does not fit
RGB).

Each print gets one line: the fit's shift and scale, and how far the flow
moved things (mean and largest, in guide pixels). A largest move at
`FLOW_MAX` means the model moved something further than that; look there
for colour across an edge.

**Previews first, then --apply.** Without `--apply`, every print is
coloured at 1600 px on its long side and written as
`colorize_preview\<name>.jpg`, B&W on the left, colour on the right, in
seconds. With `--apply`, every print is coloured at its full size into
`colorized\`, or the folder `--out` names. Both are made the same way with the same settings, the fit
measured on the print at its full size.

Step 2 sizes its worker pool from the largest print's size once read (not
its size on disk, which compression makes smaller) and the free RAM.

### Options

| Option | Default | What it does |
|---|---|---|
| `--apply` | off | write the full-size results; without it, only the previews |
| `--redo` | off | with `--apply`: write the results again, even those already written |
| `--out DIR` | `ROOT\colorized` | with `--apply`: where the results go; never the prints folder itself |
| `--seed N` | 220476035819595 | which guides: `<name>_seed<N>.png` |
| `--only NAME ...` | all | only these prints |
| `--workers N` | auto | parallel workers; `1` = no pool |

---

## calibrate_color.py

Measures step 2's colour numbers on colour prints whose real colours are
known: they are made B&W, colorized like any B&W print, and the result is
set against the real print.

```
python calibrate_color.py make COLOUR_DIR TEST_DIR
python colorize_model.py TEST_DIR                      (ComfyUI running)
python calibrate_color.py compare TEST_DIR
```

- `make` writes every colour picture in `COLOUR_DIR` (TIFF, JPEG or PNG,
  read as sRGB) to `TEST_DIR` in B&W, 16-bit, keeping its lightness exactly,
  and remembers where each came from in `calibration_sources.json`. A
  picture stored with one channel is left out.
- `compare` lays every guide on its print exactly as step 2 does -- the
  fit measured at full size, the colour made at 1200 px -- and measures:
  - **the cast**: the guide's colour where the real print is grey;
  - **the gains**: for each hue family, over the pixels where the model
    picked about the real hue and step 2's fades leave the colour whole
    (`KEEP_MIN`), the real chroma over the model's.

  It prints, for each family, how many of its pixels have the real hue and
  how many are coloured where the real print is grey; the colour error of
  each print before and after; and the numbers, ready to paste into
  `transfer_color.py`. `calibration_preview\<name>.jpg` shows the real
  print, the colour as it is now and with the new numbers, side by side.

A gain sets how strong a colour is, not which colour: where the model
picked another hue than the real one, no gain can mend it. And the real
prints must have the colours wanted: a faded print teaches the model's
colour to fade -- leave out a print that is badly faded.

---

## Running everything: run_pipeline.bat

Keep it beside the scripts and double-click it, or run it from a command
prompt. It asks:

1. **Where the prints are.** Enter = the folder the file is in. A pasted
   path with quotes or a trailing backslash is fine.
2. If Python or a package is missing, it says what to install and stops.
3. **Where to create the output folder, and its name.** Enter = the prints
   folder and `colorized`. The prints folder itself is refused.

Then it runs on its own:

4. **Step 2** writes the results whose guides an earlier run made.
5. **Step 1** counts the guides still to make (`--count`). When there are
   none, ComfyUI is not needed, and it goes on with 8.
6. **ComfyUI**: if it is not running, it starts it in a window of its own
   (`COMFY_VENV` and `COMFY_DIR` at the top of the file say where it is) and
   waits for it.
7. **Step 1** makes the guides of every B&W print still without one -- the
   long part -- and writes each print's result into the output folder as
   soon as its guide is made (`--apply`).
8. **Step 2** writes any result still missing (one that failed, say). When
   the folder holds no B&W print, it says so and stops here.

`SEED` at its top is a copy of `SEED` in `common.py`: change both together.

It can be stopped at any time -- close its window, or Ctrl+C -- and run
again: given the same output folder, it goes on from the first print not
done, and tries again a print that failed. When it is done, the ComfyUI window can be closed.

---

## Running again

Every part is safe to run again.

- Step 1 makes only the guides that are not there, or are older than their
  print; `--redo` makes them again (after changing the prompt, a hint or
  `MODEL_MP`, say).
- Step 2 writes only the results that are not there, or are older than
  their guide or their print (a guide made again gets its result made
  again); `--redo` writes them all again, after changing `CAST_AB`,
  `HUE_GAINS`, `SATURATION` or the fades, say.
- The previews are always written again.

---

## Messages

| Message | Meaning |
|---|---|
| `no prints in …` | the folder given holds no TIFF |
| `the output folder cannot be the prints folder itself` | choose another `--out` |
| `ComfyUI does not answer at …` | start ComfyUI first, and leave its window open |
| `ComfyUI is missing the node …` / `the file …` | a custom node or a model file is not installed (see Install) |
| `ComfyUI is missing the input resolution_steps of ImageScaleToTotalPixels` | ComfyUI is too old: update it |
| `another run of this step is using ComfyUI at …` | wait for that run to end, or close its window |
| `took N jobs of a stopped run off ComfyUI's queue` | a run whose window was closed left them; nothing to do |
| `ComfyUI has stopped; start it again and run this step again` | ComfyUI closed during the run; nothing is lost |
| `ComfyUI did not answer … within 60 s` | ComfyUI was too slow; that print is tried again next run |
| `ComfyUI failed: …` | the model failed on one print; the run goes on, the next run tries it again |
| `no guide after 60 min` | one print took too long (`TIMEOUT_MIN`); tried again next run |
| `N lines of colorize_hints.txt not read` | a hint line without `name: sentence` |
| `N prints could not be read; left out` | unreadable files |
| `N prints left out: the same name but for the extension or the case` | rename one of each pair |
| `N guides older than their print: made again` | the print was written again since; step 1 makes a new guide |
| `N prints with a hint, made before colorize_hints.txt last changed` | if the hint is new: `--redo --only NAME` |
| `whole fit not used: …` | the fit found something else than the guide's place; the flow alone lays the colour |
| `none of the prints given with --only is in the folder` | check the names after `--only` |
| `no guides in … named <name>_seed<N>.png, newer than their print` | step 2 before step 1, another `--seed`, or prints written again since |
| `a worker process died (most often: out of memory)` | step 2 with fewer `--workers`, or `--workers 1` |
| `no calibration_sources.json in …` | `calibrate_color.py make` first |

---

## Licence

GNU General Public License v3.0 or later. Copyright (C) 2026 omoumi.
See `LICENSE` in the repository root.
