# photo-print-restore -- reproducible, verifiable restoration of flatbed scans
# Copyright (C) 2026 omoumi
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <https://www.gnu.org/licenses/>.

# ==========================================================================
# restore_colors.py  --  STAGE 4, step 2 of 2: colour and B&W correction
#
# STAGE 4 corrects the colour of the prints: a faded colour print gets its
# colour back, a B&W print (toned, yellowed or not) becomes neutral grey:
#
#     VueScan                  step 1 of 2   levels and curves
#     restore_colors.py        step 2 of 2   colour and B&W correction
#     common.py                shared code, not a step
#     parallel.py              shared worker-count policy, not a step
#
# Requirements:
#     pip install numpy scipy tifffile opencv-python torch transformers
#                 [imagecodecs] [psutil]
#     common.py beside this file; parallel.py too, for --workers
#
# Works on the folder of TIFFs step 1 (VueScan) writes, or any folder of
# prints; it reads nothing of any other stage or step. The analysis writes
# a plan and contact sheets, --apply writes the corrected prints into a
# folder of their own (16-bit stays 16-bit, the colour profile and the
# resolution are kept), --undo removes them. The whole picture is
# measured, apart from pixels that are 0 in every channel (no data).
#
# FRONT OR BACK. A back of a print -- bare paper, writing, a stamp -- is
# written exactly as it is, a byte-for-byte copy: it keeps its colour,
# whatever the colour or B&W of its front. A back looks like paper: its
# paper white reaches BACK_MIN_WHITE L* and at least BACK_PAPER_MIN of it
# lies within BACK_PAPER_DL of that white, as on aged, foxed paper and
# under writing and stamps, but never on a photo, however dark or flat.
# Whether such a print is a back, an AI image model decides (by default
# DINOv2, base), from the sure cases most like it: paper with
# BACK_BLANK_SURE bare paper and flat is a back, a print that is no paper
# is a front. Levels and curves make the writing on a back dark and its
# contrast high, so the pixels alone cannot tell. Without torch and
# transformers, a print that looks like paper and is flat and light for its
# own white, or has no mid-tones at all, is a back. A back is scanned right
# after its front, one number higher, so no two prints in a row are both
# backs: of a run of them the ones with the strongest case, no two
# adjacent, stay backs.
#
# COLOUR OR B&W, for the fronts. The test is on the picture without
# CARD_INNER of it along every side (a border, a stamp in a corner):
#
#     ramp     a B&W print, toned or not, has one hue per lightness at most
#              (its grey ramp); a colour print varies in hue at one and the
#              same lightness. Colour when more than COLOUR_AREA_MIN of the
#              picture strays off the ramp by COLOUR_DEV in more than one
#              hue (a spread of COLOUR_SPREAD_MIN; strays all of one hue are
#              toning that varies across a B&W print), blue-violet strays
#              (INK_HUE: stamps, pen) left out -- or when the least coloured
#              pixels carry a cast of COLOUR_CAST_MIN on the red-green axis
#              (a colour print faded to one hue: red, magenta or orange as a
#              rule, green now and then), which B&W toning does not give --
#              or when FADED_AREA_MIN of it strays off the ramp and the
#              least coloured pixels carry a yellow cast of FADED_CAST_B (a
#              colour print faded to yellow or orange: a yellowed B&W print
#              has the cast, but on its ramp, not off it)
#
# It tells both a colour print faded to one hue and a B&W print toned
# unevenly. A print whose verdict is close to a limit is noted "check it"
# and listed by name. A print stored with one channel is B&W without a
# test. A side or class set in the corrections is taken as it is.
#
# A COLOUR PRINT. Two corrections, joined into one look-up table per
# channel, measured on the picture at ANALYSIS_SIDE:
#
#     levels   a black and a white point per channel, which takes most of
#              the cast away. Where the tones go is one stretch for all
#              three channels: their mean, its darkest --black % to black
#              and lightest --white % to white. The balance at each end
#              comes from the print's greyest surfaces there (the GREY_PCT
#              % least colourful of its END_PCT % darkest or lightest
#              pixels), which every channel maps to the same tone -- so a
#              coloured lightest or darkest thing (a sunlit wall, a blue
#              shadow) does not tint a white or a black that is grey. An
#              end whose greyest surfaces are not grey (chroma above
#              END_GREY_C) takes each channel's own darkest --black % or
#              lightest --white % instead. Beyond the points, a soft toe
#              and shoulder (--soft) instead of a hard clip, so no channel
#              is cut off on its own, which would tint the lightest and
#              darkest spots and lose their detail
#     curves   the picture in --bins bands of lightness, each holding as
#              many pixels as the next; in every band the least colourful
#              quarter of the pixels stands for grey, and a monotone curve
#              per channel takes --neutral of their remaining cast away
#              (faded prints have a different cast in shadows, mid-tones
#              and highlights)
#
# A B&W PRINT. Its lightness (CIELAB L*) is stretched between its darkest
# --black % and lightest --white %, and its tint is taken away: it is
# written as one grey channel, without a colour profile (one made for
# colour does not fit it).
#
# Outputs, derived from the project root -- the folder of the prints:
#
#     restore_review/restore_plan.csv     the plan: every print's class, the
#                                         numbers of the tests, the notes
#     restore_review/restore_sheet_NN.jpg every print before and after, with
#                                         its class; red when the plan
#                                         notes "check it"
#     restore_review/restore_log.csv      written by --apply, read by --undo
#     restore_review/_cache/              the measurements, the look-up
#                                         tables, small pictures and the
#                                         AI embeddings
#     restored/<name>                     written by --apply, unless --out
#
# Corrections go in ROOT\restore_overrides.txt, one per line (the extension
# of the name may be left out); then run the analysis again:
#
#     raw0012 bw                    a B&W print, whatever the tests say
#     raw0013 colour                a colour print, whatever the tests say
#     raw0014 back                  a back: copied as it is
#     raw0015 front                 not a back: the tests say colour or B&W
#     raw0020 skip                  not written
#
# Usage:
#     python restore_colors.py [ROOT] [options]   analyse, write the plan
#     python restore_colors.py [ROOT] --apply     write the corrected prints
#     python restore_colors.py [ROOT] --undo      remove what --apply wrote
#
#     ROOT                  project root: the folder of the prints (default:
#                           this folder)
#     --out DIR             with --apply: the folder for the corrected prints
#                           (default: ROOT\restored); never ROOT
#     --compress C          with --apply: keep | none | lzw | deflate
#                           (default keep: the source's own codec)
#     --black F             percentile set to black (default 0.3)
#     --white F             percentile set to white (default 99.7)
#     --soft F              share of the tone range each soft end takes,
#                           0 to 0.5 (default 0.05; 0 = a hard clip)
#     --neutral F           share of the remaining cast the curves take away
#                           (default 0.9)
#     --bins N              bands of lightness for the curves (default 6)
#     --model NAME          the Hugging Face image model that tells backs
#                           (default facebook/dinov2-base)
#     --device D            auto | cpu | cuda: where the model runs
#     --fresh               measure every print again, ignoring _cache/
#     --workers N           parallel workers (default: auto; 1 disables the
#                           pool)
#
#     python restore_colors.py D:\archive\vuescan
#     python restore_colors.py --apply --out D:\archive\restored
# ==========================================================================
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
import sys
import time

try:
    import cv2
    import numpy as np
    from scipy.interpolate import PchipInterpolator
except ModuleNotFoundError as exc:
    sys.exit(f"missing package: {exc.name} (see Requirements above)")

try:
    from common import (SCAN_EXTS, Progress, check_scans_present,
                        colour_part, console_setup, count, elapsed, error,
                        error_list, heading, info, list_prints, paint, phase,
                        pipeline_root, read_page, resolve, restored_dir,
                        review_dir, root_arg, same_dir, unit_range, warn,
                        warn_list, write_tiff)
except ModuleNotFoundError as exc:
    sys.exit("common.py must sit beside this script" if exc.name == "common"
             else f"missing package: {exc.name} (see Requirements above)")

try:
    import parallel                       # optional: enables --workers
except ModuleNotFoundError:
    parallel = None

# --------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------
# --- general ---------------------------------------------------------------
PREFIX = "restore"          # prefix of the step's files in restore_review/
SCRIPT = "restore_colors.py"
OVERRIDE_NAME = "restore_overrides.txt"  # your corrections, beside the prints
CACHE_DIR = "_cache"        # inside restore_review/
WORKER_MEM_MULTIPLE = 14.0  # peak RAM of one worker per byte of the print
                            # file: the decoded print, its float copy, the
                            # corrected picture and its CIELAB copies
CACHE_VERSION = 3           # bump whenever the measurement changes
ANALYSIS_SIDE = 1200        # the picture is measured at this long side
THUMB_SIDE = 400            # long side of the picture on the contact sheets
LUT_SIZE = 4096             # entries of the look-up table per channel

# --- colour or B&W: the ramp test ------------------------------------------
CARD_INNER = 0.10       # the tests leave out this share of every side
COLOUR_SIDE = 410       # the picture is averaged down to this long side ...
COLOUR_BLUR = 2.0       # ... and smoothed (sigma, px): grain and specks go
COLOUR_DEV = 4.0        # a pixel is off the grey ramp this far in a*b*
INK_HUE = (220.0, 320.0)    # blue-violet off the ramp is ink (a stamp), not
                            # picture colour: hue of the offset, in degrees
COLOUR_AREA_MIN = 0.07  # colour with this share of the picture off the
                        # ramp, ink left out, ...
COLOUR_SPREAD_MIN = 0.05    # ... spread over hues at least this much (all
                            # in one hue is uneven toning of a B&W print) ...
COLOUR_CAST_MIN = 4.5   # ... or a cast (|a*| of the least coloured pixels,
                        # red or green) this strong: a colour print faded
                        # to one hue
FADED_AREA_MIN = 0.05   # ... or this share off the ramp with ...
FADED_CAST_B = 12.0     # ... a yellow cast (b* of the least coloured
                        # pixels) this strong: a colour print faded to
                        # yellow or orange
COLOUR_UNSURE = (0.6, 1.6)  # an area this share of its limit is noted ...
COLOUR_CAST_UNSURE = 3.0    # ... and so is a cast from here to its limit;
                            # a cast past COLOUR_UNSURE[1] x the limit
                            # settles the verdict alone
RAMP_BIN = 4.0          # the grey ramp is read in bins of this many L* ...
RAMP_L_LO = 15.0        # ... between these two lightnesses ...
RAMP_L_HI = 92.0
RAMP_MIN_FRAC = 0.05    # ... and needs this share of the picture there
CAST_SIDE = 800         # the cast is read on about this many px a side ...
CAST_PCT = 30.0         # ... in its least chromatic pixels, this share

# --- front or back -------------------------------------------------------
BACK_INSET = 0.10       # the outer tenth of each side is not judged
BACK_DARK_DELTA = 80    # flat test: "dark" is below the print's own white
                        # (8-bit) minus this; less than ...
BACK_DARK_FRAC = 0.06   # ... this share dark ...
BACK_STD = 45.0         # ... and this flat (8-bit) passes
BACK_MIN_WHITE = 75.0   # a back's paper white reaches this L* ...
BACK_PAPER_DL = 10.0    # ... and this share of it lies within this many
BACK_PAPER_MIN = 0.45   # L* of that white (aged, foxed paper included)
BLANK_SIDE = 256        # bare paper is read at this long side ...
BLANK_DL = 4.0          # ... as within this much L* of the paper white ...
BLANK_STD = 1.5         # ... and this flat (5 px window) ...
BLANK_EDGE = 0.03       # ... without the outermost strip of the print
BACK_BLANK_SURE = 0.60  # an example back for the AI: paper, this much bare
                        # paper, and flat and light
BACK_K = 5              # a print is compared with this many nearest
                        # examples of each kind (fewer when there are fewer)
AI_MODEL = "facebook/dinov2-base"   # --model
AI_VIEW_SIDE = 224      # side of the square view the model gets as it is
                        # (no resize, no centre crop); a multiple of 14
AI_INSET = 0.06         # the print's border is trimmed before the model looks
AI_ROTS = (0, 1, 2, 3)  # k for np.rot90: all four quarter turns, since a
                        # back may lie any way up; must stay a full cycle
AI_BATCH = 8            # prints per forward pass (x len(AI_ROTS) views)
AI_CACHE_VERSION = 1    # bump whenever ai_views or the embedding changes

# --- a colour print --------------------------------------------------------
BLACK_PCT = 0.3         # --black: percentile set to black
WHITE_PCT = 99.7        # --white: percentile set to white
SOFT = 0.05             # --soft: share of the tone range each soft end takes
MIN_RANGE = 0.05        # a channel's white point stays this far above its
                        # black point
NEUTRAL = 0.9           # --neutral: share of the remaining cast taken away
BINS = 6                # --bins: bands of lightness for the curves
GREY_L = (3.0, 97.0)    # the bands lie between these lightnesses ...
GREY_PCT = 25.0         # ... and in each, this % least colourful pixels
                        # stand for grey
GREY_MIN_PX = 100       # a band with fewer pixels is left out; with fewer
                        # in all, only the levels are used
END_PCT = 5.0           # the levels' colour balance at each end comes from
                        # this % darkest and lightest pixels ...
END_GREY_C = 12.0       # ... when their greyest are this grey (chroma)

# --- contact sheets --------------------------------------------------------
SHEET_TILE = 240        # px per picture (before and after: two per print)
SHEET_COLS = 4          # prints per row
SHEET_MAX_H = 30000     # a sheet taller than this continues on the next

# --- corrections -----------------------------------------------------------
OVERRIDE_WORDS = {"bw": "bw", "colour": "colour", "color": "colour",
                  "back": "back", "front": "front",
                  "skip": "skip"}   # every word understood, and what it means
CLASS_LABEL = {"bw": "B&W", "colour": "COLOUR", "back": "BACK"}


# --------------------------------------------------------------------------
# files: the plan, the corrections and the cache
# --------------------------------------------------------------------------

def read_csv(path) -> list:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def write_csv(path, fields, rows) -> None:
    """A CSV written atomically: a crash never leaves it truncated."""
    tmp = str(path) + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    os.replace(tmp, path)


def fingerprint(path) -> str:
    """Size, modification time and the last 64 KiB: enough to tell whether
    a file is still the one that was analysed, without reading all of it."""
    st = os.stat(path)
    h = hashlib.sha1(f"{st.st_size}:{st.st_mtime_ns}".encode())
    with open(path, "rb") as fh:
        if st.st_size > 65536:
            fh.seek(-65536, os.SEEK_END)
        h.update(fh.read(65536))
    return h.hexdigest()[:16]


def read_overrides(project_root: str):
    """({stem in lower case: {"bw" | "colour" | "back" | "front" |
    "skip"}}, [lines that could not be read], [lines that contradict an
    earlier one]) from OVERRIDE_NAME beside the prints. A name may carry
    its extension or not; '#' starts a comment. Of two lines that set one
    print both B&W and colour, or both back and front, the later one is
    used."""
    path = os.path.join(resolve(project_root), OVERRIDE_NAME)
    out, bad, clashes = {}, [], []
    if not os.path.isfile(path):
        return out, bad, clashes
    with open(path, encoding="utf-8-sig") as fh:
        for n, line in enumerate(fh, 1):
            parts = line.split("#", 1)[0].split()
            if not parts:
                continue
            what = OVERRIDE_WORDS.get(parts[-1].lower())
            if len(parts) != 2 or what is None:
                bad.append(f"line {n}: {line.strip()}")
                continue
            stem = os.path.splitext(parts[0])[0].lower() \
                if parts[0].lower().endswith(SCAN_EXTS) else parts[0].lower()
            have = out.setdefault(stem, set())
            for pair in ({"bw", "colour"}, {"back", "front"}):
                if what in pair:
                    old = have & (pair - {what})
                    if old:
                        clashes.append(f"line {n}: {parts[0]} {what} (an "
                                       f"earlier line said {old.pop()})")
                        have -= pair
            have.add(what)
    return out, bad, clashes


def cache_file(project_root: str, key: str) -> str:
    return os.path.join(review_dir(project_root), CACHE_DIR, f"{key}.npz")


def load_cache(path: str):
    """The arrays saved for one print, or None."""
    try:
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except (OSError, ValueError, KeyError):
        return None


def save_cache(path: str, arrays: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# colour of a pixel
# --------------------------------------------------------------------------
# rgb2lab and lab2rgb (OpenCV) carry the corrections; lab_of (numpy, the
# same CIELAB) carries the colour-or-B&W tests. Both assume sRGB.

def rgb2lab(x):
    return cv2.cvtColor(np.ascontiguousarray(x, dtype=np.float32),
                        cv2.COLOR_RGB2LAB)


def lab2rgb(lab):
    return cv2.cvtColor(np.ascontiguousarray(lab, dtype=np.float32),
                        cv2.COLOR_LAB2RGB)


_M_RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                       [0.2126729, 0.7151522, 0.0721750],
                       [0.0193339, 0.1191920, 0.9503041]], dtype=np.float64)
_D65 = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)


def lab_of(rgb, side=None):
    """CIELAB (L, a, b) of RGB floats in 0..1; with `side`, of every n-th
    pixel only, n chosen so about `side` pixels remain along the long
    side."""
    if side:
        h, w = rgb.shape[:2]
        st = max(1, max(h, w) // side)
        rgb = rgb[::st, ::st]
    q = np.clip(rgb.astype(np.float64), 0.0, 1.0)
    lin = np.where(q <= 0.04045, q / 12.92, ((q + 0.055) / 1.055) ** 2.4)
    xyz = (lin @ _M_RGB2XYZ.T) / _D65
    eps, kap = 216.0 / 24389.0, 24389.0 / 27.0
    f = np.where(xyz > eps, np.cbrt(xyz), (kap * xyz + 16.0) / 116.0)
    return (116.0 * f[..., 1] - 16.0, 500.0 * (f[..., 0] - f[..., 1]),
            200.0 * (f[..., 1] - f[..., 2]))


def shrink(img, side):
    """img averaged down to `side` on its long side (as it is if smaller)."""
    h, w = img.shape[:2]
    f = side / float(max(h, w))
    if f >= 1.0:
        return img
    return cv2.resize(img, (max(1, round(w * f)), max(1, round(h * f))),
                      interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------
# colour or B&W
# --------------------------------------------------------------------------

def colour_area(rgb, valid=None):
    """(share of the picture off its grey ramp, blue-violet ink left out;
    how widely those pixels spread over hues, 0 = one hue, 1 = evenly), or
    (None, None) when the picture has too little mid-tone area to tell.

    The picture is averaged down to COLOUR_SIDE and smoothed with
    COLOUR_BLUR; the ramp is the median a*b* of every RAMP_BIN slice of
    lightness. A pixel is off the ramp when its a*b* lies more than
    COLOUR_DEV from its slice's median, and is ink when that offset points
    blue-violet (INK_HUE). Counted over the mid-tones (RAMP_L_LO..RAMP_L_HI).
    The spread is one minus the length of the mean unit vector of the
    offsets' hues. `valid` marks the pixels that hold picture (no-data
    pixels, 0 in every channel, and their blurred surroundings are left
    out)."""
    small = cv2.GaussianBlur(shrink(rgb.astype(np.float32), COLOUR_SIDE),
                             (0, 0), COLOUR_BLUR)
    L, a, b = (v.ravel() for v in lab_of(small))
    ok = np.ones(L.shape, bool)
    if valid is not None:
        ok = (cv2.GaussianBlur(shrink(valid.astype(np.float32), COLOUR_SIDE),
                               (0, 0), COLOUR_BLUR) > 0.99).ravel()
    bins = (L / RAMP_BIN).astype(np.int32)
    da, db = np.zeros_like(a), np.zeros_like(b)
    used = np.zeros(L.shape, bool)
    for k in np.unique(bins[ok]):
        m = (bins == k) & ok
        if int(m.sum()) < 20:
            continue
        da[m], db[m] = a[m] - np.median(a[m]), b[m] - np.median(b[m])
        used |= m
    used &= (L >= RAMP_L_LO) & (L <= RAMP_L_HI)
    if int(used.sum()) < max(200, RAMP_MIN_FRAC * L.size):
        return None, None
    hue = np.degrees(np.arctan2(db, da)) % 360.0
    ink = (hue > INK_HUE[0]) & (hue < INK_HUE[1])
    off = used & (np.hypot(da, db) > COLOUR_DEV) & ~ink
    spread = 0.0
    if int(off.sum()) > 30:
        ang = np.radians(hue[off])
        spread = 1.0 - float(np.hypot(np.cos(ang).mean(),
                                      np.sin(ang).mean()))
    return float(off.sum() / used.sum()), spread


def cast_of(rgb, valid=None):
    """(a*, b*) of the picture's cast: the median of its CAST_PCT % least
    chromatic pixels (toning of a B&W print, fading of a colour one), over
    the `valid` pixels only; (nan, nan) when there are none."""
    _, a, b = lab_of(rgb, CAST_SIDE)
    if valid is not None:
        h, w = rgb.shape[:2]
        st = max(1, max(h, w) // CAST_SIDE)
        keep = valid[::st, ::st]
        a, b = a[keep], b[keep]
    if a.size == 0:
        return float("nan"), float("nan")
    c = np.hypot(a, b)
    grey = c <= np.percentile(c, CAST_PCT)
    return float(np.median(a[grey])), float(np.median(b[grey]))


def ramp_verdict(area, spread, cast_a, cast_b=None):
    """"colour" or "bw" by the ramp test."""
    if cast_a is not None and abs(cast_a) >= COLOUR_CAST_MIN:
        return "colour"
    if (area is not None and area >= COLOUR_AREA_MIN
            and (spread or 0.0) >= COLOUR_SPREAD_MIN):
        return "colour"
    if (area is not None and area >= FADED_AREA_MIN
            and cast_b is not None and cast_b >= FADED_CAST_B):
        return "colour"                   # faded to yellow or orange
    return "bw"


def ramp_doubt(area, cast_a, verdict):
    """Why the ramp verdict needs a look, or None."""
    if (cast_a is not None and COLOUR_CAST_UNSURE <= abs(cast_a)
            < COLOUR_CAST_MIN and (area is None or area < COLOUR_AREA_MIN)):
        return "strong cast but little colour (a faded colour print?)"
    if area is None:
        if cast_a is not None and abs(cast_a) >= COLOUR_CAST_MIN:
            return None
        return "too little mid-tone area to test: B&W"
    if (area >= COLOUR_AREA_MIN and verdict == "bw"
            and (cast_a is None or abs(cast_a) < COLOUR_CAST_MIN)):
        return "colour off the grey ramp, but all of one hue (toning?): B&W"
    if cast_a is not None and abs(cast_a) >= COLOUR_CAST_MIN * \
            COLOUR_UNSURE[1]:
        return None                       # colour by its cast, well past it
    if COLOUR_UNSURE[0] <= area / COLOUR_AREA_MIN < COLOUR_UNSURE[1]:
        return "verdict close to its limit"
    return None


def classify(m):
    """(class, [notes]) of a measured print by the ramp test."""
    if bool(m["mono"]):
        return "bw", ["one channel: B&W"]
    num = lambda k: None if np.isnan(m[k]) else float(m[k])
    area, spread = num("area"), num("spread")
    cast_a, cast_b = num("cast_a"), num("cast_b")
    verdict = ramp_verdict(area, spread, cast_a, cast_b)
    notes = []
    if (verdict == "colour" and (cast_a is None
                                 or abs(cast_a) < COLOUR_CAST_MIN)
            and not (area >= COLOUR_AREA_MIN
                     and (spread or 0.0) >= COLOUR_SPREAD_MIN)):
        notes.append(f"colour faded to yellow (b* cast {cast_b:.1f})")
        return verdict, notes
    doubt = ramp_doubt(area, cast_a, verdict)
    if doubt:
        notes.append(f"{doubt}: check it")
    return verdict, notes


# --------------------------------------------------------------------------
# front or back
# --------------------------------------------------------------------------

def flat_and_light(img):
    """The flat test: little of the print is dark for its own white, and it
    is flat. Dark, low-contrast pictures pass it too, so it only counts
    for a print that looks like paper (paper_of)."""
    g = (img[..., 0] * 0.299 + img[..., 1] * 0.587
         + img[..., 2] * 0.114) * 255.0
    h, w = g.shape
    iy, ix = int(BACK_INSET * h), int(BACK_INSET * w)
    roi = g[iy:h - iy, ix:w - ix]
    if roi.size < 1000:
        return False
    rs = roi[::4, ::4]
    white = float(np.percentile(g, 90))
    return bool(float((rs < white - BACK_DARK_DELTA).mean()) < BACK_DARK_FRAC
                and float(rs.std()) < BACK_STD)


def paper_of(img):
    """(share of the print that is bare paper, paper white in L*, share
    within BACK_PAPER_DL of that white), or Nones when the print is too
    small. Bare paper is as light as the paper white and flat; the paper
    white is the 97th percentile of L*; the last share is read lightly
    smoothed, so the specks of aged paper count as paper."""
    L = lab_of(shrink(img, BLANK_SIDE))[0].astype(np.float32)
    h, w = L.shape
    if min(h, w) < 40:
        return None, None, None
    white = float(np.percentile(L, 97))
    mu = cv2.blur(L, (5, 5))
    sd = np.sqrt(np.maximum(cv2.blur(L * L, (5, 5)) - mu * mu, 0.0))
    blank = (np.abs(L - white) < BLANK_DL) & (sd < BLANK_STD)
    g = int(BLANK_EDGE * min(h, w))
    near = cv2.GaussianBlur(L, (0, 0), 1.5) >= white - BACK_PAPER_DL
    return (float(blank[g:h - g, g:w - g].mean()), white,
            float(near[g:h - g, g:w - g].mean()))


def paper_like(m):
    """Whether a measured print looks like paper: its paper white reaches
    BACK_MIN_WHITE L* and BACK_PAPER_MIN of it lies within BACK_PAPER_DL of
    that white."""
    white, paper = float(m["paper_white"]), float(m["paper"])
    if np.isnan(white) or np.isnan(paper):
        return False
    return white >= BACK_MIN_WHITE and paper >= BACK_PAPER_MIN


def looks_like_back(m):
    """The pixel test, used without the AI: paper, and flat and light or
    without mid-tones."""
    return paper_like(m) and (bool(m["flat"]) or bool(np.isnan(m["area"])))


def sure_back(m):
    """An example back for the AI: paper, BACK_BLANK_SURE bare paper, and
    flat and light."""
    blank = float(m["blank"])
    return (paper_like(m) and not np.isnan(blank)
            and blank >= BACK_BLANK_SURE and bool(m["flat"]))


def _number(name):
    """(prefix, number) of a print's name, or None without a number."""
    mt = re.match(r"^(.*?)(\d+)$", os.path.splitext(name)[0])
    return (mt.group(1).lower(), int(mt.group(2))) if mt else None


def ai_distances(names, meas, emb):
    """{name: (mean distance to its BACK_K nearest example backs, to its
    BACK_K nearest example fronts)} for every print that looks like paper.
    The example backs are the sure ones (sure_back), the example fronts the
    prints that do not look like paper; a print is not its own example.
    Comparing the means of equally many of each keeps the verdict from
    hanging on how many examples of each kind there are. Empty without
    embeddings, or with fewer than two examples of either kind."""
    if not emb:
        return {}
    seeds = [n for n in names if n in emb
             and (sure_back(meas[n]) or not paper_like(meas[n]))]
    test = [n for n in names if n in emb and paper_like(meas[n])]
    is_back = np.array([sure_back(meas[n]) for n in seeds], bool)
    if not test or is_back.sum() < 2 or (~is_back).sum() < 2:
        return {}
    D = ai_cross([emb[n] for n in test], [emb[n] for n in seeds])
    out = {}
    for i, n in enumerate(test):
        d = D[i].copy()
        d[[j for j, s in enumerate(seeds) if s == n]] = np.inf
        near = []
        for side in (is_back, ~is_back):
            ds = np.sort(d[side])
            ds = ds[np.isfinite(ds)][:BACK_K]
            near.append(float(ds.mean()) if ds.size else float("inf"))
        out[n] = (near[0], near[1])
    return out


def decide_backs(names, meas, ov, ai):
    """(the names that are backs, the would-be backs a stronger neighbour
    left fronts). A print is a back when the corrections say so; else, when
    the corrections leave its side open, when the AI finds it nearer the
    example backs than the example fronts (ai, from ai_distances), or,
    without the AI, when it passes the pixel test (looks_like_back). Then
    the archive's numbering: a back comes right after its front, one number
    higher, so of a run of would-be backs in a row the ones with the
    strongest case (the AI's margin, or the bare paper), no two adjacent,
    stay (a back set in the corrections always does)."""
    want = {}
    for n in names:
        o = ov.get(os.path.splitext(n)[0].lower(), set())
        m = meas.get(n)
        if "back" in o:
            want[n] = 1e6
        elif m is None or "front" in o or o & {"bw", "colour"}:
            continue
        elif ai:
            if n in ai and ai[n][0] < ai[n][1]:
                want[n] = max(ai[n][1] - ai[n][0], 1e-6)
        elif looks_like_back(m):
            want[n] = max(float(m["blank"]), 1e-6)
    backs, run = set(), []

    def close(run):
        best = []                         # best total over run[:i + 1]
        for i, n in enumerate(run):
            with_i = want[n] + (best[i - 2] if i >= 2 else 0.0)
            best.append(max(with_i, best[i - 1]) if i else want[n])
        i = len(run) - 1
        while i >= 0:
            if i and best[i] == best[i - 1]:
                i -= 1
            else:
                backs.add(run[i])
                i -= 2

    for n in names:
        if n in want and run:
            a, b = _number(run[-1]), _number(n)
            if a and b and a[0] == b[0] and b[1] == a[1] + 1:
                run.append(n)
                continue
        if run:
            close(run)
        run = [n] if n in want else []
    if run:
        close(run)
    return backs, set(want) - backs


# --------------------------------------------------------------------------
# front or back: the AI image model
# --------------------------------------------------------------------------

def ai_views(rgb8):
    """The model's views of one print (8-bit RGB): the border trimmed (it is
    common to every print), every channel stretched to its own 1..99
    percentiles (so the cast does not count), letterboxed to a square of
    AI_VIEW_SIDE, in all four quarter turns."""
    h, w = rgb8.shape[:2]
    iy, ix = int(AI_INSET * h), int(AI_INSET * w)
    if h - 2 * iy >= 32 and w - 2 * ix >= 32:
        rgb8 = rgb8[iy:h - iy, ix:w - ix]
    f = rgb8.astype(np.float32)
    lo = np.percentile(f, 1, axis=(0, 1))
    hi = np.percentile(f, 99, axis=(0, 1))
    rgb = np.clip((f - lo) / np.maximum(hi - lo, 1.0) * 255.0,
                  0, 255).astype(np.uint8)
    rgb = shrink(rgb, AI_VIEW_SIDE)
    h, w = rgb.shape[:2]
    s = max(h, w)
    edge = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    pad = np.full((s, s, 3), np.median(edge, axis=0), np.uint8)
    y0, x0 = (s - h) // 2, (s - w) // 2
    pad[y0:y0 + h, x0:x0 + w] = rgb
    if s != AI_VIEW_SIDE:
        pad = cv2.resize(pad, (AI_VIEW_SIDE, AI_VIEW_SIDE),
                         interpolation=cv2.INTER_CUBIC)
    return [np.ascontiguousarray(np.rot90(pad, k)) for k in AI_ROTS]


def ai_cross(sa, sb):
    """Rotation-robust cosine distances from every (R, D) view stack of sa
    to every one of sb: one minus the similarity at the best-matching
    relative turn r, where the similarity at r is the mean, over the R
    absolute turns s, of view s of the one print against view s + r of the
    other. Averaging over s keeps the model's liking for one orientation
    out of the comparison."""
    A = np.stack([np.asarray(s, np.float32) for s in sa])       # (n, R, D)
    B = np.stack([np.asarray(s, np.float32) for s in sb])       # (m, R, D)
    n, R, D = A.shape
    F = A.reshape(n, R * D)
    best = np.full((n, len(B)), -np.inf, np.float32)
    for r in range(R):
        G = np.roll(B, -r, axis=1).reshape(len(B), R * D)
        np.maximum(best, (F @ G.T) / R, out=best)
    return np.maximum(1.0 - best, 0.0).astype(np.float64)


def _features(out, n_reg):
    """A ViT's class token and the mean of its patch tokens (register
    tokens left out), each of unit length, side by side: the class token
    carries the print as a whole, the patch mean what is on it. A model
    without patch tokens gives its pooled output."""
    import torch
    hs = getattr(out, "last_hidden_state", None)
    if hs is None or hs.ndim != 3 or hs.shape[1] <= 1 + n_reg:
        v = getattr(out, "pooler_output", None)
        return v if v is not None else out[0]
    nf = torch.nn.functional.normalize
    return torch.cat([nf(hs[:, 0].float(), dim=-1),
                      nf(hs[:, 1 + n_reg:].mean(dim=1).float(), dim=-1)],
                     dim=-1)


def ai_embeddings(thumbs, cdir, model_name, device_pref):
    """{fingerprint: (R, D) view embeddings} of the prints in `thumbs`
    ({fingerprint: thumbnail, float RGB in 0..1}), read from the cache file
    of the model where it holds them. None without torch and transformers,
    or when the model cannot be loaded. The cache file keeps only the
    prints given."""
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
    except ImportError:
        warn("    ! torch or transformers is not installed: backs by the "
             "pixel test only (pip install torch transformers)")
        return None
    slug = re.sub(r"[^A-Za-z0-9]+", "_", model_name)
    path = os.path.join(cdir, f"ai_v{AI_CACHE_VERSION}_{slug}_s"
                              f"{AI_VIEW_SIDE}_r{len(AI_ROTS)}.npz")
    have = {}
    got = load_cache(path)
    if got is not None and got["vecs"].ndim == 3 \
            and got["vecs"].shape[1] == len(AI_ROTS):
        have = {str(k): v for k, v in zip(got["keys"], got["vecs"])}
    todo = [k for k in thumbs if k not in have]
    if len(todo) < len(thumbs):
        info(f"{count(len(thumbs) - len(todo), 'embedding')} read from the "
             f"cache")
    if todo:
        device = device_pref
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        info(f"loading {model_name} on {device} (the first run downloads "
             f"it)")
        try:
            model = AutoModel.from_pretrained(model_name).to(device).eval()
            proc = AutoImageProcessor.from_pretrained(model_name)
        except Exception as exc:
            warn(f"    ! the model {model_name} could not be loaded: "
                 f"{type(exc).__name__}: {exc}\n"
                 f"      backs by the pixel test only (the first run needs "
                 f"the internet; check the name given to --model)")
            return None
        # the views are already the model's size: no resize, no centre crop
        pkw = {k: False for k in ("do_resize", "do_center_crop")
               if hasattr(proc, k)}
        n_reg = int(getattr(model.config, "num_register_tokens", 0) or 0)
        bar = Progress(len(todo), "    embedding")
        with torch.no_grad():
            for i in range(0, len(todo), AI_BATCH):
                chunk = todo[i:i + AI_BATCH]
                views = [v for k in chunk for v in ai_views(np.round(
                    np.clip(thumbs[k], 0.0, 1.0) * 255.0).astype(np.uint8))]
                pv = proc(images=views, return_tensors="pt",
                          **pkw)["pixel_values"].to(device)
                f = _features(model(pixel_values=pv), n_reg)
                f = torch.nn.functional.normalize(f.float(), dim=-1)
                V = f.cpu().numpy().reshape(len(chunk), len(AI_ROTS), -1)
                for k, v in zip(chunk, V):
                    have[k] = v.astype(np.float32)
                for _ in chunk:
                    bar.update()
        bar.close()
    keys = sorted(k for k in have if k in thumbs)
    if todo or len(keys) < len(have):
        if keys:
            save_cache(path, {"keys": np.array(keys),
                              "vecs": np.stack([have[k] for k in keys])})
        elif os.path.exists(path):
            os.remove(path)
    return {k: have[k] for k in keys}


# --------------------------------------------------------------------------
# a colour print: levels and curves in one look-up table per channel
# --------------------------------------------------------------------------

def soft_ends(t, width):
    """Levels without a hard clip: t (0 at the black point, 1 at the white
    point) is left as it is between width and 1 - width; beyond, it bends
    smoothly towards 0 or 1 without reaching them, with the same slope
    where the bend starts. width 0 = a hard clip."""
    if width <= 0:
        return np.clip(t, 0, 1)
    lo, hi = width, 1.0 - width
    t = np.asarray(t, dtype=np.float32)
    out = t.copy()
    over = t > hi
    out[over] = hi + width * (1 - np.exp(-(t[over] - hi) / width))
    under = t < lo
    out[under] = lo * np.exp((t[under] - lo) / lo)
    return out


def monotone_curve(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    o = np.argsort(xs)
    xs, ys = xs[o], ys[o]
    keep = np.concatenate([[True], np.diff(xs) > 1e-3])
    xs, ys = xs[keep], np.maximum.accumulate(ys[keep])
    return PchipInterpolator(xs, ys)


def end_points(small, mask, opt):
    """(lo, hi): every channel's black and white point (3 each).

    The tones go where one stretch for all three channels puts them: the
    channels' mean, its darkest --black % to 0 and lightest --white % to 1.
    The colour balance at each end comes from the print's greyest surfaces
    there -- the GREY_PCT % least colourful of its END_PCT % darkest or
    lightest pixels -- which every channel maps to the same tone. So the
    lightest or darkest thing in the picture being coloured (a sunlit wall,
    a blue shadow) does not tint a white or a black that is grey. An end
    whose greyest surfaces are not grey after all (their median chroma
    above END_GREY_C, once the channels are levelled) keeps each channel's
    own darkest or lightest pixels as its point."""
    px = small[mask]
    own_lo = np.percentile(px, opt.black, axis=0)
    own_hi = np.percentile(px, opt.white, axis=0)
    own_hi = np.maximum(own_hi, own_lo + MIN_RANGE)
    mean = px.mean(1)
    y_lo, y_hi = np.percentile(mean, [opt.black, opt.white])
    y_hi = max(y_hi, y_lo + MIN_RANGE)
    L, a, b = (v[mask] for v in cv2.split(rgb2lab(small)))
    C = np.hypot(a, b)
    rough = np.clip((px - own_lo) / (own_hi - own_lo), 0, 1)
    rough_c = np.hypot(*cv2.split(rgb2lab(rough[None]))[1:]).ravel()
    anchors = []
    for end, own, t_own in ((L <= np.percentile(L, END_PCT), own_lo, 0.0),
                            (L >= np.percentile(L, 100 - END_PCT), own_hi,
                             1.0)):
        sel = np.flatnonzero(end)
        if len(sel) >= GREY_MIN_PX:
            grey = sel[C[sel] <= np.percentile(C[sel], GREY_PCT)]
            if float(np.median(rough_c[grey])) <= END_GREY_C:
                ref = px[grey].mean(0)
                anchors.append((ref, (float(ref.mean()) - y_lo)
                                / (y_hi - y_lo)))
                continue
        anchors.append((own, t_own))
    (ref_lo, t_lo), (ref_hi, t_hi) = anchors
    if t_hi - t_lo < MIN_RANGE or np.any(ref_hi - ref_lo < MIN_RANGE):
        return own_lo, own_hi
    span = (ref_hi - ref_lo) / (t_hi - t_lo)
    lo = ref_lo - t_lo * span
    return lo, lo + span


def build_luts(small, mask, opt):
    """Levels and neutralisation curves joined into one look-up table per
    channel ((LUT_SIZE, 3)), applied the same at any resolution."""
    grid = np.linspace(0, 1, LUT_SIZE, dtype=np.float32)
    luts = np.tile(grid[:, None], (1, 3))

    # ---- 1) per-channel levels, with a soft toe and shoulder
    lo, hi = end_points(small, mask, opt)
    x = np.empty_like(small)
    for c in range(3):
        x[..., c] = soft_ends((small[..., c] - lo[c]) / (hi[c] - lo[c]),
                              opt.soft)
        luts[:, c] = soft_ends((grid - lo[c]) / (hi[c] - lo[c]), opt.soft)

    # ---- 2) tone-dependent neutralisation curves
    lab = rgb2lab(x)
    L, C = lab[..., 0], np.hypot(lab[..., 1], lab[..., 2])
    m = mask & (L > GREY_L[0]) & (L < GREY_L[1])
    Ls, Cs, px = L[m], C[m], x[m]
    if Ls.size < GREY_MIN_PX:       # nothing usable left to measure the cast
        return luts
    nb = opt.bins
    edges = np.percentile(Ls, np.linspace(1, 99, nb + 1))
    pts = [([0.0], [0.0]) for _ in range(3)]
    for i in range(nb):
        sel = (Ls >= edges[i]) & (Ls < edges[i + 1])
        if sel.sum() < GREY_MIN_PX:
            continue
        g = sel & (Cs <= np.percentile(Cs[sel], GREY_PCT))
        mc = px[g].mean(0)
        t = mc.mean()
        for c in range(3):
            pts[c][0].append(mc[c])
            pts[c][1].append(t + (mc[c] - t) * (1 - opt.neutral))
    for c in range(3):
        f = monotone_curve(pts[c][0] + [1.0], pts[c][1] + [1.0])
        luts[:, c] = np.clip(f(luts[:, c]), 0, 1)
    return luts


def apply_luts(img, luts):
    n = luts.shape[0] - 1
    idx = np.clip(img * n, 0, n)
    xp = np.arange(n + 1, dtype=np.float32)
    out = np.empty_like(img)
    for c in range(3):
        out[..., c] = np.interp(idx[..., c], xp, luts[:, c])
    return out


# --------------------------------------------------------------------------
# a B&W print
# --------------------------------------------------------------------------

def bw_levels(small, mask, opt):
    """(lo, hi): the L* the B&W stretch maps to 0 and 100."""
    L = rgb2lab(small)[..., 0][mask]
    lo, hi = np.percentile(L, [opt.black, opt.white])
    return float(lo), float(hi)


def bw_correct(img, lo, hi):
    """The grey of a B&W print (one channel, float32 in 0..1): its L*
    stretched between lo and hi, the tint taken away."""
    L = rgb2lab(img)[..., 0]
    L = np.clip((L - lo) / max(hi - lo, 1) * 100, 0, 100).astype(np.float32)
    zero = np.zeros_like(L)
    return np.clip(lab2rgb(cv2.merge([L, zero, zero]))[..., 1], 0, 1)


# --------------------------------------------------------------------------
# reading a print, measuring it
# --------------------------------------------------------------------------

def as_rgb(pic):
    """The picture channels as float32 RGB in 0..1 (a grey print as three
    equal channels), and whether it was stored with one channel."""
    top = unit_range(pic.dtype)
    mono = pic.ndim == 2
    img = pic.astype(np.float32) / top
    if mono:
        img = np.repeat(img[..., None], 3, axis=2)
    return np.ascontiguousarray(img), mono


def _cache_key(fp, args):
    """The cache entry of a print: its fingerprint and every setting the
    measurement depends on."""
    txt = (f"{fp}|{CACHE_VERSION}|{ANALYSIS_SIDE}|{THUMB_SIDE}|{LUT_SIZE}|"
           f"{CARD_INNER}|{COLOUR_SIDE}|{COLOUR_BLUR}|{COLOUR_DEV}|"
           f"{INK_HUE}|{RAMP_BIN}|{RAMP_L_LO}|{RAMP_L_HI}|{RAMP_MIN_FRAC}|"
           f"{CAST_SIDE}|{CAST_PCT}|{BACK_INSET}|{BACK_DARK_DELTA}|"
           f"{BACK_DARK_FRAC}|{BACK_STD}|{BACK_PAPER_DL}|{BLANK_SIDE}|"
           f"{BLANK_DL}|{BLANK_STD}|{BLANK_EDGE}|"
           f"{MIN_RANGE}|{GREY_L}|{GREY_PCT}|{GREY_MIN_PX}|"
           f"{args.black}|{args.white}|{args.soft}|{args.neutral}|"
           f"{args.bins}|{END_PCT}|{END_GREY_C}")
    return hashlib.sha1(txt.encode()).hexdigest()[:16]


def _measure_job(task):
    """Measure one print, or read it from the cache. Top-level so it can be
    pickled. Returns (measurement dict or None, error or None)."""
    path, cpath, args = task
    try:
        if not args.fresh:
            got = load_cache(cpath)
            if got is not None:
                return got, None
        arr, _meta = read_page(path)
        img, mono = as_rgb(colour_part(arr))
        small = shrink(img, ANALYSIS_SIDE)
        mask = small.max(axis=2) > 0                   # 0 everywhere: no data
        if int(mask.sum()) < 100:
            return None, "no picture: every pixel is 0"
        ih, iw = small.shape[:2]
        iy, ix = int(CARD_INNER * ih), int(CARD_INNER * iw)
        inner = small[iy:ih - iy, ix:iw - ix]
        valid = mask[iy:ih - iy, ix:iw - ix]
        if not inner.size:
            inner, valid = small, mask
        nan = float("nan")
        area = spread = cast_a = cast_b = nan
        if not mono:
            a, s = colour_area(inner, valid)
            area, spread = (nan, nan) if a is None else (a, s)
            cast_a, cast_b = cast_of(inner, valid)
        blank, white, paper = paper_of(small)
        blank, white, paper = (nan if v is None else v
                               for v in (blank, white, paper))
        lo, hi = bw_levels(small, mask, args)
        got = {"lut": build_luts(small, mask, args).astype(np.float32),
               "bw_lo": np.float64(lo), "bw_hi": np.float64(hi),
               "area": np.float64(area), "spread": np.float64(spread),
               "cast_a": np.float64(cast_a), "cast_b": np.float64(cast_b),
               "mono": np.bool_(mono), "flat": np.bool_(flat_and_light(small)),
               "blank": np.float64(blank), "paper_white": np.float64(white),
               "paper": np.float64(paper),
               "small_shape": np.array(small.shape[:2], np.int64),
               "thumb": shrink(img, THUMB_SIDE).astype(np.float16)}
        save_cache(cpath, got)
        return got, None
    except Exception as exc:              # a bad file must not stop the run
        return None, f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------
# the correction (the same on the sheet and in --apply)
# --------------------------------------------------------------------------

def correct(img, m, row):
    """The corrected picture of img by the plan row and the measurement m:
    float32 in 0..1, RGB for a colour print, one grey channel for a B&W
    one."""
    if row["class"] == "back":
        return img
    if row["class"] == "colour":
        return apply_luts(img, m["lut"])
    return bw_correct(img, float(m["bw_lo"]), float(m["bw_hi"]))


def _apply_job(task):
    """Write the corrected file of one print. Top-level so it can be
    pickled. Returns (error or None, fingerprint of the written file)."""
    src, dst, row, compress = task
    tmp = dst + ".part"
    try:
        if row["class"] == "back":        # a back: the file as it is
            shutil.copyfile(src, tmp)
            os.replace(tmp, dst)
            return None, fingerprint(dst)
        root = os.path.dirname(src)
        m = load_cache(cache_file(root, row["key"]))
        if m is None:
            return ("its measurement is gone from the cache: run the "
                    "analysis again"), None
        arr, meta = read_page(src)
        pic = colour_part(arr)
        img, _mono = as_rgb(pic)
        x = correct(img, m, row)
        top = unit_range(pic.dtype)
        out = np.clip(x, 0.0, 1.0)
        if np.issubdtype(pic.dtype, np.integer):
            out = np.round(out * top)
        out = out.astype(pic.dtype)
        # the picture channels only: an alpha channel is not carried; a B&W
        # print is one grey channel, and a colour profile does not fit it
        meta = dict(meta, extrasamples=None)
        if out.ndim == 2:
            meta.update(photometric="minisblack", iccprofile=None,
                        planarconfig=None, colormap=None)
            if str(meta.get("description") or "").startswith('{"shape"'):
                meta["description"] = None    # tifffile's own shape note,
                                              # the colour shape: not true
                                              # of the grey file
        # written under a temporary name first, so a crash never leaves a
        # half-written print that a later --apply would find in its way
        write_tiff(tmp, out, meta, compress=compress)
        os.replace(tmp, dst)
        return None, fingerprint(dst)
    except Exception as exc:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        return f"{type(exc).__name__}: {exc}", None


# --------------------------------------------------------------------------
# contact sheets
# --------------------------------------------------------------------------

def to_bgr8(img: np.ndarray) -> np.ndarray:
    """A float picture in 0..1 (grey or RGB) as 8-bit BGR for a sheet."""
    a = np.round(np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    if a.ndim == 2:
        return cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(a[..., ::-1])


def _fit(img, side):
    h, w = img.shape[:2]
    f = side / float(max(h, w))
    return cv2.resize(img, (max(1, round(w * f)), max(1, round(h * f))),
                      interpolation=cv2.INTER_AREA if f < 1
                      else cv2.INTER_LINEAR)


def imwrite(path, img, quality=90) -> bool:
    """Write through numpy, so non-ASCII paths work on Windows."""
    ok, buf = cv2.imencode(os.path.splitext(str(path))[1] or ".jpg", img,
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return False
    try:
        buf.tofile(str(path))
    except OSError:
        return False
    return True


def write_sheets(rdir: str, prefix: str, pairs: list) -> int:
    """Contact sheets <prefix>_sheet_NN.jpg: every print as its picture
    before and after, side by side, SHEET_COLS pairs a row, the name under
    them -- in red when the plan notes something to check. `pairs` holds
    (label, before, after, flagged), pictures as float in 0..1. Old sheets
    of this prefix are removed first. Returns the number of sheets."""
    for f in os.listdir(rdir):
        if f.startswith(f"{prefix}_sheet_") and f.endswith(".jpg"):
            os.remove(os.path.join(rdir, f))
    tile, gap, text_h = SHEET_TILE, 12, 22
    cell_w, cell_h = 2 * tile + gap, tile + text_h
    rows_per_sheet = max(1, SHEET_MAX_H // (cell_h + gap))
    per_sheet = rows_per_sheet * SHEET_COLS
    n = 0
    for start in range(0, len(pairs), per_sheet):
        chunk = pairs[start:start + per_sheet]
        rows = (len(chunk) + SHEET_COLS - 1) // SHEET_COLS
        sheet = np.full((rows * (cell_h + gap) + gap,
                         SHEET_COLS * (cell_w + 2 * gap) + gap, 3), 235,
                        np.uint8)
        for i, (label, before, after, flagged) in enumerate(chunk):
            r, c = divmod(i, SHEET_COLS)
            x0 = gap + c * (cell_w + 2 * gap)
            y0 = gap + r * (cell_h + gap)
            for k, img in enumerate((before, after)):
                t = _fit(to_bgr8(img), tile)
                x = x0 + k * (tile + gap) + (tile - t.shape[1]) // 2
                y = y0 + tile - t.shape[0]     # just above the name
                sheet[y:y + t.shape[0], x:x + t.shape[1]] = t
            cv2.putText(sheet, label, (x0, y0 + tile + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 0, 200) if flagged else (0, 0, 0), 1,
                        cv2.LINE_AA)
        n += 1
        imwrite(os.path.join(rdir, f"{prefix}_sheet_{n:02d}.jpg"), sheet, 85)
    return n


# --------------------------------------------------------------------------
# the analysis
# --------------------------------------------------------------------------

def analyze(root, args) -> int:
    t0 = time.time()
    rdir = review_dir(root)
    os.makedirs(rdir, exist_ok=True)

    phase(1, 4, "the prints")
    names = list_prints(root)
    info(f"{count(len(names), 'print')} in {root}")
    ov, bad, clashes = read_overrides(root)
    if bad:
        warn_list(f"    ! {OVERRIDE_NAME}: {count(len(bad), 'line')} not "
                  f"understood, ignored:", bad)
    if clashes:
        warn_list(f"    ! {OVERRIDE_NAME}: lines that contradict an "
                  f"earlier line; the later one is used:", clashes)
    stems = {os.path.splitext(n)[0].lower() for n in names}
    unknown = sorted(s for s in ov if s not in stems)
    if unknown:
        warn_list(f"    ! {OVERRIDE_NAME} names prints that are not in the "
                  f"folder:", unknown)

    phase(2, 4, "measuring")
    tasks, fps = [], []
    for n in names:
        path = os.path.join(root, n)
        fp = fingerprint(path)
        fps.append(fp)
        tasks.append((path, cache_file(root, _cache_key(fp, args)), args))
    n_workers = 1
    if parallel is not None:
        biggest = max(tasks, key=lambda t: os.path.getsize(t[0]))[0]
        per = parallel.estimate_bytes_per_worker(biggest,
                                                 WORKER_MEM_MULTIPLE)
        n_workers = parallel.choose_workers(len(tasks), per, args.workers)
    info(count(n_workers, "worker"))
    bar = Progress(len(tasks), "    measuring")
    if parallel is not None:
        res = parallel.run(_measure_job, tasks, n_workers,
                           on_result=bar.update,
                           on_error=lambda t, e: (
                               None, parallel.describe_error(e)))
    else:
        res = []
        for t in tasks:
            res.append(_measure_job(t))
            bar.update()
    bar.close()

    phase(3, 4, "fronts and backs")
    print(paint(f"    {args.model}: paper-like prints compared with the sure "
                f"backs and fronts", "90"))
    meas = {n: m for n, (m, _e) in zip(names, res) if m is not None}
    fp_of = dict(zip(names, fps))
    cdir = os.path.join(rdir, CACHE_DIR)
    if args.fresh and os.path.isdir(cdir):
        for f in os.listdir(cdir):
            if f.startswith("ai_v") and f.endswith(".npz"):
                os.remove(os.path.join(cdir, f))
    emb = ai_embeddings({fp_of[n]: m["thumb"].astype(np.float32)
                         for n, m in meas.items()},
                        cdir, args.model, args.device) if meas else None
    ai = ai_distances(list(meas), meas, {n: emb[fp_of[n]] for n in meas
                                         if emb and fp_of[n] in emb})
    n_paper = sum(1 for m in meas.values() if paper_like(m))
    if emb and n_paper and not ai:
        warn("    ! too few sure backs or fronts to compare with: backs by "
             "the pixel test only")
    backs, dropped = decide_backs(names, meas, ov, ai)
    info(f"{count(n_paper, 'print')} look like paper; "
         f"{count(len(backs), 'back')} "
         + ("by the AI" if ai else "by the pixel test"))

    phase(4, 4, "the plan")
    plan, pairs, failed, flagged = [], [], [], []
    classes = {"bw": 0, "colour": 0, "back": 0}
    bar = Progress(len(names), "    planning")
    for i, (name, fp, task, (m, err)) in enumerate(zip(names, fps, tasks,
                                                       res)):
        if i:
            bar.update()                  # the print before this one
        base = {"name": name, "fingerprint": fp,
                "key": os.path.splitext(os.path.basename(task[1]))[0]}
        if m is None:
            failed.append(f"{name}: {err}")
            plan.append(dict(base, action="unreadable", notes=err))
            continue
        o = ov.get(os.path.splitext(name)[0].lower(), set())
        hand = o & {"bw", "colour"}
        d = ai.get(name)
        if name in backs:
            cls, by = "back", ("hand" if "back" in o else "tests")
            notes = ["back in the corrections" if "back" in o
                     else "back: copied as it is" + (
                         f" (AI distance to backs {d[0]:.3f}, to fronts "
                         f"{d[1]:.3f})" if d else "")]
        elif hand:
            cls, by = hand.pop(), "hand"
            notes = [f"{CLASS_LABEL[cls]} in the corrections"]
        else:
            cls, notes = classify(m)
            by = "tests"
            if name in dropped:
                notes.append("looks like a back, but a print next to it "
                             "is the stronger back: check it")
        if cls == "colour" and bool(m["mono"]):
            cls, by = "bw", "tests"
            notes.append("one channel: cannot be a colour print, B&W used")
        action = "skip" if "skip" in o else "write"
        if action == "skip":
            notes.insert(0, "skipped in the corrections")
        num = lambda k, f: "" if np.isnan(m[k]) else f"{float(m[k]):{f}}"
        row = dict(base, action=action, **{
            "class": cls, "by": by,
            "colour_area": num("area", ".4f"), "hue_spread": num("spread",
                                                                 ".3f"),
            "cast_a": num("cast_a", ".2f"), "cast_b": num("cast_b", ".2f"),
            "bare_paper": num("blank", ".3f"),
            "paper_white": num("paper_white", ".1f"),
            "ai_to_backs": f"{d[0]:.3f}" if d else "",
            "ai_to_fronts": f"{d[1]:.3f}" if d else "",
            "notes": "; ".join(notes)})
        plan.append(row)
        classes[cls] += 1
        check = "check it" in row["notes"]
        if check:
            flagged.append(name)
        thumb = m["thumb"].astype(np.float32)
        after = correct(thumb, m, row)
        if action == "skip":
            after = np.full(thumb.shape[:2], 0.5, np.float32)
        pairs.append((f"{os.path.splitext(name)[0]}  {CLASS_LABEL[cls]}",
                      thumb, after, check))
    bar.update()
    bar.close()

    if len(failed) == len(names):
        error_list("no print could be measured; no plan written", failed)
        return 1

    # the cache keeps only what this analysis used: entries of files that
    # have gone or changed, or of other settings, are dropped
    used = {os.path.basename(t[1]) for t in tasks}
    if os.path.isdir(cdir):                # the embeddings stay too
        used |= {f for f in os.listdir(cdir)
                 if f.startswith(f"ai_v{AI_CACHE_VERSION}_")}
    for f in os.listdir(cdir) if os.path.isdir(cdir) else []:
        if f.endswith(".npz") and f not in used:
            try:
                os.remove(os.path.join(cdir, f))
            except OSError:
                pass

    fields = ["name", "action", "class", "by", "colour_area", "hue_spread",
              "cast_a", "cast_b", "bare_paper", "paper_white", "ai_to_backs",
              "ai_to_fronts", "notes", "key", "fingerprint"]
    plan_path = os.path.join(rdir, f"{PREFIX}_plan.csv")
    write_csv(plan_path, fields, plan)
    n_sheets = write_sheets(rdir, PREFIX, pairs)
    info(f"{count(classes['colour'], 'colour print')}, "
         f"{count(classes['bw'], 'B&W print')}, "
         f"{count(classes['back'], 'back')} (copied as they are); "
         f"{sum(1 for p in plan if p['action'] == 'skip')} skipped in the "
         f"corrections")
    info(f"plan:   {plan_path}")
    info(f"sheets: {count(n_sheets, 'sheet')} "
         f"({os.path.join(rdir, PREFIX + '_sheet_NN.jpg')})")
    if flagged:
        warn_list(f"    ! {count(len(flagged), 'print')} noted "
                  f"\"check it\":", flagged)
    if failed:
        error_list(f"    {count(len(failed), 'print')} could not be "
                   f"measured; left out of the plan:", failed)
    print(paint(f"    done in {elapsed(t0)}", "32"))
    print(paint(f"\nnext:  python {SCRIPT}{root_arg(root)} --apply", "1"))
    return 2 if failed else 0


# --------------------------------------------------------------------------
# --apply and --undo
# --------------------------------------------------------------------------

def _stored_dir(out, root):
    """The output folder as the log stores it: relative to the prints
    folder when inside it, so the archive can be moved or copied whole;
    absolute otherwise."""
    try:
        rel = os.path.relpath(os.path.abspath(out), os.path.abspath(root))
    except ValueError:                  # another drive on Windows
        return os.path.abspath(out)
    if rel == os.pardir or rel.startswith(os.pardir + os.sep):
        return os.path.abspath(out)
    return rel


def _logged_dir(stored, root):
    """The output folder a log row names, as a path."""
    return stored if os.path.isabs(stored) else os.path.join(root, stored)


def apply_plan(root, out_arg, plan_path, log_path, workers_arg,
               compress) -> int:
    """Write every print of the plan into the output folder, through
    _apply_job. Only the rows the plan marks "write" are written, into
    the output folder. Every written file is logged at once, so an
    interrupted run can still be undone. Returns 0, 2 when some prints
    failed and the others were written, 1 when nothing was written."""
    root = resolve(root)
    if not os.path.isfile(plan_path):
        raise SystemExit(f"no plan: {plan_path}\n"
                         f"run the analysis first:\n"
                         f"    python {SCRIPT}{root_arg(root)}")
    if os.path.isfile(log_path):
        raise SystemExit(f"the plan was applied already ({log_path})\n"
                         f"remove those files first:\n"
                         f"    python {SCRIPT}{root_arg(root)} --undo")
    out = resolve(out_arg) if out_arg else restored_dir(root)
    if same_dir(out, root):
        raise SystemExit(f"the output folder cannot be the prints folder "
                         f"itself: {out}")
    phase(1, 2, "checking the plan")
    rows = [r for r in read_csv(plan_path) if r.get("action") == "write"]
    info(f"{count(len(rows), 'print')} to write in "
         f"{os.path.basename(plan_path)}")
    problems = []
    for r in rows:
        src = os.path.join(root, r["name"])
        if not os.path.isfile(src):
            problems.append(f"missing: {r['name']}")
        elif fingerprint(src) != r["fingerprint"]:
            problems.append(f"not the file that was analysed: {r['name']}")
        if os.path.exists(os.path.join(out, r["name"])):
            problems.append(f"already in {out}: {r['name']}")
    if problems:
        error("nothing written: the folder no longer matches the plan")
        warn_list(f"    ! {count(len(problems), 'problem')}:", problems)
        print(paint(f"\nrun the analysis again:\n    python {SCRIPT}"
                    f"{root_arg(root)}", "1"))
        return 1
    if not rows:
        warn("    ! the plan writes no print")
        return 0
    info("every print is still the file that was analysed")

    phase(2, 2, "writing the corrected prints")
    os.makedirs(out, exist_ok=True)
    tasks = [(os.path.join(root, r["name"]), os.path.join(out, r["name"]),
              r, compress) for r in rows]
    n_workers = 1
    if parallel is not None:
        biggest = max(tasks, key=lambda t: os.path.getsize(t[0]))[0]
        per = parallel.estimate_bytes_per_worker(biggest,
                                                 WORKER_MEM_MULTIPLE)
        n_workers = parallel.choose_workers(len(tasks), per, workers_arg)
    info(f"output: {out}")
    info(count(n_workers, "worker"))
    stored = _stored_dir(out, root)
    bar = Progress(len(tasks), "    writing")
    failed = []
    with open(log_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["out_dir", "name", "fingerprint"])

        def done(i, res):
            err, fp = res
            if err:
                failed.append(f"{rows[i]['name']}: {err}")
            else:
                w.writerow([stored, rows[i]["name"], fp])
                fh.flush()
            bar.update()

        def crashed(task, exc):
            desc = (parallel.describe_error(exc) if parallel is not None
                    else f"{type(exc).__name__}: {exc}")
            return (desc, None)

        if parallel is not None:
            parallel.run(_apply_job, tasks, n_workers, on_result=done,
                         on_error=crashed)
        else:
            for i, t in enumerate(tasks):
                try:
                    res = _apply_job(t)
                except Exception as exc:
                    res = crashed(t, exc)
                done(i, res)
    bar.close()
    written = len(tasks) - len(failed)
    if not written:
        os.remove(log_path)               # nothing to undo
        try:
            os.rmdir(out)                 # only when it is empty
        except OSError:
            pass
    print(paint(f"\nwrote {count(written, 'print')} to {out}", "32"))
    if failed:
        error_list(f"    {count(len(failed), 'print')} failed:", failed)
    if written:
        print(paint(f"\nundo:  python {SCRIPT}{root_arg(root)} --undo",
                    "1"))
        print(paint(f"\nnext, Stage 5:  python colorize_model.py "
                    f"\"{out}\"", "1"))
    if failed:
        return 2 if written else 1      # 2: the others were written
    return 0


def undo_last(root, log_path) -> int:
    """Remove the files the last --apply wrote, each only while it is still
    the file that was written."""
    if not os.path.isfile(log_path):
        raise SystemExit(f"nothing to undo: no {os.path.basename(log_path)}"
                         f" in {os.path.dirname(log_path)}")
    root = resolve(root)
    phase(1, 1, "removing the files")
    rows = read_csv(log_path)
    removed, kept = 0, []
    for r in rows:
        dst = os.path.join(_logged_dir(r["out_dir"], root), r["name"])
        if not os.path.exists(dst):
            continue
        if fingerprint(dst) != r["fingerprint"]:
            kept.append(dst)
            continue
        os.remove(dst)
        removed += 1
    for d in {r["out_dir"] for r in rows}:
        try:
            os.rmdir(_logged_dir(d, root))   # only when it is empty now
        except OSError:
            pass
    stem, ext = os.path.splitext(log_path)
    os.replace(log_path, stem + ".undone" + ext)
    print(paint(f"\nremoved {count(removed, 'file')}", "32"))
    if kept:
        warn_list(f"    ! {count(len(kept), 'file')} changed since --apply "
                  f"and left in place:", kept)
    return 2 if kept else 0


# --------------------------------------------------------------------------
# the step
# --------------------------------------------------------------------------

def _share(text):
    try:
        v = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError("a number") from None
    if not 0 <= v <= 1:
        raise argparse.ArgumentTypeError("from 0 to 1")
    return v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Stage 4 step 2 of 2 — colour and B&W correction")
    ap.add_argument("root", nargs="?", default=".",
                    help="project root: the folder of the prints (default: "
                         "this folder)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true",
                      help="write the corrected prints the plan describes")
    mode.add_argument("--undo", action="store_true",
                      help="remove the files the last --apply wrote")
    ap.add_argument("--out", default=None, metavar="DIR",
                    help="with --apply: the folder for the corrected prints "
                         "(default: ROOT/restored)")
    ap.add_argument("--compress", default="keep",
                    choices=["keep", "none", "lzw", "deflate"],
                    help="with --apply: the TIFF codec (default: keep the "
                         "source's own)")
    ap.add_argument("--black", type=float, default=BLACK_PCT,
                    help=f"percentile set to black (default "
                         f"{BLACK_PCT:g})")
    ap.add_argument("--white", type=float, default=WHITE_PCT,
                    help=f"percentile set to white (default {WHITE_PCT:g})")
    ap.add_argument("--soft", type=float, default=SOFT,
                    help=f"share of the tone range each soft end takes, "
                         f"0 to 0.5 (default {SOFT:g}; 0 = a hard clip)")
    ap.add_argument("--neutral", type=_share, default=NEUTRAL,
                    help=f"share of the remaining cast the curves take away "
                         f"(default {NEUTRAL:g})")
    ap.add_argument("--bins", type=int, default=BINS,
                    help=f"bands of lightness for the curves (default "
                         f"{BINS})")
    ap.add_argument("--model", default=AI_MODEL,
                    help=f"the Hugging Face image model that tells backs "
                         f"(default {AI_MODEL})")
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cpu", "cuda"],
                    help="where the model runs (default: auto)")
    ap.add_argument("--fresh", action="store_true",
                    help="measure every print again instead of using "
                         "_cache/ (the AI embeddings too)")
    if parallel is not None:
        parallel.add_workers_arg(ap)
    else:
        ap.add_argument("--workers", type=int, default=1,
                        help="ignored: parallel.py not found")
    args = ap.parse_args(argv)
    console_setup()
    if (args.out or args.compress != "keep") and not args.apply:
        raise SystemExit("--out and --compress go with --apply")
    if not 0 <= args.black < args.white <= 100:
        raise SystemExit("--black and --white: 0 <= black < white <= 100")
    if args.bins < 1:
        raise SystemExit("--bins must be at least 1")
    if not 0 <= args.soft <= 0.5:
        raise SystemExit("--soft must be from 0 to 0.5: each soft end takes "
                         "this share of the tone range")

    root = pipeline_root(args.root)
    check_scans_present(root)
    rdir = review_dir(root)
    log = os.path.join(rdir, f"{PREFIX}_log.csv")
    if args.apply:
        heading("STAGE 4 step 2  -  write the corrected prints")
        return apply_plan(root, args.out,
                          os.path.join(rdir, f"{PREFIX}_plan.csv"), log,
                          args.workers, args.compress)
    if args.undo:
        heading("STAGE 4 step 2  -  undo the last --apply")
        return undo_last(root, log)
    heading("STAGE 4 step 2  -  colour and B&W correction")
    return analyze(root, args)


if __name__ == "__main__":
    raise SystemExit(main())
