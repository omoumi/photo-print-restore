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
# renumber_by_similarity.py  --  STAGE 3, step 1 of 1:
#     put related prints side by side
#
# STAGE 3 puts the prints that belong together side by side, from the
# straightened, cropped files Stage 2 writes:
#
#     renumber_by_similarity.py  step 1 of 1   put related prints side by side
#     common.py                  shared code, not a step
#     parallel.py                shared worker-count policy, not a step
#
# Requirements:
#     pip install opencv-python numpy scipy tifffile torch transformers
#                 [imagecodecs] [psutil]
#     common.py beside this file; parallel.py too, for --workers
#
# Gives the prints of one folder new consecutive numbers, so that prints
# which belong together sit side by side. The crops keep their own names:
# --apply links (or, where a link is impossible, copies) each one into the
# output folder under its new name, and --undo removes what --apply made.
#
# Every print is first judged front or back. A back looks like paper:
# its lightest tone is light, and most of it lies close to that tone, as
# on aged, foxed paper and under writing, stamps and stains, but never on
# a photo, however dark or flat. Whether such a print is a back, the AI
# decides, from the sure cases most like it (much bare paper and flat:
# back; not paper-like: front). Then the archive's numbering: a back is
# scanned right after its front, one number higher with the same print
# index, so no two prints in a row are both backs; in every run of
# would-be backs the set with the strongest case, no two adjacent, stays.
# A back is filed right after its front; one without a front before it
# goes to the end. Every decision the flat-print pixel test alone would
# have made differently, and every back that is not a sure case, is noted
# in the plan as "back (...)" or "front, though flat (...)".
#
# The fronts are then split into colour and B&W, which never share a
# group. A B&W print, toned or not, keeps one hue per lightness (its grey
# ramp); a colour print is one with enough of its picture off that ramp in
# more than one hue, blue-violet ink (stamps, pen) left out, or with a
# strong cast on the red-green axis (a*): a colour print faded to one hue,
# most often red, magenta or orange, though a strong green cast counts too;
# or with some of its picture off the ramp and a strong yellow cast (b*):
# a colour print faded to yellow or orange; or with a little less of it
# off the ramp, but in many hues: a pale colour print.
# The test cannot be right on every print: a hand-coloured B&W print, a
# B&W print under a red stamp or a colour print of a grey scene can look
# like the other class. Every print whose verdict is inconclusive or close
# to a limit is listed by name and noted in the plan. A print stored with
# one channel only is B&W without a test.
#
# Inside each class, the prints are clustered on one distance: the
# picture content, told by AI image embeddings (by default DINOv2 with
# registers, large, on 448 px views), plus three capped penalties for
# what the print is made of:
#
#     print size       long x short side, in mm from the DPI (in pixels
#                      when the prints carry none); free up to --size-free
#     colour cast      a*b* of the least chromatic pixels (toning, fading)
#     corners          rounded against sharp: a rounded corner leaves the
#                      scanner lid's white in the corner of the crop; no
#                      penalty when either print's corners cannot be told
#
# A penalty only makes two prints less alike; none can keep apart two
# prints whose pictures the model finds close, so a scene printed at two
# sizes, faded apart or cut differently stays together.
#
# The AI compares every print in all four quarter turns: two prints meet
# at their best-matching relative turn, the similarity averaged over the
# four absolute turns. Those distances are then re-ranked (k-reciprocal
# re-ranking): two prints count as alike when their nearest neighbours
# overlap too, which keeps a single chance likeness from joining two
# events. In each class the merge tree of the combined distances is cut
# where the silhouette score is best (--ai-thr auto), or at the distance
# --ai-thr gives. The merge tree never undoes a merge, so every print is
# then checked against the groups that formed: one nearer on average to
# another group moves to it, with a note in the plan, until nothing moves.
# A print the cut leaves alone stays single: the group it is nearest to on
# average is not evidence that it belongs there. One farther than
# --pair-max times the split from every group pairs with its nearest
# single print, when they are as near as that and the other print is
# nearer to it than to a group (two prints of one place and time, say).
# Groups and single prints, and the prints inside every group, are put in
# the order that makes neighbours alike (optimal leaf ordering): a single
# print sits between the groups, beside the one most like it.
#
# The new numbers run: every B&W print, then every colour print, each class
# as its groups and single prints, then its prints without an AI
# embedding; after both classes the unreadable prints and the backs
# without a front. A back always takes the number right after its front.
#
# NAMES. A crop is named as Stage 2 wrote it: <scan>.tif, or <scan>_NN.tif
# when one scan held several prints. Its new name is the scan's prefix and
# its new number: raw0137_02.tif may become raw0005.tif.
#
# Outputs, derived from the project root -- the folder holding Stage 2's
# crops:
#
#     organize_review/organize_plan.csv     the plan; --apply executes
#                                           exactly this
#     organize_review/organize_summary.txt  the groups as a tree, with notes
#     organize_review/sheet_NN.jpg          contact sheets of the groups
#                                           and single prints
#     organize_review/organize_log.csv      written by --apply, read by
#                                           --undo
#     organize_review/_cache/               measurements, thumbnails and
#                                           embeddings (a rerun with other
#                                           thresholds skips the slow part)
#     organized/<new name>                  written by --apply, unless --out
#
# Corrections go in ROOT\organize_overrides.txt, one per line, by each
# crop's own name (the grey one under its thumbnail on the sheets, not its
# new number, which changes from one analysis to the next); then run the
# analysis again:
#
#     raw0012.tif back
#     raw0013_02.tif front
#     raw0020.tif bw
#     raw0021.tif colour
#
# Usage:
#     python renumber_by_similarity.py [ROOT] [options]
#                                       analyse, write the plan
#     python renumber_by_similarity.py [ROOT] --apply
#                                       link the prints as planned
#     python renumber_by_similarity.py [ROOT] --undo
#                                       remove what --apply made
#
#     ROOT              project root: the folder holding Stage 2's crops
#                       (default: this folder)
#     --out DIR         with --apply: the folder for the prints under their
#                       new names (default: ROOT\organized); never ROOT
#     --colour-thr F    colour when this share of the picture is off its
#                       grey ramp, ink left out (default 0.07)
#     --size-free F     sides this many mm apart cost nothing (default 3)
#     --size-w F        largest size penalty (default 0.15; 0 = ignore)
#     --cast-w F        largest colour-cast penalty (default 0.05)
#     --corner-w F      rounded against sharp corners (default 0.15)
#     --ai-thr F        split on the combined distance, larger = fewer
#                       groups, or auto (default)
#     --pair-max F      two single prints pair only within F x the split
#                       (default 1.4)
#     --no-grey-assist  colour embedding only
#     --model NAME      another Hugging Face image model, e.g.
#                       facebook/dinov2-base (README.md lists more)
#     --device D        auto | cpu | cuda
#     --fresh           measure every print again, ignoring _cache/
#     --workers N       parallel workers (default: auto; 1 disables the pool)
#
#     python renumber_by_similarity.py D:\archive\scans\deskewed-cropped
#     python renumber_by_similarity.py --ai-thr 0.45
#     python renumber_by_similarity.py --apply
# ==========================================================================
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import textwrap
import time
from pathlib import Path

try:
    import cv2
    import numpy as np
except ModuleNotFoundError as exc:
    sys.exit(f"missing package: {exc.name} (see Requirements above)")

try:
    from common import (REVIEW_DIR, SCAN_EXTS, Progress, check_scans_present,
                        console_setup, count, elapsed, error, heading, info,
                        organized_dir, paint, phase, pipeline_root,
                        read_page, resolve, review_dir, root_arg, same_dir,
                        warn, warn_list)
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
SCRIPT = "renumber_by_similarity.py"
WORKER_MEM_MULTIPLE = 4.0   # peak RAM of one worker per byte of the print
                            # file: the decoded print and its working copy

# --- files -----------------------------------------------------------------
NAME_RE = re.compile(r"^(.*?)(\d+)(?:_(\d{2}))?$")   # prefix | scan number
                                                     # | Stage 2's _NN
CACHE_DIR = "_cache"
PLAN_NAME = "organize_plan.csv"
SUMMARY_NAME = "organize_summary.txt"
LOG_NAME = "organize_log.csv"
OVERRIDE_NAME = "organize_overrides.txt"
CACHE_VERSION = 19      # bump whenever measure_print changes

# --- working resolution ----------------------------------------------------
WORK_SIDE = 1600        # long side a print is shrunk to before it is measured
THUMB_SIDE = 640        # long side of the thumbnail (AI, contact sheets)
STRETCH_BELOW = 49152   # a 16-bit print whose brightest sample is below this
                        # is stretched to full range first
MIN_DPI = 50            # a resolution tag below this is not trusted

# --- backs of prints -------------------------------------------------------
BASE = 800.0            # reference resolution for all px @ BASE values
BACK_INSET = 0.10       # the outer tenth of each side is not judged
BACK_DARK_DELTA = 80    # pixel test: "dark" is below the print's own
BACK_DARK_FRAC = 0.06   # white - this; less than this share dark ...
BACK_STD = 45.0         # ... and this flat passes
BACK_PAIR_TOL = 0.15    # two opposite inner margins this close (share of
                        # their mean) make it a pale mounted print instead
INSET_MAX_FRAC = 0.13   # inner-margin search depth (share of the long side)
EDGE_SKIP_FRAC = 0.08   # skip this share of each side's ends (corners)
SAMPLES_PER_SIDE = 40   # sample lines per side
HALO_GUARD = 5          # px @ BASE between the print edge and a candidate
BAND_EDGE_MIN = 50      # gradient an inner-margin edge must reach
POL_MARGIN = 10         # brightness step across that edge
MARGIN_BAND_STD = 15    # the margin must be this flat
BACK_MIN_WHITE = 75.0   # a back's paper white reaches this L* ...
BACK_PAPER_DL = 10.0    # ... and this share of it lies within this many
BACK_PAPER_MIN = 0.45   # L* of that white (aged, foxed paper included)
BACK_BLANK_SURE = 0.60  # an example back for the AI: this much bare
                        # paper and the pixel test passed
BACK_K = 5              # the AI compares a print with this many nearest
                        # examples of each kind (fewer when there are fewer)
BLANK_SIDE = 256        # bare-paper test: print shrunk to this long side
BLANK_DL = 4.0          # bare paper = within this much L* of the paper white
BLANK_STD = 1.5         # ... and this flat (5 px window)
BLANK_EDGE = 0.03       # the outermost strip of the print is ignored

# --- print size ------------------------------------------------------------
SIZE_TOL_MM = 6.0       # a back more than twice this from its front's
SIZE_TOL_REL = 0.04     # size is noted; without usable DPI, this share of
                        # the median long side instead
MAX_PRINT_MM = 600.0    # a larger "print" means the DPI tags are wrong

# --- colour or B&W ---------------------------------------------------------
CARD_INNER = 0.10       # picture area = print minus this share per side
COLOUR_SIDE = 410       # the picture is averaged down to this long side ...
COLOUR_BLUR = 2.0       # ... and smoothed (sigma, px): grain and specks go
COLOUR_DEV = 4.0        # a pixel is off the grey ramp this far in a*b*
INK_HUE = (220.0, 320.0)    # blue-violet off the ramp is ink (a stamp), not
                            # picture colour: hue of the offset, in degrees
COLOUR_AREA_MIN = 0.07  # --colour-thr: colour with this share of the picture
                        # off the ramp, ink left out, ...
COLOUR_SPREAD_MIN = 0.05    # ... spread over hues at least this much (all
                            # in one hue is uneven toning of a B&W print) ...
COLOUR_CAST_MIN = 4.5   # ... or a cast (|a_cast|, red or green) this
                        # strong: a colour print faded to one hue ...
FADED_AREA_MIN = 0.05   # ... or this share off the ramp with ...
FADED_CAST_B = 12.0     # ... a yellow cast (b_cast) this strong: a colour
                        # print faded to yellow or orange ...
MULTI_AREA_MIN = 0.04   # ... or this share off the ramp, spread over
MULTI_SPREAD_MIN = 0.3  # hues at least this much: a pale colour print
COLOUR_UNSURE = (0.6, 1.6)  # an area this share of the limit is reported ...
COLOUR_CAST_UNSURE = 3.0    # ... and so is a cast from here to the limit;
                            # a cast past COLOUR_UNSURE[1] x the limit
                            # settles the verdict alone
RAMP_BIN = 4.0          # the grey ramp is read in bins of this many L* ...
RAMP_L_LO = 15.0        # ... between these two lightnesses ...
RAMP_L_HI = 92.0
RAMP_MIN_FRAC = 0.05    # ... and needs this share of the picture there
CLASS_ORDER = ("bw", "colour")   # the order of the new numbers: every
                                 # B&W print before the first colour one

# --- penalties on the AI distance: size, cast and corners ------------------
NEUTRAL_PCT = 30.0      # the least chromatic pixels carry the colour cast
SIZE_FREE_MM = 3.0      # --size-free: sides this close cost nothing ...
SIZE_SPAN_MM = 20.0     # ... beyond it the penalty grows over this many mm
SIZE_W = 0.15           # --size-w: ... up to this
SIZE_REF_MM = 100.0     # without DPI, the median long side counts as this
CAST_SPAN = 6.0         # the cast penalty grows over this many a*b* units
CAST_W = 0.05           # --cast-w: ... up to this
CORNER_W = 0.15         # --corner-w: rounded against sharp corners
CORNER_SIDE = 512       # corners are read on the print at this long side
CORNER_PATCH = 0.12     # ... in a square this share of the short side
CORNER_LID_L = 80.0     # scanner-lid white: at least this L* ...
CORNER_LID_C = 12.0     # ... and at most this chroma
CORNER_TRIM = 0.02      # a lid-white line along an edge, at most this share
CORNER_TRIM_SHARE = 0.6  # of the short side thick, (a row this share lid
                         # white) is trimmed first

# --- picture content (AI) and the split ------------------------------------
AI_MODEL = "facebook/dinov2-with-registers-large"   # --model
AI_THR = "auto"         # --ai-thr: "auto" or a split on the combined
                        # distance (AI distance + penalties)
AI_BATCH = 8            # prints per forward pass at 224 px, fewer at a
                        # larger view (x len(AI_ROTS) views each)
GRAY_ASSIST = True      # add a greyscale embedding to the colour one
AI_CACHE_VERSION = 5    # bump whenever _ai_views or the embedding changes
AI_INSET = 0.06         # trim the print's border before the model looks
AI_ROTS = (0, 1, 2, 3)  # k for np.rot90; must stay a full cycle
AI_VIEW_SIDE = 224      # side of the square view handed to the model, which
                        # gets it as it is (no resize, no centre crop); a
                        # multiple of 14 and 32 suits DINOv2 and CLIP alike
AI_VIEW_SIDE_DINO = 448  # ... for DINOv2, which takes any multiple of 14:
                         # faces and small things come out clearer
AUTO_CANDIDATES = 24    # cuts the silhouette score is computed for, at most
PAIR_MAX = 1.4          # --pair-max: a single print farther than this
                        # many times the split from every group pairs with
                        # another single print within it
REFINE_MARGIN = 0.01    # a print moves to another group when its mean
                        # distance to it is this much smaller than to its
                        # own ...
REFINE_ROUNDS = 20      # ... checked again until nothing moves, at most
                        # this many times
RERANK_K1 = 10          # k-reciprocal re-ranking of the AI distance: the
RERANK_K2 = 4           # neighbourhood sizes, and the share of the plain
RERANK_LAMBDA = 0.5     # distance kept
OLO_MAX = 600           # above this many prints a group is ordered nearest
                        # neighbour first instead of by optimal leaf order

# --- contact sheets --------------------------------------------------------
SHEET_TILE = 200        # px per thumbnail
SHEET_COLS = 8          # thumbnails per row
SHEET_MAX_H = 30000     # a sheet taller than this continues on the next

# --- output order ----------------------------------------------------------
CLASS_LABEL = {"bw": "B&W", "colour": "COLOUR", "unsorted": "UNREADABLE",
               "orphan": "BACKS WITHOUT FRONT"}


# --------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------

def _imwrite(path, img, quality=90):
    """Write through numpy, so non-ASCII paths work on Windows."""
    ok, buf = cv2.imencode(Path(path).suffix or ".jpg", img,
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return False
    try:
        buf.tofile(str(path))
    except OSError:
        return False
    return True


def _imread(path):
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None


def _shrink(img, side):
    h, w = img.shape[:2]
    f = side / float(max(h, w))
    if f >= 1.0:
        return img
    return cv2.resize(img, (max(1, round(w * f)), max(1, round(h * f))),
                      interpolation=cv2.INTER_AREA)


def fingerprint(path):
    """Size, modification time and the last 64 KiB: enough to tell whether
    a file is still the one that was analysed, without reading all of it."""
    st = Path(path).stat()
    h = hashlib.sha1(f"{st.st_size}:{st.st_mtime_ns}".encode())
    with open(path, "rb") as fh:
        if st.st_size > 65536:
            fh.seek(-65536, os.SEEK_END)
        h.update(fh.read(65536))
    return h.hexdigest()[:16]


def scale_of(gray):
    return max(1.0, max(gray.shape) / BASE)


def compute_grad(gray):
    """|Sobel| across x and across y, scaled by the resolution factor so
    gradient thresholds stay resolution-free."""
    s = scale_of(gray)
    g = cv2.GaussianBlur(gray, (0, 0), 0.8 * s)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return np.abs(gx) * s, np.abs(gy) * s


def _local_peak(prof, i, w):
    """True when prof[i] is the maximum within +-w samples."""
    lo, hi = max(0, i - w), min(prof.size, i + w + 1)
    return prof[i] >= prof[lo:hi].max() - 1e-6


# --------------------------------------------------------------------------
# reading a print
# --------------------------------------------------------------------------

def read_print(path):
    """The print as the measurements see it, shrunk to WORK_SIDE with a
    dark 16-bit print stretched to full range. Returns (image, rgb, mono,
    dpi, (height, width)): an 8-bit BGR copy; the same pixels as RGB floats
    in 0..1, taken before the 8-bit rounding so shadows keep their colour
    exactly; whether the file holds one channel only; the resolution as
    (x, y) dots per inch or None; and the full size in pixels."""
    a, meta = read_page(str(path))
    a = np.asarray(a)
    if a.ndim == 2:
        a = a[:, :, None]
    if a.shape[2] > 3:
        a = a[..., :3]
    elif a.shape[2] == 2:
        a = a[..., :1]
    if a.dtype not in (np.uint8, np.uint16, np.float32):
        a = a.astype(np.float32)
    h0, w0 = a.shape[:2]
    top = int(a.max()) if a.dtype == np.uint16 else None
    f = max(h0, w0) / float(WORK_SIDE)
    if f > 1.0:
        w1, h1 = max(1, int(round(w0 / f))), max(1, int(round(h0 / f)))
        a = cv2.resize(np.ascontiguousarray(a), (w1, h1),
                       interpolation=cv2.INTER_AREA)
        if a.ndim == 2:
            a = a[:, :, None]
    mono = a.shape[2] == 1
    if a.dtype == np.uint16:
        if top is not None and 0 < top < STRETCH_BELOW:
            a = np.round(a.astype(np.float64) * (65535.0 / top)
                         ).astype(np.uint16)
        rgb = a.astype(np.float32) / 65535.0
        a = ((a.astype(np.uint32) + 128) // 257).clip(0, 255).astype(np.uint8)
    elif a.dtype == np.uint8:
        rgb = a.astype(np.float32) / 255.0
    else:
        lo, hi = float(a.min()), float(a.max())
        rgb = ((a - lo) / (hi - lo) if hi > lo else a * 0).astype(np.float32)
        a = (rgb * 255).astype(np.uint8)
    if mono:
        a = np.repeat(a, 3, axis=2)
        rgb = np.repeat(rgb, 3, axis=2)
    return (np.ascontiguousarray(a[..., ::-1]), np.ascontiguousarray(rgb),
            mono, _dpi_of(meta), (h0, w0))


def _dpi_of(meta):
    """(x, y) dots per inch from the resolution tags, or None."""
    res = meta.get("resolution")
    if not res or not res[0] or not res[1]:
        return None
    unit = meta.get("resolutionunit")
    try:
        unit = 2 if unit is None else int(unit)
    except (TypeError, ValueError):
        return None
    x, y = float(res[0]), float(res[1])
    if unit == 3:                         # per centimetre
        x, y = x * 2.54, y * 2.54
    elif unit != 2:
        return None
    return (x, y) if x >= MIN_DPI and y >= MIN_DPI else None


# --------------------------------------------------------------------------
# front or back
# --------------------------------------------------------------------------

def looks_like_back(gray):
    """The pixel test: a flat print, light for its own white. Dark,
    low-contrast pictures pass it too, so it only feeds decide_backs()."""
    h, w = gray.shape[:2]
    iy, ix = int(BACK_INSET * h), int(BACK_INSET * w)
    roi = gray[iy:h - iy, ix:w - ix]
    if roi.size < 1000:
        return False
    rs = roi[::4, ::4]
    white = float(np.percentile(gray, 90))
    return (float((rs < white - BACK_DARK_DELTA).mean()) < BACK_DARK_FRAC
            and float(rs.std()) < BACK_STD)


def blank_fraction(region):
    """(share of the print that is bare paper, paper white in L*, share
    within BACK_PAPER_DL of that white). Bare paper is as light as the
    paper white and flat; the paper white is the 97th percentile of L*; the
    last share is read lightly smoothed, so the specks of aged paper count
    as paper. (None, None, None) when the print is too small."""
    small = _shrink(region, BLANK_SIDE)
    L = _lab_of(small, 1 << 30)[0].astype(np.float32)
    h, w = L.shape
    if min(h, w) < 40:
        return None, None, None
    white = float(np.percentile(L, 97))
    mu = cv2.blur(L, (5, 5))
    sd = np.sqrt(np.maximum(cv2.blur(L * L, (5, 5)) - mu * mu, 0.0))
    blank = (np.abs(L - white) < BLANK_DL) & (sd < BLANK_STD)
    g = int(BLANK_EDGE * min(h, w))
    near = cv2.GaussianBlur(L, (0, 0), 1.5) >= white - BACK_PAPER_DL
    return (round(float(blank[g:h - g, g:w - g].mean()), 4),
            round(white, 2), round(float(near[g:h - g, g:w - g].mean()), 4))


def margin_pair(gray, gx, gy):
    """Do two opposite sides show an inner margin of the same depth? Then
    the light, flat print is a pale print on a mount, not a back."""
    s = scale_of(gray)
    mi = max(1, int(round(s)))
    Hg, Wg = gray.shape
    edge = {"top": 0, "bot": Hg - 1, "lef": 0, "rig": Wg - 1}
    x0, x1, y0, y1 = 0.0, Wg - 1.0, 0.0, Hg - 1.0
    dmax = int(max(12 * s, INSET_MAX_FRAC * max(x1 - x0, y1 - y0)))
    g0 = HALO_GUARD * mi
    peak_w = max(1, int(round(1.5 * mi)))
    depth = {}
    for k, off in edge.items():
        horiz = k in ("top", "bot")
        lo, hi = (x0, x1) if horiz else (y0, y1)
        skip = EDGE_SKIP_FRAC * (hi - lo)
        ts = np.linspace(lo + skip, hi - skip, SAMPLES_PER_SIDE)
        g = gy if horiz else gx
        lim_t, lim_o = (Wg, Hg) if horiz else (Hg, Wg)
        found = []
        for t in ts:
            c = int(round(t))
            if not 0 <= c < lim_t:
                continue
            a0, a1 = ((off, off + dmax) if k in ("top", "lef")
                      else (off - dmax + 1, off + 1))
            a0, a1 = max(0, a0), min(lim_o, a1)
            if a1 - a0 < g0 + 12 * mi:
                continue
            c0, c1 = max(0, c - 3 * mi), min(lim_t, c + 3 * mi + 1)
            if horiz:
                lum = gray[a0:a1, c0:c1].mean(axis=1)
                prof = g[a0:a1, c]
            else:
                lum = gray[c0:c1, a0:a1].mean(axis=0)
                prof = g[c, a0:a1]
            if k in ("bot", "rig"):
                lum, prof = lum[::-1], prof[::-1]
            for j in range(g0 + 4 * mi, prof.size - 6 * mi):
                outer = lum[g0:j - 2 * mi]
                if outer.size >= 2 * mi and \
                        outer.std() > 2 * MARGIN_BAND_STD:
                    break
                if prof[j] < BAND_EDGE_MIN or \
                        not _local_peak(prof, j, peak_w):
                    continue
                if lum[j + 2 * mi:j + 6 * mi].mean() <= \
                        outer.mean() - POL_MARGIN:
                    found.append(j)
                    break
        if len(found) >= 0.5 * len(ts):
            d = np.asarray(found, float)
            med = float(np.median(d))
            if float(np.median(np.abs(d - med))) <= max(2 * mi, 0.1 * med):
                depth[k] = med
    for a, b in (("top", "bot"), ("lef", "rig")):
        va, vb = depth.get(a), depth.get(b)
        if va is not None and vb is not None and \
                abs(va - vb) <= BACK_PAIR_TOL * 0.5 * (va + vb):
            return True
    return False


# --------------------------------------------------------------------------
# colour or B&W
# --------------------------------------------------------------------------
# A colour print varies in hue at one and the same lightness; a B&W print,
# toned or not, has one hue per lightness at most (sepia, selenium): its
# grey ramp. So the test measures how much of the picture strays off that
# ramp. Blue-violet strays are left out: they are ink (stamps, pen), which
# B&W identity photos often carry. Strays all of one hue do not count
# either: that is toning that varies across a B&W print. A colour print
# faded to one hue strays little but keeps a strong cast on the red-green
# axis -- red, magenta or orange as a rule, green now and then -- which B&W
# toning does not give. One faded to yellow or orange keeps a strong yellow
# cast with some of its picture off the ramp; a yellowed B&W print has the
# cast too, but on its ramp, not off it. A pale colour print strays a
# little less, but in many hues. All in CIELAB on the picture without its
# border, assuming sRGB.

_M_RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                       [0.2126729, 0.7151522, 0.0721750],
                       [0.0193339, 0.1191920, 0.9503041]], dtype=np.float64)
_D65 = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)


def _lab_of(rgb, side=None):
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
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return L, a, b


def colour_area(rgb):
    """(share of the picture off its grey ramp, blue-violet ink left out;
    how widely those pixels spread over hues, 0 = one hue, 1 = evenly).

    The picture is averaged down to COLOUR_SIDE and smoothed with
    COLOUR_BLUR; the ramp is the median a*b* of every RAMP_BIN slice of
    lightness. A pixel is off the ramp when its a*b* lies more than
    COLOUR_DEV from its slice's median, and is ink when that offset points
    blue-violet (INK_HUE). Counted over the mid-tones (RAMP_L_LO..RAMP_L_HI);
    (None, None) when the picture has too little of them to tell. The
    spread is one minus the length of the mean unit vector of the offsets'
    hues."""
    if rgb is None or rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.size == 0:
        return None, None
    small = cv2.GaussianBlur(_shrink(rgb.astype(np.float32), COLOUR_SIDE),
                             (0, 0), COLOUR_BLUR)
    L, a, b = (v.ravel() for v in _lab_of(small, 1 << 30))
    bins = (L / RAMP_BIN).astype(np.int32)
    da, db = np.zeros_like(a), np.zeros_like(b)
    used = np.zeros(L.shape, bool)
    for k in np.unique(bins):
        m = bins == k
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
        spread = 1.0 - float(np.hypot(np.cos(ang).mean(), np.sin(ang).mean()))
    return float(off.sum() / used.sum()), spread


def _cast_a(rec):
    return (rec.get("cast") or {}).get("a_cast")


def _cast_b(rec):
    return (rec.get("cast") or {}).get("b_cast")


def colour_verdict(rec, area_min=COLOUR_AREA_MIN):
    """Colour when enough of the picture is off its grey ramp in more than
    one hue, or the cast is strong on the red-green axis (a*): red,
    magenta or orange as a rule, green too; or with FADED_AREA_MIN of it
    off the ramp and a strong yellow cast (b*): faded to yellow or
    orange."""
    area, ca = rec.get("colour_area"), _cast_a(rec)
    if ca is not None and abs(ca) >= COLOUR_CAST_MIN:
        return True
    cb = _cast_b(rec)
    if (area is not None and area >= FADED_AREA_MIN
            and cb is not None and cb >= FADED_CAST_B):
        return True
    if (area is not None and area >= MULTI_AREA_MIN
            and (rec.get("hue_spread") or 0.0) >= MULTI_SPREAD_MIN):
        return True                      # a pale colour print
    return (area is not None and area >= area_min
            and (rec.get("hue_spread") or 0.0) >= COLOUR_SPREAD_MIN)


def colour_doubt(rec, area_min=COLOUR_AREA_MIN):
    """Why the colour verdict of this print needs a look, or None."""
    if rec.get("mono"):
        return None                      # one channel: B&W for certain
    area, ca = rec.get("colour_area"), _cast_a(rec)
    if (ca is not None and COLOUR_CAST_UNSURE <= abs(ca) < COLOUR_CAST_MIN
            and (area is None or area < area_min)):
        return "strong cast but little colour (a faded colour print?)"
    if area is None:
        if ca is not None and abs(ca) >= COLOUR_CAST_MIN:
            return None
        return "colour test inconclusive (too little mid-tone area): B&W"
    cb = _cast_b(rec)
    if area >= FADED_AREA_MIN and cb is not None and cb >= FADED_CAST_B:
        return None                      # colour, faded to yellow
    if (area >= area_min and not colour_verdict(rec, area_min)
            and (ca is None or abs(ca) < COLOUR_CAST_MIN)):
        return "colour off the grey ramp, but all of one hue (toning?): B&W"
    if ca is not None and abs(ca) >= COLOUR_CAST_MIN * COLOUR_UNSURE[1]:
        return None                      # colour by its cast, well past it
    if COLOUR_UNSURE[0] <= area / area_min < COLOUR_UNSURE[1]:
        return "colour verdict close to its limit"
    return None


# --------------------------------------------------------------------------
# penalties: print size, cast and corners
# --------------------------------------------------------------------------
# Inside one colour class the picture content decides the groups; what the
# print is made of only makes two prints a little less alike. Each penalty
# is capped, so two prints of one scene still land together when they were
# printed at another size, faded apart, or were cut differently.

def cast_features(img):
    """Colour cast of the picture area: the median a* and b* of its least
    chromatic pixels (toning of a B&W print, fading of a colour one)."""
    L, a, b = _lab_of(img, 800)
    C = np.hypot(a, b)
    neutral = C <= np.percentile(C, NEUTRAL_PCT)
    return {"a_cast": round(float(np.median(a[neutral])), 3),
            "b_cast": round(float(np.median(b[neutral])), 3)}


def corner_shape(rgb):
    """"rounded", "sharp" or None.

    A rounded corner leaves scanner-lid white in the corner of the crop: a
    patch of it joined to the corner, shaped like the outside of a quarter
    circle, that does not run along the edge. A lid-white line along an
    edge is trimmed first. A corner without lid white is sharp; one whose
    white runs on (a white border, a bright sky) cannot be told. Rounded
    when two corners or more are, sharp when three or more are, else
    None."""
    small = _shrink(rgb, CORNER_SIDE)
    L, a, b = _lab_of(small, 1 << 30)
    lid = (L >= CORNER_LID_L) & (np.hypot(a, b) <= CORNER_LID_C)
    h, w = lid.shape
    t, lim = [0, 0, 0, 0], int(CORNER_TRIM * min(h, w))
    lines = (lambda k: lid[k, :], lambda k: lid[h - 1 - k, :],
             lambda k: lid[:, k], lambda k: lid[:, w - 1 - k])
    for i, line in enumerate(lines):
        while t[i] < lim and line(t[i]).mean() >= CORNER_TRIM_SHARE:
            t[i] += 1
    lid = lid[t[0]:h - t[1], t[2]:w - t[3]]
    h, w = lid.shape
    q = int(CORNER_PATCH * min(h, w))
    if q < 8:
        return None
    radii = []
    for m in (lid[:q, :q], lid[:q, w - q:][:, ::-1],
              lid[h - q:, :q][::-1], lid[h - q:, w - q:][::-1, ::-1]):
        m = np.ascontiguousarray(m.astype(np.uint8))
        if not m[:2, :2].any():
            radii.append(0.0)
            continue
        _, lab = cv2.connectedComponents(m, connectivity=4)
        comp = lab == lab[:2, :2][m[:2, :2] > 0][0]
        rt = int(np.argmin(np.r_[comp[0], False]))     # run along the edge
        rl = int(np.argmin(np.r_[comp[:, 0], False]))  # run down the side
        if rt >= q - 1 or rl >= q - 1 or comp[q // 2:, q // 2:].any():
            radii.append(None)                         # white runs on
            continue
        r = 0.5 * (rt + rl)
        quarter = (1 - np.pi / 4) * r * r              # outside of the arc
        if r < 2:
            radii.append(0.0)
        elif (0.4 * quarter <= comp.sum() <= 2.5 * quarter
              and 0.5 <= (rt + 1) / (rl + 1) <= 2):
            radii.append(r)
        else:
            radii.append(None)
    if sum(1 for r in radii if r) >= 2:
        return "rounded"
    if sum(1 for r in radii if r == 0.0) >= 3:
        return "sharp"
    return None


def size_label(g, unit):
    lo = float(np.median([u[0]["size"][1] for u in g]))
    hi = float(np.median([u[0]["size"][0] for u in g]))
    if unit == "mm":
        return f"{lo / 10:.1f} x {hi / 10:.1f} cm"
    return f"{lo:.0f} x {hi:.0f} px"


def penalty_matrix(units, unit, size_free=SIZE_FREE_MM, size_w=SIZE_W,
                   cast_w=CAST_W, corner_w=CORNER_W):
    """What is added to the AI distance of every pair of units:

        size    the larger of the two side differences, less size_free,
                over SIZE_SPAN_MM, times size_w (at most size_w)
        corners corner_w when one has rounded corners and the other sharp
        cast    the a*b* distance of the casts over CAST_SPAN, times
                cast_w (at most cast_w)

    Without DPI the sizes are in pixels and are scaled so that the median
    long side counts as SIZE_REF_MM. What a print lacks costs nothing."""
    n = len(units)
    sz = np.array([u[0]["size"] if u[0].get("size") else (np.nan, np.nan)
                   for u in units], np.float64)
    scale = 1.0
    if unit != "mm" and np.isfinite(sz[:, 0]).any():
        med = float(np.nanmedian(sz[:, 0]))
        scale = SIZE_REF_MM / med if med > 0 else 1.0
    dl = np.abs(sz[:, None, 0] - sz[None, :, 0]) * scale
    ds = np.abs(sz[:, None, 1] - sz[None, :, 1]) * scale
    over = np.fmax(dl, ds) - size_free
    P = np.nan_to_num(size_w * np.clip(over / SIZE_SPAN_MM, 0.0, 1.0))

    cast = np.array([[(u[0]["rec"].get("cast") or {}).get(k, np.nan)
                      for k in ("a_cast", "b_cast")] for u in units],
                    np.float64)
    dc = np.hypot(cast[:, None, 0] - cast[None, :, 0],
                  cast[:, None, 1] - cast[None, :, 1])
    P += np.nan_to_num(cast_w * np.clip(dc / CAST_SPAN, 0.0, 1.0))

    cor = np.array([{"rounded": 1, "sharp": -1}.get(
        u[0]["rec"].get("corners"), 0) for u in units])
    P += np.where(cor[:, None] * cor[None, :] < 0, corner_w, 0.0)
    np.fill_diagonal(P, 0.0)
    return P.reshape(n, n)


# --------------------------------------------------------------------------
# measuring one print
# --------------------------------------------------------------------------

def measure_print(path, cache_dir, key):
    """The record the grouping works from, plus the thumbnail in the
    cache."""
    img, rgb, mono, dpi, (h, w) = read_print(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    back_px, mounted = looks_like_back(gray), False
    if back_px:
        gx, gy = compute_grad(gray)
        mounted = margin_pair(gray, gx, gy)
        back_px = not mounted
    blank, white, paper = blank_fraction(rgb)
    corners = corner_shape(rgb)
    rec = {"ok": True, "dpi": dpi, "size_px": [w, h], "back_px": back_px,
           "mounted": mounted, "blank": blank, "white": white,
           "paper": paper, "corners": corners, "mono": mono}

    # colour or B&W: the picture without its border
    ih, iw = rgb.shape[:2]
    iy, ix = int(CARD_INNER * ih), int(CARD_INNER * iw)
    inner = rgb[iy:ih - iy, ix:iw - ix]
    body = inner if inner.size else rgb
    area, spread = (None, None) if mono else colour_area(body)
    rec["colour_area"] = None if area is None else round(area, 4)
    rec["hue_spread"] = None if spread is None else round(spread, 3)

    # the colour cast, one of the penalties
    rec["cast"] = cast_features(body)

    _imwrite(Path(cache_dir) / f"{key}_thumb.jpg", _shrink(img, THUMB_SIDE),
             92)
    return rec


def _measure_worker(task):
    """Top-level (picklable) worker: one print, never raises."""
    path_s, cache_s, key, pooled = task
    if pooled:
        cv2.setNumThreads(1)
    try:
        rec = measure_print(Path(path_s), Path(cache_s), key)
    except (Exception, SystemExit) as exc:
        rec = {"ok": False, "err": f"{type(exc).__name__}: {exc}"}
    rec["v"] = CACHE_VERSION
    return rec


# --------------------------------------------------------------------------
# prints, overrides, front/back units
# --------------------------------------------------------------------------

def find_prints(root):
    """The crops in the root, in scan order, and the TIFFs whose name holds
    no number (left alone)."""
    prints, skipped = [], []
    for p in sorted(Path(root).iterdir()):
        if not p.is_file() or not p.name.lower().endswith(SCAN_EXTS):
            continue
        m = NAME_RE.match(p.stem)
        if not m:
            skipped.append(p.name)
            continue
        prints.append({"name": p.name, "path": p, "pre": m.group(1),
                       "digits": m.group(2), "ext": p.suffix,
                       "num": int(m.group(2)), "sub": int(m.group(3) or 0)})
    prints.sort(key=lambda f: (f["num"], f["sub"], f["name"]))
    return prints, skipped


def read_overrides(root):
    """{stem: {"front" | "back" | "bw" | "colour"}} from
    ROOT/organize_overrides.txt."""
    path = Path(root) / OVERRIDE_NAME
    out = {}
    if not path.exists():
        return out
    alias = {"color": "colour", "b&w": "bw", "mono": "bw", "gray": "bw",
             "grey": "bw"}
    for n, line in enumerate(path.read_text(encoding="utf-8-sig")
                             .splitlines(), 1):
        s = line.split("#", 1)[0].strip()
        if not s:
            continue
        parts = s.split()
        what = alias.get(parts[-1].lower(), parts[-1].lower())
        if len(parts) != 2 or what not in ("front", "back", "bw", "colour"):
            warn(f"    ! {OVERRIDE_NAME} line {n} ignored: {line.strip()}")
            continue
        key = Path(parts[0]).stem.lower()
        have = out.setdefault(key, set())
        for pair in ({"front", "back"}, {"bw", "colour"}):
            clash = (have & pair) - {what}
            if what in pair and clash:
                warn(f"    ! {OVERRIDE_NAME} line {n} contradicts an "
                     f"earlier line for {parts[0]} ({clash.pop()}); the "
                     f"later one, {what}, is used")
                have -= pair
        have.add(what)
    return out


def classify(prints, recs, overrides, area_min=COLOUR_AREA_MIN):
    """Attach each print's record, colour class, colour doubt (f["doubt"])
    and size, and a front / back override as f["back_ov"]; decide_backs()
    makes the front / back decision. Returns the unit of the sizes: "mm"
    when most prints carry a usable DPI, else "px"."""
    ok = [f for f in prints if recs[f["key"]].get("ok")]
    dpis = [tuple(recs[f["key"]]["dpi"]) for f in ok
            if recs[f["key"]].get("dpi")]
    use_mm = bool(ok) and 2 * len(dpis) >= len(ok)
    dpi_fill = tuple(np.median(np.array(dpis), axis=0)) if dpis else None
    for f in prints:
        rec = recs[f["key"]]
        f["rec"], f["ok"], f["notes"] = rec, bool(rec.get("ok")), []
        ov = overrides.get(Path(f["name"]).stem.lower(), set())
        if not f["ok"]:
            f.update(back=False, cls="unsorted", size=None, wh=None)
            f["notes"].append(f"unreadable: {rec.get('err', '?')}")
            if ov:
                f["notes"].append("override(s) ignored: " +
                                  ", ".join(sorted(ov)))
            continue
        f["back"] = bool(rec.get("back_px"))
        f["back_ov"] = None
        colour = colour_verdict(rec, area_min)
        f["cls"] = "colour" if colour else "bw"
        f["doubt"] = None
        for what in sorted(ov):
            if what in ("front", "back"):
                f["back_ov"] = f["back"] = what == "back"
            else:
                f["cls"] = what
            f["notes"].append(f"override: {what}")
        if not (ov & {"bw", "colour"}):
            f["doubt"] = colour_doubt(rec, area_min)
        w, h = rec["size_px"]
        if use_mm:
            dpi = rec.get("dpi")
            if not dpi:
                dpi = dpi_fill
                f["notes"].append("no DPI tag: the median DPI of the "
                                  "other prints was assumed")
            w, h = w / dpi[0] * 25.4, h / dpi[1] * 25.4
        f["wh"] = (w, h)
    if use_mm and any(f["wh"] and max(f["wh"]) > MAX_PRINT_MM for f in ok):
        use_mm = False
        for f in ok:
            f["wh"] = tuple(f["rec"]["size_px"])
    for f in prints:
        wh = f.get("wh")
        f["size"] = (max(wh), min(wh)) if wh else None
    return "mm" if use_mm else "px"


def decide_backs(prints, emb=None):
    """Which prints are backs. A candidate looks like paper: its paper white
    reaches BACK_MIN_WHITE L* and at least BACK_PAPER_MIN of it lies within
    BACK_PAPER_DL of that white, which holds for aged, foxed paper and for
    writing and stamps, but not for a photo, however dark or flat. Among
    the candidates the AI decides: a back when the mean distance to its
    BACK_K nearest example backs is below that to its BACK_K nearest
    example fronts, so the verdict does not hang on how many of each there
    are. The examples are the sure cases: candidates with BACK_BLANK_SURE
    bare paper that pass the pixel test are backs, prints that are no
    candidate are fronts. Without embeddings a back needs the example-back
    test itself. A print whose opposite margins agree (a pale
    mounted print) and an override stay as they are.

    Then the archive's numbering: a back is scanned right after its front,
    one number higher with the same print index, so no two prints in a row
    are both backs. In every run of consecutive backs the set of backs, no
    two adjacent, with the largest total AI margin stays; an override back
    always stays. Every decision that differs from the pixel test is noted,
    and so is every back that is not a sure one, to be checked on the
    sheets. Returns (fronts made backs, backs made fronts)."""
    ok = [f for f in prints if f["ok"]]
    blank = lambda f: f["rec"].get("blank") or 0.0
    cand = lambda f: ((f["rec"].get("white") or 0.0) >= BACK_MIN_WHITE
                      and (f["rec"].get("paper") or 0.0) >= BACK_PAPER_MIN)
    sure_back = lambda f: (cand(f) and blank(f) >= BACK_BLANK_SURE
                           and f["rec"].get("back_px"))
    sure_front = lambda f: not cand(f)

    ai = {}                        # id -> (to backs, to fronts)
    if emb:
        seeds = [f for f in ok if f["key"] in emb
                 and (sure_back(f) or sure_front(f))]
        test = [f for f in ok if f["key"] in emb and cand(f)]
        is_back = np.array([bool(sure_back(f)) for f in seeds])
        if test and is_back.sum() >= 2 and (~is_back).sum() >= 2:
            D = _ai_cross([emb[f["key"]] for f in test],
                          [emb[f["key"]] for f in seeds])
            skeys = [f["key"] for f in seeds]
            for i, f in enumerate(test):
                d = D[i].copy()
                d[[j for j, k in enumerate(skeys) if k == f["key"]]] = np.inf
                near = []
                for side in (is_back, ~is_back):
                    ds = np.sort(d[side])
                    ds = ds[np.isfinite(ds)][:BACK_K]
                    near.append(float(ds.mean()) if ds.size else np.inf)
                ai[id(f)] = tuple(near)

    for f in ok:
        db, df = ai.get(id(f), (None, None))
        f["back_score"] = (df - db if db is not None
                           else 1.0 if sure_back(f) else -1.0)
        if f["back_ov"] is not None:
            f["back"] = f["back_ov"]
            continue
        if f["rec"].get("mounted") or not cand(f):
            new = False
        elif db is not None:
            new = db < df
        else:
            new = bool(sure_back(f))
        f["back"] = new

    runs, run = [], []
    for f in sorted(ok, key=lambda f: (f["sub"], f["num"])):
        if f["back"] and run and (run[-1]["sub"], run[-1]["num"] + 1) == \
                (f["sub"], f["num"]):
            run.append(f)
            continue
        if len(run) > 1:
            runs.append(run)
        run = [f] if f["back"] else []
    if len(run) > 1:
        runs.append(run)
    for run in runs:
        keep = {id(g) for g in _best_backs(run)}
        for f in run:
            if id(f) not in keep:
                f["back"] = False
                f["notes"].append("front: a print next to it is the "
                                  "stronger back")

    made_back = made_front = 0
    for f in ok:
        if f["back_ov"] is not None:
            continue
        agrees = f["back"] == bool(f["rec"].get("back_px"))
        if agrees and not (f["back"] and not sure_back(f)):
            continue                   # the pixel test alone says the same
        db, df = ai.get(id(f), (None, None))
        why = (f"bare paper {blank(f):.2f}" +
               (f"; AI distance to backs {db:.3f}, to fronts {df:.3f}"
                if db is not None else ""))
        if f["back"]:
            made_back += not agrees
            f["notes"].append(f"back ({why})")
        else:
            made_front += 1
            if not any(n.startswith("front:") for n in f["notes"]):
                f["notes"].append(f"front, though flat ({why})")
    return made_back, made_front


def _best_backs(run):
    """The prints of a run of consecutive would-be backs that stay backs:
    no two adjacent, the largest total AI margin (an override back counts
    as certain)."""
    w = [1e6 if f["back_ov"] else max(f["back_score"], 1e-6) for f in run]
    best = []                           # best total over run[:i + 1]
    for i, wi in enumerate(w):
        with_i = wi + (best[i - 2] if i >= 2 else 0.0)
        best.append(max(with_i, best[i - 1]) if i else wi)
    keep, i = [], len(run) - 1
    while i >= 0:
        if i and best[i] == best[i - 1]:
            i -= 1                      # run[i] is left out
        else:
            keep.append(run[i])
            i -= 2
    return keep


def make_units(prints):
    """Fronts with the back filed after them, and backs without a front.

    A back joins the front with the scan number one lower and the same
    print index, when that front has no back yet."""
    units, front_at = [], {}
    for f in prints:
        front = front_at.get((f["num"] - 1, f["sub"]))
        if f["ok"] and f["back"] and front is not None and len(front) == 1:
            front.append(f)
            f["cls"] = front[0]["cls"]
            continue
        units.append([f])
        if f["ok"] and not f["back"]:
            front_at[(f["num"], f["sub"])] = units[-1]
    fronts = [u for u in units if not (u[0]["ok"] and u[0]["back"])]
    orphans = [u for u in units if u[0]["ok"] and u[0]["back"]]
    return fronts, orphans


# --------------------------------------------------------------------------
# picture content (AI image embeddings)
# --------------------------------------------------------------------------

def _feature_tensor(out):
    import torch
    if torch.is_tensor(out):
        return out
    for name in ("image_embeds", "pooler_output"):
        v = getattr(out, name, None)
        if v is not None:
            return v
    v = getattr(out, "last_hidden_state", None)
    return v[:, 0] if v is not None else out[0]


def _cls_and_patches(out, n_reg=0):
    """A ViT's class token and the mean of its patch tokens, each of unit
    length, side by side: the class token carries the scene as a whole, the
    patch mean what is in it. Register tokens (n_reg, after the class
    token) are left out of the mean."""
    import torch
    hs = getattr(out, "last_hidden_state", None)
    if hs is None or hs.ndim != 3 or hs.shape[1] <= 1 + n_reg:
        return _feature_tensor(out)
    cls = hs[:, 0]
    patches = hs[:, 1 + n_reg:].mean(dim=1)
    nf = torch.nn.functional.normalize
    return torch.cat([nf(cls.float(), dim=-1), nf(patches.float(), dim=-1)],
                     dim=-1)


def _view_side(model_name):
    """The side of the square view a model gets: AI_VIEW_SIDE_DINO for
    DINOv2, whose position embeddings stretch to any size, else
    AI_VIEW_SIDE."""
    return AI_VIEW_SIDE_DINO if "dinov2" in model_name.lower() \
        else AI_VIEW_SIDE


def _ai_views(bgr, side=AI_VIEW_SIDE):
    """The model's view of one print:

    * the border is trimmed: it is common to every print
    * every channel is stretched to its own 1..99 percentiles, so fading and
      toning (a magenta cast, sepia) do not count as content
    * the tile is letterboxed to a square of `side`, which the model gets
      whole: the processor neither resizes nor crops it
    * all four 90-degree rotations are returned: loose prints lie any way up
    """
    h, w = bgr.shape[:2]
    iy, ix = int(AI_INSET * h), int(AI_INSET * w)
    if h - 2 * iy >= 32 and w - 2 * ix >= 32:
        bgr = bgr[iy:h - iy, ix:w - ix]
    f = bgr.astype(np.float32)
    lo = np.percentile(f, 1, axis=(0, 1))
    hi = np.percentile(f, 99, axis=(0, 1))
    bgr = np.clip((f - lo) / np.maximum(hi - lo, 1.0) * 255.0,
                  0, 255).astype(np.uint8)
    rgb = np.ascontiguousarray(_shrink(bgr, side)[:, :, ::-1])
    h, w = rgb.shape[:2]
    s = max(h, w)
    edge = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    pad = np.full((s, s, 3), np.median(edge, axis=0), np.uint8)
    y0, x0 = (s - h) // 2, (s - w) // 2
    pad[y0:y0 + h, x0:x0 + w] = rgb
    if s != side:
        pad = cv2.resize(pad, (side, side),
                         interpolation=cv2.INTER_AREA if s > side
                         else cv2.INTER_CUBIC)
    return [np.ascontiguousarray(np.rot90(pad, k)) for k in AI_ROTS]


def _ai_cross(sa, sb):
    """Rotation-robust cosine distances from every (R, D) view stack of sa
    to every one of sb: one minus the similarity at the best-matching
    relative turn r, where the similarity at r is the mean, over the R
    absolute turns s, of view s of the one print against view s + r of the
    other. Averaging over s keeps the model's liking for one orientation
    out of the comparison.

    With the views of every print laid end to end, one product gives the sum
    over s for one relative turn r at a time."""
    A = np.stack([np.asarray(s, np.float32) for s in sa])       # (n, R, D)
    B = np.stack([np.asarray(s, np.float32) for s in sb])       # (m, R, D)
    n, R, D = A.shape
    m = len(B)
    F = np.ascontiguousarray(A.reshape(n, R * D))
    best = np.full((n, m), -np.inf, np.float32)
    step = max(1, 4_000_000 // max(1, m))
    for r in range(R):
        G = np.ascontiguousarray(np.roll(B, -r, axis=1).reshape(m, R * D))
        for i in range(0, n, step):
            blk = (F[i:i + step] @ G.T) / R
            np.maximum(best[i:i + step], blk, out=best[i:i + step])
    return np.maximum(1.0 - best, 0.0).astype(np.float64)


def _ai_matrix(stacks):
    """_ai_cross of the stacks with themselves, as a symmetric square matrix
    with a zero diagonal."""
    out = _ai_cross(stacks, stacks)
    out = np.minimum(out, out.T)
    np.fill_diagonal(out, 0.0)
    return out


def ai_embeddings(keys, cache_dir, model_name, device_pref, gray=GRAY_ASSIST):
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
    except ImportError:
        warn("    ! torch or transformers is not installed: no AI grouping "
             "(pip install torch transformers)")
        return None
    side = _view_side(model_name)
    batch = max(1, int(AI_BATCH * (AI_VIEW_SIDE / side) ** 2))
    slug = re.sub(r"[^A-Za-z0-9]+", "_", model_name)
    npz = (Path(cache_dir) / f"ai_v{AI_CACHE_VERSION}_{slug}_s{side}"
                             f"{'_g' if gray else ''}_r{len(AI_ROTS)}.npz")
    have = {}
    if npz.exists():
        try:
            z = np.load(npz)
            v = z["vecs"]
            if v.ndim == 3 and v.shape[1] == len(AI_ROTS):
                have = {str(k): x for k, x in zip(z["keys"], v)}
        except Exception:
            have = {}
    todo = [k for k in dict.fromkeys(keys) if k not in have and
            (Path(cache_dir) / f"{k}_thumb.jpg").exists()]
    cached_n = len([k for k in dict.fromkeys(keys) if k in have])
    if cached_n:
        info(f"{count(cached_n, 'embedding')} read from the cache")
    if todo:
        device = device_pref
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        info(f"loading {model_name} on {device}, views of {side} px "
              f"(the first run downloads it)")
        try:
            model = AutoModel.from_pretrained(model_name).to(device).eval()
            proc = AutoImageProcessor.from_pretrained(model_name)
        except Exception as exc:
            warn(f"    ! the model {model_name} could not be loaded: "
                 f"{type(exc).__name__}: {exc}\n"
                 f"      no AI grouping this time (the first run needs the "
                 f"internet; check the name given to --model)")
            return None
        # the views are already the model's size: no resize, no centre crop
        pkw = {k: False for k in ("do_resize", "do_center_crop")
               if hasattr(proc, k)}
        n_reg = int(getattr(model.config, "num_register_tokens", 0) or 0)

        def embed(images):
            pv = proc(images=images, return_tensors="pt",
                      **pkw)["pixel_values"]
            pv = pv.to(device)
            if hasattr(model, "get_image_features"):
                f = _feature_tensor(model.get_image_features(pixel_values=pv))
            else:
                f = _cls_and_patches(model(pixel_values=pv), n_reg)
            return torch.nn.functional.normalize(f.float(), dim=-1)

        def embed_views(flat):
            """Embeddings of a list of views; on a GPU that runs out of
            memory, one view at a time."""
            try:
                return embed(flat)
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or len(flat) < 2:
                    raise
                if device == "cuda":
                    torch.cuda.empty_cache()
                return torch.cat([embed([v]) for v in flat], dim=0)

        bar = Progress(len(todo), "    embedding")
        with torch.no_grad():
            for i in range(0, len(todo), batch):
                chunk = todo[i:i + batch]
                views, good = [], []
                for k in chunk:
                    im = _imread(Path(cache_dir) / f"{k}_thumb.jpg")
                    if im is None:
                        bar.message(paint(f"    ! unreadable thumbnail, "
                                          f"skipped: {k}", "33"))
                        continue
                    views.append(_ai_views(im, side))
                    good.append(k)
                if not views:
                    for _ in chunk:
                        bar.update()
                    continue
                flat = [v for st in views for v in st]      # n * R images
                f = embed_views(flat)
                if gray:
                    gr = [np.repeat(cv2.cvtColor(v, cv2.COLOR_RGB2GRAY)
                                    [:, :, None], 3, axis=2) for v in flat]
                    f = torch.cat([f, embed_views(gr)], dim=-1)
                f = torch.nn.functional.normalize(f, dim=-1)
                V = (f.cpu().numpy().astype(np.float32)
                      .reshape(len(good), len(AI_ROTS), -1))
                for k, v in zip(good, V):
                    have[k] = v
                for _ in chunk:
                    bar.update()
        bar.close()
    # only the prints of this folder as it is now: embeddings of files that
    # have gone or changed are dropped
    current = set(keys)
    if todo or any(k not in current for k in have):
        ks = sorted(k for k in have if k in current)
        if ks:
            np.savez(npz, keys=np.array(ks),
                     vecs=np.stack([have[k] for k in ks]))
        elif npz.exists():
            npz.unlink()
    return {k: have[k] for k in keys if k in have}


# --------------------------------------------------------------------------
# grouping and order
# --------------------------------------------------------------------------

def rerank(D, k1=RERANK_K1, k2=RERANK_K2, lam=RERANK_LAMBDA):
    """k-reciprocal re-ranking (Zhong et al., 2017) of a square distance
    matrix: two prints count as alike when their neighbourhoods overlap,
    not only when they are close. The k-reciprocal neighbours of a print
    are those among its k1 nearest that have it among theirs too, widened
    by the like sets (k1 / 2) of those neighbours that mostly lie inside
    it; every print becomes a weight vector over its set (exp(-d),
    averaged over its k2 nearest), and the Jaccard distance of two such
    vectors is blended with the plain distance: lam of the plain one is
    kept. Needs more than k1 + 1 prints; below that D comes back as it
    is."""
    D = np.asarray(D, np.float64)
    n = len(D)
    if n <= k1 + 1:
        return D
    order = np.argsort(D, axis=1, kind="stable")

    def recip(i, k):
        fwd = order[i, :k + 1]
        return fwd[np.any(order[fwd, :k + 1] == i, axis=1)]

    half = max(1, int(round(k1 / 2)))
    V = np.zeros((n, n))
    for i in range(n):
        R = recip(i, k1)
        sets = [R]
        for j in R:
            Rj = recip(j, half)
            if len(np.intersect1d(Rj, R)) > 2.0 / 3.0 * len(Rj):
                sets.append(Rj)
        R = np.unique(np.concatenate(sets))
        w = np.exp(-D[i, R])
        V[i, R] = w / w.sum()
    if k2 > 1:
        V = np.stack([V[order[i, :k2]].mean(axis=0) for i in range(n)])
    J = np.empty((n, n))
    for i in range(n):
        common = np.minimum(V[i], V).sum(axis=1)   # rows sum to 1, so the
        J[i] = 1.0 - common / (2.0 - common)       # union is 2 - common
    out = (1.0 - lam) * J + lam * D
    out = np.minimum(out, out.T)
    np.fill_diagonal(out, 0.0)
    return out


def _linkage_pre(D, method):
    """The merge tree of a precomputed square distance matrix."""
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
    return linkage(squareform(np.asarray(D, np.float64), checks=False),
                   method=method)


def _silhouette(D, labels):
    """Mean silhouette of a labelling over a square distance matrix; a print
    alone in its group scores 0. None when there is one group only, or one
    per print."""
    labs, inv = np.unique(np.asarray(labels), return_inverse=True)
    n, k = len(inv), len(labs)
    if k < 2 or k >= n:
        return None
    onehot = np.zeros((n, k))
    onehot[np.arange(n), inv] = 1.0
    cnt = onehot.sum(axis=0)
    S = np.asarray(D, np.float64) @ onehot          # (n, k) summed distances
    own = cnt[inv]
    a = S[np.arange(n), inv] / np.maximum(own - 1.0, 1.0)
    M = S / cnt
    M[np.arange(n), inv] = np.inf
    b = M.min(axis=1)
    sil = np.where(own > 1, (b - a) / np.maximum(np.maximum(a, b), 1e-12),
                   0.0)
    return float(sil.mean())


def auto_threshold(D, Z):
    """Where to cut the merge tree. The candidates are the midpoints between
    consecutive merge distances that lie between the median distance of a
    print to its nearest neighbour and the median of all distances, so one
    outlying print cannot decide it; of those (AUTO_CANDIDATES at most,
    spread evenly) the cut with the best silhouette score wins. Returns
    (split, score); without a scorable cut, the middle of the largest jump
    in that range and None."""
    from scipy.cluster.hierarchy import fcluster
    n = len(D)
    off = D + np.eye(n) * 1e9
    lo = float(np.median(off.min(axis=1)))
    hi = float(np.median(D[np.triu_indices(n, 1)]))
    h = np.sort(Z[:, 2])
    best, thr = 0.0, 0.5 * (lo + hi)
    for a, b in zip(h[:-1], h[1:]):
        a, b = max(float(a), lo), min(float(b), hi)
        if b - a > best:
            best, thr = b - a, 0.5 * (a + b)
    cands = sorted({0.5 * (float(a) + float(b)) for a, b in zip(h[:-1], h[1:])
                    if b > a and lo <= 0.5 * (a + b) <= hi})
    if len(cands) > AUTO_CANDIDATES:
        pick = np.linspace(0, len(cands) - 1, AUTO_CANDIDATES)
        cands = [cands[int(round(i))] for i in pick]
    score = None
    for t in cands:
        sc = _silhouette(D, fcluster(Z, t=t, criterion="distance"))
        if sc is not None and (score is None or sc > score):
            score, thr = sc, t
    return thr, score


def _refine(groups, D):
    """Move every print of a group of two or more to the group it is
    nearest to on average, when that is REFINE_MARGIN nearer than its own
    group without it; again until nothing moves. The merge tree never
    undoes a merge, so a print merged early stays where it was even when a
    group formed later fits it better: this puts it right. Returns the
    groups (empty ones dropped) and, per moved print, (its mean distance
    to the group it was cut into, to the group it ends in)."""
    groups = [list(g) for g in groups]
    first = {}
    for _ in range(REFINE_ROUNDS):
        moved = False
        for a, g in enumerate(groups):
            for i in list(g):
                if len(g) < 2:
                    break
                own = float(D[i, [j for j in g if j != i]].mean())
                best, to = own, None
                for b, h in enumerate(groups):
                    if b != a and len(h) >= 2:
                        d = float(D[i, h].mean())
                        if d < best - REFINE_MARGIN:
                            best, to = d, b
                if to is not None:
                    first.setdefault(i, own)
                    g.remove(i)
                    groups[to].append(i)
                    moved = True
        if not moved:
            break
    groups = [g for g in groups if g]
    where = {i: k for k, g in enumerate(groups) for i in g}
    out = {}
    for i, d0 in first.items():
        g = groups[where[i]]
        rest = [j for j in g if j != i]
        if rest:
            out[i] = (d0, float(D[i, rest].mean()))
    return groups, out


def _pair_singles(groups, D, max_d):
    """A group of one print stays single. One whose mean distance to every
    group of two prints or more is above max_d pairs with its nearest
    single print, when they are at most max_d apart and that print is
    nearer to it than to any group: two prints alike that no group fits,
    as two taken at one place and time. Returns the groups (index lists,
    the single prints as groups of one), per single print its mean
    distance to the nearest group, and per paired print (the other print,
    their distance)."""
    big = [g for g in groups if len(g) >= 2]
    singles = [g[0] for g in groups if len(g) == 1]
    near = {}                             # single -> mean distance to a group
    for i in singles:
        d = [float(D[i, b].mean()) for b in big]
        near[i] = min(d) if d else float("inf")
    pairs, paired = [], {}
    for i in sorted((i for i in singles if near[i] > max_d),
                    key=lambda i: near[i], reverse=True):
        if i in paired:
            continue
        mates = [(float(D[i, j]), j) for j in singles
                 if j != i and j not in paired
                 and float(D[i, j]) <= max_d
                 and float(D[i, j]) < near[j]]
        if mates:
            d, j = min(mates)
            pairs.append([i, j])
            paired[i], paired[j] = (j, d), (i, d)
    single = {i: near[i] for i in singles if i not in paired}
    return big + pairs + [[i] for i in single], single, paired


def _leaf_order(D, first):
    """Indices 0..n-1 of a square distance matrix in the order that makes the
    sum of neighbouring distances smallest (optimal leaf ordering of the
    average-linkage tree), read from whichever end has the lower `first`;
    nearest neighbour first, from the lowest `first`, above OLO_MAX."""
    n = len(D)
    if n <= 2:
        return sorted(range(n), key=first)
    if n > OLO_MAX:
        order, rest = [min(range(n), key=first)], set(range(n))
        rest.discard(order[0])
        while rest:
            cur = order[-1]
            nxt = min(rest, key=lambda i: (D[cur, i], first(i)))
            order.append(nxt)
            rest.discard(nxt)
        return order
    from scipy.cluster.hierarchy import leaves_list, optimal_leaf_ordering
    from scipy.spatial.distance import squareform
    y = squareform(np.asarray(D, np.float64), checks=False)
    Z = _linkage_pre(D, "average")
    order = [int(i) for i in leaves_list(optimal_leaf_ordering(Z, y))]
    if first(order[-1]) < first(order[0]):
        order.reverse()
    return order


def group_prints(units, D, thr, pair_max=PAIR_MAX):
    """Split the units at thr on the combined distance matrix D (thr None:
    auto_threshold). Returns (groups, split used, silhouette or None):
    groups is a list of unit lists in their final order, a single print a
    list of one. A single print carries its mean distance to the nearest
    group in u[0]["single"].

    Order: whole groups and single prints by the optimal leaf order of the
    mean distances between them, then the prints inside each group the
    same way, each read from the end nearer the lowest original number."""
    by_num = lambda i: (units[i][0]["num"], units[i][0]["sub"])
    n = len(units)
    if n == 1:
        units[0][0]["single"] = float("inf")
        return [list(units)], None, None
    if n == 2 and thr is None:
        return [list(units)], None, None
    from scipy.cluster.hierarchy import fcluster
    Z = _linkage_pre(D, "average")
    score = None
    if thr is None:
        thr, score = auto_threshold(D, Z)
    idx = {}
    for i, lab in enumerate(fcluster(Z, t=thr, criterion="distance")):
        idx.setdefault(int(lab), []).append(i)
    groups, moved = _refine(list(idx.values()), D)
    for i, (d0, d1) in moved.items():
        units[i][0]["moved"] = (d0, d1)
    groups, single, paired = _pair_singles(groups, D, pair_max * thr)
    for i, d in single.items():
        units[i][0]["single"] = d
    for i, (j, d) in paired.items():
        units[i][0]["paired"] = (units[j][0]["name"], d)

    m = len(groups)
    G = np.zeros((m, m))
    for a in range(m):
        for b in range(a + 1, m):
            G[a, b] = G[b, a] = float(D[np.ix_(groups[a], groups[b])].mean())
    order = _leaf_order(G, lambda a: min(by_num(i) for i in groups[a]))

    def inside(ix):
        ix = sorted(ix, key=by_num)
        o = _leaf_order(D[np.ix_(ix, ix)], lambda k: by_num(ix[k]))
        return [units[ix[k]] for k in o]

    return [inside(groups[a]) for a in order], thr, score


def size_tolerance(fronts, unit):
    """How far a back's sides may differ from its front's (twice this is
    noted)."""
    if unit == "mm":
        return SIZE_TOL_MM
    sides = [u[0]["size"][0] for u in fronts if u[0].get("size")]
    return SIZE_TOL_REL * float(np.median(sides)) if sides else 1.0


def number_leaves(leaves, start, size_tol):
    """One row per file of the plan, in the new order, with its new name."""
    total = sum(len(u) for leaf in leaves for u in leaf["units"])
    width = max([len(str(start + total - 1))]
                + [len(f["digits"]) for leaf in leaves
                   for u in leaf["units"] for f in u])
    rows, n = [], start
    for li, leaf in enumerate(leaves, 1):
        for u in leaf["units"]:
            for j, f in enumerate(u):
                side = "back" if (j or leaf["cls"] == "orphan") else "front"
                rows.append({"f": f, "leaf": li, "leaf_d": leaf, "side": side,
                             "front": u[0] if j else None,
                             "new_name": f"{f['pre']}{n:0{width}d}{f['ext']}"})
                n += 1
    by_file = {id(r["f"]): r for r in rows}
    for r in rows:
        if r["front"] is not None:
            r["front_new"] = by_file[id(r["front"])]["new_name"]
            a, b = r["f"]["size"], r["front"]["size"]
            if a and b and math.hypot(a[0] - b[0], a[1] - b[1]) > \
                    2.0 * size_tol:
                r["f"]["notes"].append("size differs from its front")
    return rows


# --------------------------------------------------------------------------
# review files
# --------------------------------------------------------------------------

def group_desc(units, unit):
    """What a group holds: its median print size, and "rounded corners"
    when most of its prints whose corners could be told have them."""
    sized = [u for u in units if u[0].get("size")]
    desc = size_label(sized, unit) if sized else "size unknown"
    cor = [u[0]["rec"].get("corners") for u in units]
    n_r, n_s = cor.count("rounded"), cor.count("sharp")
    return desc + (", rounded corners" if n_r > n_s else "")


def _leaf_label(leaf):
    k = len(leaf["units"])
    if leaf["kind"] in ("unsorted", "orphan"):
        return f"{count(k, 'file')}"
    if leaf["kind"] == "group":
        return (f"group {leaf['group']}   ({count(k, 'print')})   "
                f"{leaf['desc']}")
    if leaf["kind"] == "single":
        return "single"
    if leaf["kind"] == "noemb":
        return f"no AI embedding   ({count(k, 'print')})"
    return f"{count(k, 'print')}"


def write_sheets(rdir, cache_dir, leaves, rows):
    """Contact sheets: every group as a block of thumbnails under the new
    names: all B&W, then all colour, each class ending with its prints
    without an embedding; then the unreadable prints and
    the backs without a front. Under each thumbnail its new name, and below
    that, in grey, its own name, the one organize_overrides.txt takes. A
    sheet taller than SHEET_MAX_H continues on the next."""
    tile, cap, pad, cols, hdr, title_h = SHEET_TILE, 38, 8, SHEET_COLS, 30, 44
    width = pad + cols * (tile + pad)
    row_h = tile + cap + pad
    new_of = {id(r["f"]): r["new_name"] for r in rows}
    blocks = []                        # (kind, height, payload)
    last_cls = None
    for leaf in leaves:
        if leaf["cls"] != last_cls:
            blocks.append(("title", title_h, CLASS_LABEL[leaf["cls"]]))
            last_cls = leaf["cls"]
        blocks.append(("header", hdr, _leaf_label(leaf)))
        us = leaf["units"]
        for i in range(0, len(us), cols):
            blocks.append(("row", row_h, us[i:i + cols]))

    pages, cur, h = [], [], pad
    for b in blocks:
        if cur and h + b[1] > SHEET_MAX_H:
            pages.append((cur, h))
            cur, h = [], pad
        cur.append(b)
        h += b[1]
    if cur:
        pages.append((cur, h))

    for n, (page, height) in enumerate(pages, 1):
        sheet = np.full((height + pad, width, 3), 236, np.uint8)
        y = pad
        for kind, bh, payload in page:
            if kind == "title":
                cv2.putText(sheet, payload, (pad, y + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2,
                            cv2.LINE_AA)
            elif kind == "header":
                cv2.rectangle(sheet, (0, y), (width, y + bh - 4),
                              (200, 200, 200), -1)
                cv2.putText(sheet, payload, (pad, y + 19),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1,
                            cv2.LINE_AA)
            else:
                for i, u in enumerate(payload):
                    cx = pad + i * (tile + pad)
                    f = u[0]
                    th = (_imread(cache_dir / f"{f['key']}_thumb.jpg")
                          if f["ok"] else None)
                    if th is not None:
                        th = _shrink(th, tile)
                        th_h, th_w = th.shape[:2]
                        ox = cx + (tile - th_w) // 2
                        oy = y + (tile - th_h) // 2
                        sheet[oy:oy + th_h, ox:ox + th_w] = th
                    label = Path(new_of[id(f)]).stem + \
                        (" +back" if len(u) > 1 else "")
                    cv2.putText(sheet, label, (cx, y + tile + 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1,
                                cv2.LINE_AA)
                    cv2.putText(sheet, Path(f["name"]).stem,
                                (cx, y + tile + 32),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                (110, 110, 110), 1, cv2.LINE_AA)
            y += bh
        _imwrite(rdir / f"sheet_{n:02d}.jpg", sheet, 85)
    return len(pages)


def write_plan(rdir, rows, unit):
    """organize_plan.csv: one row per file, in the new order."""
    fields = ["new_name", "original_name", "side", "back_of", "class",
              "group", "long", "short", "unit", "bare_paper",
              "colour_area", "hue_spread", "a_cast", "b_cast", "corners",
              "notes", "fingerprint"]
    with open(rdir / PLAN_NAME, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            f, d, rec = r["f"], r["leaf_d"], r["f"]["rec"]
            cast = rec.get("cast") or {}
            row = {
                "new_name": r["new_name"], "original_name": f["name"],
                "side": r["side"],
                "back_of": r["front"]["name"] if r["front"] else "",
                "class": (d["cls"] if d["cls"] in ("unsorted", "orphan")
                          else f["cls"]),
                "group": (d["group"] if d["kind"] == "group"
                          else d["kind"] if d["kind"] in ("single",
                                                          "noemb")
                          else ""),
                "long": round(f["size"][0], 1) if f["size"] else "",
                "short": round(f["size"][1], 1) if f["size"] else "",
                "unit": unit if f["size"] else "",
                "bare_paper": "" if rec.get("blank") is None
                              else rec["blank"],
                "colour_area": ("" if rec.get("colour_area") is None
                                else rec["colour_area"]),
                "hue_spread": ("" if rec.get("hue_spread") is None
                               else rec["hue_spread"]),
                **{k: "" if cast.get(k) is None else cast[k]
                   for k in ("a_cast", "b_cast")},
                "corners": rec.get("corners") or "",
                "notes": "; ".join(f["notes"]), "fingerprint": f["key"]}
            w.writerow(row)


NOTE_KINDS = (            # how the summary line names each kind of note
    ("moved:", "moved to a nearer group"),
    ("near no group; paired", "paired"),
    ("size differs from its front", "size differs from its front"),
    ("front, though flat", "flat fronts"),
    ("back (", "backs the pixel test did not settle"),
    ("front: a print next to it", "by the back rule"),
    ("single: near no group", "single prints"),
    ("override:", "overrides"),
    ("override(s) ignored", "overrides ignored"),
    ("no DPI tag", "without a DPI tag"),
)


def note_kinds(notes_per_file):
    """[(kind, how many files)] of the notes, in NOTE_KINDS order. The ones
    to check and the unreadable prints are listed by name elsewhere, so
    they are left out here."""
    seen = {}
    for notes in notes_per_file:
        for kind in {next((label for key, label in NOTE_KINDS
                           if n.startswith(key)), None) for n in notes}:
            if kind is None:
                continue
            seen[kind] = seen.get(kind, 0) + 1
    return [(label, seen[label]) for _key, label in NOTE_KINDS
            if label in seen]


def write_summary(rdir, root, leaves, rows, head=()):
    """organize_summary.txt: the groups as a tree, every file with its new
    name and its notes."""
    by_leaf = {}
    for r in rows:
        by_leaf.setdefault(r["leaf"], []).append(r)
    n_back = sum(1 for r in rows if r["side"] == "back")
    n_bad = sum(1 for r in rows if not r["f"]["ok"])
    n_print = len(rows) - n_back - n_bad
    out = [f"STAGE 3 plan for {root}",
           f"{count(len(rows), 'file')}: {count(n_print, 'print')}, "
           f"{count(n_back, 'back')}"
           + (f", {n_bad} unreadable" if n_bad else ""), *head, ""]
    last_cls = None
    for li, leaf in enumerate(leaves, 1):
        cls = leaf["cls"]
        if cls != last_cls:
            n = sum(len(l["units"]) for l in leaves if l["cls"] == cls)
            out.append(f"{CLASS_LABEL[cls]}  ({n})")
            last_cls = cls
        ind = " " * 4
        if leaf["kind"] in ("group", "single", "noemb"):
            out.append(f"  {_leaf_label(leaf)}")
            ind = " " * 6
        for r in by_leaf[li]:
            tag = "   back" if r["side"] == "back" else ""
            note = (f"   ! {'; '.join(r['f']['notes'])}"
                    if r["f"]["notes"] else "")
            out.append(f"{ind}{r['f']['name']:<18} -> {r['new_name']}"
                       f"{tag}{note}")
    (rdir / SUMMARY_NAME).write_text("\n".join(out) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# the analysis
# --------------------------------------------------------------------------

def _load_cache(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _measure_all(prints, cache_dir, recs, args):
    """Measure every print the cache does not already hold; the records go
    into recs."""
    def cached(k):
        r = recs.get(k)
        return (isinstance(r, dict) and r.get("v") == CACHE_VERSION and
                r.get("ok") and (cache_dir / f"{k}_thumb.jpg").exists())

    todo = [f for f in prints if not cached(f["key"])]
    if len(todo) < len(prints):
        info(f"{count(len(prints) - len(todo), 'print')} read from the "
             f"cache")
    if not todo:
        return
    workers = 1
    if parallel is not None and args.workers != 1:
        sample = max(todo, key=lambda f: f["path"].stat().st_size)
        per = parallel.estimate_bytes_per_worker(str(sample["path"]),
                                                 multiple=WORKER_MEM_MULTIPLE)
        workers = parallel.choose_workers(len(todo), per, args.workers)
    info(count(workers, "worker"))
    tasks = [(str(f["path"]), str(cache_dir), f["key"], workers > 1)
             for f in todo]
    bar = Progress(len(todo), "    measuring")

    def report(i, rec):
        f = todo[i]
        recs[f["key"]] = rec
        if not rec.get("ok"):
            bar.message(paint(f"    ! {f['name']}: unreadable "
                              f"({rec.get('err')})", "1;31"))
        bar.update()

    if workers > 1 and len(tasks) > 1:
        # a worker that dies (out of memory on a huge print) must cost its
        # own prints, not the measurements of every other one
        parallel.run(_measure_worker, tasks, workers, on_result=report,
                     on_error=lambda t, exc: {
                         "ok": False, "err": parallel.describe_error(exc),
                         "v": CACHE_VERSION})
    else:
        for i, t in enumerate(tasks):
            report(i, _measure_worker(t))
    bar.close()


def analyze(root, args):
    t_all = time.time()
    if int(cv2.__version__.split(".")[0]) < 4:
        raise SystemExit(f"OpenCV 4.x is required (found {cv2.__version__})")
    if importlib.util.find_spec("scipy") is None:
        raise SystemExit("missing package: scipy (see Requirements above)")

    prints, skipped = find_prints(root)
    if not prints:
        raise SystemExit(f"no numbered prints in {root}")
    heading("STAGE 3 step 1  -  put related prints side by side")
    phase(1, 5, "reading the prints")
    print(paint("    size, bare paper, colour, thumbnail", "90"))
    info(f"folder   {root}")
    info(f"prints   {len(prints)}")
    if skipped:
        warn_list(f"    ! {count(len(skipped), 'TIFF')} without a number "
                  f"in the name, left alone:", skipped)

    rdir = Path(review_dir(root))
    cache_dir = rdir / CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    if args.fresh:
        # the AI vectors go too, or a change to the views stays hidden
        for p in list(cache_dir.glob("ai_*.npz")):
            p.unlink()
    cache_path = cache_dir / "measure.json"
    recs = {} if args.fresh else _load_cache(cache_path)

    # --------------------------------------------------------------- phase 1
    for f in prints:
        f["key"] = fingerprint(f["path"])
    _measure_all(prints, cache_dir, recs, args)
    keep = {f["key"] for f in prints}
    recs = {k: v for k, v in recs.items() if k in keep}
    cache_path.write_text(json.dumps(recs, indent=0), encoding="utf-8")
    for p in list(cache_dir.glob("*.jpg")):
        if p.stem.rsplit("_", 1)[0] not in keep:
            p.unlink()
    unit = classify(prints, recs, read_overrides(root), args.colour_thr)
    n_ok = sum(1 for f in prints if f["ok"])
    n_bad = len(prints) - n_ok
    info(f"{count(n_ok, 'print')} read"
          f"{f', {n_bad} unreadable' if n_bad else ''}")
    if not n_ok:
        error("no print could be read; no plan written")
        return 1
    if unit == "px":
        info("no usable DPI: sizes are in pixels")

    # --------------------------------------------------------------- phase 2
    phase(2, 5, "fronts and backs")
    print(paint(f"    {args.model}: AI embeddings, bare paper, a back right "
                f"after its front", "90"))
    keys = [f["key"] for f in prints if f["ok"]]
    emb = (ai_embeddings(keys, cache_dir, args.model, args.device,
                         gray=not args.no_grey_assist) if keys else None)
    made_back, made_front = decide_backs(prints, emb)
    fronts, orphans = make_units(prints)
    good = [u for u in fronts if u[0]["ok"]]
    bad = [u for u in fronts if not u[0]["ok"]]
    n_back = sum(1 for f in prints if f["ok"] and f["back"])
    info(f"{count(len(good), 'print')}, {count(n_back, 'back')} "
          f"({n_back - len(orphans)} beside their front)")
    if made_back or made_front:
        info(f"against the pixel test: {made_back} made a back, "
             f"{made_front} made a front; noted in the plan")
    if orphans:
        info(f"{count(len(orphans), 'back')} without a front before "
              f"it: filed at the end")

    # --------------------------------------------------------------- phase 3
    phase(3, 5, "colour or B&W")
    print(paint(f"    colour: {args.colour_thr:g} of the picture off its grey "
                f"ramp in more than one hue,\n    a red-green cast of "
                f"{COLOUR_CAST_MIN:g}, {FADED_AREA_MIN:g} off the ramp with a "
                f"yellow cast of {FADED_CAST_B:g},\n    or {MULTI_AREA_MIN:g} "
                f"off it in many hues", "90"))
    by_cls = {c: [u for u in good if u[0]["cls"] == c] for c in CLASS_ORDER}
    for c in CLASS_ORDER:
        info(f"{CLASS_LABEL[c]:<8}{len(by_cls[c]):>5}")
    doubt = []
    for u in good:
        if u[0].get("doubt"):
            u[0]["notes"].append(f"{u[0]['doubt']}: check it")
            doubt.append(f"{u[0]['name']}: {u[0]['doubt']}")
    if doubt:
        warn_list(f"    ! {count(len(doubt), 'print')} to check on the "
                  f"sheets (fix with bw / colour in {OVERRIDE_NAME}):",
                  doubt)

    # --------------------------------------------------------------- phase 4
    thr = None if args.ai_thr == "auto" else float(args.ai_thr)
    phase(4, 5, "grouping")
    print(paint(f"    inside each colour class: AI (re-ranked) + size, cast "
                f"and corners, average linkage, split {args.ai_thr}", "90"))
    cor = [u[0]["rec"].get("corners") for u in good]
    info(f"corners: {cor.count('rounded')} rounded, {cor.count('sharp')} "
          f"sharp, {cor.count(None)} could not be told")
    if emb is None:
        info("no AI embeddings: grouped by size, cast and corners only")
    leaves, splits = [], []
    n_single = n_paired = n_moved = 0
    for cls in CLASS_ORDER:
        have = [u for u in by_cls[cls]
                if emb is None or u[0]["key"] in emb]
        groups = []
        if have:
            A = (np.zeros((len(have), len(have))) if emb is None else
                 _ai_matrix([emb[u[0]["key"]] for u in have]))
            if emb is not None:
                A = rerank(A)
            D = A + penalty_matrix(have, unit, args.size_free, args.size_w,
                                   args.cast_w, args.corner_w)
            groups, used, score = group_prints(have, D, thr, args.pair_max)
            if used is not None:
                splits.append(f"{CLASS_LABEL[cls]} {used:.3f}" +
                              (f" (silhouette {score:.3f})"
                               if score is not None else ""))
        for gu in groups:
            d = gu[0][0].pop("single", None) if len(gu) == 1 else None
            if d is not None:
                n_single += 1
                gu[0][0]["notes"].append(
                    "single: near no group" if d == float("inf") else
                    f"single: near no group (nearest at mean distance "
                    f"{d:.3f})")
                leaves.append({"cls": cls, "kind": "single", "group": 0,
                               "units": gu, "desc": ""})
                continue
            gi = sum(1 for leaf in leaves if leaf["kind"] == "group") + 1
            for u in gu:
                mv = u[0].pop("moved", None)
                if mv is not None:
                    n_moved += 1
                    u[0]["notes"].append(f"moved: nearer to this group "
                                         f"(mean distance {mv[0]:.3f} to the "
                                         f"one it was cut into, {mv[1]:.3f} "
                                         f"to this)")
                p = u[0].pop("paired", None)
                if p is not None:
                    n_paired += 1
                    u[0]["notes"].append(f"near no group; paired with "
                                         f"{p[0]} (distance {p[1]:.3f})")
            leaves.append({"cls": cls, "kind": "group", "group": gi,
                           "units": gu, "desc": group_desc(gu, unit)})
        noemb = sorted((u for u in by_cls[cls] if emb is not None
                        and u[0]["key"] not in emb),
                       key=lambda u: (u[0]["num"], u[0]["sub"]))
        if noemb:
            leaves.append({"cls": cls, "kind": "noemb", "group": 0,
                           "units": noemb, "desc": ""})
    n_grp = sum(1 for leaf in leaves if leaf["kind"] == "group")
    split_text = ("; ".join(splits) + (" (auto)" if thr is None
                                       else " (--ai-thr)")) if splits else ""
    if split_text:
        info(f"split at {split_text}")
    if n_moved:
        info(f"{count(n_moved, 'print')} moved to a group nearer than the "
             f"one the cut put {'it' if n_moved == 1 else 'them'} in")
    if n_paired:
        info(f"{count(n_paired, 'print')} near no group paired with "
             f"{'its' if n_paired == 1 else 'their'} nearest single print")
    if n_single:
        info(f"{count(n_single, 'print')} near no group: kept single")
    for c in CLASS_ORDER:
        sz = [len(leaf["units"]) for leaf in leaves
              if leaf["cls"] == c and leaf["kind"] == "group"]
        if sz:
            info(f"{CLASS_LABEL[c]:<8}{count(len(sz), 'group')}: median "
                  f"{np.median(sz):g} prints, largest {max(sz)}")
    by_num = lambda u: (u[0]["num"], u[0]["sub"])
    for cls, us in (("unsorted", bad), ("orphan", orphans)):
        if us:
            leaves.append({"cls": cls, "kind": cls, "group": 0,
                           "units": sorted(us, key=by_num)})

    # --------------------------------------------------------------- phase 5
    phase(5, 5, "the review files")
    rows = number_leaves(leaves, min(f["num"] for f in prints),
                         size_tolerance(fronts, unit))
    for p in list(rdir.glob("sheet_*.jpg")):
        p.unlink()
    n_sheets = write_sheets(rdir, cache_dir, leaves, rows)
    write_plan(rdir, rows, unit)
    write_summary(rdir, root, leaves, rows,
                  [f"split: {split_text}" if split_text else
                   "split: none (too few prints)",
                   f"prints moved to a nearer group: {n_moved}",
                   f"prints near no group, paired with a single print: "
                   f"{n_paired}",
                   f"prints near no group, kept single: {n_single}"])

    info(f"{count(len(good), 'print')} in {count(n_grp, 'group')}"
          + (f", {n_single} single" if n_single else ""))
    if orphans:
        info(f"{count(len(orphans), 'back')} without a front, filed at "
              f"the end")
    info()
    info(f"plan      {rdir / PLAN_NAME}")
    info(f"tree      {rdir / SUMMARY_NAME}")
    info(f"sheets    {rdir / 'sheet_*.jpg'}  ({n_sheets})")
    kinds = note_kinds(r["f"]["notes"] for r in rows)
    if kinds:
        print(paint(textwrap.fill(
            "notes in the tree: " + ", ".join(f"{n} {k}" for k, n in kinds),
            width=70, initial_indent="    ", subsequent_indent="    "),
            "90"))
    if bad:
        warn_list(f"    ! {count(len(bad), 'print')} could not be read and "
                  f"{'is' if len(bad) == 1 else 'are'} filed at the end:",
                  [u[0]["name"] for u in bad])
    print(paint(f"\nwrote the plan in {rdir}", "32"))
    print(paint(f"    done in {elapsed(t_all)}", "32"))
    print(paint(f"\nnext, after looking at the sheets:\n    python {SCRIPT}"
                f"{root_arg(root)} --apply", "1"))
    return 2 if bad else 0


# --------------------------------------------------------------------------
# --apply and --undo
# --------------------------------------------------------------------------

def _read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _stored_dir(out, root):
    """The output folder as the log stores it: relative to the crops folder
    when inside it, so the archive can be moved or copied whole; absolute
    otherwise."""
    try:
        rel = os.path.relpath(os.path.abspath(out), os.path.abspath(root))
    except ValueError:                  # another drive on Windows
        return os.path.abspath(out)
    if rel == os.pardir or rel.startswith(os.pardir + os.sep):
        return os.path.abspath(out)
    return rel


def _logged_dir(stored, root):
    """The output folder a log row names, as a path."""
    return Path(stored) if os.path.isabs(stored) else Path(root) / stored


def apply_plan(root, out_arg):
    """Link every print into the output folder under its planned name; copy
    it where a link is impossible (another drive, a file system without
    links). The crops keep their own names."""
    rdir = Path(review_dir(root))
    plan, log = rdir / PLAN_NAME, rdir / LOG_NAME
    if not plan.exists():
        raise SystemExit(f"no plan in {rdir}\n"
                         "run the analysis first:\n"
                         f"    python {SCRIPT}{root_arg(root)}")
    if log.exists():
        raise SystemExit(f"the plan was applied already ({log})\n"
                         "remove those files first:\n"
                         f"    python {SCRIPT}{root_arg(root)} "
                         f"--undo")
    out = Path(resolve(out_arg) if out_arg else organized_dir(root))
    if same_dir(out, root):
        raise SystemExit(f"the output folder cannot be the prints folder "
                         f"itself: {out}")
    phase(1, 2, "checking the plan")
    rows = _read_csv(plan)
    info(f"{count(len(rows), 'file')} in {PLAN_NAME}")
    problems = []
    new = [r["new_name"].lower() for r in rows]
    if len(set(new)) != len(new):
        problems.append("the plan gives two files the same new name")
    for r in rows:
        src = Path(root) / r["original_name"]
        if not src.is_file():
            problems.append(f"missing: {r['original_name']}")
        elif r["fingerprint"] and fingerprint(src) != r["fingerprint"]:
            problems.append(f"not the file that was analysed: "
                            f"{r['original_name']}")
        if (out / r["new_name"]).exists():
            problems.append(f"already in {out}: {r['new_name']}")
    if problems:
        error("nothing written: the folder no longer matches the plan")
        warn_list(f"    ! {count(len(problems), 'problem')}:", problems)
        print(paint(f"\nrun the analysis again:\n    python {SCRIPT}"
                    f"{root_arg(root)}", "1"))
        return 1

    info("every crop is still the file that was analysed")

    phase(2, 2, "linking the prints")
    out.mkdir(parents=True, exist_ok=True)
    links = copies = 0
    failed = None
    stored = _stored_dir(out, root)
    bar = Progress(len(rows), "    linking")
    # written row by row, so an interrupted run can still be undone
    with open(log, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["out_dir", "new_name", "original_name", "fingerprint",
                    "how"])
        for r in rows:
            src, dst = Path(root) / r["original_name"], out / r["new_name"]
            try:
                try:
                    os.link(src, dst)
                    how, links = "link", links + 1
                except OSError:
                    shutil.copy2(src, dst)
                    how, copies = "copy", copies + 1
            except OSError as exc:
                failed = f"{r['original_name']} -> {dst}: {exc}"
                try:
                    if dst.exists():
                        dst.unlink()          # a half-written copy
                except OSError:
                    pass
                break
            w.writerow([stored, r["new_name"], r["original_name"],
                        r["fingerprint"], how])
            fh.flush()
            bar.update()
    bar.close()
    if failed:
        error(f"stopped: could not write {failed}")
        print(paint(f"\nthe {count(links + copies, 'file')} written so far "
                    f"{'is' if links + copies == 1 else 'are'} in the log; "
                    f"remove {'it' if links + copies == 1 else 'them'} "
                    f"with:\n    python {SCRIPT}{root_arg(root)} --undo",
                    "1"))
        return 1
    print(paint(f"\nwrote {count(len(rows), 'file')} to {out}", "32")
          + f"  ({links} linked, {copies} copied)")
    if links:
        info("A linked file and its crop are one file on disk: change "
             "neither in place.")
    print(paint(f"\nundo:  python {SCRIPT}{root_arg(root)} --undo", "1"))
    print(paint(f"\nnext, Stage 4: VueScan on the prints in \"{out}\" (the "
                f"settings are in Stage 4's README)", "1"))
    return 0


def undo_last(root):
    """Remove the files the last --apply made, each only while it is still
    the file that was written."""
    rdir = Path(review_dir(root))
    log = rdir / LOG_NAME
    if not log.exists():
        raise SystemExit(f"nothing to undo: no {LOG_NAME} in {rdir}")
    phase(1, 1, "removing the files")
    rows = _read_csv(log)
    removed, kept = 0, []
    for r in rows:
        dst = _logged_dir(r["out_dir"], root) / r["new_name"]
        if not dst.exists():
            continue
        if r["fingerprint"] and fingerprint(dst) != r["fingerprint"]:
            kept.append(str(dst))
            continue
        dst.unlink()
        removed += 1
    for d in {r["out_dir"] for r in rows}:
        try:
            os.rmdir(_logged_dir(d, root))   # only when it is empty now
        except OSError:
            pass
    log.replace(rdir / (Path(LOG_NAME).stem + ".undone.csv"))
    print(paint(f"\nremoved {count(removed, 'file')}", "32"))
    if kept:
        warn_list(f"    ! {count(len(kept), 'file')} changed since --apply "
                  f"and left in place:", kept)
    return 2 if kept else 0


# --------------------------------------------------------------------------
# the step
# --------------------------------------------------------------------------

def _thr_arg(text):
    """--ai-thr: "auto" or a positive number."""
    if text.lower() == "auto":
        return "auto"
    try:
        v = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError("'auto' or a number") from None
    if v <= 0:
        raise argparse.ArgumentTypeError("must be above 0")
    return v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Stage 3 step 1 of 1 — put related prints side by side")
    ap.add_argument("root", nargs="?", default=".",
                    help="project root: the folder holding Stage 2's crops "
                         "(default: this folder)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true",
                      help=f"link the prints into the output folder under "
                           f"the names {REVIEW_DIR}/{PLAN_NAME} gives")
    mode.add_argument("--undo", action="store_true",
                      help="remove the files the last --apply made")
    ap.add_argument("--out", default=None, metavar="DIR",
                    help="with --apply: the folder for the prints under "
                         "their new names (default: ROOT/organized)")
    ap.add_argument("--colour-thr", type=float, default=COLOUR_AREA_MIN,
                    help=f"a print is colour when this share of its picture "
                         f"is off its grey ramp (default "
                         f"{COLOUR_AREA_MIN:g}), or its cast is strong on "
                         f"the red-green axis, or it is faded to yellow, or "
                         f"a little less of it is off in many hues; "
                         f"colour_area, a_cast and b_cast are in "
                         f"{PLAN_NAME}")
    ap.add_argument("--size-free", type=float, default=SIZE_FREE_MM,
                    help=f"sides this many mm apart cost nothing (default "
                         f"{SIZE_FREE_MM:g})")
    ap.add_argument("--size-w", type=float, default=SIZE_W,
                    help=f"largest size penalty (default {SIZE_W:g}; 0 "
                         f"ignores the size)")
    ap.add_argument("--corner-w", type=float, default=CORNER_W,
                    help=f"penalty for rounded against sharp corners "
                         f"(default {CORNER_W:g}; 0 ignores the corners, a "
                         f"large value keeps them apart)")
    ap.add_argument("--cast-w", type=float, default=CAST_W,
                    help=f"largest colour-cast penalty (default {CAST_W:g}; "
                         f"0 ignores the cast)")
    ap.add_argument("--pair-max", type=float, default=PAIR_MAX,
                    help=f"a single print farther than this many times the "
                         f"split from every group pairs with another "
                         f"single print within it (default {PAIR_MAX:g})")
    ap.add_argument("--ai-thr", type=_thr_arg, default=AI_THR,
                    help="split on the combined distance (AI distance + "
                         "penalties), larger = fewer groups; 'auto' (the "
                         "default) takes the cut of the merge tree with the "
                         "best silhouette score")
    ap.add_argument("--no-grey-assist", "--no-gray-assist",
                    dest="no_grey_assist", action="store_true",
                    help="colour embedding only, without the greyscale "
                         "half (twice as fast)")
    ap.add_argument("--model", default=AI_MODEL,
                    help=f"Hugging Face model (default {AI_MODEL})")
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cpu", "cuda"],
                    help="where the model runs (default: auto)")
    ap.add_argument("--fresh", action="store_true",
                    help="measure every print again instead of using "
                         "_cache/ (also drops the AI embeddings)")
    if parallel is not None:
        parallel.add_workers_arg(ap)
    else:
        ap.add_argument("--workers", type=int, default=1,
                        help="ignored: parallel.py not found")
    args = ap.parse_args(argv)
    console_setup()
    if args.out and not args.apply:
        raise SystemExit("--out goes with --apply")

    root = Path(pipeline_root(args.root))
    check_scans_present(str(root))
    if args.apply:
        heading("STAGE 3 step 1  -  link the prints under their new names")
        return apply_plan(root, args.out)
    if args.undo:
        heading("STAGE 3 step 1  -  undo the last --apply")
        return undo_last(root)
    return analyze(root, args)


if __name__ == "__main__":
    raise SystemExit(main())
