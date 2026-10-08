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
# detect_borders.py  --  STAGE 2, step 1 of 4: find the print
#
# STAGE 2 turns a folder of flatbed scans (Stage 1's repaired scans)
# into straightened, cropped image files:
#
#     detect_borders.py        step 1 of 4   find the print
#     suggest_orientation.py   step 2 of 4   suggest the way up
#     review.py                step 3 of 4   approve the box, turn and way up
#     deskew_crop.py           step 4 of 4   straighten, crop and orient
#     common.py                shared code, not a step
#     parallel.py              shared worker-count policy, not a step
#     anchor_report.py         diagnostic for step 1, not a step
#
# Requirements:
#     pip install opencv-python numpy tifffile [imagecodecs] [psutil]
#     common.py beside this file; parallel.py too, for --workers
#
# This step only measures: what it writes are coordinates. Two fixed
# outputs, both derived from the project root -- the folder holding the
# scans, the one path argument this step takes:
#
#     json files/boxes_detect_borders.json   always: the coordinates every
#                                            later step builds on
#     detect_review/*_marked.jpg             only with --review: a lightweight
#                                            preview per scan, for the eye
#
# TIFF pixels are read exactly as stored (no Orientation tag applied), which
# is what the project file's coordinates promise. Each scan is processed
# independently, in parallel when parallel.py sits beside this script.
#
# THE RING FINDER (--ring, off unless asked for) is a simpler method: the
# largest closed outline of colour edges on the scan is the whole print, and
# a rectangle is placed on it. One print per scan,
# measured at RING_WORK_SIDE, so less exact than the side-by-side search. A
# light, flat print is taken for a back, framed blue in the previews; a
# front is framed red.
#
# To see why a side was placed where it was, run anchor_report.py on that
# scan: it prints the per-side statistics this step works from and, with
# --draw, the points each side was fitted on.
#
# Usage:
#     python detect_borders.py [ROOT] [options]
#
#     ROOT              project root: the folder holding the scans
#                       (default: this folder)
#     --review          write the detect_review/ previews; without it this
#                       step writes coordinates only, nothing to look at
#     --back            treat the scans as card BACKS (no emulsion to find)
#     --ring            find the print by its edge ring instead (see THE RING
#                       FINDER above); not with --back
#     --new-only        measure only the scans the project file does not hold
#                       yet, and keep every other record as it is (what
#                       run_pipeline.bat does, so scans added to the folder
#                       later are picked up); without it every scan is
#                       measured again
#     --workers N       parallel workers (default: auto; 1 disables the pool)
#
#     python detect_borders.py
#     python detect_borders.py D:\archive\scans
#     python detect_borders.py --review
#     python detect_borders.py --back
#     python detect_borders.py --ring --review
#     python detect_borders.py --new-only
# ==========================================================================
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

try:
    import cv2
    import numpy as np
    import tifffile
except ModuleNotFoundError as exc:
    sys.exit(f"missing package: {exc.name} (see Requirements above)")

try:
    from common import (SCAN_EXTS, SCHEMA_VERSION, STAGE_DETECT, TIFF_EXT,
                        Progress, check_scans_present, console_setup, count,
                        elapsed, error, heading, info, paint, phase,
                        pipeline_root, read_page, review_dir, root_arg,
                        stage_json, warn_list, write_json)
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
# Pixel and gradient values marked "@ BASE" are calibrated at BASE
# resolution and scaled to the scan by scale_of().

# --- general ---------------------------------------------------------------
SCRIPT = "detect_borders.py"
CROP = 4                # hard crop of the extreme outer scan border
                        # (px @ BASE)
BASE = 800.0            # reference resolution for all px @ BASE values
THICKNESS_SCALE = 2000  # preview frame thickness = image width / this (min 2)
WORKER_MEM_MULTIPLE = 8.0   # peak RAM of one worker per byte of the scan
                            # file

# --- rough photo localization (Otsu core box + soft growth + per-photo gate)
SOFT_DELTA = 80         # pixels darker than paper-this count as photo content
SOFT_EXPAND_CAP = 0.30  # max one-side box growth, fraction of box size
SOFT_GROW_MIN = 0.40    # the soft blob must fill this much of the core box
CORE_DARK_SLACK = 60    # a core box must contain a pixel darker than this
MIN_EDGE_STRENGTH = 80  # a photo ROI must contain one edge this strong
BOX_MIN_AREA = 0.02     # a box must cover this fraction of the page ...
BOX_MIN_SIDE = 0.06     # ... be at least this fraction wide and high ...
BOX_MAX_SIDE = 0.96     # ... and not this much in both (the whole page)
MERGE_GAP = 0.03        # boxes closer than this (fraction of short side) merge

# --- backs of prints (edge-based localization + auto detection) ------------
BACK_GRAD_THR = 100     # gradient of the shadow groove along a card's edge
BACK_CLOSE_STRONG = 10.0  # closing kernel of the strong-ring rung (px @ BASE)
BACK_MIN_AREA = 0.02    # the card must cover at least this fraction of page
BACK_DARK_DELTA = 80    # "dark" inside a card: below paper - this
BACK_DARK_FRAC = 0.06   # a card whose interior holds less than this much
BACK_STD = 45.0         # dark tone, and is this flat, carries no emulsion
BACK_BORDER_GUARD = 6   # tone rung: blobs this close to the scan edge are
                        # page shading (px @ BASE)
BACK_INK_OPEN = 3       # tone rung: opening that removes handwriting
                        # (px @ BASE)
# Low-contrast backs (cream or white cards on pale pages) show an edge
# groove of only 10-25 grey levels, below BACK_GRAD_THR. back_rois() then
# tries weaker evidence in rungs; each rung fires only when the previous
# one found nothing.
BACK_GRAD_THR_FAINT = 70    # faint-ring bar; bare-page noise stays under it
BACK_CLOSE_K = 22           # closing kernel of the faint/tone rungs
                            # (px @ BASE)
BACK_DILATE_K = 2           # ring fattening so nearby arcs merge (px @ BASE)
BACK_TONE_DELTA = 8         # tone rung: |level - paper| that counts as
                            # "not page"

# --- back-mode frame placement ---------------------------------------------
BACK_TANGENT_PCT = 90.0 # outward percentile where the back-mode frame sits
BACK_ENV_MAX_EXC = 12.0 # cap on the envelope's outward excursion (px @ BASE)
# back mode is refused when two opposite margins agree
BACK_PAIR_TOL = 0.15    # two opposite insets this close (as a fraction of
                        # their mean) are a real mount, not a card back

# --- search geometry along one side ----------------------------------------
INSET_FRAC = 0.14       # inward extent of the search window (fraction of side)
MARGIN_FRAC = 0.30      # outward search margin, fraction of bbox max side
INSET_MAX_FRAC = 0.13   # max expected white print margin (fraction of
                        # max side)
INSET_CAP = 110.0       # absolute cap of the search inset (px @ BASE)
SIDE_INSET_LIM = 0.25   # max plausible margin between image edge and outline
HALO_GUARD = 5          # min px between the print edge and inner candidates
BORDER_GUARD = 6        # ignore candidates closer than this to the image
                        # border
EDGE_SKIP_FRAC = 0.08   # skip this fraction of side ends (corner effects)
SAMPLES_PER_SIDE = 40   # target number of sample points along each side
SCAN_EDGE_DARK = 60     # a scan border darker than paper - this is the
                        # scanner's own edge, not picture (CROP stays on)

# --- deep search for a genuinely wide mount --------------------------------
# INSET_CAP / INSET_MAX_FRAC size the search band for an ORDINARY print
# margin and must stay small: every strong tone boundary inside a picture
# looks like an image edge. A card with a wide mount on one side gets a
# SECOND, deeper trace, bounded by the margin the print shows on its other
# sides, whose result must be a clean full-length fit.
DEEP_REACH_K = 3.0        # a deep trace may look this many * m_est inward
SEARCH_HARD_FRAC = 0.40   # ... but never past this fraction of the long side
DEEP_THIN_SPREAD = 0.5    # core scatter, px @ BASE
DEEP_THIN_MIN = 12        # core must hold at least this many samples
DEEP_THIN_SPAN = 0.60     # ... spanning this fraction of the side
DEEP_WEAK_FRAC = 0.60     # deepen sides measured on fewer lines
DEEP_CORE_FRAC = 0.70     # fraction of scan lines that must AGREE
DEEP_MAX_SPREAD = 2.0     # ... and those scatter by under this (px @ BASE)

# --- print-edge anchor detection -------------------------------------------
EDGE_MIN_THR = 300      # min gradient for a strong print-edge anchor
FAINT_ANCHOR_THR = 70   # gradient bar for recovering a faint edge
ANCHOR_NEAR_BOX = 0.5   # an anchor deeper than this fraction of `inset`
                        # inside the rough box is an internal content edge

# --- image-edge candidate checks -------------------------------------------
BAND_EDGE_MIN = 50      # min gradient for an inner-edge candidate
POL_MARGIN = 10         # required brightness step across the line
PAPER_SLACK = 50        # outer side of a valid line must be near paper level
STEP_PAPER_SLACK = 30   # step-only accepts need a REALLY paper-bright outer
MARGIN_BAND_SLACK = 60  # strip right before an inner candidate must really be
MARGIN_BAND_STD = 15    # paper-bright ... and flat (tolerates JPEG ringing)
INNER_DARK_SLACK = 70   # inner darker than paper-70 => content
INNER_TEXTURE_STD = 30  # ... or inner std above this => textured content
STRONG_DARK_EXTRA = 10  # decisive-dark bar for the borderless relaxations
LOCAL_PAPER_DELTA_LO = 15  # the local paper level may deviate from the global
LOCAL_PAPER_DELTA_HI = 10  # estimate by at most this much (shading cap)

# --- margin consensus (m_est) ----------------------------------------------
MWIN_LO = 0.45          # hint window around the expected margin width ...
MWIN_HI = 2.2           # ... (fractions of m_est; also the strict search cap)
MWIN_WEAK = (0.6, 1.6)  # tighter hint window for the weak pass

# --- corridor uniformity and prior-edge refinement -------------------------
CORRIDOR_STEP = 12      # max brightness drop, outer -> inner corridor half
CORRIDOR_STEP_UP = 2 * CORRIDOR_STEP   # same bound for a brightness RISE
CORRIDOR_RESID = 12.0   # toned card: max |deviation| from a straight ramp
EDGE_REFINE_THR = 25    # min gradient to snap a prior-placed edge
WIDE_RUN_FACTOR = 1.25  # bright smooth strip wider than this * inset = content
WIDE_RUN_CAP = 2.5      # how far (in * inset) the strip measurement looks

# --- line fitting ----------------------------------------------------------
MIN_POINTS = 8          # min accepted sample points to fit a side line
FIT_TOL = 3.0           # inlier distance (px @ BASE) for the refit
MIN_INLIER_FRAC = 0.5   # majority of samples must support the fitted line
MAX_TILT_DEG = 4.0      # max tilt of a fitted side line (deg)
WEAK_MIN_POINTS = 3     # min samples for the fixed-angle offset fit
FIXED_ANGLE_TOL = 2.5   # max median residual (px @ BASE) for it
CONSENSUS_TRIM_DEG = 0.5  # sides further than this from the median tilt are
                          # left out of the consensus ...
CONSENSUS_MAX_DEG = 2.0   # ... which is clipped to +/- this

# --- weak-contrast pass ----------------------------------------------------
WEAK_BAND_MIN = 14      # gradient bar for weak-contrast boundaries
DENSE_FRAC = 0.35       # a side is "weak" when accepted samples fall below
SPAN_FRAC = 0.60        # this density OR cover less than this span

# --- cross-side consistency (passes 2/3/5, clean-margin guard) -------------
DEV_ABS = 5.0           # re-trace trigger: a side inset deviates from m_est
DEV_REL = 0.7           # by more than max(DEV_ABS*s, DEV_REL*m_est)
EVIDENCE_RATIO = 0.5    # a well-supported line is not traded for a weaker one
CLEAN_MARGIN_FRAC = 0.6 # clean-band fraction required by the margin guards

# --- outward-cluster preference --------------------------------------------
# A side's samples can split into two well-populated groups: the print edge
# and a tone boundary INSIDE the picture. Nothing outside the print edge can
# be mistaken for it, while every strong boundary inside the picture can,
# so when the groups are clearly separated the OUTWARD one is the edge.
OUTWARD_GAP = 2.0       # residual jump (in * FIT_TOL) that separates groups
OUTWARD_MIN_SPAN = 0.35  # the outward group must cover this fraction of
                         # the side
OUTWARD_PCT = 20.0      # offset read off the group's OUTWARD end
# The outward group is fitted at the FIXED consensus angle, so it fixes an
# offset, not a slope. When one side has pinned the angle (dense samples,
# tight scatter), the span bar that protects the slope can come down. A cut
# edge may also carry less gradient than a tone boundary inside the picture;
# the strength test is waived when a real card margin lies behind the
# candidate (margin_behind_ok), which such a boundary cannot fake.
TILT_ANCHORED_FRAC = 0.80   # a side with this fraction of its scan lines ...
TILT_ANCHORED_SPREAD = 1.0  # ... and this scatter (px @ BASE) pins the angle
OUTWARD_MIN_SPAN_ANCHORED = 0.12  # span demanded once the angle is pinned

# --- sources that carry their own acceptance test --------------------------
# The geometric prior is a last resort and must not outrank a side that a
# pass with its own evidence bars already measured.
VERIFIED_SRC = ("pass3-deep", "outward-cluster", "outward-smear",
                "pass5-cluster", "pass2b-ghost",
                "pass5-fill (cluster beats prior)")

# --- outward / outlier re-fit of a smeared side ----------------------------
# On a strongly textured print with a low-contrast emulsion edge, a side's
# samples smear and the majority fit can lock onto a weak ridge inside the
# print's white border, making the frame too big. The true-edge samples are
# then the outward minority. Route a (smear_outward_fit) fits the outward
# tail; route b clusters the fit's outliers when there is no tail, and
# accepts them only on an inward move with a clean margin behind.
SMEAR_INLIER_FRAC = 0.80 # inlier fraction below which a side is smeared
SMEAR_FLANK_TOL = 1.0    # tail starts this * FIT_TOL outward of the flank
SMEAR_TAIL_MIN = 6       # the tail / outlier cluster must hold this many
SMEAR_FRINGE_SKIP = 0.25 # fraction of the anchor->candidate gap
                         # skipped at its anchor end (deckle fringe)

# --- solid-anchor completion & ghost outline -------------------------------
PRIOR_STARVED_FRAC = 0.5   # pass-1 samples under this fraction of the side's
                           # scan lines = the edge was never measured
PRIOR_ANCHOR_FRAC = 0.9    # ... while the outline is anchored on at least
                           # this fraction: anchor + m_est outranks a thin line
GHOST_INSET_FRAC = 0.30    # margin under this fraction of m_est on a solid
                           # outline = ghost-suspect
GHOST_DIN_FRAC = 0.25      # ghost re-anchor trigger, fraction of inset
                           # (ordinary traces use ANCHOR_NEAR_BOX)

# --- a verified cluster beats the geometric prior --------------------------
# Before the geometric prior fires on a starved side, the side's own cluster
# is tried on stricter terms than a pass-5 replacement gets: a strong ridge
# OR geometric coherence (tight scatter AND a consensus margin).
FILL_SPAN = 0.60        # the cluster must span this fraction of the side,
                        # unless the angle is pinned ...
FILL_PINNED_SPREAD = 0.50   # ... and then its scatter is under this
                            # (px @ BASE)
FILL_PINNED_MARGIN = 0.15   # ... and its inset matches m_est this closely
FILL_MIN_PTS = 4        # the cluster needs at least this many samples
FILL_TIGHT_SPREAD = 1.0 # cluster scatter (px @ BASE) that counts as
                        # geometrically coherent on its own
FILL_TIGHT_GRAD = BAND_EDGE_MIN   # ... it must still sit on a real ridge

# --- card-margin reference level -------------------------------------------
# PAPER_SLACK & co ask "is the outer side of this candidate still margin?"
# against the ALBUM PAGE. On a toned or aged card the margin sits far below
# that, so every image-edge candidate would be refused. The strip right
# inside the outline is therefore measured too and used as the reference
# when it looks like card stock. On an ordinary scan it never applies.
MARGIN_REF_MIN_DROP = 30.0  # at least this far below paper, else ordinary
MARGIN_REF_DROP = 100.0 # how far below paper_loc a margin reference may sit
MARGIN_REF_STD = 12.0   # the strip must be this flat to count as card stock
MARGIN_REF_SIDE_FRAC = 0.5  # share of scan lines that must agree per side

# --- red-channel cross-check -----------------------------------------------
# A SILVER photographic edge shows in red at least as strongly as in
# luminance; ink and foxing barely register there. On a colour print that
# does not hold, so the check is off for the whole scan.
ALT_GRAD_RATIO = 0.90   # red-channel gradient must reach this fraction of
                        # luminance, else the edge is chromatic (ink, foxing)

# --- is this print silver-based or colour? ---------------------------------
# A scan is colour only when chroma AND hue spread both say so: a strong
# sepia reaches colour-level chroma, but on one locked hue.
COLOR_C_P90_MIN = 26.0    # C_p90 at or above this MAY be colour ...
COLOR_HUE_R_MAX = 0.90    # ... but only if the hue is not locked to one
                          # direction (resultant length below this)
CHROMA_ANALYSIS_SIDE = 1100   # long side the scan is decimated to first

# --- final quad sanity -----------------------------------------------------
QUAD_MIN_SIDE = 25      # every side at least this long (px @ BASE)
QUAD_MAX_GROWTH = 1.6   # no side longer than this * the rough box ...
QUAD_GROWTH_PAD = 60    # ... + this (px @ BASE)
QUAD_MAX_COS = 0.35     # |cos| of the corner angle must stay under this
MIN_SIDE_GAP = 20       # opposite sides closer than this (px @ BASE) collapse

# --- the ring finder (--ring only) -----------------------------------------
RING_WORK_SIDE = 1600   # long side the scan is shrunk to before it is measured
RING_STRETCH_BELOW = 49152  # a 16-bit scan whose brightest sample is below
                            # this is stretched to full range first
RING_THR = 30.0         # colour gradient that counts as part of the ring ...
RING_THR_FAINT = 18.0   # ... and the lower bar, tried when nothing passes
RING_BLUR = 1.0         # blur before the gradient (px @ BASE)
RING_DUST = 0.03        # a piece shorter than this share of the long side is
                        # dust
RING_CLOSE_K = 22       # closing kernel that joins the ring (px @ BASE)
RING_BORDER_BAND = 0.02  # thin strips this wide along the scan edge are
                         # dropped (share of the short side)
RING_MIN_AREA = 0.02    # a print covers at least this share of the scan ...
RING_MIN_SIDE = 0.06    # ... is at least this share wide and high ...
RING_MAX_SIDE = 0.96    # ... and not this much in both (the whole page)
RING_SIDE_MID = 0.80    # each side is placed from this middle share of its
                        # outline, away from rounded corners
RING_EDGE_GUARD = 1     # a side this close to the scan edge is reported as
                        # cut off (px @ BASE)
RING_FRONT_COLOR = (0, 0, 255)    # preview frame, BGR: red on a front ...
RING_BACK_COLOR = (255, 128, 0)   # ... blue on a back


# --------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------

# one verdict per scan line (was the card-margin reference used?), filled by
# _profile_edge and read by trace_side for the side being traced
_MARGIN_REF_LOG = []

def scale_of(gray):
    """Scale factor of the current image relative to the BASE resolution."""
    return max(1.0, max(gray.shape) / BASE)


def paper_level_of(gray):
    """Estimate the album-paper brightness level."""
    return float(np.percentile(gray, 90))


def compute_grad(gray):
    """|Sobel| magnitude plus directional components, scaled by the
    resolution factor so gradient thresholds stay resolution-free."""
    s = scale_of(gray)
    g = cv2.GaussianBlur(gray, (0, 0), 0.8 * s)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return (cv2.magnitude(gx, gy) * s, np.abs(gx) * s, np.abs(gy) * s)


# --- colour/mono classification helpers ------------------------------------

_M_RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                       [0.2126729, 0.7151522, 0.0721750],
                       [0.0193339, 0.1191920, 0.9503041]], dtype=np.float64)
_D65 = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)


def srgb_chroma_stats(img):
    """(C_p90, hue_R) of a BGR scan, in CIE L*a*b*.

    C_p90 is the 90th percentile of chroma - how colourful the frame is at
    all - and hue_R is the resultant length of the chromatic pixels' hue
    directions: 1.0 means every coloured pixel points the same way, which is
    what a toned monochrome does and a colour photograph does not.  Both are
    needed: a strong sepia reaches colour-level chroma on one locked hue."""
    if img is None or img.ndim != 3 or img.shape[2] != 3:
        return None, None
    h, w = img.shape[:2]
    st = max(1, max(h, w) // CHROMA_ANALYSIS_SIDE)
    q = img[::st, ::st][:, :, ::-1].astype(np.float64) / 255.0
    lin = np.where(q <= 0.04045, q / 12.92, ((q + 0.055) / 1.055) ** 2.4)
    xyz = (lin @ _M_RGB2XYZ.T) / _D65
    eps, kap = 216.0 / 24389.0, 24389.0 / 27.0
    f = np.where(xyz > eps, np.cbrt(xyz), (kap * xyz + 16.0) / 116.0)
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    C = np.hypot(a, b)
    c90 = float(np.percentile(C, 90.0))
    m = C > max(1.5, float(np.median(C)))
    if int(m.sum()) < 500:
        m = C > 1.5
    if int(m.sum()) < 50:
        return c90, None
    ang = np.arctan2(b[m], a[m])
    R = float(np.hypot(np.mean(np.cos(ang)), np.mean(np.sin(ang))))
    return c90, R


def scan_is_colour(img):
    """True only when chroma AND hue spread both say colour.

    Every refusal path returns False (silver), so the red-channel veto - a
    measured guard against ink and foxing - stays on unless the scan proves
    it is a colour print where the veto's premise does not hold."""
    c90, R = srgb_chroma_stats(img)
    if c90 is None:
        return False                  # single channel: no chroma exists
    if c90 < COLOR_C_P90_MIN:
        return False                  # not colourful enough, whatever the hue
    if R is None:
        return False                  # unmeasurable hue: stay conservative
    return R < COLOR_HUE_R_MAX


# --------------------------------------------------------------------------
# box finding
# --------------------------------------------------------------------------

def merge_boxes(boxes, gap, gray=None, paper=None):
    """Union boxes that overlap or lie closer than `gap` pixels.

    Fragments of one photo are separated by image content; distinct photos
    are separated by a strip of bare album page.  When `gray`/`paper` are
    available, a non-overlapping pair whose separating corridor is
    paper-bright and flat is therefore left alone."""
    def strip_is_page(a, b):
        if gray is None or paper is None:
            return False
        H, W = gray.shape
        ax0, ay0, ax1, ay1 = a[0], a[1], a[0] + a[2], a[1] + a[3]
        bx0, by0, bx1, by1 = b[0], b[1], b[0] + b[2], b[1] + b[3]
        if ax1 <= bx0 or bx1 <= ax0:                    # horizontal corridor
            u0, u1 = (ax1, bx0) if ax1 <= bx0 else (bx1, ax0)
            v0, v1 = max(ay0, by0), min(ay1, by1)
            sl = (slice(max(0, v0), min(H, v1)), slice(max(0, u0), min(W, u1)))
        elif ay1 <= by0 or by1 <= ay0:                  # vertical corridor
            v0, v1 = (ay1, by0) if ay1 <= by0 else (by1, ay0)
            u0, u1 = max(ax0, bx0), min(ax1, bx1)
            sl = (slice(max(0, v0), min(H, v1)), slice(max(0, u0), min(W, u1)))
        else:
            return False                                 # boxes overlap
        seg = gray[sl]
        if seg.size == 0 or min(seg.shape) < 2:
            return False                                 # touching: no strip
        return (float(seg.mean()) >= paper - PAPER_SLACK and
                float(seg.std()) <= 2 * MARGIN_BAND_STD)

    def near(a, b):
        if not (a[0] - gap < b[0] + b[2] and b[0] - gap < a[0] + a[2] and
                a[1] - gap < b[1] + b[3] and b[1] - gap < a[1] + a[3]):
            return False
        return not strip_is_page(a, b)

    merged = list(boxes)
    changed = True
    while changed:
        changed = False
        out = []
        while merged:
            cur = merged.pop()
            for i, o in enumerate(out):
                if near(cur, o):
                    x0 = min(o[0], cur[0]); y0 = min(o[1], cur[1])
                    x1 = max(o[0] + o[2], cur[0] + cur[2])
                    y1 = max(o[1] + o[3], cur[1] + cur[3])
                    out[i] = (x0, y0, x1 - x0, y1 - y0)
                    changed = True
                    break
            else:
                out.append(cur)
        merged = out
    merged.sort(key=lambda b: (b[1], b[0]))
    return merged


def looks_like_back(gray, boxes, paper):
    """Does this scan show the BACK of a print?  A back carries no emulsion:
    the interior is neither dark nor textured (both by an order of
    magnitude against any picture)."""
    if not boxes:
        return False
    bx, by, bw, bh = max(boxes, key=lambda b: b[2] * b[3])
    ix, iy = int(0.1 * bw), int(0.1 * bh)
    roi = gray[by + iy:by + bh - iy, bx + ix:bx + bw - ix]
    if roi.size < 1000:
        return False
    rs = roi[::4, ::4]
    return (float((rs < paper - BACK_DARK_DELTA).mean()) < BACK_DARK_FRAC and
            float(rs.std()) < BACK_STD)


def _back_components(mask, s, close_k, hull):
    """Close a candidate mask, optionally fatten it, contour it.

    Returns [(x, y, w, h, area), ...].  With hull=True the area and
    bbox come from the CONVEX HULL: findContours on an open arc yields a
    polygon whose implicit closure hugs the arc, so contourArea then
    measures the thin stroke instead of the card it surrounds."""
    k = max(3, int(round(close_k * s)) | 1)
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    if hull:
        kd = max(3, int(round(BACK_DILATE_K * s)) | 1)
        m = cv2.dilate(m, np.ones((kd, kd), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for ct in cnts:
        obj = cv2.convexHull(ct) if hull else ct
        x, y, w, h = cv2.boundingRect(obj)
        out.append((x, y, w, h, float(cv2.contourArea(obj))))
    return out


def _back_gate(comps, H, W, s, border_guard=False):
    """BACK_MIN_AREA / dimension gates.  (Tone rung only) blobs
    hugging the scan border are page shading or the scanner's own edge."""
    keep, t = [], max(2, int(round(BACK_BORDER_GUARD * s)))
    for x, y, w, h, a in comps:
        if a < BACK_MIN_AREA * H * W:
            continue
        if w < BOX_MIN_SIDE * W or h < BOX_MIN_SIDE * H:
            continue
        if border_guard and (x <= t or y <= t or
                             x + w >= W - t or y + h >= H - t):
            continue
        keep.append((x, y, w, h))
    return keep


def back_rois(gray, grad, paper=None):
    """Locate a card by its EDGE, with a tone fallback.

    Three rungs, each firing only when the previous one found nothing:
      1. the strong-ring finder (grad > BACK_GRAD_THR, close 10s);
      2. a faint ring at BACK_GRAD_THR_FAINT with wide closing, credited by
         the CONVEX HULL of each component (an open arc still encloses its
         card - see _back_components);
      3. the tone field |level - paper| > BACK_TONE_DELTA with ink strokes
         opened away and blobs hugging the scan border rejected."""
    s = scale_of(gray)
    H, W = gray.shape
    gap = int(MERGE_GAP * min(H, W))

    comps = _back_components((grad > BACK_GRAD_THR).astype(np.uint8),
                             s, BACK_CLOSE_STRONG, hull=False)
    boxes = _back_gate(comps, H, W, s)
    if boxes:
        return merge_boxes(boxes, gap)

    comps = _back_components((grad > BACK_GRAD_THR_FAINT).astype(np.uint8),
                             s, BACK_CLOSE_K, hull=True)
    boxes = _back_gate(comps, H, W, s)
    if boxes or paper is None:
        return merge_boxes(boxes, gap)

    tone = (np.abs(gray.astype(np.float32) - float(paper)) >
            BACK_TONE_DELTA).astype(np.uint8)
    ko = max(3, int(round(BACK_INK_OPEN * s)) | 1)   # drop handwriting
    tone = cv2.morphologyEx(tone, cv2.MORPH_OPEN, np.ones((ko, ko), np.uint8))
    comps = _back_components(tone, s, BACK_CLOSE_K, hull=True)
    boxes = _back_gate(comps, H, W, s, border_guard=True)
    return merge_boxes(boxes, gap)


def rough_rois(gray, paper):
    """Rough photo localization with Otsu; rejects specks/bands/shading.

    Otsu sees only the DARK parts of a photo, so light regions truncate the
    core box.  Each core box is therefore grown with a SOFT paper-relative
    mask (gray < paper - SOFT_DELTA), capped at SOFT_EXPAND_CAP per side."""
    H, W = gray.shape
    _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    soft = ((gray < paper - SOFT_DELTA) | (m > 0)).astype(np.uint8) * 255
    soft = cv2.morphologyEx(soft, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    _, lab_soft, st_soft, _ = cv2.connectedComponentsWithStats(soft, 8)

    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if cv2.contourArea(c) < BOX_MIN_AREA * H * W:
            continue
        if w > BOX_MAX_SIDE * W and h > BOX_MAX_SIDE * H:
            continue
        if w < BOX_MIN_SIDE * W or h < BOX_MIN_SIDE * H:
            continue
        if gray[y:y + h, x:x + w].min() > paper - CORE_DARK_SLACK:
            continue
        bx, by, bw, bh = x, y, w, h
        ids, counts = np.unique(lab_soft[y:y + h, x:x + w], return_counts=True)
        best, best_n = 0, 0
        for sid, cn in zip(ids, counts):
            if sid != 0 and cn > best_n:
                best, best_n = sid, cn
        if best > 0 and best_n > SOFT_GROW_MIN * w * h:
            sx, sy, sw, sh = st_soft[best][:4]
            grow_x = int(SOFT_EXPAND_CAP * w)
            grow_y = int(SOFT_EXPAND_CAP * h)
            bx = max(x - grow_x, min(x, sx))
            by = max(y - grow_y, min(y, sy))
            bw = min(x + w + grow_x, max(x + w, sx + sw)) - bx
            bh = min(y + h + grow_y, max(y + h, sy + sh)) - by
            if (bw > BOX_MAX_SIDE * W and bh > BOX_MAX_SIDE * H) or \
                    bw < BOX_MIN_SIDE * W or bh < BOX_MIN_SIDE * H:
                bx, by, bw, bh = x, y, w, h   # unsafe growth -> keep the core
        boxes.append((bx, by, bw, bh))
    return merge_boxes(boxes, int(MERGE_GAP * min(H, W)), gray, paper)


# --------------------------------------------------------------------------
# line and cluster fitting
# --------------------------------------------------------------------------

def _fit_line_robust(pts, kind, tol=FIT_TOL):
    """Theil-Sen start + least-squares inlier refit + inlier-majority
    requirement + tilt cap.  Returns (px, py, vx, vy) or None."""
    n = len(pts)
    if n < MIN_POINTS:
        return None
    pts = np.asarray(pts, np.float64)
    if kind in ("top", "bot"):
        t, off = pts[:, 0], pts[:, 1]
    else:
        t, off = pts[:, 1], pts[:, 0]
    if np.unique(t).size < MIN_POINTS:
        return None
    i, j = np.triu_indices(n, 1)
    dt = t[j] - t[i]
    ok = np.abs(dt) > 1e-6
    if not ok.any():
        return None
    slope = float(np.median((off[j] - off[i])[ok] / dt[ok]))
    b = float(np.median(off - slope * t))
    need = max(MIN_POINTS, int(round(MIN_INLIER_FRAC * n)))
    inl = np.zeros(n, bool)
    for _ in range(3):
        r = np.abs(off - (slope * t + b))
        inl = r < tol
        if inl.sum() < need:
            return None
        A = np.stack([t[inl], np.ones(int(inl.sum()))], axis=1)
        (slope, b), *_ = np.linalg.lstsq(A, off[inl], rcond=None)
    inl = np.abs(off - (slope * t + b)) < tol
    if inl.sum() < need:
        return None
    if abs(np.degrees(np.arctan(slope))) > MAX_TILT_DEG:
        return None
    tc = float(np.median(t[inl]))
    oc = slope * tc + b
    if kind in ("top", "bot"):
        return (tc, oc, 1.0, slope)
    return (oc, tc, slope, 1.0)


def _refit_at_angle(pts, kind, slope, tol, min_pts=None):
    """Offset-only robust fit of a line whose ANGLE is given."""
    floor = MIN_POINTS if min_pts is None else min_pts
    if len(pts) < floor:
        return None
    a = np.asarray(pts, np.float64)
    t, off = ((a[:, 0], a[:, 1]) if kind in ("top", "bot")
              else (a[:, 1], a[:, 0]))
    b = float(np.median(off - slope * t))
    need = max(floor, int(round(MIN_INLIER_FRAC * len(t))))
    inl = None
    for _ in range(3):
        inl = np.abs(off - (slope * t + b)) < tol
        if inl.sum() < need:
            return None
        b = float(np.median((off - slope * t)[inl]))
    tc = float(np.median(t[inl]))
    oc = slope * tc + b
    if kind in ("top", "bot"):
        return (tc, oc, 1.0, slope)
    return (oc, tc, slope, 1.0)


def _fit_line_fixed_angle(pts, kind, slope, tol):
    """Few-point rescue: estimate only the OFFSET of a line whose ANGLE is
    fixed to the consensus tilt."""
    if len(pts) < WEAK_MIN_POINTS:
        return None
    pts = np.asarray(pts, np.float64)
    if kind in ("top", "bot"):
        t, off = pts[:, 0], pts[:, 1]
    else:
        t, off = pts[:, 1], pts[:, 0]
    b = float(np.median(off - slope * t))
    r = np.abs(off - (slope * t + b))
    if float(np.median(r)) > tol or float(r.max()) > 3.0 * tol:
        return None                          # samples disagree -> not one edge
    tc = float(np.median(t))
    oc = slope * tc + b
    if kind in ("top", "bot"):
        return (tc, oc, 1.0, slope)
    return (oc, tc, slope, 1.0)


def cluster_at_angle(pts, kind, slope, tol):
    """Largest MUTUALLY CONSISTENT cluster of samples at a given angle:
    the densest band of width 2*tol in the residuals.  The CALLER must
    check the result sits on a gradient before believing it."""
    if len(pts) < WEAK_MIN_POINTS:
        return None, []
    a = np.asarray(pts, np.float64)
    tt, off = ((a[:, 0], a[:, 1]) if kind in ("top", "bot")
               else (a[:, 1], a[:, 0]))
    res = off - slope * tt
    order_i = np.argsort(res)
    r = res[order_i]
    best_lo = best_n = 0
    j = 0
    for i in range(len(r)):                       # widest-window sweep
        while j < len(r) and r[j] - r[i] <= 2 * tol:
            j += 1
        if j - i > best_n:
            best_n, best_lo = j - i, i
    if best_n < WEAK_MIN_POINTS:
        return None, []
    keep = order_i[best_lo:best_lo + best_n]
    cl = [tuple(a[i]) for i in keep]
    return _refit_at_angle(cl, kind, slope, tol,
                           min_pts=WEAK_MIN_POINTS), cl


def outward_cluster_at_angle(pts, kind, slope, tol, gap=OUTWARD_GAP):
    """Group of samples lying clearly OUTWARD of the densest band.

    Returns (line, cluster_points, n_groups); n_groups < 2 means the samples
    do not split and the caller must leave the side alone."""
    if len(pts) < 2 * WEAK_MIN_POINTS:
        return None, [], 0
    _main_line, main_cl = cluster_at_angle(pts, kind, slope, tol)
    if len(main_cl) < WEAK_MIN_POINTS:
        return None, [], 0
    a = np.asarray(pts, np.float64)
    tt, off = ((a[:, 0], a[:, 1]) if kind in ("top", "bot")
               else (a[:, 1], a[:, 0]))
    res = off - slope * tt
    m = np.asarray(main_cl, np.float64)
    m_tt, m_off = ((m[:, 0], m[:, 1]) if kind in ("top", "bot")
                   else (m[:, 1], m[:, 0]))
    m_res = float(np.median(m_off - slope * m_tt))
    sgn = -1.0 if kind in ("top", "lef") else 1.0     # outward, in residual
    keep = np.where(sgn * (res - m_res) > gap * tol)[0]
    if len(keep) < WEAK_MIN_POINTS:
        return None, [], 1
    cl = [tuple(a[i]) for i in keep]
    cand = _refit_at_angle(cl, kind, slope, tol, min_pts=WEAK_MIN_POINTS)
    if cand is None:
        return None, [], 1                # outward samples do not line up
    # read the offset off the OUTWARD end of the group
    r_cl = np.array([(q[1] if kind in ("top", "bot") else q[0]) -
                     line_off_at(cand, kind,
                                 q[0] if kind in ("top", "bot") else q[1])
                     for q in cl], np.float64)
    shift = float(np.percentile(r_cl, OUTWARD_PCT if kind in ("top", "lef")
                                else 100.0 - OUTWARD_PCT))
    px, py, vx, vy = cand
    cand = ((px, py + shift, vx, vy) if kind in ("top", "bot")
            else (px + shift, py, vx, vy))
    return cand, cl, 2


def smear_outward_fit(pts, kind, slope, tol):
    """Outward-tail refit of a SMEARED sample cloud (route a).

    Splits at the dense band's outward FLANK - every sample at least
    SMEAR_FLANK_TOL * tol outward of the densest band's outermost extent -
    and refits only the tail.  The offset is read off the tail's outward
    end.  Returns (line, tail_points) or (None, [])."""
    if len(pts) < 2 * WEAK_MIN_POINTS:
        return None, []
    _main_line, main_cl = cluster_at_angle(pts, kind, slope, tol)
    if len(main_cl) < WEAK_MIN_POINTS:
        return None, []
    a = np.asarray(pts, np.float64)
    tt, off = ((a[:, 0], a[:, 1]) if kind in ("top", "bot")
               else (a[:, 1], a[:, 0]))
    sgn = -1.0 if kind in ("top", "lef") else 1.0     # outward, in residual
    ro = sgn * (off - slope * tt)
    m = np.asarray(main_cl, np.float64)
    m_tt, m_off = ((m[:, 0], m[:, 1]) if kind in ("top", "bot")
                   else (m[:, 1], m[:, 0]))
    flank = float(np.max(sgn * (m_off - slope * m_tt)))
    keep = np.where(ro >= flank + SMEAR_FLANK_TOL * tol)[0]
    if len(keep) < SMEAR_TAIL_MIN:
        return None, []
    cl = [tuple(a[i]) for i in keep]
    cand = _refit_at_angle(cl, kind, slope, tol, min_pts=WEAK_MIN_POINTS)
    if cand is None:
        return None, []               # the tail is not one straight ridge
    r_cl = np.array([(q[1] if kind in ("top", "bot") else q[0]) -
                     line_off_at(cand, kind,
                                 q[0] if kind in ("top", "bot") else q[1])
                     for q in cl], np.float64)
    shift = float(np.percentile(r_cl, OUTWARD_PCT if kind in ("top", "lef")
                                else 100.0 - OUTWARD_PCT))
    px, py, vx, vy = cand
    cand = ((px, py + shift, vx, vy) if kind in ("top", "bot")
            else (px + shift, py, vx, vy))
    return cand, cl


def _line_dev(l, k):
    """Deviation of a side line from the scan axis, in radians."""
    vx, vy = l[2], l[3]
    if k == "h":
        if vx < 0:
            vx, vy = -vx, -vy
        return float(np.arctan2(vy, vx)) if vx > 1e-6 else None
    if vy < 0:
        vx, vy = -vx, -vy
    return float(-np.arctan2(vx, vy)) if vy > 1e-6 else None


def _side_devs(l_top, l_bot, l_lef, l_rig):
    """Deviation of every fitted side from the scan axis, in radians."""
    devs = []
    for l, k in ((l_top, "h"), (l_bot, "h"), (l_lef, "v"), (l_rig, "v")):
        if l is not None:
            d = _line_dev(l, k)
            if d is not None:
                devs.append(d)
    return devs


def _consensus_tilt(l_top, l_bot, l_lef, l_rig):
    """Common rotation angle of the fitted sides (pooled median with outlier
    trimming).  0.0 when nothing is available."""
    devs = _side_devs(l_top, l_bot, l_lef, l_rig)
    if not devs:
        return 0.0
    th = float(np.median(devs))
    keep = [d for d in devs if abs(d - th) < np.radians(CONSENSUS_TRIM_DEG)]
    if len(keep) >= 2:
        th = float(np.median(keep))
    return float(np.clip(th, -np.radians(CONSENSUS_MAX_DEG),
                         np.radians(CONSENSUS_MAX_DEG)))


def _local_peak(prof, i, w):
    """True when prof[i] is the maximum within +-w samples."""
    lo, hi = max(0, i - w), min(prof.size, i + w + 1)
    return prof[i] >= prof[lo:hi].max() - 1e-6


# --------------------------------------------------------------------------
# per-scan-line image-edge search
# --------------------------------------------------------------------------

def _profile_edge(gray, prof, base, outside_low, paper, inset, kind, coord,
                  m_hint=None, box_edge=None, band_min=BAND_EDGE_MIN,
                  m_win=None, need_step=False, anchor_win=None, search=None,
                  prof_alt=None, card_ref=True, ghost=False):
    """Find the image-area edge on one scan line.

    Steps:
      1. anchor = outermost gradient peak seen from the page (with a faint
         bar retry and an internal-content back-search);
      2. local paper level from the strip beyond the anchor;
      3. re-anchor on a faint outer print edge when the strong anchor sits
         well inside the rough-box edge (ghost: lower trigger,
         softer bar, and a no-album-page p10 test on the strip between);
      4. a "borderless" anchor is verified against the margin consensus
         [the outside band may be the print's white border when
         the inside is decisively dark and the anchor sits ON the content];
      5. image-edge search inside the anchor: position-verified candidate
         nearest m_hint, else the outermost candidate passing all semantic
         checks (ghost: only real content steps are trusted);
      6. nothing found and dark right inside the anchor => borderless edge;
      6b. ghost accept: a deep re-anchored outline over a uniform
         light field IS the image edge;
      7. wide bright strip behind the anchor + local refinement.

    Returns (edge_t, anchor_t); either may be None."""
    H, W = gray.shape
    if search is None or search < inset:
        search = inset
    deep = search > inset
    mi = max(1, int(round(scale_of(gray))))
    n = prof.size
    if n < 12 * mi:
        return None, None
    dim = H if kind in ("top", "bot") else W
    sgn = 1 if outside_low else -1              # inward direction
    peak_w = max(1, int(round(1.5 * mi)))

    def seg_stats(t0, t1):
        t0 = max(0, int(t0)); t1 = min(dim, int(t1))
        if t1 - t0 < 1:
            return None
        if kind in ("top", "bot"):
            c0 = max(0, coord - 3 * mi); c1 = min(W, coord + 3 * mi)
            seg = gray[t0:t1, c0:c1]
        else:
            r0 = max(0, coord - 3 * mi); r1 = min(H, coord + 3 * mi)
            seg = gray[r0:r1, t0:t1]
        if seg.size == 0:
            return None
        return float(seg.mean()), float(seg.std())

    # The bot/rig outer band sits one pixel further into the edge ramp than
    # the top/lef one (half-open slices). The thresholds are tuned against
    # these bands as they are; making them symmetric needs retuning.
    def outer_stats(t):                         # band toward the album page
        a, b = ((t - 5 * mi, t - 1 * mi) if sgn > 0
                else (t + 1 * mi, t + 5 * mi))
        return seg_stats(a, b)

    def inner_stats(t):                         # band toward photo interior
        a, b = ((t + 2 * mi, t + 6 * mi) if sgn > 0
                else (t - 6 * mi, t - 2 * mi))
        return seg_stats(a, b)

    def seg_pct(t0, t1, q=10.0):
        """Low percentile of a band - unlike the mean it is not rescued by a
        bright majority, so a band that is part page and part photo fails."""
        t0 = max(0, int(t0)); t1 = min(dim, int(t1))
        if t1 - t0 < 4 * mi:
            return None
        if kind in ("top", "bot"):
            c0 = max(0, coord - 3 * mi); c1 = min(W, coord + 3 * mi)
            seg = gray[t0:t1, c0:c1]
        else:
            r0 = max(0, coord - 3 * mi); r1 = min(H, coord + 3 * mi)
            seg = gray[r0:r1, t0:t1]
        if seg.size < 20 * mi * mi:
            return None
        return float(np.percentile(seg, q))

    def outer_run_ok(i, strict=False):
        """Everything from the page-side end of the window out to a candidate
        anchor must look like bare album page."""
        if sgn > 0:
            u0, u1 = base, base + i - 3 * mi
        else:
            u0, u1 = base + i + 3 * mi, base + n
        p10 = seg_pct(u0, u1)
        if p10 is None:
            return not strict           # unmeasurable: judge nothing
        return p10 >= paper - PAPER_SLACK

    # 1) anchor: outermost gradient peak scanning from the page side
    rng = range(n) if sgn > 0 else range(n - 1, -1, -1)
    if anchor_win is not None:
        a_lo = max(0, int(round(anchor_win[0] - base)))
        a_hi = min(n - 1, int(round(anchor_win[1] - base)))
        if a_hi < a_lo:
            return None, None
        rng = range(a_lo, a_hi + 1) if sgn > 0 else range(a_hi, a_lo - 1, -1)
    ai = -1
    strong_first = -1
    for i in rng:
        if prof[i] >= EDGE_MIN_THR:
            if strong_first < 0:
                strong_first = i
            if anchor_win is not None or outer_run_ok(i):
                ai = i
                break
    if ai < 0 and strong_first >= 0:
        back = (range(0, strong_first) if sgn > 0
                else range(n - 1, strong_first, -1))
        for i in back:
            if prof[i] >= FAINT_ANCHOR_THR and _local_peak(prof, i, peak_w) \
                    and outer_run_ok(i, strict=True):
                ai = i
                break
        if ai < 0:
            ai = strong_first           # nothing better: take the strong edge
    if ai < 0:
        for i in rng:
            if prof[i] >= FAINT_ANCHOR_THR and _local_peak(prof, i, peak_w):
                ai = i
                break
        if ai < 0:
            return None, None
    lo, hi = max(0, ai - 2 * mi), min(n, ai + 2 * mi + 1)
    if anchor_win is not None:
        lo = max(lo, int(round(anchor_win[0] - base)))
        hi = min(hi, int(round(anchor_win[1] - base)) + 1)
    ai = lo + int(np.argmax(prof[lo:hi]))
    anchor = base + ai

    # 2) local paper level from the page strip beyond the print outline
    def local_paper_level():
        if sgn > 0:
            u0, u1 = base, base + ai - 3 * mi
        else:
            u0, u1 = base + ai + 3 * mi, base + n
        u0 = max(0, int(u0)); u1 = min(dim, int(u1))
        if u1 - u0 < 8 * mi:
            return paper
        if kind in ("top", "bot"):
            c0 = max(0, coord - 3 * mi); c1 = min(W, coord + 3 * mi)
            seg = gray[u0:u1, c0:c1]
        else:
            r0 = max(0, coord - 3 * mi); r1 = min(H, coord + 3 * mi)
            seg = gray[r0:r1, u0:u1]
        if seg.size < 20 * mi * mi:
            return paper
        pl = float(np.percentile(seg, 90))
        return float(np.clip(pl, paper - LOCAL_PAPER_DELTA_LO,
                             paper + LOCAL_PAPER_DELTA_HI))

    paper_loc = local_paper_level()

    def margin_ref_level(a_i):
        """Reference level for "is the outer side still margin?"."""
        if not card_ref:
            return paper_loc
        if sgn > 0:
            u0, u1 = base + a_i + 2 * mi, base + a_i + 8 * mi
        else:
            u0, u1 = base + a_i - 8 * mi, base + a_i - 2 * mi
        st = seg_stats(u0, u1)
        if st is None:
            return paper_loc
        lvl, sd = st
        used = not (sd > MARGIN_REF_STD or
                    lvl > paper_loc - MARGIN_REF_MIN_DROP or
                    lvl < paper_loc - MARGIN_REF_DROP)
        _MARGIN_REF_LOG.append(bool(used))
        return float(lvl) if used else paper_loc

    ai0 = ai

    # 3) re-anchor on a faint outer print edge when the strong anchor sits
    # well inside the rough-box edge (it is an internal content edge then)
    if box_edge is not None:
        box_i = box_edge - base
        din = (ai - box_i) if sgn > 0 else (box_i - ai)
        # a ghost retrace lowers the trigger: its purpose is
        # hunting an outline the strong bar walked past
        thr_din = max(6 * mi, int((GHOST_DIN_FRAC if ghost
                                   else ANCHOR_NEAR_BOX) * inset))
        if din > thr_din:
            cap = max(6 * mi, int(inset))
            if sgn > 0:
                lo_j = max(0, box_i - cap)
                hi_j = max(0, ai - 4 * mi)
                order = range(hi_j, lo_j - 1, -1)
            else:
                hi_j = min(n - 1, box_i + cap)
                lo_j = min(n - 1, ai + 4 * mi)
                order = range(lo_j, hi_j + 1)
            best = None
            for j in order:
                if prof[j] < FAINT_ANCHOR_THR:
                    continue
                if not _local_peak(prof, j, peak_w):
                    continue
                if sgn > 0:
                    st = seg_stats(base + j + 2 * mi, base + ai - 2 * mi)
                else:
                    st = seg_stats(base + ai + 2 * mi, base + j - 2 * mi)
                if st is None:
                    continue
                bm, bstd = st
                if bm >= paper_loc - (MARGIN_BAND_SLACK if ghost
                                      else PAPER_SLACK) and \
                        bstd <= 2 * MARGIN_BAND_STD:
                    if ghost:
                        # the strip between must NOT contain album
                        # page: a real outline has border/card behind it, a
                        # page speck has page (p10 separates them)
                        p10 = seg_pct(base + j + 2 * mi, base + ai - 2 * mi) \
                            if sgn > 0 else \
                            seg_pct(base + ai + 2 * mi, base + j - 2 * mi)
                        if p10 is None or \
                                p10 > paper_loc - LOCAL_PAPER_DELTA_LO:
                            continue
                    best = j                  # keep scanning: take outermost
            if best is not None:
                ai = best
                anchor = base + ai

    if ai != ai0:
        paper_loc = local_paper_level()
        ai0 = ai

    def borderless_at(idx):
        """Probe: content (dark) right inside, page right outside.

        The page-bright bar for the outside band has a gated second chance:
        a UNIFORM bright-ish strip is accepted as the print's white border
        behind a borderless edge whose outline was too soft to anchor on,
        when the inside is decisively dark. An anchor far OUTSIDE the Otsu
        content box is not on a borderless image edge -- what is "right
        inside" is then the print's own white border reading a few levels
        under the paper bar -- so such a side must meet the decisive-dark
        bar, or the claim is refused and the geometric prior places the
        edge. A genuine borderless anchor sits ON the content and is
        untouched."""
        t = base + idx
        ost = outer_stats(t)
        ist = inner_stats(t)
        if ost is None or ist is None:
            return False
        if not (ist[0] < paper_loc - PAPER_SLACK and
                ost[0] - ist[0] >= POL_MARGIN):
            return False
        if ost[0] >= paper_loc - PAPER_SLACK:
            # A print's white border can read a few levels under the paper
            # bar, and the borderless probe then fires on an ordinary
            # bordered side, so an anchor far outside the Otsu content box
            # must meet the decisive-dark bar.
            if box_edge is not None:
                din_b = ((base + idx) - box_edge) if sgn > 0 \
                    else (box_edge - (base + idx))
                if din_b < -max(6 * mi, int(GHOST_DIN_FRAC * inset)) and \
                        ist[0] >= paper_loc - (INNER_DARK_SLACK +
                                               STRONG_DARK_EXTRA):
                    return False
            return True
        a, b = ((t - 8 * mi, t - 2 * mi) if sgn > 0
                else (t + 2 * mi, t + 8 * mi))
        wide = seg_stats(a, b)
        return (wide is not None and
                ist[0] < paper_loc - (INNER_DARK_SLACK +
                                      STRONG_DARK_EXTRA) and
                ost[0] >= paper_loc - MARGIN_BAND_SLACK and
                wide[0] >= paper_loc - MARGIN_BAND_SLACK and
                ost[1] <= 2 * MARGIN_BAND_STD and
                wide[1] <= 2 * MARGIN_BAND_STD)

    margin_ref = margin_ref_level(ai)

    # 4) borderless claim vs margin consensus
    allow_borderless = True
    if m_hint is not None and m_hint >= 3 * mi and borderless_at(ai):
        reach = (int(MWIN_HI * m_hint) if ghost
                 else int(m_hint) + 3 * int(inset))
        if sgn > 0:
            j_hi = max(0, ai - 2 * mi)
            j_lo = max(0, ai - reach)
        else:
            j_lo = min(n - 1, ai + 2 * mi)
            j_hi = min(n - 1, ai + reach)
        if box_edge is not None:
            lim_o = int(1.5 * inset) + 2 * mi
            if sgn > 0:
                j_lo = max(j_lo, (box_edge - base) - lim_o)
            else:
                j_hi = min(j_hi, (box_edge - base) + lim_o)
        order_j = (range(j_hi, j_lo - 1, -1) if sgn > 0
                   else range(j_lo, j_hi + 1))
        best_j = None
        for j in order_j:
            if prof[j] < FAINT_ANCHOR_THR or not _local_peak(prof, j, peak_w):
                continue
            if sgn > 0:
                st = seg_stats(base + j + 2 * mi, base + ai - 2 * mi)
            else:
                st = seg_stats(base + ai + 2 * mi, base + j - 2 * mi)
            if st is None:
                continue
            if st[0] >= paper_loc - PAPER_SLACK and \
                    st[1] <= 2 * MARGIN_BAND_STD:
                if ghost:
                    # the strip must not contain album page (p10, as in
                    # step 3)
                    p10 = seg_pct(base + j + 2 * mi, base + ai - 2 * mi) \
                        if sgn > 0 else \
                        seg_pct(base + ai + 2 * mi, base + j - 2 * mi)
                    if p10 is None or \
                            p10 > paper_loc - LOCAL_PAPER_DELTA_LO:
                        continue
                best_j = j                    # keep scanning: outermost wins
        if best_j is not None:
            ai = best_j
            anchor = base + ai
        else:
            allow_borderless = False   # consensus margin contradicts it

    if ai != ai0:
        paper_loc = local_paper_level()

    if sgn > 0:
        b0, b1 = ai + 3 * mi, min(n, ai + search)
    else:
        b0, b1 = max(0, ai - search), ai - 3 * mi
    band = prof[b0:b1] if b1 - b0 >= 6 * mi else None

    def corridor_ok(t, ramp=False):
        """The corridor from the print outline to the candidate must be ONE
        uniform margin (no brightness step in either direction; the upward
        bound keeps headroom for scanner shading).  `ramp` judges the
        corridor by its departure from a straight line instead; only the
        toned-card retry uses it."""
        a0 = anchor + sgn * 2 * mi
        a1 = t - sgn * 2 * mi
        lo_c, hi_c = (a0, a1) if sgn > 0 else (a1, a0)
        if hi_c - lo_c < 6 * mi:
            return True                     # too short to judge
        if ramp:
            step_c = max(1, mi)
            xs, ys = [], []
            q = lo_c
            while q + step_c <= hi_c:
                st = seg_stats(q, q + step_c)
                if st is not None:
                    xs.append(float(q)); ys.append(st[0])
                q += step_c
            if len(xs) >= 4:
                xa = np.asarray(xs); ya = np.asarray(ys)
                A = np.stack([xa, np.ones(xa.size)], axis=1)
                (k_, b_), *_ = np.linalg.lstsq(A, ya, rcond=None)
                return bool(np.abs(ya - (k_ * xa + b_)).max()
                            <= CORRIDOR_RESID)
        mid = lo_c + (hi_c - lo_c) // 2
        s_lo = seg_stats(lo_c, mid)
        s_hi = seg_stats(mid, hi_c)
        if s_lo is None or s_hi is None:
            return True
        # lo_c/hi_c are sorted by COORDINATE; on bot/rig the corridor runs
        # the other way, so the halves swap roles
        so, si = (s_lo, s_hi) if sgn > 0 else (s_hi, s_lo)
        return (si[0] >= so[0] - CORRIDOR_STEP and
                si[0] <= so[0] + CORRIDOR_STEP_UP)

    def alt_ok(c):
        """True unless the candidate's step is chromatic."""
        if prof_alt is None:
            return True
        if c < 0 or c >= prof_alt.size or c >= prof.size:
            return True
        lum = float(prof[c])
        if lum <= 0:
            return True
        return float(prof_alt[c]) >= ALT_GRAD_RATIO * lum

    def strict_search(ref=None, ramp=False):
        """Outermost candidate passing all semantic checks."""
        if ref is None:
            ref = paper_loc
        if band is None:
            return None
        dist_cap = (None if deep or m_hint is None else MWIN_HI * m_hint)
        order = range(band.size) if sgn > 0 else range(band.size - 1, -1, -1)
        for j in order:
            if band[j] < band_min:            # same bar in weak mode
                continue
            if not _local_peak(band, j, peak_w):
                continue
            c = b0 + j
            if abs(c - ai) < HALO_GUARD * mi:
                continue                        # JPEG halo hugging print edge
            if not alt_ok(c):
                continue
            if dist_cap is not None:
                dist = (c - ai) if sgn > 0 else (ai - c)
                if dist > dist_cap:
                    continue
            t = base + c
            ost, ist = outer_stats(t), inner_stats(t)
            if ost is None or ist is None:
                continue
            om, im, istd = ost[0], ist[0], ist[1]
            if om < ref - PAPER_SLACK:
                continue                        # outer side is not paper
            textured = istd >= INNER_TEXTURE_STD
            strong_dark = im < paper_loc - (INNER_DARK_SLACK +
                                            STRONG_DARK_EXTRA)
            dark = im < paper_loc - INNER_DARK_SLACK
            if ghost:
                # a ghost retrace crosses a light picture field
                # (a flat white wall passes as card stock), so only a real
                # content step with a paper-bright outer side is trusted
                if om - im < POL_MARGIN or om < ref - STEP_PAPER_SLACK:
                    continue
            elif not (dark or textured):
                if om - im < POL_MARGIN:
                    continue
                if om < ref - STEP_PAPER_SLACK:
                    continue        # a bright image strip is no margin
            if not strong_dark and abs(c - ai) >= 8 * mi:
                if sgn > 0:
                    g0, g1 = c - 7 * mi, c - 2 * mi   # margin right before
                else:                                  # the edge
                    g0, g1 = c + 2 * mi, c + 7 * mi
                mst = seg_stats(base + g0, base + g1)
                if mst is None:
                    continue
                bm, bstd = mst
                if bm < ref - MARGIN_BAND_SLACK or \
                        bstd > MARGIN_BAND_STD:
                    continue
            if not corridor_ok(t, ramp=ramp):
                continue
            return t
        return None

    # 5) image-edge search inside the anchor
    edge = None
    if m_hint is not None and band is not None:
        # Position-verified search: LOCAL-PEAK candidate closest to m_hint.
        lo_f, hi_f = m_win if m_win is not None else (MWIN_LO, MWIN_HI)
        lo_d, hi_d = lo_f * m_hint, hi_f * m_hint
        if deep:
            # in deep mode the upper bound is the deep reach
            hi_d = max(hi_d, float(search))
        best = None
        order = range(band.size) if sgn > 0 else range(band.size - 1, -1, -1)
        for j in order:
            if band[j] < band_min:
                continue
            if not _local_peak(band, j, peak_w):
                continue
            c = b0 + j
            if abs(c - ai) < HALO_GUARD * mi:
                continue
            if not alt_ok(c):
                continue
            dist = (c - ai) if sgn > 0 else (ai - c)   # inward distance
            if not (lo_d <= dist <= hi_d):
                continue
            ost = outer_stats(base + c)
            if ost is None or ost[0] < paper_loc - PAPER_SLACK:
                continue                # the outer side must still be paper
            ist = inner_stats(base + c)
            if ist is None:
                continue
            om, im, istd = ost[0], ist[0], ist[1]
            if ghost:
                if om - im < POL_MARGIN or \
                        om < paper_loc - STEP_PAPER_SLACK:
                    continue        # see strict_search
            elif need_step:
                if om - im < POL_MARGIN:
                    continue                    # no content step => noise
            elif not (im < paper_loc - INNER_DARK_SLACK or
                      istd >= INNER_TEXTURE_STD):
                if om - im < POL_MARGIN or om < paper_loc - STEP_PAPER_SLACK:
                    continue
            if not corridor_ok(base + c):
                continue
            d = abs(dist - m_hint)
            if best is None or d < best[0]:
                best = (d, base + c)
        if best is not None:
            edge = best[1]
    if edge is None:
        edge = strict_search()
    if margin_ref < paper_loc and \
            (edge is None or abs(edge - anchor) <= 2 * mi):
        # Second chance on a toned card only
        e2 = strict_search(ref=margin_ref, ramp=True)
        if e2 is not None:
            edge = e2

    # 6) no white margin: the anchor itself is the image edge (gated above)
    if edge is None and allow_borderless and borderless_at(ai):
        edge = base + ai

    # 6b) ghost accept: the anchor sits on the print outline and
    # the image area behind it is a uniform light field (a white wall in
    # the picture) - the emulsion edge is the anchor itself
    if edge is None and ghost and box_edge is not None:
        box_i = box_edge - base
        din_g = (ai - box_i) if sgn > 0 else (box_i - ai)
        if din_g > max(6 * mi, int(GHOST_DIN_FRAC * inset)):
            u0, u1 = ((base + ai + 2 * mi, base + ai + 8 * mi) if sgn > 0
                      else (base + ai - 8 * mi, base + ai - 2 * mi))
            st_in = seg_stats(u0, u1)
            ost_g = outer_stats(base + ai)
            if (st_in is not None and ost_g is not None and
                    st_in[0] >= paper_loc - MARGIN_BAND_SLACK and
                    st_in[1] <= 2 * MARGIN_BAND_STD and
                    ost_g[0] >= paper_loc - MARGIN_BAND_SLACK):
                edge = base + ai

    # 7) wide bright strip behind the anchor + local refinement
    if edge is None and m_hint is not None and not deep and not ghost:
        step = 4 * mi
        end = base + ai + sgn * int(WIDE_RUN_CAP * inset)
        end = min(end, base + n - 1) if sgn > 0 else max(end, base)
        run = 0
        pos = base + ai
        while True:
            seg_a = pos if sgn > 0 else pos - step
            seg_b = pos + step if sgn > 0 else pos
            if (sgn > 0 and seg_b > end) or (sgn < 0 and seg_a < end):
                break
            st = seg_stats(seg_a, seg_b)
            if st is None or st[0] < paper_loc - PAPER_SLACK or \
                    st[1] > 2 * MARGIN_BAND_STD:
                break
            run += step
            pos += sgn * step
        if run > WIDE_RUN_FACTOR * inset:
            off = int(round(min(max(m_hint, 3.0 * mi), float(inset))))
            pi = ai + sgn * off
            w = 3 * mi
            if sgn > 0:
                lo_r = max(ai + 2 * mi, pi - w)
                hi_r = min(ai + int(inset), pi + w)
            else:
                lo_r = max(ai - int(inset), pi - w)
                hi_r = min(ai - 2 * mi, pi + w)
            lo_r = max(0, lo_r)
            hi_r = min(n - 1, hi_r)
            best_r = None
            if lo_r <= hi_r:
                for j in range(lo_r, hi_r + 1):
                    if prof[j] < EDGE_REFINE_THR:
                        continue
                    if not _local_peak(prof, j, peak_w):
                        continue
                    if best_r is None or prof[j] > prof[best_r]:
                        best_r = j
            edge_i = best_r if best_r is not None else pi
            edge = int(min(max(base + edge_i, base), base + n - 1))
    return edge, anchor


def trace_side(gray, gdir, paper, bbox, kind, m_hint=None, weak=False,
               reach=None, gdir_alt=None, ghost=False):
    """Trace one photo side; fit lines through image-edge samples and
    print-edge anchors.  Returns (image_line, anchor_line, pts, anchors).

    `ghost` turns the trace into a ghost-outline retrace: the
    card reference is disabled, the step-3 re-anchor trigger is lowered and
    made page-free, both edge searches trust only real content steps, and
    step 6b accepts a deep outline over a uniform light field."""
    H, W = gray.shape
    s = scale_of(gray)
    mi = max(1, int(round(s)))
    bx, by, bw, bh = bbox
    margin = int(max(40 * s, MARGIN_FRAC * max(bw, bh)))
    ins_x = max(6 * mi, int(INSET_FRAC * bw))
    ins_y = max(6 * mi, int(INSET_FRAC * bh))
    cap_px = INSET_CAP * s
    inset = int(max(12 * s, min(cap_px, INSET_MAX_FRAC * max(bw, bh))))
    search = inset if reach is None else int(
        min(max(inset, reach), SEARCH_HARD_FRAC * max(bw, bh)))
    guard = int(BORDER_GUARD * mi)
    pts = []
    anchors = []
    pe_kw = (dict(band_min=WEAK_BAND_MIN, m_win=MWIN_WEAK, need_step=True)
             if weak else {})

    if kind in ("top", "bot"):
        if kind == "top":
            w_lo, w_hi = max(0, by - margin), by + ins_y
            box_t = by
            outside_low = True
        else:
            w_lo, w_hi = by + bh - ins_y, min(H, by + bh + margin)
            box_t = by + bh - 1
            outside_low = False
        s0 = bx + int(EDGE_SKIP_FRAC * bw)
        s1 = bx + bw - int(EDGE_SKIP_FRAC * bw)
        row_prof, lim = False, H
    else:
        if kind == "lef":
            w_lo, w_hi = max(0, bx - margin), bx + ins_x
            box_t = bx
            outside_low = True
        else:
            w_lo, w_hi = bx + bw - ins_x, min(W, bx + bw + margin)
            box_t = bx + bw - 1
            outside_low = False
        s0 = by + int(EDGE_SKIP_FRAC * bh)
        s1 = by + bh - int(EDGE_SKIP_FRAC * bh)
        row_prof, lim = True, W

    if w_lo <= 0 or w_hi >= lim:
        guard = mi     # the picture runs off the scan; keep the samples

    step = max(1, (s1 - s0) // SAMPLES_PER_SIDE)
    coords = list(range(s0, s1 + 1, step))

    card_ref = not ghost

    def run(c, aw=None):
        prof = gdir[c, w_lo:w_hi] if row_prof else gdir[w_lo:w_hi, c]
        p_alt = None
        if gdir_alt is not None:
            p_alt = (gdir_alt[c, w_lo:w_hi] if row_prof
                     else gdir_alt[w_lo:w_hi, c])
        return _profile_edge(gray, prof, w_lo, outside_low, paper, inset,
                             kind, c, m_hint, box_edge=box_t,
                             anchor_win=aw, search=search, prof_alt=p_alt,
                             card_ref=card_ref, ghost=ghost, **pe_kw)

    del _MARGIN_REF_LOG[:]
    res = [(c,) + tuple(run(c)) for c in coords]
    if _MARGIN_REF_LOG and card_ref:
        # the card reference must hold on a majority of scan lines
        n_all = len(_MARGIN_REF_LOG)
        n_use = sum(_MARGIN_REF_LOG)
        if 0 < n_use < MARGIN_REF_SIDE_FRAC * n_all:
            card_ref = False
            del _MARGIN_REF_LOG[:]
            res = [(c,) + tuple(run(c)) for c in coords]

    # Per-side anchor consensus: contamination is always inward, so the
    # consensus is read off the OUTWARD end of the anchor distribution.
    av = np.array([a for _, _, a in res if a is not None], float)
    ac = np.array([c for c, _, a in res if a is not None], float)
    if av.size >= 6:
        ref = float(np.percentile(av, 20.0 if outside_low else 80.0))
        tol = max(6.0 * s, 0.25 * inset)
        core = np.abs(av - ref) <= tol
        cl = None
        if int(core.sum()) >= MIN_POINTS:
            idx = np.where(core)[0]
            cpts = [((ac[i], av[i]) if kind in ("top", "bot")
                     else (av[i], ac[i])) for i in idx]
            cl = _fit_line_robust(cpts, kind, FIT_TOL * s)
        span = abs(coords[-1] - coords[0]) if len(coords) > 1 else 0
        tol_c = tol if cl is not None else \
            tol + 0.5 * float(np.tan(np.radians(MAX_TILT_DEG))) * span
        inward = 1.0 if outside_low else -1.0
        for i, (c, t, a) in enumerate(res):
            if a is None:
                continue
            e = line_off_at(cl, kind, c) if cl is not None else ref
            if inward * (a - e) <= tol_c:
                continue                # not an inward outlier: leave alone
            res[i] = (c,) + tuple(run(c, (e - tol, e + tol)))

    for c, t, a in res:
        if a is not None and guard <= a <= lim - 1 - guard:
            anchors.append((float(c), float(a)) if kind in ("top", "bot")
                           else (float(a), float(c)))
        if t is None or t < guard or t > lim - 1 - guard:
            continue
        pts.append((float(c), float(t)) if kind in ("top", "bot")
                   else (float(t), float(c)))

    line = None
    if len(pts) >= MIN_POINTS:
        line = _fit_line_robust(pts, kind, FIT_TOL * s)
    aline = None
    if len(anchors) >= MIN_POINTS:
        aline = _fit_line_robust(anchors, kind, FIT_TOL * s)
    return line, aline, pts, anchors


# --------------------------------------------------------------------------
# quad assembly
# --------------------------------------------------------------------------

def _rectify_sides(l_top, l_bot, l_lef, l_rig):
    """Snap all four side lines to one common rotation angle."""
    devs = _side_devs(l_top, l_bot, l_lef, l_rig)
    if not devs:
        return l_top, l_bot, l_lef, l_rig, 0.0
    th = _consensus_tilt(l_top, l_bot, l_lef, l_rig)
    if max(abs(d - th) for d in devs) < np.radians(0.02):
        return l_top, l_bot, l_lef, l_rig, th

    def rot(l, dth):
        px, py, vx, vy = l
        c, sn = np.cos(dth), np.sin(dth)
        return (px, py, vx * c + vy * sn, -vx * sn + vy * c)

    def pull(l, k):
        if l is None:
            return None
        d = _line_dev(l, k)
        if d is None or abs(d - th) < np.radians(0.005):
            return l
        return rot(l, d - th)

    return (pull(l_top, "h"), pull(l_bot, "h"),
            pull(l_lef, "v"), pull(l_rig, "v"), th)


def line_intersect(l1, l2):
    """Intersection point of two lines given as (px, py, vx, vy)."""
    x1, y1, dx1, dy1 = l1
    x2, y2, dx2, dy2 = l2
    det = dx1 * dy2 - dy1 * dx2
    if abs(det) < 1e-6:
        return None
    t = ((x2 - x1) * dy2 - (y2 - y1) * dx2) / det
    return (x1 + t * dx1, y1 + t * dy1)


def line_off_at(l, kind, t):
    """Offset coordinate of a side line at position `t` along the side."""
    px, py, vx, vy = l
    if kind in ("top", "bot"):
        return py if abs(vx) < 1e-9 else py + (vy / vx) * (t - px)
    return px if abs(vy) < 1e-9 else px + (vx / vy) * (t - py)


def _back_quad_lines(apts, anch, order, s):
    """Back mode: the ANGLE comes from all sides jointly (pooled consensus),
    the OFFSET is the OUTWARD envelope of the anchor scatter (tangent to the
    card edge).  Sides with too few anchors fall through to the bbox."""
    out = {"top": -1.0, "bot": 1.0, "lef": -1.0, "rig": 1.0}
    temps = {}
    for k in order:
        p = apts.get(k) or []
        temps[k] = None
        if len(p) < WEAK_MIN_POINTS:
            continue
        a = np.asarray(p, np.float64)
        t, off = ((a[:, 0], a[:, 1]) if k in ("top", "bot")
                  else (a[:, 1], a[:, 0]))
        if np.unique(t).size < WEAK_MIN_POINTS:
            continue
        i, j = np.triu_indices(len(t), 1)
        dt = t[j] - t[i]
        ok = np.abs(dt) > 1e-6
        if not ok.any():
            continue
        slope = float(np.median((off[j] - off[i])[ok] / dt[ok]))
        if abs(np.degrees(np.arctan(slope))) > MAX_TILT_DEG:
            slope = 0.0        # a wild seed must not poison the consensus
        tc = float(np.median(t))
        b = float(np.median(off - slope * t))
        temps[k] = ((tc, b, 1.0, slope) if k in ("top", "bot")
                    else (b, tc, slope, 1.0))
    th = _consensus_tilt(temps["top"], temps["bot"],
                         temps["lef"], temps["rig"])
    sl_h = float(np.tan(th))
    sl_v = float(-np.tan(th))
    lines, pts, src = {}, {}, {}
    for k in order:
        p = apts.get(k) or []
        if temps[k] is not None:
            slope = sl_h if k in ("top", "bot") else sl_v
            a = np.asarray(p, np.float64)
            t, off = ((a[:, 0], a[:, 1]) if k in ("top", "bot")
                      else (a[:, 1], a[:, 0]))
            r = out[k] * (off - slope * t)          # residual, outward > 0
            med = float(np.median(r))
            env = float(np.percentile(r, BACK_TANGENT_PCT))
            env = max(env, med)                     # never inside the bulk
            env = med + min(env - med, BACK_ENV_MAX_EXC * s)
            b = out[k] * env
            tc = float(np.median(t))
            oc = slope * tc + b
            lines[k] = ((tc, oc, 1.0, slope) if k in ("top", "bot")
                        else (oc, tc, slope, 1.0))
            pts[k] = []
            src[k] = "back-envelope (tangent to the card edge)"
        elif anch.get(k) is not None:
            lines[k] = anch[k]
            pts[k] = []
            src[k] = "back-anchor"
        else:
            lines[k] = None
            pts[k] = []
            src[k] = "back-bbox"
    return lines, pts, src


def detect_quad(gray, gx, gy, paper, bbox, want_dbg=False, prov=None,
                back=False, gx_alt=None, gy_alt=None):
    """Detect the image-area quad of one photo.

    Passes: 1 trace all sides; 1b outward-group / smear refit; 2 re-trace
    with the margin hint; 2b ghost-outline retrace; 3 weak
    re-trace; 3d deep trace; 4 few-point rescue; 5 minority-cluster
    replacement; then solid-anchor completion, the clean-margin
    guard, the anchor+m_est prior, the back-mode envelope, rectification
    and corner intersection."""
    s = scale_of(gray)
    bx, by, bw, bh = bbox
    gdir = {"top": gy, "bot": gy, "lef": gx, "rig": gx}
    gdir_alt = {"top": gy_alt, "bot": gy_alt,
                "lef": gx_alt, "rig": gx_alt}
    order = ("top", "bot", "lef", "rig")

    lines, pts, anch, apts = {}, {}, {}, {}
    src = {k: "none" for k in order}
    for k in order:                                     # pass 1
        lines[k], anch[k], pts[k], apts[k] = trace_side(
            gray, gdir[k], paper, bbox, k, gdir_alt=gdir_alt[k])
        if lines[k] is not None:
            src[k] = "pass1"

    lim = SIDE_INSET_LIM * max(bw, bh)

    def side_inset(l, a, k, lim_k=None):
        """Margin between the image edge and the print outline, both lines
        evaluated at the SAME position along the side."""
        if l is None or a is None:
            return None
        t = l[0] if k in ("top", "bot") else l[1]
        off_l = l[1] if k in ("top", "bot") else l[0]
        off_a = line_off_at(a, k, t)
        d = (off_l - off_a) if k in ("top", "lef") else (off_a - off_l)
        return d if 2 * s < d < (lim if lim_k is None else lim_k) else None

    def expected_samples(k):
        """Scan lines trace_side offers on this side."""
        length = ((bw if k in ("top", "bot") else bh)
                  * (1.0 - 2 * EDGE_SKIP_FRAC))
        step = max(1, int(length) // SAMPLES_PER_SIDE)
        return int(length) // step + 1

    def deep_ok_thin(line, samples, k):
        """Signature of a cut print edge on a thin side."""
        if line is None or not samples:
            return False
        a = np.asarray(samples, np.float64)
        tt = a[:, 0] if k in ("top", "bot") else a[:, 1]
        off = a[:, 1] if k in ("top", "bot") else a[:, 0]
        res = np.abs(off - np.array([line_off_at(line, k, x) for x in tt]))
        inl = res < FIT_TOL * s
        if int(inl.sum()) < DEEP_THIN_MIN:
            return False
        if float(np.percentile(res[inl], 90)) / s > DEEP_THIN_SPREAD:
            return False
        length = ((bw if k in ("top", "bot") else bh) *
                  (1.0 - 2 * EDGE_SKIP_FRAC))
        core = tt[inl]
        if float(core.max() - core.min()) < DEEP_THIN_SPAN * length:
            return False
        st = line_strength(line, [tuple(v) for v in a[inl]], k)
        return st is not None and st >= EDGE_MIN_THR

    def deep_ok(line, samples, k):
        """Evidence bar for a deep-trace candidate."""
        if line is None:
            return False
        n_exp = expected_samples(k)
        a = np.asarray(samples, np.float64)
        t = a[:, 0] if k in ("top", "bot") else a[:, 1]
        off = a[:, 1] if k in ("top", "bot") else a[:, 0]
        res = np.abs(off - np.array([line_off_at(line, k, tt) for tt in t]))
        inl = res < FIT_TOL * s
        if int(inl.sum()) < DEEP_CORE_FRAC * n_exp:
            return False
        return float(np.percentile(res[inl], 90)) / s <= DEEP_MAX_SPREAD

    def line_strength(line, samples, k):
        """Median gradient magnitude ON a line, at its own positions."""
        if line is None or not samples:
            return None
        Hg, Wg = gray.shape
        w = max(1, int(round(1.5 * s)))
        vals = []
        for (sx, sy) in samples:
            tt = sx if k in ("top", "bot") else sy
            off = line_off_at(line, k, tt)
            x, y = ((int(round(tt)), int(round(off))) if k in ("top", "bot")
                    else (int(round(off)), int(round(tt))))
            if k in ("top", "bot"):
                y0, y1, x0, x1 = y - w, y + w + 1, x, x + 1
            else:
                y0, y1, x0, x1 = y, y + 1, x - w, x + w + 1
            y0, y1 = max(0, y0), min(Hg, y1)
            x0, x1 = max(0, x0), min(Wg, x1)
            if y1 > y0 and x1 > x0:
                vals.append(float(gdir[k][y0:y1, x0:x1].max()))
        return float(np.median(vals)) if vals else None

    def weaker_evidence(cur_line, cur_pts, new_line, new_pts, k):
        """True when a well-supported line would be swapped for one with
        much less gradient behind it."""
        st_cur = line_strength(cur_line, cur_pts, k)
        st_new = line_strength(new_line, new_pts, k)
        return (st_cur is not None and st_new is not None and
                st_cur >= EDGE_MIN_THR and st_new < EVIDENCE_RATIO * st_cur)

    def fit_support(line, samples, k):
        """Fraction of samples lying within FIT_TOL of the line."""
        if line is None or len(samples) < 2:
            return 0.0
        t = np.array([p[0] if k in ("top", "bot") else p[1] for p in samples],
                     np.float64)
        o = np.array([p[1] if k in ("top", "bot") else p[0] for p in samples],
                     np.float64)
        r = np.abs(o - np.array([line_off_at(line, k, tt) for tt in t]))
        return float(np.mean(r < FIT_TOL * s))

    def margin_behind_ok(line, anch_line, k, samples):
        """Is the strip between the print outline and `line`
        a clean bright margin on most of the side?  The first
        SMEAR_FRINGE_SKIP of the gap is skipped (deckle fringe)."""
        if anch_line is None:
            return True
        Hg, Wg = gray.shape
        mi_q = max(1, int(round(s)))
        ok_n = tot = 0
        for (sx, sy) in samples:
            t = sx if k in ("top", "bot") else sy
            off_l = line_off_at(line, k, t)
            off_a = line_off_at(anch_line, k, t)
            lo, hi = ((off_a, off_l) if k in ("top", "lef")
                      else (off_l, off_a))
            gap_full = hi - lo
            if gap_full < 6 * mi_q:
                continue
            lo += max(2 * mi_q, int(SMEAR_FRINGE_SKIP * gap_full))
            hi -= 2 * mi_q
            if hi - lo < 2 * mi_q:
                continue
            tot += 1
            lo_i, hi_i = int(lo), int(hi)   # slice indices must be integers
            if k in ("top", "bot"):
                seg = gray[lo_i:hi_i, max(0, int(sx) - 3 * mi_q):
                                min(Wg, int(sx) + 3 * mi_q)]
            else:
                seg = gray[max(0, int(sy) - 3 * mi_q):
                                min(Hg, int(sy) + 3 * mi_q), lo_i:hi_i]
            if seg.size == 0:
                continue
            if float(seg.mean()) >= paper - PAPER_SLACK and \
                    float(seg.std()) <= 2 * MARGIN_BAND_STD:
                ok_n += 1
        return tot >= 3 and ok_n >= CLEAN_MARGIN_FRAC * tot

    # Rescue an ANCHOR line whose free fit was rejected.
    if sum(lines[k] is not None for k in order) >= 1:
        th_a = _consensus_tilt(lines["top"], lines["bot"],
                               lines["lef"], lines["rig"])
        for k in order:
            if anch[k] is not None or len(apts[k]) < MIN_POINTS:
                continue
            sl = (float(np.tan(th_a)) if k in ("top", "bot")
                  else float(-np.tan(th_a)))
            cand = _refit_at_angle(apts[k], k, sl, FIT_TOL * s)
            if cand is not None:
                anch[k] = cand

    pts1 = {k: list(pts[k]) for k in order}   # pass-1 samples, kept for pass 5
    dev = {k: side_inset(lines[k], anch[k], k) for k in order}

    # Rescue a side whose FREE slope broke the tilt cap.
    if sum(lines[k] is not None for k in order) >= 1:
        th0 = _consensus_tilt(lines["top"], lines["bot"],
                              lines["lef"], lines["rig"])
        for k in order:
            if lines[k] is not None or len(pts[k]) < MIN_POINTS:
                continue
            sl = (float(np.tan(th0)) if k in ("top", "bot")
                  else float(-np.tan(th0)))
            cand = _refit_at_angle(pts[k], k, sl, FIT_TOL * s)
            if cand is not None:
                lines[k] = cand
                dev[k] = side_inset(cand, anch[k], k)
                src[k] = "tilt-rescue"

    def tilt_anchored():
        """True when some side has pinned the consensus angle.

        A side measured on nearly every scan line, whose samples scatter by
        under a pixel at BASE, fixes the slope for the whole card.  Once
        that is true, an outward group is only being asked for an offset,
        and the span bar that protects the slope can come down."""
        for kk in order:
            if lines[kk] is None or not pts[kk]:
                continue
            if len(pts[kk]) < TILT_ANCHORED_FRAC * expected_samples(kk):
                continue
            a_p = np.asarray(pts[kk], np.float64)
            t_p = a_p[:, 0] if kk in ("top", "bot") else a_p[:, 1]
            o_p = a_p[:, 1] if kk in ("top", "bot") else a_p[:, 0]
            r_p = np.abs(o_p - np.array(
                [line_off_at(lines[kk], kk, x) for x in t_p]))
            if float(np.percentile(r_p, 90)) / s <= TILT_ANCHORED_SPREAD:
                return True
        return False

    # Pass 1b: outward-group preference / smear refit.
    if not back:
        span_min = (OUTWARD_MIN_SPAN_ANCHORED
                    if tilt_anchored()
                    else OUTWARD_MIN_SPAN)
        th_o = _consensus_tilt(lines["top"], lines["bot"],
                               lines["lef"], lines["rig"])
        for k in order:
            slope = (float(np.tan(th_o)) if k in ("top", "bot")
                     else float(-np.tan(th_o)))
            length = ((bw if k in ("top", "bot") else bh) *
                      (1.0 - 2 * EDGE_SKIP_FRAC))
            cand, cl, n_g = outward_cluster_at_angle(pts[k], k, slope,
                                                     FIT_TOL * s)
            smeared = False
            if (cand is None or n_g < 2) \
                    and lines[k] is not None \
                    and len(pts[k]) >= 2 * WEAK_MIN_POINTS \
                    and fit_support(lines[k], pts[k], k) < SMEAR_INLIER_FRAC:
                # route a: the outward tail beyond the dense band's flank
                cand, cl = smear_outward_fit(pts[k], k, slope, FIT_TOL * s)
                smeared = cand is not None
                if cand is None:
                    # route b: the fit's outliers
                    t_arr = np.array([p[0] if k in ("top", "bot") else p[1]
                                      for p in pts[k]], np.float64)
                    o_arr = np.array([p[1] if k in ("top", "bot") else p[0]
                                      for p in pts[k]], np.float64)
                    inl = np.abs(
                        o_arr - np.array([line_off_at(lines[k], k, t)
                                          for t in t_arr])) < FIT_TOL * s
                    out_pts = [pts[k][i] for i in np.where(~inl)[0]]
                    if len(out_pts) >= SMEAR_TAIL_MIN:
                        alt, alt_cl = cluster_at_angle(out_pts, k, slope,
                                                       FIT_TOL * s)
                        if alt is not None:
                            r_cl = np.array(
                                [(q[1] if k in ("top", "bot") else q[0]) -
                                 line_off_at(alt, k,
                                             q[0] if k in ("top", "bot")
                                             else q[1])
                                 for q in alt_cl], np.float64)
                            shift = float(np.percentile(
                                r_cl, OUTWARD_PCT if k in ("top", "lef")
                                else 100.0 - OUTWARD_PCT))
                            px, py, vx, vy = alt
                            alt = ((px, py + shift, vx, vy)
                                   if k in ("top", "bot")
                                   else (px + shift, py, vx, vy))
                            st_alt = line_strength(alt, alt_cl, k)
                            st_cur = line_strength(lines[k], pts[k], k)
                            d_cur = side_inset(lines[k], anch[k], k)
                            d_alt = side_inset(alt, anch[k], k)
                            if (st_alt is not None
                                    and st_alt >= BAND_EDGE_MIN
                                    and st_cur is not None
                                    and st_alt >= st_cur
                                    and d_cur is not None
                                    and d_alt is not None
                                    and d_alt - d_cur >= FIT_TOL * s
                                    and margin_behind_ok(alt, anch[k], k,
                                                         alt_cl)):
                                cand, cl = alt, alt_cl
                                smeared = True
            if cand is None or (n_g < 2 and not smeared):
                continue
            ts = [q[0] if k in ("top", "bot") else q[1] for q in cl]
            if max(ts) - min(ts) < span_min * length:
                continue
            st_new = line_strength(cand, cl, k)
            if st_new is None or st_new < BAND_EDGE_MIN:
                continue
            if lines[k] is not None:
                st_cur = line_strength(lines[k], pts[k], k)
                if smeared:
                    if st_cur is not None and st_new < EVIDENCE_RATIO * st_cur:
                        continue
                else:
                    t_mid = float(np.median(ts))
                    d_out = (line_off_at(lines[k], k, t_mid) -
                             line_off_at(cand, k, t_mid))
                    if k in ("bot", "rig"):
                        d_out = -d_out
                    if d_out <= FIT_TOL * s:
                        continue
                    # a cut edge may carry less gradient than a tone
                    # boundary inside the picture, so the strength test is
                    # waived when the strip behind the candidate is a real
                    # card margin, which such a boundary cannot fake
                    if st_cur is not None and st_new < st_cur and not \
                            margin_behind_ok(cand, anch[k], k, cl):
                        continue
            lines[k], pts[k] = cand, cl
            src[k] = "outward-smear" if smeared else "outward-cluster"
            dev[k] = side_inset(cand, anch[k], k)

    ms = [d for d in dev.values() if d is not None]
    m_est = float(np.median(ms)) if ms else None

    # Pass 2: re-trace missing / implausible sides with the margin hint.
    if m_est is not None:
        dev_lim = max(DEV_ABS * s, DEV_REL * m_est)
        for k in order:
            if (lines[k] is None or dev[k] is None or
                    abs(dev[k] - m_est) > dev_lim):
                l2, a2, p2, ap2 = trace_side(gray, gdir[k], paper, bbox, k,
                                             m_est,
                                             gdir_alt=gdir_alt[k])
                if l2 is None:
                    continue
                d2 = side_inset(l2, a2 if a2 is not None else anch[k], k)
                take2 = (lines[k] is None or
                         (d2 is not None and
                          (dev[k] is None or
                           abs(d2 - m_est) <= abs(dev[k] - m_est))) or
                         (d2 is None and len(p2) > len(pts[k])))
                if take2 and lines[k] is not None and weaker_evidence(
                        lines[k], pts[k], l2, p2, k):
                    take2 = False
                if take2:
                    lines[k], pts[k] = l2, p2
                    if a2 is not None:
                        anch[k], apts[k] = a2, ap2
                    elif ap2 and not apts[k]:
                        apts[k] = ap2
                    src[k] = "pass2"
                    dev[k] = d2

    # Pass 2b: ghost-outline retrace.  A side whose margin is
    # unmeasurable or a small fraction of m_est while its print outline is
    # solidly anchored is ghost-suspect.  The swap happens ONLY when the
    # ghost line moved OUTWARD with real support AND sits at a plausible
    # margin inside its OWN re-anchored outline with a clean border strip
    # behind it - a line parked on the print groove itself has no
    # such margin and would only add the border to the frame.
    if not back and m_est is not None:
        for k in order:
            if anch[k] is None or len(apts[k]) < \
                    PRIOR_ANCHOR_FRAC * expected_samples(k):
                continue
            if dev[k] is not None and dev[k] >= GHOST_INSET_FRAC * m_est:
                continue
            l5, a5, p5, ap5 = trace_side(gray, gdir[k], paper, bbox, k,
                                         m_est, ghost=True,
                                         gdir_alt=gdir_alt[k])
            if l5 is None or len(p5) < MIN_POINTS or lines[k] is None:
                continue
            t_mid = l5[0] if k in ("top", "bot") else l5[1]
            off_new = l5[1] if k in ("top", "bot") else l5[0]
            off_old = line_off_at(lines[k], k, t_mid)
            outward = -1.0 if k in ("top", "lef") else 1.0
            if outward * (off_new - off_old) < FIT_TOL * s:
                continue      # the same line (borderless-correct) or inward
            # plausible margin + clean border behind the ghost line
            a_use5 = a5 if a5 is not None else anch[k]
            if side_inset(l5, a_use5, k) is None or \
                    not margin_behind_ok(l5, a_use5, k, p5):
                continue
            lines[k], pts[k] = l5, p5
            if a5 is not None:
                anch[k], apts[k] = a5, ap5
            src[k] = "pass2b-ghost"
            dev[k] = side_inset(l5, anch[k], k)

    # Pass 3: rescue weak-contrast (or still suspicious) sides.
    def _is_weak(samples, k):
        if len(samples) < 2:
            return True
        if k in ("top", "bot"):
            ts = [p[0] for p in samples]
            length = bw * (1.0 - 2 * EDGE_SKIP_FRAC)
        else:
            ts = [p[1] for p in samples]
            length = bh * (1.0 - 2 * EDGE_SKIP_FRAC)
        step = max(1, int(length) // SAMPLES_PER_SIDE)
        expected = int(length) // step + 1
        dens = len(samples) / float(max(1, expected))
        span = (max(ts) - min(ts)) / max(length, 1.0)
        return dens < DENSE_FRAC or span < SPAN_FRAC

    if m_est is not None:
        dev_lim = max(DEV_ABS * s, DEV_REL * m_est)
        for k in order:
            susp = (lines[k] is not None and
                    (dev[k] is None or abs(dev[k] - m_est) > dev_lim))
            # a fit that still fails to hold its samples is
            # suspicious even when its inset agrees with the consensus
            weakfit = (lines[k] is not None and
                       fit_support(lines[k], pts[k], k) < SMEAR_INLIER_FRAC)
            if not _is_weak(pts[k], k) and not susp and not weakfit:
                continue
            l3, a3, p3, ap3 = trace_side(gray, gdir[k], paper, bbox, k, m_est,
                                         weak=True, gdir_alt=gdir_alt[k])
            if l3 is None:
                if lines[k] is None and len(p3) > len(pts[k]):
                    pts[k] = p3
                continue
            ins3 = side_inset(l3, a3 if a3 is not None else anch[k], k)
            take = False
            if lines[k] is None:
                take = True
            elif ins3 is None:
                take = False
            else:
                if weakfit:
                    # the incumbent is a failed fit; judge the
                    # weak retrace by its own evidence gates
                    take = True
                else:
                    d_cur = (abs(dev[k] - m_est)
                             if dev[k] is not None else None)
                    d_new = abs(ins3 - m_est)
                    if d_cur is None or d_new < d_cur:
                        take = True
                    elif (not susp and len(p3) > len(pts[k]) and
                            d_new <= d_cur):
                        take = True
                if take and weaker_evidence(lines[k], pts[k], l3, p3, k):
                    take = False
            if take:
                lines[k], pts[k] = l3, p3
                src[k] = "pass3-weak"
                if a3 is not None:
                    anch[k], apts[k] = a3, ap3
                elif ap3 and not apts[k]:
                    apts[k] = ap3
                dev[k] = ins3

    # Pass 3d: deep trace for a genuinely wide mount.
    deep_sides = set()
    if m_est is not None and not back:
        dev_lim = max(DEV_ABS * s, DEV_REL * m_est)
        deep_lim = max(lim, DEEP_REACH_K * m_est)
        for k in order:
            thin = len(pts[k]) < DEEP_WEAK_FRAC * expected_samples(k)
            if dev[k] is not None and abs(dev[k] - m_est) <= dev_lim \
                    and not thin:
                continue
            reach_k = deep_lim
            l4, a4, p4, ap4 = trace_side(gray, gdir[k], paper, bbox, k,
                                         m_est, reach=reach_k,
                                         gdir_alt=gdir_alt[k])
            if not (deep_ok(l4, p4, k) or
                    (thin and deep_ok_thin(l4, p4, k))):
                continue
            a_use = a4 if a4 is not None else anch[k]
            d4 = side_inset(l4, a_use, k, lim_k=deep_lim)
            if d4 is None:
                continue
            if lines[k] is not None and weaker_evidence(lines[k], pts[k],
                                                        l4, p4, k):
                continue
            if thin and len(p4) <= len(pts[k]):
                continue
            lines[k], pts[k], dev[k] = l4, p4, d4
            if a4 is not None:
                anch[k], apts[k] = a4, ap4
            elif ap4 and not apts[k]:
                apts[k] = ap4
            src[k] = "pass3-deep"
            deep_sides.add(k)

    # Pass 4: few-point rescue.
    th0 = _consensus_tilt(lines["top"], lines["bot"],
                          lines["lef"], lines["rig"])
    sl_h = float(np.tan(th0))
    sl_v = float(-np.tan(th0))

    def few_point_line(samples, k):
        """Reject outliers BEFORE the few-point rescue."""
        if len(samples) < WEAK_MIN_POINTS:
            return None
        slope = sl_h if k in ("top", "bot") else sl_v
        cand = _refit_at_angle(samples, k, slope, FIT_TOL * s,
                               min_pts=WEAK_MIN_POINTS)
        if cand is not None:
            return cand
        return _fit_line_fixed_angle(samples, k, slope, FIXED_ANGLE_TOL * s)

    def cluster_line(samples, k):
        """Pass-5 candidate: believed only when it sits on a real ridge."""
        slope = sl_h if k in ("top", "bot") else sl_v
        cand, cl = cluster_at_angle(samples, k, slope, FIT_TOL * s)
        if cand is None:
            return None, None, []
        st = line_strength(cand, cl, k)
        if st is None or st < BAND_EDGE_MIN:
            return None, None, []
        return cand, st, cl

    for k in order:
        if lines[k] is not None:
            continue
        cand = few_point_line(pts[k], k)
        if cand is None:
            continue
        if anch[k] is not None:
            ins_c = side_inset(cand, anch[k], k)
            if ins_c is None:
                continue
            # solid outline: only a plausible margin may fill the side;
            # an implausible few-point guess is refused so the geometric
            # prior (anchor + m_est) fills it instead
            if m_est is not None and \
                    len(apts[k]) >= PRIOR_ANCHOR_FRAC * expected_samples(k) \
                    and not (MWIN_LO * m_est <= ins_c <= MWIN_HI * m_est):
                continue
        lines[k] = cand
        src[k] = "pass4-fewpoint"

    # Pass 5: REPLACEMENT ONLY (minority cluster with far more gradient).
    for k in order:
        if not pts1[k] or k in deep_sides:
            continue
        cand, st_new, cl_pts = cluster_line(pts1[k], k)
        if cand is None or lines[k] is None:
            continue
        if src[k].startswith("pass5") or src[k] in ("outward-cluster",
                                                    "outward-smear"):
            continue
        st_cur = line_strength(lines[k], pts[k], k)
        if st_cur is not None and st_new < st_cur / EVIDENCE_RATIO:
            continue
        if anch[k] is not None and side_inset(cand, anch[k], k) is None:
            continue
        lines[k], pts[k], src[k] = cand, cl_pts, "pass5-cluster"
        dev[k] = side_inset(cand, anch[k], k)

    # Geometric prior: anchor shifted inward by the consensus margin.
    def from_anchor(aline, k):
        if aline is None or m_est is None:
            return None
        px, py, vx, vy = aline
        if k == "top":  return (px, py + m_est, vx, vy)
        if k == "bot":  return (px, py - m_est, vx, vy)
        if k == "lef":  return (px + m_est, py, vx, vy)
        return (px - m_est, py, vx, vy)

    # Solid-anchor completion.  A starved side (the image edge was
    # never really measured) whose outline is solidly anchored takes the
    # geometric prior; a side whose incumbent carries real support
    # (>= SMEAR_INLIER_FRAC of its samples) is not touched.
    # A side a verified pass already carried is left alone.
    # Before the prior fires, the side's own cluster is tried on
    # stricter terms than a pass-5 replacement gets - a strong ridge OR
    # geometric coherence plus a consensus margin.
    if not back and m_est is not None:
        for k in order:
            if src[k] in VERIFIED_SRC:
                continue              # a pass with its own bars carried it
            if len(pts1[k]) >= PRIOR_STARVED_FRAC * expected_samples(k):
                continue              # pass 1 measured this side
            if anch[k] is None or len(apts[k]) < \
                    PRIOR_ANCHOR_FRAC * expected_samples(k):
                continue              # outline not solidly anchored
            if (lines[k] is not None
                    and len(pts[k]) >= PRIOR_STARVED_FRAC
                    * expected_samples(k)
                    and fit_support(lines[k], pts[k], k)
                    >= SMEAR_INLIER_FRAC):
                continue              # the incumbent carries real support
            cand, st_new, cl_pts = cluster_line(pts1[k], k)
            ok_fill = False
            if cand is not None and len(cl_pts) >= FILL_MIN_PTS:
                ok_fill = st_new >= EDGE_MIN_THR
                if not ok_fill and st_new >= FILL_TIGHT_GRAD:
                    # geometric coherence instead of strength
                    a_c = np.asarray(cl_pts, np.float64)
                    t_c = (a_c[:, 0] if k in ("top", "bot")
                           else a_c[:, 1])
                    o_c = (a_c[:, 1] if k in ("top", "bot")
                           else a_c[:, 0])
                    r_c = np.abs(o_c - np.array(
                        [line_off_at(cand, k, x) for x in t_c]))
                    ins_c = (side_inset(cand, anch[k], k)
                             if anch[k] is not None else None)
                    ok_fill = (
                        float(np.percentile(r_c, 90)) / s
                        <= FILL_TIGHT_SPREAD
                        and m_est is not None and ins_c is not None
                        and MWIN_LO * m_est <= ins_c <= MWIN_HI * m_est)
            if ok_fill:
                ts = [q[0] if k in ("top", "bot") else q[1]
                      for q in cl_pts]
                length = ((bw if k in ("top", "bot") else bh) *
                          (1.0 - 2 * EDGE_SKIP_FRAC))
                ins_f = (side_inset(cand, anch[k], k)
                         if anch[k] is not None else None)
                span_ok = max(ts) - min(ts) >= FILL_SPAN * length
                if not span_ok and ins_f is not None \
                        and m_est is not None and tilt_anchored():
                    # The angle is pinned elsewhere, so this
                    # cluster is only fixing an offset - judge it on
                    # scatter and margin, which a drifting boundary
                    # cannot fake, rather than on how much of the side
                    # happens to be visible.
                    a_f = np.asarray(cl_pts, np.float64)
                    t_f = (a_f[:, 0] if k in ("top", "bot")
                           else a_f[:, 1])
                    o_f = (a_f[:, 1] if k in ("top", "bot")
                           else a_f[:, 0])
                    r_f = np.abs(o_f - np.array(
                        [line_off_at(cand, k, x) for x in t_f]))
                    span_ok = (
                        float(np.percentile(r_f, 90)) / s
                        <= FILL_PINNED_SPREAD
                        and abs(ins_f - m_est)
                        <= FILL_PINNED_MARGIN * m_est)
                if span_ok and (anch[k] is None or ins_f is not None):
                    lines[k], pts[k] = cand, cl_pts
                    dev[k] = ins_f
                    src[k] = "pass5-fill (cluster beats prior)"
                    continue
            repl = from_anchor(anch[k], k)
            if repl is not None:
                src[k] = "prior (solid anchor, unmeasured edge)"
                lines[k], pts[k], dev[k] = repl, [], None

    # Clean-margin guard: a line far beyond the consensus margin whose
    # page-side band is not a clean margin is replaced by anchor + m_est.
    if m_est is not None and not back:
        mi_q = max(1, int(round(s)))
        Hg, Wg = gray.shape
        for k in order:
            l = lines[k]
            if (l is None or anch[k] is None or dev[k] is None or
                    dev[k] <= MWIN_HI * m_est or not apts[k]):
                continue
            px, py, vx, vy = l
            if k in ("top", "bot"):
                slope = vy / vx if abs(vx) > 1e-9 else 0.0
            else:
                slope = vx / vy if abs(vy) > 1e-9 else 0.0
            # measure every band first, then decide what "clean"
            # means (card stock may sit far below the album page)
            bands = []
            for (ax, ay) in apts[k]:
                if k == "top":
                    yl = py + slope * (ax - px)
                    t0, t1 = yl - 6 * mi_q, yl - 2 * mi_q
                    u0, u1 = ax - 3 * mi_q, ax + 3 * mi_q
                elif k == "bot":
                    yl = py + slope * (ax - px)
                    t0, t1 = yl + 2 * mi_q, yl + 6 * mi_q
                    u0, u1 = ax - 3 * mi_q, ax + 3 * mi_q
                elif k == "lef":
                    xl = px + slope * (ay - py)
                    t0, t1 = ay - 3 * mi_q, ay + 3 * mi_q
                    u0, u1 = xl - 6 * mi_q, xl - 2 * mi_q
                else:
                    xl = px + slope * (ay - py)
                    t0, t1 = ay - 3 * mi_q, ay + 3 * mi_q
                    u0, u1 = xl + 2 * mi_q, xl + 6 * mi_q
                t0 = int(max(0, t0)); t1 = int(min(Hg, t1))
                u0 = int(max(0, u0)); u1 = int(min(Wg, u1))
                if t1 - t0 < 1 or u1 - u0 < 1:
                    continue
                seg = gray[t0:t1, u0:u1]
                bands.append((float(seg.mean()), float(seg.std())))
            ref_c = float(paper)
            if bands:
                med = float(np.median([b[0] for b in bands]))
                if paper - MARGIN_REF_DROP <= med <= \
                        paper - MARGIN_REF_MIN_DROP:
                    ref_c = med           # card stock, not the album page
            tot = len(bands)
            clean = sum(1 for m_, sd_ in bands
                        if m_ >= ref_c - PAPER_SLACK and
                        sd_ <= 2 * MARGIN_BAND_STD)
            if tot >= 3 and clean < CLEAN_MARGIN_FRAC * tot:
                repl = from_anchor(anch[k], k)
                if repl is not None:
                    src[k] = "replaced (anchor + m_est)"
                    lines[k], pts[k], dev[k] = repl, [], None

    for k in order:
        if lines[k] is None and not back:
            lines[k] = from_anchor(anch[k], k)
            if lines[k] is not None:
                src[k] = "from_anchor (anchor + m_est)"
                pts[k] = []    # prior line: no measured samples to show

    # Back mode: the frame is the tangent envelope of the anchor scatter.
    if back:
        b_lines, b_pts, b_src = _back_quad_lines(apts, anch, order, s)
        for k in order:
            lines[k] = b_lines[k]
            pts[k] = b_pts[k]
            src[k] = b_src[k]
            dev[k] = None

    # Enforce one common rotation angle over all sides before corners.
    l_top, l_bot, l_lef, l_rig, th_c = _rectify_sides(
        lines["top"], lines["bot"], lines["lef"], lines["rig"])
    t_c = float(np.tan(th_c))

    cx, cy = bx + bw / 2.0, by + bh / 2.0
    y_top = by if l_top is None else line_off_at(l_top, "top", cx)
    y_bot = by + bh - 1 if l_bot is None else line_off_at(l_bot, "bot", cx)
    x_lef = bx if l_lef is None else line_off_at(l_lef, "lef", cy)
    x_rig = bx + bw - 1 if l_rig is None else line_off_at(l_rig, "rig", cy)

    if y_top >= y_bot - MIN_SIDE_GAP * s:
        y_top, y_bot, l_top, l_bot = by, by + bh - 1, None, None
    if x_lef >= x_rig - MIN_SIDE_GAP * s:
        x_lef, x_rig, l_lef, l_rig = bx, bx + bw - 1, None, None

    if l_top is None: l_top = (cx, float(y_top), 1.0, t_c)
    if l_bot is None: l_bot = (cx, float(y_bot), 1.0, t_c)
    if l_lef is None: l_lef = (float(x_lef), cy, -t_c, 1.0)
    if l_rig is None: l_rig = (float(x_rig), cy, -t_c, 1.0)

    def inter(lh, lv, fx, fy):
        p = line_intersect(lh, lv)
        return p if p is not None else (float(fx), float(fy))

    quad = np.float32([inter(l_top, l_lef, x_lef, y_top),
                       inter(l_top, l_rig, x_rig, y_top),
                       inter(l_bot, l_rig, x_rig, y_bot),
                       inter(l_bot, l_lef, x_lef, y_bot)])

    w1 = np.linalg.norm(quad[1] - quad[0])
    w2 = np.linalg.norm(quad[2] - quad[3])
    h1 = np.linalg.norm(quad[3] - quad[0])
    h2 = np.linalg.norm(quad[2] - quad[1])
    ok = min(w1, w2, h1, h2) > QUAD_MIN_SIDE * s
    ok = ok and max(w1, w2) < QUAD_MAX_GROWTH * bw + QUAD_GROWTH_PAD * s \
        and max(h1, h2) < QUAD_MAX_GROWTH * bh + QUAD_GROWTH_PAD * s
    v1, v2 = quad[1] - quad[0], quad[3] - quad[0]
    cosang = abs(float(np.dot(v1, v2)) /
                 (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6))
    ok = ok and cosang < QUAD_MAX_COS
    if not ok:
        # the fitted quad is implausible: fall back to the rough box, and
        # say so, so no side is reported as measured when it was not
        quad = np.float32([[bx, by], [bx + bw - 1, by],
                           [bx + bw - 1, by + bh - 1], [bx, by + bh - 1]])
        src = {k: "rough box (fitted quad failed the sanity check)"
               for k in order}

    dbg_items = None
    if want_dbg:
        dbg_items = [(k, list(pts[k]), list(apts[k])) for k in order]
    if prov is not None:
        prov.update(source=dict(src), inset={k: dev[k] for k in order},
                    m_est=m_est, n_pts={k: len(pts[k]) for k in order},
                    tilt_deg=float(np.degrees(th_c)))
    return np.round(quad).astype(np.int32), dbg_items


# --------------------------------------------------------------------------
# drawing, I/O, reports
# --------------------------------------------------------------------------

def draw_quads(image_bgr, quads, color=(0, 0, 255)):
    """Draw red borders around detected photo regions."""
    out = image_bgr.copy()
    t = max(2, out.shape[1] // max(1, THICKNESS_SCALE))
    for q in quads:
        cv2.polylines(out, [q.reshape(-1, 1, 2)], True, color, t, cv2.LINE_AA)
    return out


_DBG_COLOR = {"top": (0, 255, 255), "bot": (255, 0, 255),
              "lef": (0, 200, 0), "rig": (0, 160, 255)}


def _draw_dbg(img, items, s, off):
    """Draw anchor_report.py --draw overlays on TOP of the red frame."""
    ox, oy = off
    r = max(2, int(round(3 * s)))
    for kind, pts, anchors in items:
        col = _DBG_COLOR.get(kind, (255, 255, 255))
        for x, y in pts:
            cv2.circle(img, (int(x) + ox, int(y) + oy), r, col, 1, cv2.LINE_AA)
        for x, y in anchors:
            cv2.drawMarker(img, (int(x) + ox, int(y) + oy), col,
                           cv2.MARKER_CROSS, 2 * r + 1, 1, cv2.LINE_AA)


def read_image(path):
    """8-bit BGR copy of page 0 in the STORED raster frame, for measuring
    only, or None (the reason is printed).

    common.read_page reads the pixels exactly as stored: no Orientation tag
    applied, planar storage interleaved, non-ASCII paths fine on Windows."""
    a = read_pixels(path)
    return None if a is None else to_bgr8(a)


def read_pixels(path):
    """Page 0 exactly as stored, through common.read_page, or None (the
    reason is printed)."""
    try:
        a, _meta = read_page(str(path))
    except (Exception, SystemExit) as exc:
        print(f"    ! {Path(path).name}: {exc}")
        return None
    return np.asarray(a)


def to_bgr8(a):
    """8-bit BGR copy of the pixels read_pixels returned."""
    if a.ndim == 2:
        a = a[:, :, None]
    if a.shape[2] > 3:
        a = a[..., :3]
    elif a.shape[2] == 2:
        a = a[..., :1]
    if a.dtype == np.uint16:
        a = ((a.astype(np.uint32) + 128) // 257).clip(0, 255).astype(np.uint8)
    elif a.dtype != np.uint8:
        f = a.astype(np.float32)
        lo, hi = float(f.min()), float(f.max())
        a = (((f - lo) / (hi - lo) * 255) if hi > lo else f * 0
             ).astype(np.uint8)
    if a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    return np.ascontiguousarray(a[..., ::-1])          # RGB -> BGR


def orientation_tag(path):
    """The file's TIFF Orientation tag, reported for information only."""
    if not str(path).lower().endswith(TIFF_EXT):
        return 1
    try:
        with tifffile.TiffFile(str(path)) as tf:
            tag = tf.pages[0].tags.get("Orientation")
            return int(tag.value) if tag is not None else 1
    except Exception:
        return 1


def quad_report(q):
    """Everything worth recording about one quad, derived from its corners."""
    p = np.asarray(q, np.float64)
    order = np.argsort(p[:, 1])
    top = p[order[:2]][np.argsort(p[order[:2]][:, 0])]
    bot = p[order[2:]][np.argsort(p[order[2:]][:, 0])]
    tl, tr = top[0], top[1]
    bl, br = bot[0], bot[1]

    def ang(a, b):
        return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))

    bottom = ang(bl, br)
    tilt = ((bottom + 45.0) % 90.0) - 45.0
    x0, y0 = int(p[:, 0].min()), int(p[:, 1].min())
    x1, y1 = int(p[:, 0].max()), int(p[:, 1].max())
    w_top, w_bot = np.hypot(*(tr - tl)), np.hypot(*(br - bl))
    h_left, h_right = np.hypot(*(bl - tl)), np.hypot(*(br - tr))
    return {
        "corners": [[int(round(v)) for v in pt] for pt in (tl, tr, br, bl)],
        "bbox": [x0, y0, x1 + 1, y1 + 1],
        "tilt_deg": round(tilt, 4),
        "angle_bottom_deg": round(bottom, 4),
        "angle_top_deg": round(ang(tl, tr), 4),
        "angle_left_deg": round(ang(tl, bl) - 90.0, 4),
        "angle_right_deg": round(ang(tr, br) - 90.0, 4),
        "side_px": {"top": round(w_top, 1), "bottom": round(w_bot, 1),
                    "left": round(h_left, 1), "right": round(h_right, 1)},
        "skew_px": round(max(abs(w_top - w_bot), abs(h_left - h_right)), 1),
        "bbox_waste_pct": round(
            100.0 * (1.0 - (w_top * h_left)
                     / max(1.0, (x1 + 1 - x0) * (y1 + 1 - y0))),
            2),
    }


def _imwrite(path, img):
    """Write an image through numpy so non-ASCII output paths work."""
    ok, buf = cv2.imencode(Path(path).suffix or ".jpg", img)
    if not ok:
        return False
    try:
        buf.tofile(str(path))
    except OSError:
        return False
    return True


# --------------------------------------------------------------------------
# the ring finder (--ring)
# --------------------------------------------------------------------------
# An alternative to the detector above: the whole print (frame plus
# emulsion) is the largest closed outline of strong COLOUR gradient on the
# scan, and its rectangle is placed from the middle of each side of that
# outline. One print per scan,
# measured on a copy shrunk to RING_WORK_SIDE, so its corners are exact to
# about one pixel of that copy. A light, flat interior makes it a back
# (framed blue in the previews), unless opposite margins agree, which is a
# pale mounted print instead.

def ring_prepare(a):
    """The scan as the ring finder measures it: shrunk to RING_WORK_SIDE,
    a dark 16-bit scan stretched to full range, 8-bit BGR. Returns (image,
    (fx, fy) from the copy to the scan, stretch factor)."""
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
    f = max(h0, w0) / float(RING_WORK_SIDE)
    if f > 1.0:
        w1, h1 = max(1, int(round(w0 / f))), max(1, int(round(h0 / f)))
        a = cv2.resize(np.ascontiguousarray(a), (w1, h1),
                       interpolation=cv2.INTER_AREA)
        if a.ndim == 2:
            a = a[:, :, None]
    h1, w1 = a.shape[:2]
    stretch = 1.0
    if a.dtype == np.uint16:
        if top is not None and 0 < top < RING_STRETCH_BELOW:
            stretch = 65535.0 / top
            a = np.round(a.astype(np.float64) * stretch).astype(np.uint16)
        a = ((a.astype(np.uint32) + 128) // 257).clip(0, 255).astype(np.uint8)
    elif a.dtype != np.uint8:
        lo, hi = float(a.min()), float(a.max())
        a = (((a - lo) / (hi - lo) * 255) if hi > lo else a * 0
             ).astype(np.uint8)
    if a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    return (np.ascontiguousarray(a[..., ::-1]),
            (w0 / float(w1), h0 / float(h1)), stretch)


def _ring_colour_grad(img):
    """The strongest Sobel magnitude of the three colour channels."""
    s = scale_of(img)
    blur = cv2.GaussianBlur(img, (0, 0), RING_BLUR * s).astype(np.float32)
    g = None
    for c in range(3):
        m = cv2.magnitude(cv2.Sobel(blur[:, :, c], cv2.CV_32F, 1, 0, ksize=3),
                          cv2.Sobel(blur[:, :, c], cv2.CV_32F, 0, 1, ksize=3))
        g = m if g is None else np.maximum(g, m)
    return g * s


def _ring_regions(cgrad, thr, s):
    """Convex hulls of the closed rings above `thr`, largest first."""
    H, W = cgrad.shape
    m = (cgrad > thr).astype(np.uint8)
    _, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    x, y, w, h = st[:, 0], st[:, 1], st[:, 2], st[:, 3]
    band, t = RING_BORDER_BAND * min(H, W), 2
    strip = ((((y <= t) | (y + h >= H - t)) & (h <= band)) |
             (((x <= t) | (x + w >= W - t)) & (w <= band)))
    keep = (np.maximum(w, h) >= RING_DUST * max(H, W)) & ~strip
    keep[0] = False
    m = keep[lab].astype(np.uint8)
    k = max(3, int(round(RING_CLOSE_K * s)) | 1)
    m = cv2.copyMakeBorder(m, k, k, k, k, cv2.BORDER_CONSTANT, value=0)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    m = np.ascontiguousarray(m[k:-k, k:-k])
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        hull = cv2.convexHull(c)
        x, y, w, h = cv2.boundingRect(hull)
        area = float(cv2.contourArea(hull))
        if area < RING_MIN_AREA * H * W or w < RING_MIN_SIDE * W \
                or h < RING_MIN_SIDE * H:
            continue
        if w > RING_MAX_SIDE * W and h > RING_MAX_SIDE * H:
            continue
        out.append((area, hull))
    out.sort(key=lambda t: -t[0])
    return [h for _, h in out]


def _ring_envelope_quad(hull):
    """(corners clockwise from top-left, tilt): the rectangle whose sides
    sit at the median of the middle part of each side of the hull."""
    (cx, cy), _, ang = cv2.minAreaRect(hull)
    ang = (ang + 45.0) % 90.0 - 45.0
    th = math.radians(ang)
    ct, st = math.cos(th), math.sin(th)
    poly = hull.reshape(-1, 2).astype(np.float64)
    pts = []
    for p, q in zip(poly, np.roll(poly, -1, axis=0)):
        n = max(1, int(math.ceil(float(np.linalg.norm(q - p)))))
        pts.append(p + (np.arange(n)[:, None] / n) * (q - p))
    d = np.vstack(pts) - (cx, cy)
    u = d[:, 0] * ct + d[:, 1] * st
    v = -d[:, 0] * st + d[:, 1] * ct

    def place(off, along, outer):
        if not np.any(outer):            # degenerate (collinear) hull
            raise ValueError("degenerate hull side")
        lo, hi = float(along.min()), float(along.max())
        cut = 0.5 * (1.0 - RING_SIDE_MID) * (hi - lo)
        sel = off[(along >= lo + cut) & (along <= hi - cut) & outer]
        if sel.size:
            return float(np.median(sel))
        return float(off[outer].min() if off[outer].mean() < 0
                     else off[outer].max())

    top, bot = place(v, u, v < 0), place(v, u, v > 0)
    lef, rig = place(u, v, u < 0), place(u, v, u > 0)
    cu = np.array([lef, rig, rig, lef])
    cv_ = np.array([top, top, bot, bot])
    quad = np.stack([cu * ct - cv_ * st + cx, cu * st + cv_ * ct + cy], axis=1)
    return quad, ang


def _ring_warp(img, quad):
    """The print cut out and squared, for the back test."""
    q = np.asarray(quad, np.float32)
    w = int(round(0.5 * (np.linalg.norm(q[1] - q[0])
                         + np.linalg.norm(q[2] - q[3]))))
    h = int(round(0.5 * (np.linalg.norm(q[3] - q[0])
                         + np.linalg.norm(q[2] - q[1]))))
    w, h = max(w, 2), max(h, 2)
    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    M = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)


def _ring_looks_like_back(card, paper):
    """A light, flat interior: no emulsion, so the back of a print."""
    h, w = card.shape[:2]
    iy, ix = int(0.1 * h), int(0.1 * w)
    roi = card[iy:h - iy, ix:w - ix]
    if roi.size < 1000:
        return False
    rs = roi[::4, ::4]
    return (float((rs < paper - BACK_DARK_DELTA).mean()) < BACK_DARK_FRAC and
            float(rs.std()) < BACK_STD)


def _ring_margin_pair(gray, gx, gy, quad):
    """Do two opposite sides show the same inner margin? Then it is a pale
    mounted print, not a back."""
    s = scale_of(gray)
    mi = max(1, int(round(s)))
    Hg, Wg = gray.shape
    q = np.asarray(quad, np.float64)
    tl, tr, br, bl = q
    lines = {"top": (tl[0], tl[1], tr[0] - tl[0], tr[1] - tl[1]),
             "bot": (bl[0], bl[1], br[0] - bl[0], br[1] - bl[1]),
             "lef": (tl[0], tl[1], bl[0] - tl[0], bl[1] - tl[1]),
             "rig": (tr[0], tr[1], br[0] - tr[0], br[1] - tr[1])}
    x0, x1 = float(q[:, 0].min()), float(q[:, 0].max())
    y0, y1 = float(q[:, 1].min()), float(q[:, 1].max())
    dmax = int(max(12 * s, INSET_MAX_FRAC * max(x1 - x0, y1 - y0)))
    g0 = HALO_GUARD * mi
    peak_w = max(1, int(round(1.5 * mi)))
    depth = {}
    for k, l in lines.items():
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
            off = int(round(line_off_at(l, k, t)))
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


def ring_find(img):
    """The print on one scan prepared by ring_prepare, in that copy's
    pixels: a dict with paper, n_regions and quad -- None when no ring is
    found, else with tilt, cut (sides touching the scan edge), back and
    why. None when the scan is too small to measure."""
    H, W = img.shape[:2]
    crop = int(round(CROP * max(H, W) / BASE))
    crop = max(0, min(crop, (min(H, W) - 1) // 2))
    img_c = img[crop:H - crop, crop:W - crop]
    if img_c.size == 0:
        return None
    gray = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
    paper = paper_level_of(gray)
    s = scale_of(gray)
    Hc, Wc = gray.shape
    cgrad = _ring_colour_grad(img_c)
    hulls = _ring_regions(cgrad, RING_THR, s) or \
        _ring_regions(cgrad, RING_THR_FAINT, s)
    none = {"quad": None, "paper": paper, "n_regions": len(hulls)}
    if not hulls:
        return none
    quad = tilt = None
    for h in hulls:                  # skip degenerate hulls
        try:
            quad, tilt = _ring_envelope_quad(h)
            break
        except (ValueError, cv2.error):
            continue
    if quad is None:
        return none
    g = RING_EDGE_GUARD * s
    cut = [k for k, hit in (("top", quad[:2, 1].min() <= g),
                            ("bot", quad[2:, 1].max() >= Hc - 1 - g),
                            ("lef", quad[[0, 3], 0].min() <= g),
                            ("rig", quad[[1, 2], 0].max() >= Wc - 1 - g))
           if hit]
    quad[:, 0] = np.clip(quad[:, 0], 0, Wc - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, Hc - 1)

    back, why = False, ""
    if _ring_looks_like_back(_ring_warp(gray, quad), paper):
        back, why = True, "light, flat interior"
        _, gx, gy = compute_grad(gray)
        if _ring_margin_pair(gray, gx, gy, quad):
            back, why = False, "opposite margins agree (pale mounted print)"
    return {"quad": quad + crop, "tilt": tilt, "cut": cut, "back": back,
            "why": why, "n_regions": len(hulls), "paper": paper}


def process_ring(path, out_dir, review=False):
    """process_one for --ring: the same console block and the same record,
    from the ring finder."""
    a = read_pixels(path)
    if a is None:
        return None
    H, W = a.shape[:2]
    work, (fx, fy), stretch = ring_prepare(a)
    card = ring_find(work)
    if card is None:
        print(f"    ! {Path(path).name}: too small to process")
        return None

    quads = []
    if card["quad"] is not None:
        q = np.round(card["quad"] * np.array([fx, fy])).astype(np.int32)
        q[:, 0] = np.clip(q[:, 0], 0, W - 1)
        q[:, 1] = np.clip(q[:, 1], 0, H - 1)
        quads.append(q)

    print(f"    {Path(path).name}: {count(len(quads), 'print')}")
    for i, q in enumerate(quads):
        rep = quad_report(q)
        x0, y0, x1, y1 = rep["bbox"]
        print(f"        print {i}: x={x0} y={y0} w={x1 - x0} h={y1 - y0} "
              f"| corners={q.tolist()}")
        side = "back" if card["back"] else "front"
        why = f" ({card['why']})" if card["why"] else ""
        print(f"        ring: {side}{why}, "
              f"{count(card['n_regions'], 'region')}, "
              f"tilt={rep['tilt_deg']:.2f}deg")
        if card["cut"]:
            print(f"        ! the print touches the scan edge: "
                  f"{', '.join(card['cut'])}")

    if review:
        colour = RING_BACK_COLOR if card.get("back") else RING_FRONT_COLOR
        out = draw_quads(to_bgr8(a), quads, colour)
        dst = Path(out_dir) / f"{Path(path).stem}_marked.jpg"
        if not _imwrite(dst, out):
            print(f"        ! {Path(path).name}: could not write the "
                  f"preview {dst}")

    # the paper level as the unstretched scan has it, like every other scan
    return {"size": [H, W],
            "paper_level": round(float(card["paper"] / stretch), 1),
            "orientation": orientation_tag(path),
            "photos": [dict(quad_report(q),
                            sides={k: "ring" for k in
                                   ("top", "bot", "lef", "rig")})
                       for q in quads]}


# --------------------------------------------------------------------------
# one scan
# --------------------------------------------------------------------------

def prepare_scan(img, back=False):
    """Everything detect_quad needs for one scan, and the rough boxes.

    Returns a dict: gray, grad, gx, gy, gx_alt, gy_alt (the red-channel
    gradients, None on a colour print), paper, boxes, back (the mode the
    boxes were found in), back_forced (--back), off (x, y offset from the
    measured frame to the scan's own pixels). anchor_report.py calls this
    too, so it always looks at exactly what the detector looked at."""
    back_forced = bool(back)
    H, W = img.shape[:2]
    crop = int(round(CROP * max(H, W) / BASE))
    crop = max(0, min(crop, (min(H, W) - 1) // 2))
    gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray_full[crop:H - crop, crop:W - crop]
    if gray.size == 0:
        return None
    paper = paper_level_of(gray)
    grad, gx, gy = compute_grad(gray)
    # The red-channel veto assumes a silver image; decide once, for the
    # whole scan, whether this print is one.
    colour_print = scan_is_colour(img)
    use_alt = not colour_print
    gx_alt = gy_alt = None
    if use_alt:
        _, gx_alt, gy_alt = compute_grad(
            np.ascontiguousarray(img[crop:H - crop, crop:W - crop, 2]))
    if back:
        boxes = back_rois(gray, grad, paper)
    else:
        boxes = rough_rois(gray, paper)
        if not boxes:
            # a white-on-white card leaves Otsu nothing: edge-based finder
            boxes = back_rois(gray, grad, paper)
            back = bool(boxes)
        elif looks_like_back(gray, boxes, paper):
            eboxes = back_rois(gray, grad, paper)
            if eboxes:
                boxes, back = eboxes, True
            else:
                # every back finder empty but looks_like_back says card:
                # keep back mode on the Otsu box
                back = True

    cl = ct = crop
    if not back and crop and boxes:
        gh, gw = gray.shape
        touch = {"lef": any(bx <= 0 for bx, _, _, _ in boxes),
                 "top": any(by <= 0 for _, by, _, _ in boxes),
                 "rig": any(bx + bw >= gw for bx, _, bw, _ in boxes),
                 "bot": any(by + bh >= gh for _, by, _, bh in boxes)}
        strip = {"lef": gray_full[:, :3], "rig": gray_full[:, -3:],
                 "top": gray_full[:3, :], "bot": gray_full[-3:, :]}
        # Lift CROP only on a border the picture actually reaches, and only
        # when that border is not the scanner's own dark edge.
        keep = {k: touch[k] and
                float(np.median(strip[k])) >= paper - SCAN_EDGE_DARK
                for k in ("lef", "top", "rig", "bot")}
        if any(keep.values()):
            cl = 0 if keep["lef"] else crop
            ct = 0 if keep["top"] else crop
            cr = 0 if keep["rig"] else crop
            cb = 0 if keep["bot"] else crop
            gray = gray_full[ct:H - cb, cl:W - cr]
            if use_alt:
                _, gx_alt, gy_alt = compute_grad(
                    np.ascontiguousarray(img[ct:H - cb, cl:W - cr, 2]))
            paper = paper_level_of(gray)
            grad, gx, gy = compute_grad(gray)
            boxes = [(bx + crop - cl, by + crop - ct, bw, bh)
                     for bx, by, bw, bh in boxes]
    return {"gray": gray, "grad": grad, "gx": gx, "gy": gy,
            "gx_alt": gx_alt, "gy_alt": gy_alt, "paper": paper,
            "boxes": boxes, "back": back, "back_forced": back_forced,
            "off": (cl, ct)}


def _has_margin_pair(prov):
    """Do two OPPOSITE sides agree on a margin?

    One side with an inset proves nothing - a bare card back throws up a
    lone edge readily enough. Two facing each other at the same depth is a
    mounted print, because that is what a mount is."""
    d = (prov or {}).get("inset") or {}
    for a, b in (("top", "bot"), ("lef", "rig")):
        va, vb = d.get(a), d.get(b)
        if va is None or vb is None or va <= 0 or vb <= 0:
            continue
        if abs(va - vb) <= BACK_PAIR_TOL * (0.5 * (va + vb)):
            return True
    return False


def measure_box(sc, bb, want_dbg=False):
    """(quad in the measured frame, debug items, provenance) of one rough
    box of a prepared scan, or None when the box holds no real edge.

    Back mode never traces the sides -- it builds the quad from the anchors
    alone -- so the margins that would disprove it are looked for BEFORE
    committing to it: one probe in normal mode, and if two opposite sides
    agree on a margin, this is a mounted print Otsu could not threshold."""
    bx, by, bw, bh = bb
    if bw < 1 or bh < 1:
        return None
    roi = sc["grad"][by:by + bh, bx:bx + bw]
    if roi.size == 0 or roi.max() < MIN_EDGE_STRENGTH:
        return None
    args = (sc["gray"], sc["gx"], sc["gy"], sc["paper"], bb)
    alt = dict(gx_alt=sc["gx_alt"], gy_alt=sc["gy_alt"], want_dbg=want_dbg)
    back = sc["back"]
    if back and not sc["back_forced"]:
        prov = {}
        q, dbg = detect_quad(*args, prov=prov, back=False, **alt)
        if _has_margin_pair(prov):
            return q, dbg, prov
    prov = {}
    q, dbg = detect_quad(*args, prov=prov, back=back, **alt)
    return q, dbg, prov


def process_one(path, out_dir, review=False, back=False):
    """Process a single scan: detect, optionally write the review preview,
    print the result, and hand the caller the per-photo records the project
    file needs. With review off, nothing is drawn or written."""
    img = read_image(path)
    if img is None:
        return None
    H, W = img.shape[:2]
    sc = prepare_scan(img, back)
    if sc is None:
        print(f"    ! {Path(path).name}: too small to process")
        return None
    cl, ct = sc["off"]

    quads, provs = [], []
    for bb in sc["boxes"]:
        res = measure_box(sc, bb)
        if res is None:
            continue
        q, _dbg, prov = res
        q = q + np.array([cl, ct], np.int32)
        q[:, 0] = np.clip(q[:, 0], 0, W - 1)
        q[:, 1] = np.clip(q[:, 1], 0, H - 1)
        quads.append(q)
        provs.append(prov)

    print(f"    {Path(path).name}: {count(len(quads), 'print')}")
    for i, (q, p) in enumerate(zip(quads, provs)):
        x0, y0 = int(q[:, 0].min()), int(q[:, 1].min())
        x1, y1 = int(q[:, 0].max()), int(q[:, 1].max())
        print(f"        print {i}: x={x0} y={y0} w={x1 - x0 + 1} "
              f"h={y1 - y0 + 1} | corners={q.tolist()}")
        src = p.get("source", {})
        sides = ", ".join(f"{k}={src.get(k, '-')}"
                          for k in ("top", "bot", "lef", "rig"))
        m = p.get("m_est")
        mstr = "n/a" if m is None else f"{m:.1f}px"
        print(f"        sides: {sides}")
        print(f"        m_est={mstr} tilt={p.get('tilt_deg', 0.0):.2f}deg "
              f"n_pts={p.get('n_pts')}")

    if review:
        out = draw_quads(img, quads)
        dst = Path(out_dir) / f"{Path(path).stem}_marked.jpg"
        if not _imwrite(dst, out):
            print(f"        ! {Path(path).name}: could not write the "
                  f"preview {dst}")

    return {"size": [H, W], "paper_level": round(float(sc["paper"]), 1),
            "orientation": orientation_tag(path),
            "photos": [dict(quad_report(q), sides=p.get("source", {}))
                       for q, p in zip(quads, provs)]}


# --------------------------------------------------------------------------
# the step
# --------------------------------------------------------------------------

def _worker(task):
    """Top-level (picklable) worker: process one scan, and return its
    console output together with the record, so the parent can print each
    file's block in one piece."""
    path_s, out_s, review, back, ring = task
    buf = io.StringIO()
    with redirect_stdout(buf):
        try:
            if ring:
                rec = process_ring(Path(path_s), Path(out_s), review)
            else:
                rec = process_one(Path(path_s), Path(out_s), review, back)
        except Exception as exc:
            # one unreadable scan must cost its own file, not the batch
            rec = None
            print(f"    ! {Path(path_s).name}: {exc}")
    return buf.getvalue(), rec


def _colour(line):
    """How one line of a scan's block is shown: a failed scan red, a
    warning under a scan (a preview not written, say) yellow, the scan's
    own line bold -- yellow when it holds other than one print -- and its
    details grey. The words are unchanged."""
    if line.startswith("    ! "):
        return paint(line, "1;31")
    if line.startswith("        ! "):
        return paint(line, "33")
    if line.startswith("        "):
        return paint(line, "90")
    if line.endswith((" print", " prints")):
        n = line.rsplit(":", 1)[-1].split()[0]
        return paint(line, "1" if n == "1" else "1;33")
    return line


def run(root, args) -> int:
    t0 = time.time()

    phase(1, 3, "the scans")
    if args.ring:
        info("the edge-ring finder (--ring): one print per scan")
    # only files directly in the root; the previews go to their own folder
    files = sorted(p for p in Path(root).iterdir()
                   if p.is_file() and p.name.lower().endswith(SCAN_EXTS))
    json_path = stage_json(root, STAGE_DETECT)
    kept = {}
    if args.new_only and os.path.isfile(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as fh:
                old = json.load(fh).get("items", {})
        except (OSError, ValueError) as exc:
            error(f"cannot read {json_path}: {exc}")
            return 1
        present = {f.name for f in files}
        kept = {n: r for n, r in old.items() if n in present}
        files = [f for f in files if f.name not in old]
        info(f"--new-only: {count(len(kept), 'scan')} measured already, "
             f"{len(files)} new")
        if not files:
            print(paint(f"\nnothing new to measure; {json_path} is left as "
                        f"it is", "32"))
            print(paint(f"\nnext:  python suggest_orientation.py"
                        f"{root_arg(root)}", "1"))
            return 0
    else:
        info(count(len(files), "scan"))

    # Step 4 is TIFF-only, so say so HERE rather than at the crop.
    other = [p.name for p in files
             if not p.name.lower().endswith(TIFF_EXT)]
    if other:
        warn_list(f"    ! {count(len(other), 'file')} not TIFF: measured "
                  f"here, but deskew_crop.py, which is TIFF-only, will skip "
                  f"{'it' if len(other) == 1 else 'them'}:", other)

    rdir = review_dir(root)
    if args.review:
        os.makedirs(rdir, exist_ok=True)
    tasks = [(str(f), str(rdir), args.review, args.back, args.ring)
             for f in files]
    workers = 1
    if parallel is not None and args.workers != 1:
        per = parallel.estimate_bytes_per_worker(str(files[0]),
                                                 multiple=WORKER_MEM_MULTIPLE)
        workers = parallel.choose_workers(len(tasks), per, args.workers)
    info(count(workers, "worker"))

    phase(2, 3, "the prints on each scan")
    # collect first, print in INPUT order, so two runs of the same folder
    # can be compared line by line
    bar = Progress(len(tasks), "    measuring")
    if workers <= 1 or len(tasks) <= 1:
        results = []
        for t in tasks:
            results.append(_worker(t))
            bar.update()
    else:
        # a worker that dies (out of memory on a huge scan) must cost its
        # own scans, not the boxes already found for every other one
        results = parallel.run(
            _worker, tasks, workers, on_result=bar.update,
            on_error=lambda t, exc: (
                f"    ! {Path(t[0]).name}: "
                f"{parallel.describe_error(exc)}\n", None))
    bar.close()

    items, failed = dict(kept), []
    for f, (text, rec) in zip(files, results):
        if text:
            print("\n".join(_colour(ln)
                            for ln in text.rstrip("\n").split("\n")))
        if rec is not None:
            items[f.name] = rec
        else:
            failed.append(f.name)

    phase(3, 3, "the project file")
    if failed:
        warn_list(f"    ! {count(len(failed), 'scan')} could not be "
                  f"processed and {'is' if len(failed) == 1 else 'are'} left "
                  f"out:", failed)
    if not items:
        error("no scan could be processed; nothing written")
        return 1

    doc = {
        "version": SCHEMA_VERSION,
        "source_dir": str(root),
        "detector": SCRIPT,
        "note": "corners clockwise from top-left, in pixels of the "
                "STORED raster. bbox is [x0, y0, x1, y1] with x1 and y1 "
                "exclusive. tilt_deg is measured from the bottom edge, "
                "positive clockwise, folded into +/-45. The TIFF "
                "Orientation tag is NOT applied to these coordinates, so "
                "they line up with what tifffile reads; it is reported per "
                "file for information only. No step applies it: the way "
                "up is the flip and quarter turns approved in review.py.",
        "items": dict(sorted(items.items())),
    }
    write_json(json_path, doc)
    n = sum(len(v["photos"]) for v in items.values())
    print(paint(f"\nwrote {json_path}  ({count(len(items), 'scan')}, "
                f"{count(n, 'print')})", "32"))
    if args.review:
        print(paint(f"    previews in {rdir}", "32"))
    print(paint(f"    done in {elapsed(t0)}", "32"))
    print(paint(f"\nnext:  python suggest_orientation.py{root_arg(root)}",
                "1"))
    return 2 if failed else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Stage 2 step 1 of 4 — find the print")
    ap.add_argument("root", nargs="?", default=".",
                    help="project root: the folder holding the scans "
                         "(default: this folder)")
    ap.add_argument("--review", action="store_true",
                    help="write a lightweight _marked JPEG per scan into "
                         "detect_review/, for visual checking; without it "
                         "this step writes coordinates only")
    ap.add_argument("--back", action="store_true",
                    help="backs of prints: frame the physical card edge")
    ap.add_argument("--ring", action="store_true",
                    help="find the print by its edge ring instead: one "
                         "print per scan, framed red, or blue on a back, in "
                         "the previews")
    ap.add_argument("--new-only", action="store_true",
                    help="measure only the scans the project file does not "
                         "hold yet, and keep every other record as it is")
    if parallel is not None:
        parallel.add_workers_arg(ap)
    else:
        ap.add_argument("--workers", type=int, default=1,
                        help="ignored: parallel.py not found")
    args = ap.parse_args(argv)
    console_setup()

    if int(cv2.__version__.split(".")[0]) < 4:
        raise SystemExit(f"OpenCV 4.x is required (found {cv2.__version__})")
    if args.ring and args.back:
        raise SystemExit("--ring finds fronts and backs itself: not with "
                         "--back")

    root = pipeline_root(args.root)
    check_scans_present(root)
    heading("STAGE 2 step 1  -  find the print")
    return run(root, args)


if __name__ == "__main__":
    raise SystemExit(main())
