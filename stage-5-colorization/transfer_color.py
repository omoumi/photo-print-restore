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
# transfer_color.py  --  STAGE 5, step 2 of 2: the colour on the print
#
# STAGE 5 colorizes the B&W prints: a model paints each of them, and only
# the model's colour is kept, laid on the print's own lightness:
#
#     colorize_model.py        step 1 of 2   colour from the model
#     transfer_color.py        step 2 of 2   the colour on the print
#     calibrate_color.py       sets step 2's colour numbers (not a step)
#     common.py                shared code, not a step
#     parallel.py              shared worker-count policy, not a step
#     colorize_by_hand.json    step 1's graph, for ComfyUI's own page
#
# Requirements:
#     pip install numpy tifffile opencv-python [imagecodecs] [psutil]
#     common.py beside this file; parallel.py too, for --workers
#
# Works on any folder of prints that has guides beside it, in
# colorize_guides/<name>_seed<N>.png (step 1 writes them). Only a B&W print
# -- one picture channel, as in step 1 -- is coloured, so a guide beside a
# colour print or a back is never used; a print without a guide is left
# out and counted. The results go to folders of their own. Step 1 run with
# --apply writes each print's result with this step's code as soon as the
# print's guide is made; run here, the step writes the results still
# missing.
#
# FIRST A PREVIEW, THEN --apply. Run without --apply, the step colours
# every print brought down to PREVIEW_SIDE on its long side and writes
#
#     colorize_preview/<name>.jpg     the print in B&W on the left, in
#                                     colour on the right
#
# in seconds, so the settings can be tried in turn. With --apply it colours
# every print at its full size and writes
#
#     colorized/<name>.tif            16-bit RGB when the print is 16-bit
#                                     (--out DIR writes them anywhere else)
#
# Both are made the same way with the same settings -- the guided filter's
# radius follows the size -- so the preview shows the result, only smaller.
# A preview is written again on every run. A result already written, and
# newer than its print and its guide, is left as it is -- so a run that was
# stopped goes on where it stopped, and a guide made again (step 1, --redo)
# or a print written again gets its result made again -- unless --redo,
# which writes them all again (after changing the settings, say).
#
# THE COLOUR AND THE LIGHTNESS. Every result keeps the print's lightness
# -- CIE L*, the print read as sRGB grey -- and takes only the guide's
# colour (a*, b*). Detail, grain and faces are the print's own, whatever
# the model redrew. A colour sRGB cannot show at a pixel's lightness is
# made fainter until it can (GAMUT_STEPS halvings), rather than clipped,
# which would change the lightness; so L* is kept to within the rounding
# of the output's bits. Lab is computed here exactly (CIE, D65 white), not
# with OpenCV's approximation.
#
# THE MODEL'S CAST. The model tints what should be white or grey -- walls,
# white shirts, tiles -- by about CAST_AB (a*, b*; yellow as measured).
# This is taken off every pixel's colour, nearly all of it where the colour
# is faint and less the stronger it is: the share taken off is
# exp(-(C/CAST_C)^2) at chroma C -- about 0.9 at the cast's own chroma,
# 0.37 at CAST_C -- so whites turn white while skin and flowers keep their
# colour.
#
# HOW STRONG THE COLOUR IS. The guide's colour is multiplied by SATURATION,
# and by its hue family's gain in HUE_GAINS: red, orange (skin), yellow,
# green, cyan, blue and purple, each at its hue angle in HUE_CENTRES; a
# colour between two families takes a gain between theirs, so there is no
# step. calibrate_color.py measures these numbers on colour prints made
# B&W; the yellow, cyan and blue gains here are lower than it measured,
# chosen by eye, so that white walls and shirts the model tinted cream or
# turquoise turn white -- at the cost of some real yellow and blue (a
# pool's water, a pale sky).
#
# LIGHT AND DARK. The model most often colours what is white or near it,
# and tints deep shadow blue. So, by the print's own lightness, the colour
# fades out above LIGHT_FADE (from the first L* to none at the second), and
# below DARK_FADE (from its second L* down to its third number, a share of
# the colour, at its first L*); the mid-tones keep it.
#
# The print keeps its own lightness and detail: the guide's colour is laid
# on it. That takes three passes, the first two measured at the smaller of
# the two sizes -- the guide brought down to the print's size when the print
# is the smaller (a small print, whose guide the model drew larger), else
# the print's lightness brought down to the guide's. Which size is chosen
# by the print's full size, also for a preview, so a preview's fit is the
# result's fit; only the third pass runs at the preview's size:
#
#     1. the whole guide onto the print: an affine fit (ECC, at ECC_SIDE)
#        of the guide's lightness to the print's -- a shift, a scale, a
#        slight turn. A fit that scales outside WARP_SCALE, turns more than
#        WARP_TURN degrees or shifts more than WARP_SHIFT of the picture's
#        side has found something else than the guide's place: it is not
#        used, and the report line says so;
#     2. every part of it onto the print: dense optical flow (DIS) between
#        the two, for what the model moved here and there; the flow is
#        smoothed over FLOW_SMOOTH, then each local move is held to
#        FLOW_MAX guide pixels;
#     3. the colour onto the print's edges: the colour, at the print's own
#        size, goes through a guided filter with the print's lightness as
#        the guide (radius GF_RADIUS guide pixels, GF_EPS), so it changes
#        where the print has an edge and stays even inside.
#
# FLOW_SMOOTH, FLOW_MAX and GF_RADIUS are in guide pixels at either size.
#
# Each print gets one line: the fit's shift and scale, and how far the
# flow moved things (mean and largest, in guide pixels). A large flow is
# where to look first.
#
# Usage:
#     python transfer_color.py [ROOT] [options]
#
#     ROOT                  project root: the folder of the prints (default:
#                           this folder)
#     --apply               write the full-size results (default: only the
#                           previews)
#     --redo                with --apply: write the results again, even
#                           those already written
#     --out DIR             with --apply: the folder for the results
#                           (default: ROOT\colorized); never ROOT
#     --workers N           parallel workers (default: auto, from the cores
#                           and the free RAM; 1 disables the pool)
#     --seed N              which guide: <name>_seed<N>.png (default: SEED
#                           in common.py, the seed step 1 uses)
#     --only NAME [NAME ...]
#                           only these prints (the extension may be left out)
#
#     python transfer_color.py D:\archive\restored
#     python transfer_color.py D:\archive\restored --apply --only raw0012
# ==========================================================================
from __future__ import annotations

import argparse
import os
import sys
import time

try:
    import cv2
    import numpy as np
except ModuleNotFoundError as exc:
    sys.exit(f"missing package: {exc.name} (see Requirements above)")

try:
    from common import (PREVIEW_DIR, SEED, Progress,
                        check_scans_present, console_setup, count,
                        decoded_bytes, elapsed, error, error_list, grey_of,
                        guide_dir, guide_name, heading, info, is_bw, newer,
                        paint, phase, pick_prints, pipeline_root, read_page,
                        resolve, result_dir, root_arg, unit_range, warn,
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
SCRIPT = "transfer_color.py"
WORKER_MEM_MULTIPLE = 40.0  # peak RAM of one worker per byte of the print
                            # once read (not on disk), full size (--apply)
PREVIEW_MEM_MULTIPLE = 8.0   # ... and for a preview

# --- the preview -----------------------------------------------------------
PREVIEW_SIDE = 1600         # the print's long side in the preview
PREVIEW_GAP = 16            # white between the B&W and the colour halves
PREVIEW_QUALITY = 90        # JPEG quality

# --- 1. the whole guide onto the print (ECC) -------------------------------
ECC_SIDE = 512              # the fit runs at this long side (then scaled)
ECC_ITERS = 100
ECC_EPS = 1e-5
ECC_BLUR = 5                # Gaussian size ECC smooths both with
WARP_SCALE = (0.9, 1.1)     # a fit that scales the guide outside this,
WARP_TURN = 3.0             # turns it more than this (degrees), or
WARP_SHIFT = 0.05           # shifts it more than this share of the
                            # picture's side, is not used

# --- 2. every part of it onto the print (DIS optical flow) ----------------
FLOW_MAX = 8.0              # a local move is held to this, in guide pixels
FLOW_SMOOTH = 3.0           # sigma the flow is smoothed with, guide pixels
CLAHE_CLIP = 2.0            # both are evened out (CLAHE) before the flow,
CLAHE_TILES = 8             # so the model's other brightness matters less

# --- 3. the colour onto the print's edges (guided filter) -----------------
GF_RADIUS = 3.0             # in guide pixels (scaled to the print's size)
GF_EPS = 1e-3               # larger: smoother colour, across weak edges too

# --- the colour ------------------------------------------------------------
CAST_AB = (0.0, 12.4)       # the model's cast, taken off (a*, b*)
CAST_C = 40.0               # share of the cast taken off: exp(-(C/CAST_C)^2)
                            # at chroma C -- 0.37 at CAST_C
SATURATION = 1.0            # every colour times this
HUE_GAINS = {               # each hue family's colour times this
    "red": 1.08, "orange": 0.69, "yellow": 0.17, "green": 0.63,
    "cyan": 0.15, "blue": 0.41, "purple": 0.67,
}
LIGHT_FADE = (72.0, 92.0)   # L*: colour starts to fade, and is gone
DARK_FADE = (15.0, 35.0, 0.3)   # L*: below the second, colour fades to
                                # the third share at the first
HUE_CENTRES = {             # each family's hue angle (CIE Lab, degrees)
    "red": 20, "orange": 60, "yellow": 95, "green": 145,
    "cyan": 205, "blue": 265, "purple": 320,
}
GAMUT_STEPS = 12            # halvings that find how much fainter a colour
                            # sRGB cannot show must be
ROWS = 256                  # rows coloured at a time, to hold RAM down

# --- CIE Lab, D65 white, sRGB primaries (not settings) ---------------------
_M = np.array([[0.412453, 0.357580, 0.180423],      # linear sRGB -> XYZ
               [0.212671, 0.715160, 0.072169],
               [0.019334, 0.119193, 0.950227]])
_M_INV = np.linalg.inv(_M)
_WHITE = _M.sum(axis=1)                             # Xn, Yn (= 1), Zn
_EPS, _KAPPA = 216.0 / 24389.0, 24389.0 / 27.0


# --------------------------------------------------------------------------
# lightness and colour
# --------------------------------------------------------------------------

def print_grey(path: str):
    """(grey in 0..1 as float32, the source array, its metadata)."""
    arr, meta = read_page(path)
    return grey_of(arr), arr, meta


def _linear(c):
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _encode(lin):
    lin = np.clip(lin, 0.0, 1.0)
    return np.where(lin <= 0.0031308, lin * 12.92,
                    1.055 * lin ** (1 / 2.4) - 0.055)


def _f(t):
    return np.where(t > _EPS, np.cbrt(t), (_KAPPA * t + 16.0) / 116.0)


def _l_of_y(y):
    return 116.0 * _f(y) - 16.0


def grey_to_l(grey: np.ndarray) -> np.ndarray:
    """CIE L* (0..100) of sRGB grey, float32."""
    return _l_of_y(_linear(grey)).astype(np.float32)


def l_to_grey(l_star: np.ndarray) -> np.ndarray:
    """sRGB grey (0..1) of CIE L*, float32: grey_to_l backwards."""
    fy = (np.asarray(l_star, np.float64) + 16.0) / 116.0
    y = np.where(fy ** 3 > _EPS, fy ** 3, (116.0 * fy - 16.0) / _KAPPA)
    return _encode(y).astype(np.float32)


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """CIE Lab (float32) of sRGB in 0..1, shape (..., 3)."""
    xyz = _linear(rgb) @ _M.T / _WHITE
    fx, fy, fz = _f(xyz[..., 0]), _f(xyz[..., 1]), _f(xyz[..., 2])
    return np.stack([116.0 * fy - 16.0, 500.0 * (fx - fy),
                     200.0 * (fy - fz)], axis=-1).astype(np.float32)


def _lab_to_linear(l_star, a, b):
    fy = (np.asarray(l_star, np.float64) + 16.0) / 116.0
    fx, fz = fy + np.asarray(a, np.float64) / 500.0, \
        fy - np.asarray(b, np.float64) / 200.0

    def inv(t):
        return np.where(t ** 3 > _EPS, t ** 3, (116.0 * t - 16.0) / _KAPPA)

    xyz = np.stack([inv(fx), inv(fy), inv(fz)], axis=-1) * _WHITE
    return xyz @ _M_INV.T


def lab_to_srgb(l_star, a, b) -> np.ndarray:
    """sRGB (0..1, float32) of CIE Lab. A colour sRGB cannot show at its
    lightness is made fainter until it can, so L* is kept."""
    lin = _lab_to_linear(l_star, a, b)
    out = ((lin < -1e-6) | (lin > 1 + 1e-6)).any(axis=-1)
    if out.any():
        l_o = np.asarray(l_star)[out]
        a_o, b_o = np.asarray(a)[out], np.asarray(b)[out]
        lo, hi = np.zeros(l_o.shape), np.ones(l_o.shape)
        for _ in range(GAMUT_STEPS):
            mid = (lo + hi) / 2
            t = _lab_to_linear(l_o, a_o * mid, b_o * mid)
            fits = ((t >= -1e-6) & (t <= 1 + 1e-6)).all(axis=-1)
            lo, hi = np.where(fits, mid, lo), np.where(fits, hi, mid)
        lin[out] = _lab_to_linear(l_o, a_o * lo, b_o * lo)
    return _encode(lin).astype(np.float32)


def read_guide(path: str) -> np.ndarray:
    """The guide as float32 Lab."""
    buf = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if bgr is None:
        raise OSError(f"could not read the guide {path}")
    return srgb_to_lab(bgr[..., ::-1].astype(np.float32) / 255.0)


# --------------------------------------------------------------------------
# 1. the whole guide onto the print
# --------------------------------------------------------------------------

def fit_whole(print_l: np.ndarray, guide_l: np.ndarray):
    """The affine warp that lays the guide on the print (both L*, same
    size), or None when ECC does not converge."""
    h, w = print_l.shape
    f = min(1.0, ECC_SIDE / float(max(h, w)))
    small = (max(8, round(w * f)), max(8, round(h * f)))
    p_s = cv2.resize(print_l, small, interpolation=cv2.INTER_AREA) / 100.0
    g_s = cv2.resize(guide_l, small, interpolation=cv2.INTER_AREA) / 100.0
    warp = np.eye(2, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ECC_ITERS,
            ECC_EPS)
    try:
        _, warp = cv2.findTransformECC(p_s, g_s, warp, cv2.MOTION_AFFINE,
                                       crit, None, ECC_BLUR)
    except cv2.error:
        return None
    warp[:, 2] /= f                     # the shift, back at the guide's size
    return warp


def warp_off(warp, size):
    """Why a fit is not used (see WARP_SCALE), or None when it is fine."""
    sx = float(np.hypot(warp[0, 0], warp[1, 0]))
    sy = float(np.hypot(warp[0, 1], warp[1, 1]))
    turn = float(np.degrees(np.arctan2(warp[1, 0], warp[0, 0])))
    shift = max(abs(float(warp[0, 2])) / size[0],
                abs(float(warp[1, 2])) / size[1])
    if not (WARP_SCALE[0] <= sx <= WARP_SCALE[1]
            and WARP_SCALE[0] <= sy <= WARP_SCALE[1]):
        return f"scale {sx:.2f}/{sy:.2f}"
    if abs(turn) > WARP_TURN:
        return f"turn {turn:+.1f} degrees"
    if shift > WARP_SHIFT:
        return f"shift {shift:.0%} of the side"
    return None


def apply_whole(img: np.ndarray, warp, size):
    return cv2.warpAffine(img, warp, size,
                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)


# --------------------------------------------------------------------------
# 2. every part of it onto the print
# --------------------------------------------------------------------------

def _even(l_star: np.ndarray) -> np.ndarray:
    u8 = np.clip(l_star * 2.55, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP,
                            tileGridSize=(CLAHE_TILES, CLAHE_TILES))
    return clahe.apply(u8)


def fit_parts(print_l: np.ndarray, guide_l: np.ndarray, px: float):
    """(map_x, map_y) that bring every part of the guide onto the print
    (both at the working size, `px` of its pixels to a guide pixel), and
    the flow's mean and largest move, in guide pixels."""
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow = dis.calc(_even(print_l), _even(guide_l), None)
    if FLOW_SMOOTH > 0:
        flow = cv2.GaussianBlur(flow, (0, 0), FLOW_SMOOTH * px)
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    limit = FLOW_MAX * px
    over = mag > limit
    if over.any():
        flow[over] *= (limit / mag[over])[:, None]
        mag = np.minimum(mag, limit)
    h, w = print_l.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    return (gx + flow[..., 0], gy + flow[..., 1],
            float(mag.mean()) / px, float(mag.max()) / px)


# --------------------------------------------------------------------------
# 3. the colour onto the print's edges
# --------------------------------------------------------------------------

def guided_filter(guide: np.ndarray, src: np.ndarray, r: int,
                  eps: float) -> np.ndarray:
    """He's guided filter: `src` smoothed so its changes follow the edges
    of `guide` (both float32, same size)."""
    k = (2 * r + 1, 2 * r + 1)

    def mean(x):
        return cv2.boxFilter(x, -1, k, borderType=cv2.BORDER_REFLECT)

    m_i, m_p = mean(guide), mean(src)
    var_i = mean(guide * guide) - m_i * m_i
    cov = mean(guide * src) - m_i * m_p
    a = cov / (var_i + eps)
    b = m_p - a * m_i
    return mean(a) * guide + mean(b)


# --------------------------------------------------------------------------
# the colour: the model's cast, the hue families' gains, light and dark
# --------------------------------------------------------------------------

def hue_segment(a: np.ndarray, b: np.ndarray):
    """(the families in hue order, each pixel's segment k, and t): a
    pixel's hue lies between family k's centre and the next one's, t of
    the way along."""
    names = sorted(HUE_CENTRES, key=lambda n: HUE_CENTRES[n] % 360)
    base = HUE_CENTRES[names[0]] % 360
    rel = np.array([(HUE_CENTRES[n] - base) % 360 for n in names] + [360.0])
    hue = (np.degrees(np.arctan2(b.astype(np.float64), a.astype(np.float64)))
           - base) % 360
    k = np.clip(np.searchsorted(rel, hue, side="right") - 1, 0,
                len(names) - 1)
    t = ((hue - rel[k]) / (rel[k + 1] - rel[k])).astype(np.float32)
    return names, k, t


def hue_weights(a: np.ndarray, b: np.ndarray):
    """[(family, weight)] for every pixel; the weights sum to 1."""
    names, k, t = hue_segment(a, b)
    n = len(names)
    return [(name, (np.where(k == i, 1.0 - t, 0.0)
                    + np.where(k == (i - 1) % n, t, 0.0)).astype(np.float32))
            for i, name in enumerate(names)]


def colour_gain(a: np.ndarray, b: np.ndarray, saturation: float,
                gains: dict) -> np.ndarray:
    """The factor every pixel's colour is multiplied by: the gains of the
    two families its hue lies between, shared by nearness, times
    `saturation`."""
    names, k, t = hue_segment(a, b)
    g = np.array([float(gains.get(n, 1.0)) for n in names] + [0.0],
                 np.float32)
    g[-1] = g[0]
    return (g[k] * (1.0 - t) + g[k + 1] * t) * np.float32(saturation)


def uncast(a: np.ndarray, b: np.ndarray, cast=None):
    """The guide's colour with the model's cast taken off: nearly all of it
    where the colour is faint, less where it is strong (see CAST_C)."""
    ca, cb = CAST_AB if cast is None else cast
    if not (ca or cb):
        return a, b
    w = np.exp(-np.square(np.hypot(a, b) / np.float32(CAST_C)))
    return a - np.float32(ca) * w, b - np.float32(cb) * w


def _smooth(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def light_fade(l_star: np.ndarray) -> np.ndarray:
    """The share of colour kept at each lightness (see LIGHT AND DARK)."""
    keep = 1.0 - _smooth(LIGHT_FADE[0], LIGHT_FADE[1], l_star)
    d0, d1, low = DARK_FADE
    keep *= 1.0 - (1.0 - low) * (1.0 - _smooth(d0, d1, l_star))
    return keep.astype(np.float32)


def finish(grey: np.ndarray, a: np.ndarray, b: np.ndarray,
           saturation: float = None, gains: dict = None,
           cast=None) -> np.ndarray:
    """The print in colour: its own L*, the colour without the model's
    cast, times its gain, faded in the lights and darks; sRGB float32
    0..1. ROWS rows at a time."""
    sat = SATURATION if saturation is None else saturation
    gains = HUE_GAINS if gains is None else gains
    out = np.empty(grey.shape + (3,), np.float32)
    for y0 in range(0, grey.shape[0], ROWS):
        rows = slice(y0, y0 + ROWS)
        ua, ub = uncast(a[rows], b[rows], cast)
        g = colour_gain(ua, ub, sat, gains)
        l_star = grey_to_l(grey[rows])
        g *= light_fade(l_star)
        out[rows] = lab_to_srgb(l_star, ua * g, ub * g)
    return out


# --------------------------------------------------------------------------
# one print
# --------------------------------------------------------------------------

def colour(grey: np.ndarray, guide_path: str, size=None):
    """The guide's colour on the print: `grey` is the print at full size
    (0..1); the colour is made at `size` (width, height; default: full).
    (sRGB float32 0..1, the grey at that size, the one-line report)."""
    a, b, small, line = colour_ab(grey, guide_path, size)
    return finish(small, a, b), small, line


def lay_guide(grey: np.ndarray, guide_path: str):
    """The guide's a*, b* laid on the print, measured on `grey` (the print
    at full size) at the smaller of the two sizes: (a*b* at that size, the
    guide's width, the one-line report)."""
    big_h, big_w = grey.shape
    guide = read_guide(guide_path)
    gh, gw = guide.shape[:2]
    if big_w * big_h < gw * gh:           # a small print: the guide comes down
        size = (big_w, big_h)
        guide = cv2.resize(guide, size, interpolation=cv2.INTER_AREA)
        print_l = grey_to_l(grey)
    else:                                 # the print's lightness comes down
        size = (gw, gh)
        print_l = grey_to_l(cv2.resize(grey, size,
                                       interpolation=cv2.INTER_AREA))
    px = size[0] / float(gw)              # working pixels per guide pixel
    notes = []

    warp = fit_whole(print_l, guide[..., 0])
    if warp is None:
        notes.append("whole fit failed: none applied")
        warp = np.eye(2, 3, dtype=np.float32)
    else:
        why = warp_off(warp, size)
        if why:
            notes.append(f"whole fit not used: {why}")
            warp = np.eye(2, 3, dtype=np.float32)
    guide = apply_whole(guide, warp, size)
    sx = float(np.hypot(warp[0, 0], warp[1, 0]))
    sy = float(np.hypot(warp[0, 1], warp[1, 1]))

    mx, my, f_mean, f_max = fit_parts(print_l, guide[..., 0], px)
    ab = cv2.remap(np.ascontiguousarray(guide[..., 1:]), mx, my,
                   cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    line = (f"shift {warp[0, 2] / px:+.1f},{warp[1, 2] / px:+.1f} px, scale "
            f"{sx:.3f}/{sy:.3f}; flow mean {f_mean:.1f}, max {f_max:.1f} px")
    return ab, gw, line + (f"  ({'; '.join(notes)})" if notes else "")


def colour_ab(grey: np.ndarray, guide_path: str, size=None):
    """The guide's colour laid on the print, before any gain: `grey` is the
    print at full size; the colour is made at `size` (width, height;
    default: full). (a*, b*, the grey at that size, the one-line
    report)."""
    ab, gw, line = lay_guide(grey, guide_path)
    h, w = grey.shape
    size = tuple(size or (w, h))
    if size != (w, h):
        grey = cv2.resize(grey, size, interpolation=cv2.INTER_AREA)
    if ab.shape[1::-1] != size:
        ab = cv2.resize(ab, size, interpolation=cv2.INTER_LINEAR)
    r = max(1, int(round(GF_RADIUS * size[0] / float(gw))))
    a = guided_filter(grey, np.ascontiguousarray(ab[..., 0]), r, GF_EPS)
    b = guided_filter(grey, np.ascontiguousarray(ab[..., 1]), r, GF_EPS)
    return a, b, grey, line


def write_preview(src: str, guide_path: str, dst: str) -> str:
    """The print at PREVIEW_SIDE, B&W beside colour, as a JPEG."""
    grey, _, _ = print_grey(src)
    h, w = grey.shape
    f = min(1.0, PREVIEW_SIDE / float(max(h, w)))
    size = (max(1, round(w * f)), max(1, round(h * f)))
    rgb, small, line = colour(grey, guide_path, size)
    del grey
    ph, pw = small.shape
    sheet = np.ones((ph, 2 * pw + PREVIEW_GAP, 3), np.float32)
    sheet[:, :pw] = small[..., None]
    sheet[:, pw + PREVIEW_GAP:] = rgb
    u8 = np.round(sheet * 255.0).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(u8[..., ::-1]),
                           [cv2.IMWRITE_JPEG_QUALITY, PREVIEW_QUALITY])
    if not ok:
        raise OSError(f"could not encode {dst}")
    tmp = dst + ".tmp"
    buf.tofile(tmp)
    os.replace(tmp, dst)
    return line


def write_result(src: str, guide_path: str, dst: str) -> str:
    """The print at its full size, in colour, as a TIFF."""
    grey, arr, meta = print_grey(src)
    rgb, _, line = colour(grey, guide_path)
    del grey
    if np.issubdtype(arr.dtype, np.integer):
        out = np.round(rgb * unit_range(arr.dtype)).astype(arr.dtype)
    else:
        out = rgb.astype(np.float32)
    del rgb, arr

    # a grey colour profile does not fit RGB: the result carries none
    meta = dict(meta, photometric="rgb", planarconfig=1, extrasamples=None,
                colormap=None, iccprofile=None)
    if str(meta.get("description") or "").startswith('{"shape"'):
        meta["description"] = None        # tifffile's own shape note: the
                                          # grey's shape, not the result's
    tmp = dst + ".tmp"
    write_tiff(tmp, out, meta)
    os.replace(tmp, dst)
    return line


# --------------------------------------------------------------------------
# the step
# --------------------------------------------------------------------------

def _job(task):
    """One print, in a worker: (name, its report line or None, the failure
    or None). Top level, so the pool can pickle it."""
    name, src, guide, dst, apply, single = task
    if not single:
        cv2.setNumThreads(1)        # the pool spreads the work already
    try:
        line = (write_result if apply else write_preview)(src, guide, dst)
        return name, line, None
    except Exception as exc:
        return name, None, f"{type(exc).__name__}: {exc}"


def run(root, args) -> int:
    t0 = time.time()
    gdir = guide_dir(root)
    odir = (args.out_dir if args.apply
            else os.path.join(resolve(root), PREVIEW_DIR))
    ext = ".tif" if args.apply else ".jpg"

    def guide(n):
        return os.path.join(gdir, guide_name(n, args.seed))

    phase(1, 2, "the prints and their guides")
    names, missing, clashes = pick_prints(root, args.only)
    if missing:
        warn_list(f"    ! {count(len(missing), 'name')} given with --only "
                  f"not found:", missing)
    if clashes:
        warn_list(f"    ! {count(len(clashes), 'print')} left out: the same "
                  f"name but for the extension or the case:", clashes)
    bw, other, unreadable = [], [], []
    for n in names:
        one, err = is_bw(os.path.join(root, n))
        if err:
            unreadable.append(f"{n}: {err}")
        elif one:
            bw.append(n)
        else:
            other.append(n)
    guided = [n for n in bw if os.path.isfile(guide(n))]
    todo = [n for n in guided if newer(guide(n), os.path.join(root, n))]
    stale = len(guided) - len(todo)
    info(f"{count(len(bw), 'B&W print')} (one channel); {len(todo)} with a "
         f"guide (seed {args.seed}), {len(bw) - len(todo)} without")
    info(f"left out: {count(len(other), 'print')} in RGB (colour prints, "
         f"backs)")
    if stale:
        warn(f"    ! {count(stale, 'guide')} older than the print: left "
             f"out until step 1 makes {'it' if stale == 1 else 'them'} "
             f"again")
    if unreadable:
        warn_list(f"    ! {count(len(unreadable), 'print')} could not be "
                  f"read; left out:", unreadable)
    if args.only and not names:
        error("none of the prints given with --only is in the folder")
        return 1
    if not todo:
        error(f"no guides in {gdir} named <name>_seed{args.seed}.png, "
              f"newer than their print")
        return 1
    if args.apply and not args.redo:
        fresh = [n for n in todo if newer(
            os.path.join(odir, os.path.splitext(n)[0] + ext),
            guide(n), os.path.join(root, n))]
        todo = [n for n in todo if n not in fresh]
        info(f"{count(len(fresh), 'result')} written already, "
             f"{len(todo)} to write")
        if not todo:
            print(paint(f"\nnothing to write; the results are in {odir}",
                        "32"))
            return 0

    phase(2, 2, "the colour on the prints" if args.apply
          else f"the previews ({PREVIEW_SIDE} px)")
    os.makedirs(odir, exist_ok=True)
    workers = 1
    if parallel is not None and args.workers != 1:
        biggest = max(decoded_bytes(os.path.join(root, n)) for n in todo)
        per = max(parallel.MIN_BYTES_PER_WORKER, int(biggest * (
            WORKER_MEM_MULTIPLE if args.apply else PREVIEW_MEM_MULTIPLE)))
        workers = parallel.choose_workers(len(todo), per, args.workers)
    info(count(workers, "worker"))
    tasks = [(n, os.path.join(root, n), guide(n),
              os.path.join(odir, os.path.splitext(n)[0] + ext),
              args.apply, workers == 1) for n in todo]
    failed, made = [], 0
    bar = Progress(len(todo), "    colouring")

    def report(_i, res):
        nonlocal made
        name, line, err = res
        if err:
            failed.append(f"{name}: {err}")
            bar.message(paint(f"    ! {name}: failed", "33"))
        else:
            made += 1
            bar.message(paint(f"    {name}: {line}", "90"))
        bar.update()

    try:
        if parallel is not None and workers > 1 and len(tasks) > 1:
            parallel.run(_job, tasks, workers, on_result=report,
                         on_error=lambda task, exc: (
                             task[0], None, parallel.describe_error(exc)))
        else:
            for i, t in enumerate(tasks):
                try:
                    res = _job(t)
                except Exception as exc:
                    res = (t[0], None, f"{type(exc).__name__}: {exc}")
                report(i, res)
    except KeyboardInterrupt:
        bar.close()
        warn(f"\nstopped: {count(made, 'print')} written; run the step "
             f"again to go on")
        return 1
    bar.close()

    print(paint(f"\nwrote {count(made, 'print')} in {odir}", "32"))
    if failed:
        error_list(f"    {count(len(failed), 'print')} failed:", failed)
    print(paint(f"    done in {elapsed(t0)}", "32"))
    if not args.apply and made:
        only = (" --only " + " ".join(args.only)) if args.only else ""
        seed = f" --seed {args.seed}" if args.seed != SEED else ""
        print(paint(f"\nnext, after looking at the previews:\n    python "
                    f"{SCRIPT}{root_arg(root)} --apply{seed}{only}", "1"))
    return 2 if failed else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Stage 5 step 2 of 2 — the colour on the print")
    ap.add_argument("root", nargs="?", default=".",
                    help="project root: the folder of the prints (default: "
                         "this folder)")
    ap.add_argument("--apply", action="store_true",
                    help="write the full-size results (default: only the "
                         "previews)")
    ap.add_argument("--redo", action="store_true",
                    help="with --apply: write the results again, even those "
                         "already written")
    ap.add_argument("--out", default=None, metavar="DIR",
                    help="with --apply: the folder for the results "
                         "(default: ROOT/colorized)")
    if parallel is not None:
        parallel.add_workers_arg(ap)
    else:
        ap.add_argument("--workers", type=int, default=1,
                        help="ignored: parallel.py not found")
    ap.add_argument("--seed", type=int, default=SEED,
                    help=f"which guide: <name>_seed<N>.png (default {SEED})")
    ap.add_argument("--only", nargs="+", metavar="NAME",
                    help="only these prints (the extension may be left out)")
    args = ap.parse_args(argv)
    console_setup()
    if args.redo and not args.apply:
        raise SystemExit("--redo goes with --apply (the previews are always "
                         "written again)")
    if args.out and not args.apply:
        raise SystemExit("--out goes with --apply (the previews always go to "
                         "colorize_preview/)")

    root = pipeline_root(args.root)
    check_scans_present(root)
    args.out_dir = result_dir(root, args.out) if args.apply else None
    heading("STAGE 5 step 2  -  the colour on the print")
    return run(root, args)


if __name__ == "__main__":
    raise SystemExit(main())
