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
# deskew_crop.py  --  STAGE 2, step 4 of 4: straighten, crop and orient
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
#     pip install numpy tifffile opencv-python [imagecodecs] [psutil]
#     common.py beside this file; parallel.py too, for --workers
#
# FOR EVERY PRINT, in four parts:
#
#   1. DECIDE the tilt in the frame review.py showed. The approved corners go
#      through the approved flip (the archive flip XOR the scan's own V) and
#      quarter turns, and common.tilt_from_corners measures the tilt there.
#      The way up is already settled by that flip and turn, so the tilt is
#      only ever the small crooked remainder: the search covers -45..+45
#      degrees and cannot add a quarter turn of its own. Nothing else feeds
#      the angle -- not the `angle` field of the box, not a detector.
#   2. STRAIGHTEN the scan as it sits on disk by the same tilt, only if it is
#      at least --min-angle; otherwise the print is sliced out untouched and
#      keeps every sample exactly. A quarter turn does not change a tilt and
#      a flip mirrors it, so the tilt in the scan's own frame is the decided
#      tilt, negated when the scan is flipped.
#   3. CROP the full bounding box of the straightened print.
#   4. ORIENT the crop by the approved flip and quarter turns. Both only move
#      pixels, so this part is lossless.
#
# THE ONE PLACE ANY SAMPLE IS RECOMPUTED is part 2. The scan is turned about
# its own centre onto a canvas large enough to hold all of it
# (rotation_matrix), and only the crop's pixels of that canvas are computed.
# Each is read from its exact position in the scan -- the position is never
# rounded -- by Lanczos interpolation with a = 4: the 8 x 8 scan samples
# around that position, weighted by the Lanczos kernel, the weights
# normalised to sum to one, all in float64. The value is rounded once, to
# the scan's own sample type, when the crop is written.
#
# THE SCAN'S EDGE. Where the kernel of a pixel reaches past the edge of the
# scan, the edge row or column is repeated for the missing samples. A pixel
# whose own position lies outside the scan -- part of a print that was off
# the glass -- has no data at all; it is written as 0 (black) and counted in
# the manifest as no_data_pixels. Nothing of the print that was scanned is
# ever cut off.
#
# THE CROP IS THE FULL BOUNDING BOX of the straightened quad, not the inner
# rectangle. A print is never a perfect rectangle, so an inner rectangle
# would shave real emulsion off two edges to avoid a sliver of album page at
# the corners. A sliver can be dealt with later; emulsion thrown away here
# cannot. The manifest records where the print's own outline lies inside
# each crop, so a later stage can tell the two apart without measuring
# again.
#
# Every print on a sheet is a separate piece of paper, so each is measured
# and straightened by its own corners.
#
# Outputs, derived from the project root -- the folder holding the scans --
# unless --out names another folder for the crops:
#
#     deskewed-cropped/<n>[_NN].tif      the finished crops
#     json files/boxes_deskew_crop.json  the manifest: every number needed
#                                        to rebuild each crop exactly
#
# The manifest describes every crop of the current plan that is in the
# output folder: the ones this run wrote, plus the ones an earlier run wrote
# into the same folder and this run left in place.
#
# The flip and turn decisions (archive-wide and per-scan) are read from the
# project file; there is no flag. What this step writes is by construction
# what review.py showed.
#
# Usage:
#     python deskew_crop.py [ROOT] [options]
#
#     ROOT              project root: the folder holding the scans
#                       (default: this folder)
#     --min-angle DEG   below this the print is sliced without straightening,
#                       so every sample is kept exactly (default:
#                       common.STRAIGHTEN_MIN_DEG; review.py warns from the
#                       same value)
#     --compress NAME   keep (default), none, lzw or deflate. All lossless.
#     --all             also crop files never confirmed in review.py
#     --overwrite       replace existing outputs
#     --out DIR         write the crops to DIR instead of
#                       ROOT\deskewed-cropped. Created if missing; never the
#                       scans folder itself, whose files the crops are named
#                       after
#     --verify          afterwards, rebuild every crop in the manifest from
#                       its raw scan and compare SHA-256 (see the --verify
#                       section below for what is and is not independent).
#                       About as long again as the crop itself. Run again
#                       with the same --out when every crop is already there
#                       and it checks them without making any. Exit code 3
#                       if it fails
#     --workers N       parallel workers (default: auto; 1 disables the pool)
#
#     python deskew_crop.py
#     python deskew_crop.py D:\archive\scans --out D:\archive\crops --verify
# ==========================================================================
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time

try:
    import cv2
    import numpy as np
    import tifffile
except ModuleNotFoundError as exc:
    sys.exit(f"missing package: {exc.name} (see Requirements above)")

try:
    from common import (JSON_DIR, SCHEMA_VERSION, STAGE_DESKEW_CROP,
                        STAGE_DETECT, STAGE_REVIEW, STRAIGHTEN_MIN_DEG,
                        check_scans_present, console_setup, count,
                        cropped_dir, elapsed, error, heading, info,
                        load_project, orient_image, orient_points, paint,
                        phase, pipeline_root, read_page, resolve, root_arg,
                        same_dir, stage_json, swap_resolution,
                        tilt_from_corners, warn, warn_list, write_json,
                        write_tiff)
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
SCRIPT = "deskew_crop.py"
WORKER_MEM_MULTIPLE = 6.0   # peak RAM of one worker per byte of the scan
                            # file: the decoded scan, the crop in float64
                            # and one block's weights, about 4x the file
                            # for an uncompressed 16-bit RGB scan
VERIFY_MEM_MULTIPLE = 6.0   # the same for --verify: the scan, the rebuilt
                            # crop and the file under test

# --- straightening and cropping --------------------------------------------
METHOD = "lanczos4-exact"   # recorded in the manifest
LANCZOS_A = 4               # kernel half-width: 2a = 8 samples each way
BLOCK_PIXELS = 131072       # pixels resampled at a time (bounds the memory)
MIN_CROP_PX = 8             # a crop narrower than this is refused

# --------------------------------------------------------------------------
# straightening, in the scan's own frame
# --------------------------------------------------------------------------
# angle_deg follows OpenCV: positive turns counter-clockwise on screen. A
# print tilted clockwise by t is straightened by angle_deg = t.

def rotation_matrix(w, h, angle):
    """Rotation about the frame centre, onto a canvas large enough to hold
    the whole rotated source -- no pixel is pushed off the edge. Returns
    (M, out_w, out_h), M taking a scan position to its canvas position.

    --verify rebuilds this from the same two numbers rather than trusting
    the matrix in the manifest, so this function is part of the contract:
    change the rounding here and every stored crop stops verifying."""
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    c, s = abs(m[0, 0]), abs(m[0, 1])
    nw = int(math.ceil(h * s + w * c))
    nh = int(math.ceil(h * c + w * s))
    m[0, 2] += nw / 2.0 - w / 2.0
    m[1, 2] += nh / 2.0 - h / 2.0
    return m, nw, nh


def map_points(pts, m):
    """Push (x, y) pairs through a 2x3 affine matrix."""
    p = np.asarray(pts, np.float64).reshape(-1, 2)
    return p @ m[:, :2].T + m[:, 2]


def inverse_affine(m):
    """The 2x3 matrix taking a canvas position back to the scan position it
    is read from."""
    a = np.asarray(m, np.float64)[:, :2]
    ai = np.linalg.inv(a)
    return np.hstack([ai, -(ai @ np.asarray(m, np.float64)[:, 2:])])


def _lanczos_taps(s, a=LANCZOS_A):
    """For source coordinates `s` along one axis (any array shape): the index
    of the first of the 2a samples the kernel reads, and their 2a weights."""
    base = np.floor(s).astype(np.int64) - (a - 1)
    weights = []
    for k in range(2 * a):
        x = s - (base + k)                 # distance to that sample
        weights.append(np.sinc(x) * np.sinc(x / a))
    return base, weights


def lanczos_crop(arr, inv, box, a=LANCZOS_A):
    """(crop in float64, no-data mask) for the canvas pixels of `box`
    (x0, y0, x1, y1), each read from the scan `arr` at inv @ (x, y, 1).

    Samples a kernel needs from beyond the scan's edge repeat the edge.
    A pixel whose own position lies outside the scan's area -- more than
    half a sample past its first or last sample -- has no data: it is
    marked in the mask and set to 0. Rows are done a block at a time so the
    weight arrays stay small.

    The 2a x 2a weights are the product of a row weight and a column weight,
    so each kernel row is summed along x first and then weighted once in y,
    and the normalisation is (sum of x weights) * (sum of y weights). The
    samples are gathered from a flat view with np.take, which is faster
    than two-axis fancy indexing."""
    x0, y0, x1, y1 = box
    planes = arr[..., None] if arr.ndim == 2 else arr
    h, w, nch = planes.shape
    flat = planes.reshape(-1, nch)
    out = np.empty((y1 - y0, x1 - x0, nch), np.float64)
    nodata = np.zeros((y1 - y0, x1 - x0), bool)
    u = np.arange(x0, x1, dtype=np.float64)[None, :]
    rows = max(1, BLOCK_PIXELS // max(1, x1 - x0))
    for r0 in range(y0, y1, rows):
        r1 = min(y1, r0 + rows)
        v = np.arange(r0, r1, dtype=np.float64)[:, None]
        sx = inv[0, 0] * u + inv[0, 1] * v + inv[0, 2]
        sy = inv[1, 0] * u + inv[1, 1] * v + inv[1, 2]
        bx, wx = _lanczos_taps(sx, a)
        by, wy = _lanczos_taps(sy, a)
        xi = [np.clip(bx + i, 0, w - 1) for i in range(2 * a)]
        acc = np.zeros(sx.shape + (nch,), np.float64)
        row_sum = np.empty_like(acc)
        for j in range(2 * a):
            base = np.clip(by + j, 0, h - 1) * w
            row_sum[...] = 0.0
            for i in range(2 * a):
                row_sum += wx[i][..., None] * np.take(flat, base + xi[i],
                                                      axis=0)
            acc += wy[j][..., None] * row_sum
        norm = sum(wx) * sum(wy)
        out[r0 - y0:r1 - y0] = acc / norm[..., None]
        nodata[r0 - y0:r1 - y0] = ((sx < -0.5) | (sx > w - 0.5)
                                   | (sy < -0.5) | (sy > h - 0.5))
    out[nodata] = 0.0
    return (out if arr.ndim == 3 else out[..., 0]), nodata


def to_dtype(a, dtype):
    """Round once to the scan's own sample type."""
    dt = np.dtype(dtype)
    if dt.kind == "f":
        return a.astype(dt)
    info = np.iinfo(dt)
    return np.clip(np.rint(a), info.min, info.max).astype(dt)


def crop_box_of(quad, bounds=None):
    """The box that is actually cut out: the quad's FULL bounding box --
    nothing of the print is thrown away here. `bounds`, when given, limits
    it (a slice cannot reach past the scan)."""
    q = np.asarray(quad, float)
    b = [int(math.floor(q[:, 0].min())), int(math.floor(q[:, 1].min())),
         int(math.ceil(q[:, 0].max())), int(math.ceil(q[:, 1].max()))]
    if bounds is not None:
        b = [max(bounds[0], b[0]), max(bounds[1], b[1]),
             min(bounds[2], b[2]), min(bounds[3], b[3])]
    if b[2] - b[0] < MIN_CROP_PX or b[3] - b[1] < MIN_CROP_PX:
        raise ValueError(f"the approved box barely overlaps the scan: "
                         f"{max(0, b[2] - b[0])}x{max(0, b[3] - b[1])} px "
                         f"would be left. Check this box in review.py")
    return b


def out_name(name: str, index: int, total: int) -> str:
    stem, ext = os.path.splitext(name)
    return f"{stem}{ext}" if total == 1 else f"{stem}_{index + 1:02d}{ext}"


def plan_crop(size, corners, flip, rot, min_angle):
    """Everything about one print, from its approved corners in the scan's
    own frame. Pure geometry: no pixels are touched here.

    A sliced print is cut from the scan itself, so its box stops at the
    scan's edge: beyond it nothing was scanned. A straightened print keeps
    its full box, and the few pixels of it that lie outside the scan are the
    no-data pixels of lanczos_crop."""
    h, w = int(size[0]), int(size[1])
    tilt, keep = tilt_from_corners(orient_points(corners, (h, w), flip, rot))
    angle = -tilt if flip else tilt            # the same tilt, scan's frame
    if abs(tilt) < min_angle:
        m = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        nw, nh, act, angle = w, h, "sliced", 0.0
        quad = np.asarray(corners, np.float64)
        box = crop_box_of(quad, [0, 0, w, h])
    else:
        m, nw, nh = rotation_matrix(w, h, angle)
        act = "straightened"
        quad = map_points(corners, m)
        box = crop_box_of(quad)
    q_out = orient_points(quad - np.array([box[0], box[1]], float),
                          (box[3] - box[1], box[2] - box[0]), flip, rot)
    return {"action": act, "tilt": tilt, "angle": angle, "keep": keep,
            "m": m, "inv": inverse_affine(m), "canvas": [nw, nh],
            "crop_box": box, "quad": q_out}


# --------------------------------------------------------------------------
# one scan
# --------------------------------------------------------------------------

def _job(task):
    """Straighten, crop and orient every print on one scan. Top-level so it
    can be pickled."""
    (src, flip, rot, jobs, compress, min_angle) = task
    tmp = None
    try:
        arr, meta = read_page(src)
        h, w = arr.shape[:2]
        meta = swap_resolution(meta, rot)
        written = []
        for (dst, corners) in jobs:
            g = plan_crop((h, w), corners, flip, rot, min_angle)
            x0, y0, x1, y1 = g["crop_box"]
            no_data = 0
            if g["action"] == "sliced":
                piece = arr[y0:y1, x0:x1]
            else:
                val, mask = lanczos_crop(arr, g["inv"], g["crop_box"])
                piece = to_dtype(val, arr.dtype)
                no_data = int(mask.sum())
            piece = orient_image(piece, flip, rot)
            # written under a temporary name first, so a crash never leaves
            # a half-written crop that a later run would take as done
            tmp = dst + ".part"
            write_tiff(tmp, piece, meta, compress=compress)
            os.replace(tmp, dst)
            tmp = None
            written.append({
                "file": os.path.basename(dst),
                "action": g["action"],
                "tilt_deg": g["tilt"], "angle_deg": g["angle"],
                "kept_fraction": round(g["keep"], 5),
                "canvas": g["canvas"], "crop_box": g["crop_box"],
                "inverse": g["inv"].tolist(), "no_data_pixels": no_data,
                "size": [int(piece.shape[0]), int(piece.shape[1])],
                "quad": [[round(float(v), 2) for v in pt] for pt in g["quad"]],
                "numpy_version": np.__version__,
            })
        return (os.path.basename(src), w, h, written, None)
    except Exception as exc:            # a bad file must not stop the run
        if tmp is not None:
            try:
                os.remove(tmp)
            except OSError:
                pass
        return (os.path.basename(src), 0, 0, [],
                f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------
# --verify: rebuild every crop from its raw scan and compare
# --------------------------------------------------------------------------
# What this check rebuilds with its own code: the TIFF reader, the rotation
# matrix and canvas size from the recorded angle (by hand, not
# getRotationMatrix2D), a check that the recorded inverse matrix is that
# rotation's inverse, the flip and the quarter turns (a transpose plus a
# reversal, not np.rot90), and the count of no-data pixels.
#
# What it shares with step 4: the Lanczos resampling itself, lanczos_crop,
# which is fed the recorded inverse so the rebuild repeats the step's
# arithmetic exactly. A bug in lanczos_crop would therefore not be caught
# here.
#
# The resampling is plain float64 arithmetic in numpy, repeatable on one
# installation; a manifest made under a different numpy version is reported
# rather than silently failed. On a print that was only sliced --
# tilt below --min-angle -- no sample was computed at all and a stronger
# claim holds: the multiset of sample values cannot have changed, only their
# positions.

# The tags a crop carries over from its scan, listed here independently of
# common.write_tiff. Every other tag of the scan is left behind on purpose:
# the structural ones are computed afresh for the crop, and Orientation
# would make every viewer turn the already-turned pixels again.
_CARRIED_TAGS = {
    "PhotometricInterpretation", "PlanarConfiguration", "BitsPerSample",
    "SamplesPerPixel", "ExtraSamples", "XResolution", "YResolution",
    "ResolutionUnit", "InterColorProfile", "ImageDescription", "DateTime",
    "Software", "ColorMap", "Make", "Model", "Artist", "Copyright",
    "HostComputer", "PageName",
}


def _read_checked(path):
    """Pixels plus the metadata worth comparing, in the file's own dtype.

    Written without common.read_page on purpose: a checker should not share
    a reader with the thing it checks."""
    meta = {"icc": None, "xres": None, "yres": None, "resunit": None,
            "depth": None, "photometric": None, "tags": set(), "pages": 1}
    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        a = page.asarray()
        if a.ndim == 3 and getattr(page, "planarconfig", 1) == 2 and \
                a.shape[0] <= 4 < a.shape[1]:
            a = np.transpose(a, (1, 2, 0))
        meta["pages"] = len(tf.pages)
        meta["tags"] = set(t.name for t in page.tags.values())
        meta["photometric"] = str(page.photometric).split(".")[-1]
        bps = page.bitspersample
        meta["depth"] = (bps if isinstance(bps, int) else bps[0]) \
            * page.samplesperpixel
        tg = page.tags
        if "InterColorProfile" in tg:
            meta["icc"] = hashlib.sha256(
                bytes(tg["InterColorProfile"].value)).hexdigest()[:16]

        def _num(v):
            """A TIFF rational and a plain number mean the same thing;
            compare what they denote, not how they are stored."""
            try:
                if isinstance(v, tuple) and len(v) == 2 and v[1]:
                    return round(float(v[0]) / float(v[1]), 6)
                return round(float(v), 6)
            except Exception:
                return v

        for k, name in (("xres", "XResolution"), ("yres", "YResolution")):
            if name in tg:
                meta[k] = _num(tg[name].value)
        if "ResolutionUnit" in tg:
            try:
                meta["resunit"] = int(tg["ResolutionUnit"].value)
            except Exception:
                meta["resunit"] = str(tg["ResolutionUnit"].value)
    a = np.asarray(a)
    return (a[:, :, None] if a.ndim == 2 else a), meta


def _own_orient(a, flip, rot):
    """The flip and quarter turns, written independently of common.py."""
    if flip:
        a = a[::-1]
    for _ in range((int(rot) // 90) % 4):          # one quarter turn clockwise
        a = np.transpose(a, (1, 0, 2))[:, ::-1]
    return a


def _own_matrix(w, h, angle):
    """The same rotation, built from the angle by hand.

    Not getRotationMatrix2D, and not the matrix in the manifest: if either
    the stage or the manifest has the geometry wrong, this has to disagree."""
    t = math.radians(angle)
    c, s = math.cos(t), math.sin(t)
    m = np.array([[c, s, 0.0], [-s, c, 0.0]], np.float64)
    m[0, 2] = (1 - c) * (w / 2.0) - s * (h / 2.0)
    m[1, 2] = s * (w / 2.0) + (1 - c) * (h / 2.0)
    nw = int(math.ceil(h * abs(s) + w * abs(c)))
    nh = int(math.ceil(h * abs(c) + w * abs(s)))
    m[0, 2] += nw / 2.0 - w / 2.0
    m[1, 2] += nh / 2.0 - h / 2.0
    return m, nw, nh


def _inverse_problem(ph, m):
    """None, or why the recorded inverse is not the inverse of `m`: every
    corner of the crop box must go to the canvas and back to itself."""
    inv = np.asarray(ph["inverse"], np.float64)
    x0, y0, x1, y1 = ph["crop_box"]
    for (px, py) in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
        sx, sy = inv @ np.array([px, py, 1.0])
        bx, by = m @ np.array([sx, sy, 1.0])
        if abs(bx - px) > 1e-6 or abs(by - py) > 1e-6:
            return (f"the recorded inverse matrix does not undo the rotation "
                    f"at ({px}, {py})")
    return None


def _rebuild(raw, ph, flip, rot, a):
    """(what the crop SHOULD be, its no-data count) from the raw scan
    (H, W, C)."""
    x0, y0, x1, y1 = ph["crop_box"]
    if ph["action"] == "sliced":
        piece, nodata = raw[y0:y1, x0:x1], 0
    else:
        v, mask = lanczos_crop(raw, np.asarray(ph["inverse"], np.float64),
                               ph["crop_box"], a)
        if raw.dtype.kind == "f":
            piece = v.astype(raw.dtype)
        else:
            lim = np.iinfo(raw.dtype)
            piece = np.minimum(np.maximum(np.rint(v), lim.min),
                               lim.max).astype(raw.dtype)
        nodata = int(mask.sum())
    return np.ascontiguousarray(_own_orient(piece, flip, rot)), nodata


def _sha(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _verify_job(task):
    """Check one crop against its raw scan. Top-level so it can be pickled.

    Both files are re-read inside the worker, so only paths and a handful of
    numbers cross a pipe and the large arrays never do."""
    raw_path, out_path, flip, rot, ph, lanczos_a = task
    name = os.path.basename(out_path)
    bad = []
    try:
        a_in, m_in = _read_checked(raw_path)
        a_out, m_out = _read_checked(out_path)
    except Exception as exc:
        return (name, ["cannot read: %s" % exc], None, None)

    if a_in.dtype != a_out.dtype:
        bad.append(f"dtype changed {a_in.dtype} -> {a_out.dtype}")
    if m_in["depth"] != m_out["depth"]:
        bad.append(f"bit depth {m_in['depth']} -> {m_out['depth']}")
    if a_in.shape[2] != a_out.shape[2]:
        bad.append("channel count changed")
    if m_in["photometric"] != m_out["photometric"]:
        bad.append(f"photometric {m_in['photometric']} -> "
                   f"{m_out['photometric']}")
    if m_out["pages"] > 1:
        bad.append("output has more than one page")
    shape = f"{a_out.shape[1]}x{a_out.shape[0]}"
    h, w = a_in.shape[:2]
    m = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    if not bad and ph["action"] != "sliced":
        m, nw, nh = _own_matrix(w, h, ph["angle_deg"])
        if [nw, nh] != list(ph["canvas"]):
            bad.append(f"canvas should be {nw}x{nh} for this angle, the "
                       f"manifest says {ph['canvas'][0]}x{ph['canvas'][1]}")
        else:
            why = _inverse_problem(ph, m)
            if why:
                bad.append(why)
    if not bad:
        want, nodata = _rebuild(a_in, ph, flip, rot, lanczos_a)
        if nodata != int(ph.get("no_data_pixels", 0)):
            bad.append(f"{count(nodata, 'pixel')} outside the scan, the "
                       f"manifest says {ph.get('no_data_pixels', 0)}")
        if want.shape != a_out.shape:
            bad.append(f"size is {shape}, expected "
                       f"{want.shape[1]}x{want.shape[0]}")
        elif _sha(want) != _sha(a_out):
            d = np.abs(want.astype(np.int64) - a_out.astype(np.int64))
            k = int((d > 0).sum())
            bad.append(f"PIXELS CHANGED: {count(k, 'sample')} "
                       f"{'differs' if k == 1 else 'differ'}, worst by "
                       f"{int(d.max())}")
        elif ph["action"] == "sliced" and not np.array_equal(
                np.sort(a_out.ravel()), np.sort(want.ravel())):
            bad.append("sample values were computed, not just moved")

    if m_in["icc"] and m_in["icc"] != m_out["icc"]:
        bad.append("ICC profile not carried over")
    quarter = (int(rot) // 90) % 2 == 1
    want_x = m_in["yres"] if quarter else m_in["xres"]
    want_y = m_in["xres"] if quarter else m_in["yres"]
    if want_x is not None and want_x != m_out["xres"]:
        bad.append(f"XResolution should be {want_x}, is {m_out['xres']}")
    if want_y is not None and want_y != m_out["yres"]:
        bad.append(f"YResolution should be {want_y}, is {m_out['yres']}")
    if m_in["resunit"] is not None and m_in["resunit"] != m_out["resunit"]:
        bad.append(f"resolution unit {m_in['resunit']} -> {m_out['resunit']}")
    lost = sorted((m_in["tags"] & _CARRIED_TAGS) - m_out["tags"])
    if lost:
        bad.append("tags not carried over: " + ",".join(lost))
    return (name, bad, shape, str(a_out.dtype))


def verify_manifest(root, doc, out_dir, workers=None):
    """Check every crop the manifest describes. Returns the failure count."""
    tr = doc.get("transform", {})
    other = sorted({ph.get("numpy_version") or tr.get("numpy_version")
                    for it in doc.get("items", {}).values()
                    for ph in it.get("photos", [])}
                   - {None, np.__version__})
    if other:
        warn(f"    ! some crops were made with numpy {', '.join(other)}, "
             f"this is {np.__version__}.\n"
             f"      A library change could move the last bit of a rare "
             f"sample, so a straightened\n      print may not reproduce "
             f"bit for bit here. A sliced one is unaffected.")
    lanczos_a = int(tr.get("lanczos_a", LANCZOS_A))

    tasks, missing = [], 0
    for name in sorted(doc.get("items", {})):
        it = doc["items"][name]
        raw = os.path.join(root, name)
        if not os.path.isfile(raw):
            error(f"    {name}: the raw scan is gone - cannot verify")
            missing += 1
            continue
        for ph in it.get("photos", []):
            out = os.path.join(out_dir, ph["file"])
            if not os.path.isfile(out):
                error(f"    {ph['file']}: not in "
                      f"{os.path.basename(out_dir)}/")
                missing += 1
                continue
            tasks.append((raw, out, bool(it["flip"]), int(it["rot"]), ph,
                          lanczos_a))
    if not tasks:
        warn("    nothing to verify")
        return missing

    n = 1
    if parallel is not None and workers != 1:
        per = parallel.estimate_bytes_per_worker(tasks[0][0],
                                                 multiple=VERIFY_MEM_MULTIPLE)
        n = parallel.choose_workers(len(tasks), per, workers)
    failed = missing

    def report(_i, res):
        nonlocal failed
        name, bad, shape, dtype = res
        if bad:
            failed += 1
            print(f"    {name:28s} {shape or '?'} {dtype or '?'}  "
                  + paint("[FAIL]", "1;31"))
            for b in bad:
                print(paint(f"          ! {b}", "31"))
        else:
            print(f"    {name:28s} {shape} {dtype}  identical  "
                  + paint("[OK]", "32"))

    info(f"rebuilding {count(len(tasks), 'crop')} from the raw scans")
    info(count(n, "worker"))
    if n > 1:
        parallel.run(_verify_job, tasks, n, on_result=report,
                     on_error=lambda t, exc: (
                         os.path.basename(t[1]),
                         ["not checked: " + parallel.describe_error(exc)],
                         None, None))
    else:
        for i, t in enumerate(tasks):
            report(i, _verify_job(t))

    checked_bad = failed - missing          # crops that were compared and lost
    ok = len(tasks) - checked_bad
    print(paint(f"\n{'VERIFIED OK' if failed == 0 else 'VERIFICATION'}: "
                f"{ok} of {count(len(tasks), 'crop')} identical"
                + (f", {missing} could not be checked at all"
                   if missing else ""), "1;32" if failed == 0 else "1;31"))
    if failed == 0:
        info("Every crop reproduces its raw scan exactly through the recorded")
        info("flip, turn, straightening and crop (SHA-256 match), with bit")
        info("depth, colour model, ICC profile and resolution intact. The")
        info("files match the manifest; whether the way up is right is what")
        info("review.py is for.")
    else:
        if checked_bad:
            print(paint(f"    {count(checked_bad, 'crop')} FAILED and should "
                        f"not be used.", "1;31"))
        if missing:
            error(f"    {count(missing, 'file')} could not be checked: a raw "
                  f"scan or a crop is missing.")
    return failed


# --------------------------------------------------------------------------
# the step
# --------------------------------------------------------------------------

def _manifest_dir(doc, root):
    """The folder a manifest's crops are in; ROOT/deskewed-cropped when it
    names none. A relative out_dir is taken from the scans folder."""
    d = doc.get("out_dir")
    if not d:
        return cropped_dir(root)
    return d if os.path.isabs(d) else os.path.join(resolve(root), d)


def _stored_dir(out_dir, root):
    """out_dir as the manifest stores it: relative to the scans folder when
    inside it, so the archive can be moved or copied whole; absolute
    otherwise."""
    try:
        rel = os.path.relpath(os.path.abspath(out_dir), resolve(root))
    except ValueError:                  # another drive on Windows
        return os.path.abspath(out_dir)
    if rel == os.pardir or rel.startswith(os.pardir + os.sep):
        return os.path.abspath(out_dir)
    return rel


def _write_manifest(dst_json, proj, root, min_angle, items,
                    out_dir) -> None:
    """The manifest, written atomically so a crash never truncates it."""
    doc = {
        "version": SCHEMA_VERSION,
        "stage": STAGE_DESKEW_CROP,
        "source_dir": os.path.abspath(root),
        "tool": SCRIPT,
        # where the crops of THIS manifest live; --out can move them away
        # from ROOT/deskewed-cropped, and a later run or --verify must know
        "out_dir": _stored_dir(out_dir, root),
        "flip": proj.flip,
        "transform": {
            "order": "straighten (in the scan's own frame), then crop, "
                     "then orient (flip, quarter turns)",
            "method": METHOD,
            "lanczos_a": LANCZOS_A,
            "min_angle_deg": float(min_angle),
            "numpy_version": np.__version__,
        },
        "note": "numpy_version, per crop, is the numpy that computed it. "
                "tilt_deg is the print's tilt as review.py showed it (after "
                "flip and rot), positive = clockwise on screen. angle_deg is "
                "the turn applied to the scan as it sits on disk, in "
                "OpenCV's convention (positive = counter-clockwise), 0 when "
                "sliced: rotation_matrix(width, height, angle_deg) turns the "
                "scan about its centre onto a canvas of the recorded size, "
                "and crop_box is cut from that canvas before the flip and "
                "the quarter turns; inverse takes a canvas pixel to the "
                "exact scan position it was read from. no_data_pixels "
                "counts the crop's pixels that lie outside the scan, "
                "written as 0. quad is the approved print outline in "
                "the finished crop's own pixel indices: it says where the "
                "print itself ends inside the crop, which is what tells "
                "picture from album page in the corners. out_dir is "
                "relative to the scans folder when the crops are inside it.",
        "items": items,
    }
    write_json(dst_json, doc)


def _belongs_to(fname, stem, ext):
    """Is `fname` a crop name the scan `stem`+`ext` can produce: STEM.ext
    or STEM_NN.ext?"""
    f_stem, f_ext = os.path.splitext(fname)
    if f_ext.lower() != ext.lower():
        return False
    return f_stem == stem or re.fullmatch(re.escape(stem) + r"_\d{2}",
                                          f_stem) is not None


def _report_orphans(out_dir, planned, expected, scan_stems):
    """Crops in the folder that no box of the current plan produces.

    The name of a crop depends on how many prints share its sheet: one box
    writes NAME.tif, two write NAME_01.tif and NAME_02.tif. Add a box in
    review.py, run again, and yesterday's NAME.tif stays behind, looking
    exactly like a current crop. Only names a planned scan can produce are
    considered, and never a name that is itself another scan's own name."""
    try:
        have = set(os.listdir(out_dir))
    except OSError:
        return
    orphan = []
    for name in planned:
        stem, ext = os.path.splitext(name)
        orphan += [f for f in have - expected
                   if _belongs_to(f, stem, ext)
                   and (os.path.splitext(f)[0] == stem
                        or os.path.splitext(f)[0] not in scan_stems)]
    orphan = sorted(set(orphan))
    if orphan:
        warn_list(f"    ! {count(len(orphan), 'file')} in {out_dir} "
                  f"{'belongs' if len(orphan) == 1 else 'belong'} to no box "
                  f"of the current plan (left over from an earlier "
                  f"review?):", orphan)
        warn("      " + ("It is" if len(orphan) == 1 else "They are")
             + " left alone; delete "
             + ("it" if len(orphan) == 1 else "them")
             + " once you are sure.")


def _previous_items(dst_json, root, out_dir):
    """Manifest items of an earlier run into the same folder -- the entries
    that describe crops this run leaves in place. Empty when there is
    nothing usable. An entry made under another numpy version is kept:
    --verify says so when it checks that crop."""
    try:
        with open(dst_json, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError):
        return {}
    if not same_dir(_manifest_dir(doc, root), out_dir):
        return {}
    return doc.get("items", {})


def _describe(written):
    """One log fragment per print."""
    out = []
    for p in written:
        s = f"tilt {p['tilt_deg']:+.3f} deg, {p['action']}"
        s += f", crop {p['size'][1]}x{p['size'][0]}"
        s += f", keeps {p['kept_fraction'] * 100:.2f}% of the print"
        if p.get("no_data_pixels"):
            s += (f", {p['no_data_pixels']} px outside the scan "
                  f"(written black)")
        out.append(s)
    return "; ".join(out)


def run_from_project(proj, root, out_dir, min_angle=STRAIGHTEN_MIN_DEG,
                     compress="keep", take_all=False, overwrite=False,
                     dst_json=None, workers=None, phases=2) -> int:
    """Phases 1 and 2 of `phases`. Returns the exit code."""

    todo = [n for n in proj.names()
            if proj.items[n].status == "confirmed"
            or (take_all and proj.items[n].status != "skipped")]
    if not todo:
        error(f"nothing confirmed yet; run review.py first, or pass --all:\n"
              f"    python review.py{root_arg(root)}")
        return 1

    tasks, names, skipped, expected = [], [], 0, set()
    kept = {}              # scan name -> crop files left in place
    for name in todo:
        it = proj.items[name]
        if not it.boxes:
            warn(f"    {name}: no box, skipped")
            skipped += 1
            continue
        path = os.path.join(root, name)
        if not os.path.isfile(path):
            error(f"    {name}: the scan is gone from {root}")
            skipped += 1
            continue
        try:
            with tifffile.TiffFile(path) as tf:
                p = tf.pages[0]
                shape = [p.imagelength, p.imagewidth]
        except Exception as exc:
            error(f"    {name}: cannot read as TIFF ({exc}) - this step is "
                  f"TIFF-only")
            skipped += 1
            continue
        if shape != list(it.size):
            error(f"    {name}: the scan is {shape[1]}x{shape[0]} but "
                  f"review.py recorded {it.size[1]}x{it.size[0]} - rerun "
                  f"review.py")
            skipped += 1
            continue

        jobs, n_have = [], len(it.boxes)
        for k, b in enumerate(it.boxes):
            dst = os.path.join(out_dir, out_name(name, k, n_have))
            expected.add(os.path.basename(dst))
            if os.path.exists(dst) and not overwrite:
                print(paint(f"    {os.path.basename(dst)}  exists, skipped",
                            "90"))
                kept.setdefault(name, set()).add(os.path.basename(dst))
                continue
            jobs.append((dst, b.corners))
        if not jobs:
            continue
        # archive-wide flip XOR per-scan V, and the approved quarter turn:
        # exactly what review.py showed
        tasks.append((path, proj.flipped(it), it.rot, jobs, compress,
                      float(min_angle)))
        names.append(name)

    _report_orphans(out_dir, todo, expected,
                    {os.path.splitext(n)[0] for n in proj.names()})

    if not tasks:
        # Every crop is already on disk. That is a finished run, not a
        # failure: the manifest of the run that wrote them is left as it
        # is, for --verify.
        phase(2, phases, "the crops")
        print(paint(f"    nothing to do: {count(len(expected), 'crop')} "
                    f"already in {out_dir} (--overwrite to redo them)",
                    "32"))
        return 2 if skipped else 0
    info(f"{count(len(tasks), 'scan')} to crop")

    phase(2, phases, "the crops")
    n_workers = 1
    if parallel is not None and workers != 1:
        per = parallel.estimate_bytes_per_worker(tasks[0][0],
                                                 multiple=WORKER_MEM_MULTIPLE)
        n_workers = parallel.choose_workers(len(tasks), per, workers)
    info(count(n_workers, "worker"))

    items = {}
    n_ok = failed = 0

    def report(i, res):
        nonlocal n_ok, failed
        name = names[i]
        it = proj.items[name]
        _base, w, h, written, err = res
        if err:
            error(f"    [{i + 1}/{len(tasks)}] {name}: FAILED {err}")
            failed += 1
            return
        items[name] = {
            "source_size": [h, w],
            "flip": bool(tasks[i][1]), "rot": int(tasks[i][2]),
            "status": it.status,
            "orientation": it.orientation, "paper_level": it.paper_level,
            "photos": written,
        }
        n_ok += 1
        turn = f", rot {tasks[i][2]}" if tasks[i][2] else ""
        flip = ", flipped" if tasks[i][1] else ""
        print(paint(f"    [{i + 1}/{len(tasks)}]", "90")
              + f" {name}: {w}x{h}{flip}{turn} -> {_describe(written)}")

    if n_workers > 1:
        # a worker that dies (out of memory on a huge scan) must cost its
        # own scans, not the manifest entries of every scan already done
        parallel.run(_job, tasks, n_workers, on_result=report,
                     on_error=lambda t, exc: (
                         os.path.basename(t[0]), 0, 0, [],
                         parallel.describe_error(exc)))
    else:
        for i, t in enumerate(tasks):
            report(i, _job(t))

    # crops left in place keep their earlier manifest entries, so --verify
    # still checks them
    n_carried, n_lost = 0, 0
    prev = (_previous_items(dst_json, root, out_dir) if dst_json and kept
            else {})
    for name, files in kept.items():
        old = prev.get(name)
        now = items.get(name)
        keep_ph = []
        if old and (now is None or (bool(old.get("flip")) == now["flip"]
                                    and int(old.get("rot", 0)) == now["rot"])):
            keep_ph = [ph for ph in old.get("photos", [])
                       if ph.get("file") in files]
        n_carried += len(keep_ph)
        n_lost += len(files) - len(keep_ph)
        if not keep_ph:
            continue
        if now is None:
            items[name] = dict(old, photos=keep_ph)
        else:
            now["photos"] = sorted(keep_ph + now["photos"],
                                   key=lambda ph: ph["file"])
    if n_lost:
        warn(f"    ! {count(n_lost, 'crop')} left in place "
             f"{'has' if n_lost == 1 else 'have'} no earlier manifest entry "
             f"and cannot be verified; --overwrite remakes "
             f"{'it' if n_lost == 1 else 'them'}")

    if dst_json and items:
        items = dict(sorted(items.items()))
        _write_manifest(dst_json, proj, root, min_angle, items, out_dir)
        print(paint(f"\nwrote {dst_json}", "32") + f"  ({n_ok} of "
              f"{count(len(tasks), 'scan')} this run, {len(items)} in the "
              f"manifest"
              + (f" incl. {count(n_carried, 'crop')} left in place"
                 if n_carried else "")
              + f", {count(failed + skipped, 'problem')})")
        if failed + skipped == 0:
            print(paint(f"    the crops in {out_dir} are the input of "
                        f"Stage 3", "32"))
    return 0 if failed + skipped == 0 else 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Stage 2 step 4 of 4 — straighten, crop and orient")
    ap.add_argument("root", nargs="?", default=".",
                    help="project root: the folder holding the scans "
                         "(default: this folder)")
    ap.add_argument("--min-angle", type=float, default=STRAIGHTEN_MIN_DEG,
                    help="below this tilt the print is sliced without "
                         "straightening, so every sample is kept exactly "
                         f"(default: {STRAIGHTEN_MIN_DEG} deg)")
    ap.add_argument("--compress", choices=["keep", "none", "lzw", "deflate"],
                    default="keep",
                    help="output codec (default: same as the source); all "
                         "choices are lossless")
    ap.add_argument("--all", action="store_true",
                    help="also crop files you never confirmed in review.py")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace existing outputs in the output folder")
    ap.add_argument("--out", default=None, metavar="DIR",
                    help="folder for the finished crops (default: "
                         "ROOT/deskewed-cropped). Created if missing")
    ap.add_argument("--verify", action="store_true",
                    help="afterwards, rebuild every crop in the manifest "
                         "from its raw scan and compare SHA-256, along with "
                         "bit depth, colour model, ICC profile, resolution "
                         "and the other TIFF tags")
    if parallel is not None:
        parallel.add_workers_arg(ap)
    else:
        ap.add_argument("--workers", type=int, default=1,
                        help="ignored: parallel.py not found")
    args = ap.parse_args(argv)
    console_setup()

    if args.min_angle < 0:
        raise SystemExit(f"--min-angle must be 0 or more, not "
                         f"{args.min_angle}")
    root = pipeline_root(args.root)
    check_scans_present(root)

    proj = load_project(root)
    if proj is None:
        raise SystemExit(
            f"no project file in {os.path.join(root, JSON_DIR)}\n"
            "run detect_borders.py (and ideally review.py) first:\n"
            f"    python detect_borders.py{root_arg(root)}\n"
            f"    python review.py{root_arg(root)}")
    out_dir = resolve(args.out) if args.out else cropped_dir(root)
    if same_dir(out_dir, root):
        raise SystemExit("--out must not be the scans folder itself: the "
                         "crops are named after the scans")
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as exc:
        raise SystemExit(f"cannot create the output folder {out_dir}: {exc}")

    heading("STAGE 2 step 4  -  straighten, crop and orient")
    t0 = time.time()
    phases = 3 if args.verify else 2
    phase(1, phases, "the plan")
    info(f"from {os.path.basename(proj.path)}")
    info(f"output: {out_dir}")
    if args.all and proj.path != stage_json(root, STAGE_REVIEW) and \
            proj.counts().get("confirmed", 0) == 0:
        info(f"note: nothing was confirmed in review.py yet; the boxes of "
             f"boxes_{STAGE_DETECT}.json are used as detected")
    if proj.added:
        warn_list(f"    ! {count(len(proj.added), 'scan')} measured by "
                  f"step 1 after the review began "
                  f"{'is' if len(proj.added) == 1 else 'are'} added, not yet "
                  f"confirmed:", proj.added)
    dst_json = stage_json(root, STAGE_DESKEW_CROP)
    code = run_from_project(
        proj, root, out_dir, args.min_angle, args.compress, args.all,
        args.overwrite, dst_json, args.workers, phases)

    if code == 1:
        return code
    if args.verify:
        phase(3, phases, "verification: raw scan against crop")
        # read the manifest back from disk rather than using the one just
        # built in memory: what a later stage will trust is the file
        if not os.path.isfile(dst_json):
            error("nothing to verify: no manifest yet")
            return 1
        with open(dst_json, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        rec = _manifest_dir(doc, root)
        if not same_dir(rec, out_dir):
            error(f"cannot verify: the manifest describes the crops in\n"
                  f"    {rec}\nnot the ones in\n    {out_dir}\n"
                  f"to verify those, run:\n"
                  f"    python {SCRIPT}{root_arg(root)} --out "
                  f"\"{rec}\" --verify\nor remake this folder with "
                  f"--overwrite")
            return 3
        if verify_manifest(root, doc, out_dir, args.workers):
            return 3          # distinct from 2 (a crop failed or was skipped)
    print(paint(f"    done in {elapsed(t0)}", "32"))
    print(paint(f"\nnext, Stage 3:  python renumber_by_similarity.py "
                f"\"{out_dir}\"", "1"))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
