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
# repair_seams.py  --  STAGE 1, step 3 of 3: put the dropped column back
#
# STAGE 1 repairs the vertical seams a CIS flatbed leaves in every scan it
# makes:
#
#     calibrate_scanner.py     step 1 of 3   measure pitch and phase
#     detect_seams.py          step 2 of 3   locate the seams in each scan
#     repair_seams.py          step 3 of 3   put the dropped column back
#     common.py                shared code, not a step
#     parallel.py              shared worker-count policy, not a step
#     pixel_ruler.py           helper for step 2, not a step
#
# Requirements:
#     pip install numpy tifffile [pillow] [imagecodecs] [psutil]
#                 # pillow only for scans that are not TIFF
#     common.py beside this file; parallel.py too, for --workers
#
# EVERY SCANNED SAMPLE IS KEPT, AND CHECKED. At each seam the sensor
# dropped one column, so exactly one column is inserted there and every
# original sample is copied through as it is: the file grows by one pixel
# per seam and by nothing else. Taking the inserted columns out again gives
# the input back bit for bit; that is checked on every scan before it is
# written, and can be checked again at any time with --verify.
#
# THE NEW COLUMN. "directional" (the default) blends the horizontal average
# of the two neighbours with the two diagonal averages, each weighted by how
# well its direction matches across the gap, so a slanted edge is bridged
# instead of smeared. It is computed in float64 and rounded once to the
# scan's own dtype. "linear" is the plain average of the two neighbours,
# rounded half up for integer data.
#
# THE FILE SURVIVES. Bit depth, colour model, compression (a lossy one is
# written as deflate), ICC profile, resolution and every other tag of the
# scan are carried to the output. A scan that is not TIFF is written as
# TIFF.
#
# It reads the seam columns from json files/boxes_detect_seams.json.
# Outputs, derived from the project root -- the folder holding the scans --
# unless --out names another folder for the repairs:
#
#     seam-repaired/<n>.tif               the repaired scans: the input of
#                                         Stage 2
#     json files/boxes_repair_seams.json  the manifest: every number needed
#                                         to rebuild each output exactly,
#                                         and the SHA-256 of its original
#                                         samples
#
# The manifest describes every output in the output folder: the ones this
# run wrote, plus the ones an earlier run wrote into the same folder and
# this run left in place.
# Outputs that exist are skipped unless --overwrite is given; one whose
# seams step 2 has changed since is named, so it can be redone. Two scans
# that would write the same output name (raw0001.tif and raw0001.png) stop
# the step before anything is written.
#
# Usage:
#     python repair_seams.py [ROOT] [options]
#
#     ROOT              project root: the folder holding the scans
#                       (default: this folder)
#     --method NAME     directional (default) or linear
#     --overwrite       replace outputs that exist
#     --out DIR         write the repairs to DIR instead of
#                       ROOT\seam-repaired. Created if missing; never the
#                       scans folder itself
#     --verify          afterwards, rebuild every output in the manifest
#                       from its raw scan and compare SHA-256 (see the
#                       --verify section below for what is and is not
#                       independent). Run again when every output is
#                       already there and it checks them without making
#                       any. Exit code 3 if it fails
#     --workers N       parallel workers (default: auto; 1 disables the pool)
#
#     python repair_seams.py --verify
#     python repair_seams.py D:\archive\scans --verify
# ==========================================================================
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

try:
    import numpy as np
    import tifffile
except ModuleNotFoundError as exc:
    sys.exit(f"missing package: {exc.name} (see Requirements above)")

try:
    from common import (JSON_DIR, REPAIRED_DIR, SCHEMA_VERSION, STAGE_DETECT,
                        STAGE_REPAIR, TIFF_EXT, Image, as_hwc,
                        check_scans_present, console_setup, count, elapsed,
                        error, heading, info, paint, phase, pipeline_root,
                        read_scan, repaired_dir, resolve, root_arg, same_dir,
                        stage_json, warn, warn_list, write_json, write_tiff)
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
SCRIPT = "repair_seams.py"
WORKER_MEM_MULTIPLE = 6.0   # peak RAM of one worker per byte of the scan
                            # file: the scan, the wider output, the check
VERIFY_MEM_MULTIPLE = 8.0   # the same for --verify: the scan, the output,
                            # the rebuilt output, the samples taken back out

# --- filling the new column ------------------------------------------------
METHODS = ("directional", "linear")   # --method; the first is the default
DIAGONAL_WEIGHT = 0.75      # weight of a diagonal against the horizontal,
                            # at equal match
MATCH_EPS = 1e-6            # added to each mismatch, 8-bit scale, so a
                            # perfect match does not divide by zero
LUMA = (0.299, 0.587, 0.114)   # how R, G, B are mixed to compare directions


# --------------------------------------------------------------------------
# filling the new column
# --------------------------------------------------------------------------

def _dtype_limits(dt):
    """The dtype's own value range, so 16-bit data is never clipped at 255;
    (None, None) for float data."""
    if np.issubdtype(dt, np.integer):
        info = np.iinfo(dt)
        return float(info.min), float(info.max)
    return None, None


def directional_column(arr, x):
    """The missing column between arr[:, x] and arr[:, x+1], in arr's dtype.

    Each row blends three candidates: the horizontal average and the two
    diagonals (left row above with right row below, and the reverse), each
    weighted by the inverse of how far apart its two ends are in luminance.
    The first and last rows have no diagonal and keep the horizontal average.
    Only the two columns beside the seam are read."""
    dt = arr.dtype
    lo, hi = _dtype_limits(dt)
    h, _, channels = arr.shape

    left = arr[:, x, :].astype(np.float64)
    right = arr[:, x + 1, :].astype(np.float64)
    horizontal = (left + right) / 2.0

    def _finish(vals):
        if lo is not None:
            vals = np.rint(vals).clip(lo, hi)
        return vals.astype(dt)

    if h < 3:
        return _finish(horizontal)

    if channels >= 3:
        left_y = LUMA[0] * left[:, 0] + LUMA[1] * left[:, 1] \
            + LUMA[2] * left[:, 2]
        right_y = LUMA[0] * right[:, 0] + LUMA[1] * right[:, 1] \
            + LUMA[2] * right[:, 2]
    else:
        left_y = left.mean(axis=1)
        right_y = right.mean(axis=1)

    down = (arr[:-2, x, :].astype(np.float64)          # left[y-1], right[y+1]
            + arr[2:, x + 1, :].astype(np.float64)) / 2.0
    up = (arr[2:, x, :].astype(np.float64)             # left[y+1], right[y-1]
          + arr[:-2, x + 1, :].astype(np.float64)) / 2.0

    err_h = np.abs(right_y - left_y)
    err_d = np.zeros(h)
    err_u = np.zeros(h)
    err_d[1:-1] = np.abs(right_y[2:] - left_y[:-2])
    err_u[1:-1] = np.abs(right_y[:-2] - left_y[2:])

    eps = MATCH_EPS * ((hi - lo) / 255.0 if lo is not None else 1.0)
    wh = 1.0 / (err_h + eps)
    wd = DIAGONAL_WEIGHT / (err_d + eps)
    wu = DIAGONAL_WEIGHT / (err_u + eps)
    total = wh + wd + wu
    wh /= total
    wd /= total
    wu /= total

    result = horizontal.copy()
    result[1:-1] = (wh[1:-1, None] * horizontal[1:-1]
                    + wd[1:-1, None] * down
                    + wu[1:-1, None] * up)
    return _finish(result)


def linear_column(arr, x):
    """The average of the two neighbours in arr's dtype: rounded half up for
    integer data, exact for float data."""
    if not np.issubdtype(arr.dtype, np.integer):
        return ((arr[:, x, :].astype(np.float64)
                 + arr[:, x + 1, :].astype(np.float64)) / 2.0
                ).astype(arr.dtype)
    left = arr[:, x, :].astype(np.int64)
    right = arr[:, x + 1, :].astype(np.int64)
    return ((left + right + 1) // 2).astype(arr.dtype)


# --------------------------------------------------------------------------
# one scan
# --------------------------------------------------------------------------

def valid_seams(seams, width):
    """The seam columns a scan of this width can take, sorted, once each."""
    return sorted(set(int(x) for x in seams if 0 <= int(x) < width - 1))


def repair_image(arr, seams, method="directional"):
    """arr (H x W x C) with one new column after every seam column. The
    original samples are copied in blocks, unchanged."""
    h, w, c = arr.shape
    cols = valid_seams(seams, w)
    make_col = directional_column if method == "directional" else linear_column
    out = np.empty((h, w + len(cols), c), dtype=arr.dtype)
    start = 0
    for k, x in enumerate(cols):
        out[:, start + k:x + 1 + k] = arr[:, start:x + 1]
        out[:, x + 1 + k] = make_col(arr, x)
        start = x + 1
    out[:, start + len(cols):] = arr[:, start:]
    return out


def check_repair(a, out, seams):
    """Does `out` hold every sample of `a` unchanged, plus one column per
    seam and nothing else? The inserted columns are taken out with
    np.delete, a construction other than repair_image's block copy."""
    cols = valid_seams(seams, a.shape[1])
    if out.dtype != a.dtype or out.shape != (a.shape[0],
                                             a.shape[1] + len(cols),
                                             a.shape[2]):
        return False
    inserted = [x + 1 + k for k, x in enumerate(cols)]
    return bool(np.array_equal(np.delete(out, inserted, axis=1), a))


def _sha(arr):
    """SHA-256 of the samples, in memory order."""
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


PALETTE = 3                 # PhotometricInterpretation of a palette image


def _is_palette(meta):
    """Is this a palette image? Its samples are indices into a colour map,
    so a new column cannot be interpolated from them."""
    p = meta.get("photometric")
    try:
        return int(p) == PALETTE
    except (TypeError, ValueError):
        return str(p).lower() == "palette"


def _job(task):
    """One scan. Top-level so it can be pickled. Returns a record, never
    raises: a bad file must not stop the run.

    The worker reads the scan and writes the output itself, so no image
    crosses a pipe. The output is written under a temporary name and renamed
    only when complete, and only when the check has passed, so a file in
    the output folder is always a finished, checked repair."""
    src, dst, seams, method = task
    name = os.path.basename(src)
    tmp = dst + ".part"
    try:
        a, meta = read_scan(src)
        if _is_palette(meta):
            raise RuntimeError("palette images are not supported")
        a3 = as_hwc(a)
        out = repair_image(a3, seams, method)
        ok = check_repair(a3, out, seams)
        if ok:
            write_tiff(tmp, out[:, :, 0] if a.ndim == 2 else out, meta,
                       "keep", meta.get("extratags"))
            os.replace(tmp, dst)
    except Exception as exc:            # a bad file must not stop the run
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        return {"file": name, "error": f"{type(exc).__name__}: {exc}"}
    return {"file": name, "error": None, "ok": ok,
            "in_size": [int(a3.shape[0]), int(a3.shape[1])],
            "out_size": [int(out.shape[0]), int(out.shape[1])],
            "dtype": str(a3.dtype), "seams": len(valid_seams(seams,
                                                             a3.shape[1])),
            "sha256_originals": _sha(a3)}


# --------------------------------------------------------------------------
# --verify: rebuild every output from its scan and compare
# --------------------------------------------------------------------------
# What this check rebuilds with its own code: the reader (tifffile directly,
# not common.read_scan), which columns were inserted (an index array, not
# repair_image's block copy), and every check of dtype, bit depth, size,
# colour model, ICC profile, resolution and tags.
#
# What it shares with step 3: the fill of the inserted columns, repair_image
# with directional_column and linear_column, and Pillow for a scan that is
# not TIFF. A bug in those would therefore not be caught here.
#
# The decisive test is the hash of the ORIGINAL samples, recorded in the
# manifest when the repair was made. The raw scan must still have it, and so
# must the repair with its inserted columns taken out: then every sample
# that was scanned is provably untouched, which also settles the minimum,
# the maximum and the histogram, since one changed bit would change the
# hash. The directional fill is plain float64 arithmetic in numpy,
# repeatable on one installation; an output made under a different numpy
# version is reported rather than silently failed.

# Tags that may differ or be missing in the output by design: the layout of
# the data, the codec, and pointers to other IFDs, which are not carried.
_IGNORED_TAGS = {
    "StripOffsets", "StripByteCounts", "RowsPerStrip", "ImageWidth",
    "ImageLength", "Compression", "Predictor", "TileOffsets",
    "TileByteCounts", "TileWidth", "TileLength", "NewSubfileType",
    "SubfileType", "FillOrder", "MinSampleValue", "MaxSampleValue",
    "SMinSampleValue", "SMaxSampleValue", "JPEGTables", "YCbCrCoefficients",
    "YCbCrSubSampling", "YCbCrPositioning", "ReferenceBlackWhite",
    "ExifTag", "GPSTag", "InteroperabilityTag", "SubIFDs",
}


def _num(v):
    """What a TIFF rational or a plain number denotes, for comparing."""
    try:
        if isinstance(v, tuple) and len(v) == 2 and v[1]:
            return round(float(v[0]) / float(v[1]), 6)
        return round(float(v), 6)
    except Exception:
        return v


def _read_checked(path):
    """Pixels plus the metadata worth comparing, in the file's own dtype.

    Written without common.read_scan on purpose: a checker should not share
    a reader with the thing it checks."""
    m = {"icc": None, "xres": None, "yres": None, "resunit": None,
         "depth": None, "photometric": None, "tags": set(), "pages": 1}
    if path.lower().endswith(TIFF_EXT):
        with tifffile.TiffFile(path) as tf:
            page = tf.pages[0]
            a = page.asarray()
            if a.ndim == 3 and page.planarconfig == 2:
                a = np.moveaxis(a, 0, -1)
            m["pages"] = len(tf.pages)
            m["tags"] = set(t.name for t in page.tags.values())
            bps = page.bitspersample
            m["depth"] = ((bps if isinstance(bps, int) else bps[0])
                          * page.samplesperpixel)
            m["photometric"] = int(page.photometric)
            tg = page.tags
            if "InterColorProfile" in tg:
                m["icc"] = hashlib.sha256(
                    bytes(tg["InterColorProfile"].value)).hexdigest()[:16]
            for k, nm in (("xres", "XResolution"), ("yres", "YResolution")):
                if nm in tg:
                    m[k] = _num(tg[nm].value)
            if "ResolutionUnit" in tg:
                m["resunit"] = int(tg["ResolutionUnit"].value)
    else:
        with Image.open(path) as im:
            a = np.array(im)
            if im.info.get("icc_profile"):
                m["icc"] = hashlib.sha256(
                    im.info["icc_profile"]).hexdigest()[:16]
    return (a[:, :, None] if a.ndim == 2 else a), m


def _verify_job(task):
    """Check one output against its scan. Top-level so it can be pickled.
    Returns (name, problems, size, dtype), never raises."""
    raw_path, out_path, seams, method, recorded = task
    name = os.path.basename(out_path)
    try:
        a_in, m_in = _read_checked(raw_path)
        a_out, m_out = _read_checked(out_path)
    except Exception as exc:
        return name, [f"cannot read: {exc}"], None, None

    bad = []
    n = len(valid_seams(seams, a_in.shape[1]))
    if a_in.dtype != a_out.dtype:
        bad.append(f"dtype changed {a_in.dtype} -> {a_out.dtype}")
    if m_in["depth"] is not None and m_in["depth"] != m_out["depth"]:
        bad.append(f"bit depth {m_in['depth']} -> {m_out['depth']}")
    if a_in.shape[0] != a_out.shape[0]:
        bad.append("height changed")
    if a_in.shape[2] != a_out.shape[2]:
        bad.append("channel count changed")
    if a_out.shape[1] != a_in.shape[1] + n:
        bad.append(f"width is {a_out.shape[1]}, expected "
                   f"{a_in.shape[1]}+{n}={a_in.shape[1] + n}")
    if m_out["pages"] > 1:
        bad.append("the output has more than one page")

    if recorded is None:
        bad.append("the manifest records no SHA-256 for this repair")
    elif _sha(a_in) != recorded:
        bad.append("the raw scan has changed since it was repaired: its "
                   "SHA-256 differs from the manifest")
    if not bad:
        pos = np.asarray(valid_seams(seams, a_in.shape[1]), dtype=np.int64)
        inserted = pos + 1 + np.arange(len(pos))
        keep = np.setdiff1d(np.arange(a_out.shape[1]), inserted)
        rec = a_out[:, keep, :]
        if _sha(rec) != recorded:
            d = np.abs(rec.astype(np.int64) - a_in.astype(np.int64))
            k = int((d > 0).sum())
            bad.append(f"PIXELS CHANGED: {count(k, 'sample')} "
                       f"{'differs' if k == 1 else 'differ'}, worst by "
                       f"{int(d.max())}")
        elif _sha(repair_image(a_in, seams, method)) != _sha(a_out):
            bad.append("the inserted columns do not match the recorded "
                       "method")

    if m_in["photometric"] is not None and \
            m_in["photometric"] != m_out["photometric"]:
        bad.append(f"colour model {m_in['photometric']} -> "
                   f"{m_out['photometric']}")
    if m_in["icc"] and m_in["icc"] != m_out["icc"]:
        bad.append("ICC profile not carried over")
    for k, label in (("xres", "XResolution"), ("yres", "YResolution"),
                     ("resunit", "resolution unit")):
        if m_in[k] is not None and m_in[k] != m_out[k]:
            bad.append(f"{label} {m_in[k]} -> {m_out[k]}")
    lost = sorted((m_in["tags"] - m_out["tags"]) - _IGNORED_TAGS)
    if lost:
        bad.append("tags not carried over: " + ", ".join(lost))
    return (name, bad, f"{a_out.shape[1]}x{a_out.shape[0]}",
            str(a_out.dtype))


def verify_manifest(root, doc, out_dir, workers=None):
    """Check every output the manifest describes. Returns the failure
    count."""
    items = doc.get("items", {})
    other = sorted({it.get("numpy_version") for it in items.values()}
                   - {None, np.__version__})
    if other:
        warn(f"    ! some outputs were made with numpy {', '.join(other)}, "
             f"this is {np.__version__}.\n"
             f"      A library change could move the last bit of a rare "
             f"sample in an inserted\n      column; the original samples "
             f"are compared all the same.")

    tasks, missing = [], 0
    for name in sorted(items):
        it = items[name]
        raw = os.path.join(root, name)
        out = os.path.join(out_dir, it["file"])
        if not os.path.isfile(raw):
            error(f"    {name}: the raw scan is gone - cannot verify")
            missing += 1
            continue
        if not os.path.isfile(out):
            error(f"    {it['file']}: not in {os.path.basename(out_dir)}/")
            missing += 1
            continue
        tasks.append((raw, out, it.get("seams", []),
                      it.get("method", doc.get("method", METHODS[0])),
                      it.get("sha256_originals")))
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

    info(f"rebuilding {count(len(tasks), 'repair')} from the raw scans")
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

    checked_bad = failed - missing        # repairs that were compared and lost
    ok = len(tasks) - checked_bad
    print(paint(f"\n{'VERIFIED OK' if failed == 0 else 'VERIFICATION'}: "
                f"{ok} of {count(len(tasks), 'repair')} identical"
                + (f", {missing} could not be checked at all"
                   if missing else ""), "1;32" if failed == 0 else "1;31"))
    if failed == 0:
        info("Every raw scan and every repair still hold the original samples")
        info("recorded in the manifest (SHA-256 match), with one column per")
        info("recorded seam filled by the recorded method, and bit depth,")
        info("colour model, ICC profile, resolution and tags intact.")
    else:
        if checked_bad:
            print(paint(f"    {count(checked_bad, 'repair')} FAILED and "
                        f"should not be used.", "1;31"))
        if missing:
            error(f"    {count(missing, 'file')} could not be checked: a raw "
                  f"scan or a repair is missing.")
    return failed


# --------------------------------------------------------------------------
# the step
# --------------------------------------------------------------------------

def _load(path):
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _manifest_dir(doc, root):
    """The folder a manifest's repairs are in; ROOT/seam-repaired when it
    names none. A relative out_dir is taken from the scans folder."""
    d = doc.get("out_dir")
    if not d:
        return repaired_dir(root)
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


def _write_manifest(dst_json, proj, root, out_dir, method, items) -> None:
    """The manifest, written atomically so a crash never truncates it."""
    doc = {
        "version": SCHEMA_VERSION,
        "stage": STAGE_REPAIR,
        "source_dir": str(root),
        # where the repairs of THIS manifest live; --out can move them away
        # from ROOT/seam-repaired, and a later run or --verify must know
        "out_dir": _stored_dir(out_dir, root),
        "tool": SCRIPT,
        "note": "seams are the columns of the raw scan after which one "
                "column was inserted, in the order of the scan: the column "
                "inserted after seam c lies between the scan's columns c "
                "and c+1. method is how it was filled, numpy_version the "
                "numpy that computed it. sha256_originals is the SHA-256 of "
                "the raw scan's samples (H x W x C, in memory order); the "
                "output with its inserted columns taken out has the same "
                "hash. out_dir is relative to the scans folder when the "
                "repairs are inside it.",
        "method": method,
        "pitch": proj.get("pitch"),
        "items": dict(sorted(items.items())),
    }
    write_json(dst_json, doc)


def _name_clashes(names):
    """Groups of scans that would write the same output name, compared
    without case as Windows does."""
    by_out = {}
    for name in names:
        out = (os.path.splitext(name)[0] + ".tif").lower()
        by_out.setdefault(out, []).append(name)
    return [sorted(g) for g in by_out.values() if len(g) > 1]


def run_from_project(proj, prev, root, out_dir, method, overwrite, dst_json,
                     workers, phases=2) -> int:
    """Phases 1 and 2 of `phases`. Returns the exit code."""
    phase(1, phases, "the plan")
    info(f"method: {method}")
    info(f"output: {out_dir}")
    items_in = proj.get("items", {})
    if not items_in:
        error(f"the step-2 file lists no scan; run it again:\n"
              f"    python detect_seams.py{root_arg(root)}")
        return 1
    clashes = _name_clashes(items_in)
    if clashes:
        error("these scans would write the same output name; rename one of "
              "each group and run step 2 again:")
        for group in clashes:
            error("    " + ", ".join(group))
        return 1

    # an output left in place keeps its entry from the run that wrote it,
    # so --verify still checks it
    items = {k: v for k, v in prev.items()
             if os.path.isfile(os.path.join(out_dir, v.get("file", "")))}
    tasks, names, skipped, kept, changed, unknown = [], [], 0, 0, [], []
    for name in sorted(items_in):
        it = items_in[name]
        seams = [int(x) for x in it.get("seams", [])]
        src = os.path.join(root, name)
        dst = os.path.join(out_dir, os.path.splitext(name)[0] + ".tif")
        if not os.path.isfile(src):
            error(f"    {name}: the scan is gone")
            skipped += 1
            continue
        if os.path.exists(dst) and not overwrite:
            print(paint(f"    {os.path.basename(dst)}  exists, skipped",
                        "90"))
            kept += 1
            old = prev.get(name)
            if old is None:
                unknown.append(os.path.basename(dst))
            elif old.get("seams") != seams or old.get("method") != method:
                changed.append(os.path.basename(dst))
            continue
        tasks.append((src, dst, seams, method))
        names.append(name)

    for head, files in (
            (f"    ! {count(len(changed), 'output')} left in place "
             f"{'was' if len(changed) == 1 else 'were'} repaired with other "
             f"seams or another method than step 2 now gives; --overwrite "
             f"repairs {'it' if len(changed) == 1 else 'them'} again:",
             changed),
            (f"    ! {count(len(unknown), 'output')} left in place "
             f"{'has' if len(unknown) == 1 else 'have'} no earlier manifest "
             f"entry and cannot be verified; --overwrite remakes "
             f"{'it' if len(unknown) == 1 else 'them'}:", unknown)):
        if files:
            warn_list(head, files)

    if not tasks:
        # Every output is already on disk. That is a finished run, not a
        # failure: the manifest of the run that wrote them is left as it
        # is, for --verify.
        phase(2, phases, "the repairs")
        print(paint(f"    nothing to do: {count(kept, 'repair')} already in "
                    f"{out_dir} (--overwrite to redo them)", "32"))
        return 2 if skipped else 0
    info(f"{count(len(tasks), 'scan')} to repair, {kept} left in place")

    phase(2, phases, "the repairs")
    n_workers = 1
    if parallel is not None and workers != 1:
        per = parallel.estimate_bytes_per_worker(tasks[0][0],
                                                 multiple=WORKER_MEM_MULTIPLE)
        n_workers = parallel.choose_workers(len(tasks), per, workers)
    info(count(n_workers, "worker"))

    n_ok = failed = 0

    def report(i, r):
        nonlocal n_ok, failed
        name = names[i]
        tag = paint(f"    [{i + 1}/{len(tasks)}]", "90")
        if r.get("error"):
            error(f"    [{i + 1}/{len(tasks)}] {name}: FAILED {r['error']}")
            failed += 1
            return
        size = (f"{r['in_size'][1]}x{r['in_size'][0]} -> "
                f"{r['out_size'][1]}x{r['out_size'][0]} {r['dtype']}, "
                f"{count(r['seams'], 'column')} inserted")
        if not r["ok"]:
            error(f"    [{i + 1}/{len(tasks)}] {name}: {size} - FAILED THE "
                  f"CHECK, not written")
            failed += 1
            return
        n_ok += 1
        items[name] = {
            "file": os.path.basename(tasks[i][1]), "method": method,
            "seams": tasks[i][2], "in_size": r["in_size"],
            "out_size": r["out_size"], "dtype": r["dtype"],
            "sha256_originals": r["sha256_originals"],
            "numpy_version": np.__version__}
        print(tag + f" {name}: {size}")

    if n_workers > 1:
        # a worker that dies (out of memory on a huge scan) must cost its
        # own scans, not the manifest entries of every scan already done
        parallel.run(_job, tasks, n_workers, on_result=report,
                     on_error=lambda t, exc: {
                         "file": os.path.basename(t[0]),
                         "error": parallel.describe_error(exc)})
    else:
        for i, t in enumerate(tasks):
            report(i, _job(t))

    _write_manifest(dst_json, proj, root, out_dir, method, items)
    problems = failed + skipped
    print(paint(f"\nwrote {dst_json}", "32") + f"  ({n_ok} of "
          f"{count(len(tasks), 'scan')} this run, {len(items)} in the "
          f"manifest, {count(problems, 'problem')})")
    if problems == 0:
        print(paint(f"    the repaired scans in {out_dir} are the input of "
                    f"Stage 2", "32"))
    return 0 if problems == 0 else 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Stage 1 step 3 of 3 — put the dropped column back")
    ap.add_argument("root", nargs="?", default=".",
                    help="project root: the folder holding the scans "
                         "(default: this folder)")
    ap.add_argument("--method", choices=METHODS, default=METHODS[0],
                    help="how the new column is filled (default: directional, "
                         "which follows slanted edges; linear is the plain "
                         "average of the two neighbours)")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace existing outputs in the output folder")
    ap.add_argument("--out", default=None, metavar="DIR",
                    help=f"folder for the repaired scans (default: "
                         f"ROOT/{REPAIRED_DIR}). Created if missing")
    ap.add_argument("--verify", action="store_true",
                    help="afterwards, rebuild every output in the manifest "
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

    root = pipeline_root(args.root)
    check_scans_present(root)
    proj = _load(stage_json(root, STAGE_DETECT))
    if proj is None:
        raise SystemExit(
            f"no {os.path.basename(stage_json(root, STAGE_DETECT))} in "
            f"{os.path.join(root, JSON_DIR)}\n"
            "run step 2 first:\n"
            f"    python detect_seams.py{root_arg(root)}")

    out_dir = resolve(args.out) if args.out else repaired_dir(root)
    if same_dir(out_dir, root):
        raise SystemExit(f"the output folder cannot be the scans folder "
                         f"itself: {out_dir}\nthe repairs would take the "
                         f"raw scans' names")
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as exc:
        raise SystemExit(f"cannot create the output folder {out_dir}: {exc}")
    heading("STAGE 1 step 3  -  put the dropped column back")
    t0 = time.time()
    phases = 3 if args.verify else 2
    dst_json = stage_json(root, STAGE_REPAIR)
    old = _load(dst_json) or {}
    # outputs an earlier run left in place are carried over only when that
    # run wrote into this same folder
    prev = (old.get("items", {})
            if same_dir(_manifest_dir(old, root), out_dir) else {})
    code = run_from_project(proj, prev, root, out_dir, args.method,
                            args.overwrite, dst_json, args.workers, phases)

    if code == 1:
        return code
    if args.verify:
        phase(3, phases, "verification: raw scan against repair")
        # read the manifest back from disk rather than using the one just
        # built in memory: what a later stage will trust is the file
        doc = _load(dst_json)
        if not doc:
            error("nothing to verify: no manifest yet")
            return 1
        rec = _manifest_dir(doc, root)
        if not same_dir(rec, out_dir):
            error(f"cannot verify: the manifest describes the repairs in\n"
                  f"    {rec}\nnot the ones in\n    {out_dir}\n"
                  f"to verify those, run:\n"
                  f"    python {SCRIPT}{root_arg(root)} --out \"{rec}\" "
                  f"--verify\nor remake this folder with --overwrite")
            return 3
        if verify_manifest(root, doc, out_dir, args.workers):
            return 3          # distinct from 2 (a scan failed or was skipped)
    print(paint(f"    done in {elapsed(t0)}", "32"))
    print(paint(f"\nnext, Stage 2:  python detect_borders.py \"{out_dir}\"",
                "1"))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
