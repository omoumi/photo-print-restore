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
# calibrate_color.py  --  STAGE 5: step 2's colour numbers (not a step)
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
#     pip install numpy tifffile opencv-python [imagecodecs]
#     transfer_color.py and common.py beside this file
#
# The model's cast that step 2 takes off, and how strong it makes each
# colour -- CAST_AB and HUE_GAINS in transfer_color.py -- are measured here
# on colour prints whose real colours are known: they are made B&W,
# colorized like any B&W print, and the result is set against the real
# print. In three runs:
#
#     1. python calibrate_color.py make COLOUR_DIR TEST_DIR
#        Every colour picture in COLOUR_DIR -- TIFF, JPEG or PNG, read as
#        sRGB -- is written to TEST_DIR in B&W as <name>.tif, 16-bit,
#        keeping its lightness (CIE L*) exactly, the way Stage 4 writes a
#        B&W print; TEST_DIR/calibration_sources.json remembers where each
#        one came from. A picture stored with one channel is left out.
#
#     2. python colorize_model.py TEST_DIR
#        Step 1 makes the guides, as for any folder of prints (ComfyUI must
#        be running).
#
#     3. python calibrate_color.py compare TEST_DIR
#        Every guide is laid on its print exactly as step 2 lays it -- the
#        fit measured on the print at its full size -- with the colour made
#        at COMPARE_SIDE on the long side, and set against the real colour.
#
# THE NUMBERS. Step 2 first takes the model's cast off the guide's colour
# (CAST_AB, see THE MODEL'S CAST there), then multiplies each pixel's
# colour by a gain that depends on its hue family (see HOW STRONG THE
# COLOUR IS there). Measured here:
#
#     the cast   the median colour of the guide, laid on the print, where
#                the real print is grey (chroma below CHROMA_MIN, L* in
#                CAST_L), less the real print's own there;
#     the gains  for each family, over the pixels where the model picked
#                about the real hue (within HUE_AGREE degrees, both
#                coloured) and step 2 keeps at least KEEP_MIN of the colour
#                (its LIGHT_FADE and DARK_FADE): the real chroma over the
#                model's, with the cast taken off -- how much stronger or
#                fainter the model makes a colour it gets right. A family
#                with fewer than MIN_AGREE_PX such pixels keeps 1; every
#                gain stays between GAIN_MIN and GAIN_MAX.
#
# Only the pixels with the right hue set a gain: where the model picked
# another hue -- a shirt blue that was red -- no gain can mend it, and
# letting those pixels in would only make every colour fainter. So the
# report gives, for each family, how many of its pixels have about the
# real hue, and how many are coloured where the real print is grey; and
# the colour error (mean CIE76 distance in a*, b*) of each print, before
# and after. The numbers are printed ready to paste into transfer_color.py.
#
# The real prints must have the colours wanted: a faded print teaches the
# model's colour to fade. The gains are a measurement, not an order: step 2
# may keep lower ones chosen by eye (it does for yellow, cyan and blue).
# Its LIGHT_FADE and DARK_FADE are used as they are, in "now" and "new";
# the gains are measured only where they leave the colour whole, so a gain
# is right where it counts fully, and the fades do the rest as in step 2.
#
# compare also writes, for a look:
#
#     calibration_preview/<name>.jpg   the real print, step 2's colour as
#                                      it is now, and with the new gains
#
# Usage:
#     python calibrate_color.py make COLOUR_DIR TEST_DIR
#     python calibrate_color.py compare TEST_DIR [--seed N]
#
#     python calibrate_color.py make D:\archive\colour_prints D:\calib
#     python calibrate_color.py compare D:\calib
# ==========================================================================
from __future__ import annotations

import argparse
import json
import os
import sys
import time

try:
    import cv2
    import numpy as np
except ModuleNotFoundError as exc:
    sys.exit(f"missing package: {exc.name} (see Requirements above)")

try:
    from common import (SCHEMA_VERSION, SEED, Progress, colour_part,
                        console_setup, count, elapsed, error, error_list,
                        guide_dir, guide_name, heading, info, paint, phase,
                        read_page, resolve, unit_range, warn_list,
                        write_json, write_tiff)
    import transfer_color as step2
except ModuleNotFoundError as exc:
    sys.exit(f"{exc.name}.py must sit beside this script"
             if exc.name in ("common", "transfer_color")
             else f"missing package: {exc.name} (see Requirements above)")

# --------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------
# --- general ---------------------------------------------------------------
SCRIPT = "calibrate_color.py"
SOURCES_FILE = "calibration_sources.json"    # in TEST_DIR, written by make
COMPARE_PREVIEW_DIR = "calibration_preview"  # in TEST_DIR, written by
                                             # compare
PREVIEW_QUALITY = 88

# --- compare ---------------------------------------------------------------
COMPARE_SIDE = 1200     # prints are compared at this long side
CHROMA_MIN = 5.0        # a pixel with less chroma (C*ab) counts as grey
HUE_AGREE = 30.0        # degrees: about the real hue
CAST_L = (30.0, 95.0)   # the cast is read on grey of this lightness
MIN_AGREE_PX = 2000     # a family with fewer right-hue pixels keeps 1
KEEP_MIN = 0.99         # gains are measured where step 2's fades keep at
                        # least this share of the colour
GAIN_MIN, GAIN_MAX = 0.2, 1.5
COLOUR_EXTS = (".tif", ".tiff", ".jpg", ".jpeg", ".png")


# --------------------------------------------------------------------------
# make: colour prints in B&W
# --------------------------------------------------------------------------

def to_bw(arr: np.ndarray) -> np.ndarray | None:
    """The print in B&W, 16-bit: sRGB grey with each pixel's own L*; None
    when it is not a colour print."""
    pic = colour_part(arr)
    if pic.ndim == 2:
        return None
    rgb = np.clip(pic.astype(np.float32) / unit_range(arr.dtype), 0, 1)
    grey = step2.l_to_grey(step2.srgb_to_lab(rgb)[..., 0])
    return np.round(grey * 65535.0).astype(np.uint16)


def make(colour_dir: str, test_dir: str) -> int:
    t0 = time.time()
    src_dir, dst_dir = resolve(colour_dir), resolve(test_dir)
    if os.path.normcase(src_dir) == os.path.normcase(dst_dir):
        error("COLOUR_DIR and TEST_DIR must be two folders")
        return 1
    if not os.path.isdir(src_dir):
        error(f"no folder {src_dir}")
        return 1
    names = sorted(f for f in os.listdir(src_dir)
                   if f.lower().endswith(COLOUR_EXTS)
                   and os.path.isfile(os.path.join(src_dir, f)))
    if not names:
        error(f"no colour pictures (TIFF, JPEG, PNG) in {src_dir}")
        return 1

    phase(1, 1, "the colour prints in B&W")
    os.makedirs(dst_dir, exist_ok=True)
    sources, skipped, failed = {}, [], []
    bar = Progress(len(names), "    writing")
    for name in names:
        try:
            arr, meta = read_page(os.path.join(src_dir, name))
            bw = to_bw(arr)
            if bw is None:
                skipped.append(name)
            else:
                meta = dict(meta, photometric="minisblack", planarconfig=None,
                            extrasamples=None, colormap=None, iccprofile=None)
                if str(meta.get("description") or "").startswith('{"shape"'):
                    meta["description"] = None    # tifffile's own shape
                                                  # note: not the grey's
                out = os.path.splitext(name)[0] + ".tif"
                if out in sources:
                    raise ValueError(f"a second picture named {out}")
                dst = os.path.join(dst_dir, out)
                write_tiff(dst + ".tmp", bw, meta)
                os.replace(dst + ".tmp", dst)
                sources[out] = os.path.join(src_dir, name)
        except (Exception, SystemExit) as exc:
            failed.append(f"{name}: {type(exc).__name__}: {exc}")
        bar.update()
    bar.close()
    write_json(os.path.join(dst_dir, SOURCES_FILE),
               {"version": SCHEMA_VERSION, "sources": sources})

    print(paint(f"\nwrote {count(len(sources), 'print')} in B&W in "
                f"{dst_dir}", "32"))
    if skipped:
        warn_list(f"    ! {count(len(skipped), 'picture')} with one channel "
                  f"only, left out:", skipped)
    if failed:
        error_list(f"    {count(len(failed), 'picture')} could not be "
                   f"written:", failed)
    print(paint(f"    done in {elapsed(t0)}", "32"))
    print(paint(f"\nnext, with ComfyUI running:\n    python colorize_model.py "
                f"\"{dst_dir}\"\nthen:\n    python {SCRIPT} compare "
                f"\"{dst_dir}\"", "1"))
    return 2 if failed else 0


# --------------------------------------------------------------------------
# compare: the colour against the real one
# --------------------------------------------------------------------------

def small(img: np.ndarray, size) -> np.ndarray:
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def chroma_hue(a, b):
    return np.hypot(a, b), np.degrees(np.arctan2(b, a)) % 360


def ab_error(a, b, ta, tb) -> float:
    return float(np.mean(np.hypot(a - ta, b - tb)))


def compare(test_dir: str, seed: int) -> int:
    t0 = time.time()
    root = resolve(test_dir)
    try:
        with open(os.path.join(root, SOURCES_FILE), encoding="utf-8") as f:
            sources = json.load(f)["sources"]
    except (OSError, ValueError, KeyError):
        error(f"no {SOURCES_FILE} in {root}; run 'make' first")
        return 1
    gdir = guide_dir(root)

    phase(1, 3, "the prints and their guides")
    todo, lacking = [], []
    for name, src in sorted(sources.items()):
        g = os.path.join(gdir, guide_name(name, seed))
        if not os.path.isfile(g):
            lacking.append(name)
        elif not os.path.isfile(src):
            lacking.append(f"{name} (its colour print {src} is gone)")
        else:
            todo.append((name, src, g))
    info(f"{count(len(sources), 'print')}; {len(todo)} with a guide "
         f"(seed {seed})")
    if lacking:
        warn_list(f"    ! {count(len(lacking), 'print')} left out:", lacking)
    if not todo:
        error("nothing to compare; make the guides first (colorize_model.py)")
        return 1

    phase(2, 3, "the colour laid on the prints")
    fams = sorted(step2.HUE_CENTRES, key=lambda n: step2.HUE_CENTRES[n] % 360)
    per_print = []
    bar = Progress(len(todo), "    laying")
    for name, src, gpath in todo:
        try:
            grey, _, _ = step2.print_grey(os.path.join(root, name))
            h, w = grey.shape
            f = min(1.0, COMPARE_SIDE / float(max(h, w)))
            size = (max(1, round(w * f)), max(1, round(h * f)))
            a, b, grey, _ = step2.colour_ab(grey, gpath, size)
            arr, _ = read_page(src)
            rgb = colour_part(arr).astype(np.float32) / unit_range(arr.dtype)
            truth = step2.srgb_to_lab(np.clip(small(rgb, size), 0, 1))
            per_print.append((name, grey, a, b, truth))
        except (Exception, SystemExit) as exc:
            bar.message(paint(f"    ! {name}: {type(exc).__name__}: {exc}",
                              "33"))
        bar.update()
    bar.close()
    if not per_print:
        error("no print could be compared")
        return 1

    phase(3, 3, "the numbers")
    # the cast: the guide's colour where the real print is grey
    da, db = [], []
    for _, _, a, b, truth in per_print:
        tc = np.hypot(truth[..., 1], truth[..., 2])
        g = (tc < CHROMA_MIN) & (truth[..., 0] >= CAST_L[0]) \
            & (truth[..., 0] <= CAST_L[1])
        da.append((a - truth[..., 1])[g])
        db.append((b - truth[..., 2])[g])
    da, db = np.concatenate(da), np.concatenate(db)
    cast = ((round(float(np.median(da)), 1) + 0.0,
             round(float(np.median(db)), 1) + 0.0)
            if da.size >= MIN_AGREE_PX else (0.0, 0.0))
    info(f"the model's cast, on {da.size} px where the real print is grey: "
         f"a* {cast[0]:+.1f}, b* {cast[1]:+.1f}")

    # the gains: the real chroma over the model's, where the hue is right
    sums = {n: np.zeros(4) for n in fams}   # px, coloured, agree, grey
    num = {n: 0.0 for n in fams}
    den = {n: 0.0 for n in fams}
    used = {n: 0 for n in fams}             # right hue, colour kept whole
    for _, _, a, b, truth in per_print:
        ua, ub = step2.uncast(a, b, cast)
        c, hue = chroma_hue(ua, ub)
        tc, thue = chroma_hue(truth[..., 1], truth[..., 2])
        weights = step2.hue_weights(ua, ub)
        fam = np.argmax(np.stack([w for _, w in weights]), axis=0)
        near = np.abs((hue - thue + 180) % 360 - 180) <= HUE_AGREE
        coloured = c >= CHROMA_MIN
        whole = step2.light_fade(truth[..., 0]) >= KEEP_MIN
        right = coloured & (tc >= CHROMA_MIN) & near
        for i, (n, _) in enumerate(weights):
            mine = fam == i
            sums[n] += [mine.sum(), (mine & coloured).sum(),
                        (mine & right).sum(),
                        (mine & coloured & (tc < CHROMA_MIN)).sum()]
            sel = mine & right & whole
            used[n] += int(sel.sum())
            num[n] += float(tc[sel].sum())
            den[n] += float(c[sel].sum())
    new = {}
    for n in fams:
        if used[n] >= MIN_AGREE_PX and den[n] > 0:
            new[n] = round(float(np.clip(num[n] / den[n], GAIN_MIN,
                                         GAIN_MAX)), 2)
        else:
            new[n] = 1.0

    total = sum(v[0] for v in sums.values())
    print()
    print(f"    {'family':<8}{'pixels':>8}{'right hue':>11}{'grey made':>11}"
          f"{'gain':>8}")
    for n in fams:
        px, col, agree, grey_made = sums[n]
        few = "  (few: kept 1)" if used[n] < MIN_AGREE_PX else ""
        if col == 0:
            print(f"    {n:<8}{px / total:>7.0%}{'-':>11}{'-':>11}"
                  f"{new[n]:>8.2f}{few}")
            continue
        print(f"    {n:<8}{px / total:>7.0%}{agree / col:>10.0%} "
              f"{grey_made / col:>10.0%} {new[n]:>7.2f}{few}")
    print(paint(f"    right hue: coloured pixels within {HUE_AGREE:.0f} "
                "degrees of the real hue; grey made: coloured where the "
                "real print is grey; the gain is measured on the right-hue "
                "pixels step 2 does not fade", "90"))

    print()
    print(f"    {'print':<24}{'raw':>8}{'now':>8}{'new':>8}   "
          f"(mean colour error, CIE76 in a*b*)")
    os.makedirs(os.path.join(root, COMPARE_PREVIEW_DIR), exist_ok=True)
    totals = np.zeros(3)
    for name, grey, a, b, truth in per_print:
        ta, tb = truth[..., 1], truth[..., 2]
        errs = []
        for cst, sat, gn in (((0.0, 0.0), 1.0, {}),
                             (step2.CAST_AB, step2.SATURATION,
                              step2.HUE_GAINS),
                             (cast, 1.0, new)):
            ua, ub = step2.uncast(a, b, cst)
            g = step2.colour_gain(ua, ub, sat, gn)
            if gn:                          # step 2's fades, but not raw
                g = g * step2.light_fade(truth[..., 0])
            errs.append(ab_error(ua * g, ub * g, ta, tb))
        totals += errs
        print(f"    {name[:23]:<24}{errs[0]:>8.1f}{errs[1]:>8.1f}"
              f"{errs[2]:>8.1f}")
        tiles = [step2.lab_to_srgb(truth[..., 0], truth[..., 1],
                                   truth[..., 2]),
                 step2.finish(grey, a, b),
                 step2.finish(grey, a, b, 1.0, new, cast)]
        gap = np.ones((grey.shape[0], 12, 3), np.float32)
        sheet = np.hstack([tiles[0], gap, tiles[1], gap, tiles[2]])
        u8 = np.round(sheet[..., ::-1] * 255).astype(np.uint8)
        ok, buf = cv2.imencode(".jpg", u8, [cv2.IMWRITE_JPEG_QUALITY,
                                            PREVIEW_QUALITY])
        if ok:
            buf.tofile(os.path.join(root, COMPARE_PREVIEW_DIR,
                                    os.path.splitext(name)[0] + ".jpg"))
    n = len(per_print)
    print(f"    {'all':<24}{totals[0] / n:>8.1f}{totals[1] / n:>8.1f}"
          f"{totals[2] / n:>8.1f}")
    print(paint("    raw: the guide's colour as it is; now: with the "
                "numbers in transfer_color.py; new: with the numbers "
                "below", "90"))

    print(paint("\nthe numbers measured, as transfer_color.py writes them "
                "(its own may stay\nlower by choice, see HOW STRONG THE "
                "COLOUR IS at the top of transfer_color.py):", "1"))
    print(f"CAST_AB = ({cast[0]:.1f}, {cast[1]:.1f})       "
          f"# the model's cast, taken off (a*, b*)")
    print("SATURATION = 1.0            # every colour times this")
    print("HUE_GAINS = {               # each hue family's colour times this")
    items = [f'"{n}": {new[n]:.2f}' for n in fams]
    for i in range(0, len(items), 4):
        print("    " + ", ".join(items[i:i + 4]) + ",")
    print("}")
    print(paint(f"\nthe real print, now and new side by side: "
                f"{os.path.join(root, COMPARE_PREVIEW_DIR)}", "32"))
    print(paint(f"    done in {elapsed(t0)}", "32"))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Stage 5 — step 2's colour numbers, measured on colour "
                    "prints")
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make", help="write colour prints in B&W")
    mk.add_argument("colour_dir", help="the folder of colour prints")
    mk.add_argument("test_dir", help="the folder to write them to, in B&W")
    cp = sub.add_parser("compare", help="set the colour against the real "
                                        "one; print the numbers")
    cp.add_argument("test_dir", help="the folder 'make' wrote")
    cp.add_argument("--seed", type=int, default=SEED,
                    help=f"which guides: <name>_seed<N>.png (default "
                         f"{SEED})")
    args = ap.parse_args(argv)
    console_setup()
    if args.cmd == "make":
        heading("STAGE 5  -  calibration: colour prints in B&W")
        return make(args.colour_dir, args.test_dir)
    heading("STAGE 5  -  calibration: the colour against the real one")
    return compare(args.test_dir, args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
