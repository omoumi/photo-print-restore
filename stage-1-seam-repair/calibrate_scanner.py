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
# calibrate_scanner.py  --  STAGE 1, step 1 of 3: measure pitch and phase
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
# Measures the seam PITCH and PHASE of the scanner once, from many scans.
# This step only measures: what it writes is two numbers.
#
# WHY MANY SCANS. In one photo the seam signal is a few grey levels, buried
# under the photo's own texture. The texture differs from photo to photo
# while the seams never move, so averaging the seam evidence of many scans
# cancels the texture and leaves the grid standing.
#
# THE FUNDAMENTAL. A grid of pitch P also scores well at 2P, 3P..., because
# every other line still lands on a seam, so the smallest pitch scoring near
# the best is the one chosen. Those multiples are the same grid, so the
# margin over "the next candidate" is taken against the best pitch that is
# not one of them.
#
# NOTHING IS WRITTEN UNLESS THE ANSWER IS CLEAR. A wrong pitch would mis-grid
# every scan in silence, so a weak result says what to do instead and writes
# nothing.
#
# One fixed output, derived from the project root -- the folder holding the
# scans, the one path argument this step takes:
#
#     json files/boxes_calibrate_scanner.json   pitch, phase and how clear
#                                               the result is; step 2 reads
#                                               it on its own
#
# Usage:
#     python calibrate_scanner.py [ROOT] [options]
#
#     ROOT              project root: the folder holding the scans
#                       (default: this folder)
#     --max-images N    how many scans to measure (default 40); more is
#                       better
#     --min-pitch N     narrowest pitch to consider, px (default 80)
#     --max-pitch N     widest pitch to consider, px (default 1200)
#     --exts .tif,...   which files count as scans
#     --workers N       parallel workers (default: auto; 1 disables the pool)
#
#     python calibrate_scanner.py
#     python calibrate_scanner.py D:\archive\scans --max-images 200
# ==========================================================================
from __future__ import annotations

import argparse
import os
import sys
import time

try:
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view
except ModuleNotFoundError as exc:
    sys.exit(f"missing package: {exc.name} (see Requirements above)")

try:
    from common import (SCAN_EXTS, SCHEMA_VERSION, STAGE_CALIBRATE, Progress,
                        check_scans_present, console_setup, count, elapsed,
                        error, heading, info, list_scans, paint, parse_exts,
                        phase, pipeline_root, read_scan, root_arg,
                        seam_evidence, stage_json, warn, warn_list,
                        weighted_median_offset, write_json)
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
SCRIPT = "calibrate_scanner.py"
WORKER_MEM_MULTIPLE = 3.0   # peak RAM of one worker per byte of the scan
                            # file: the scan and one band at a time

# --- the scans measured ----------------------------------------------------
MAX_IMAGES = 40             # --max-images
FEW_SCANS = 5               # below this many, warn that the result is weak
MIN_READABLE = 3            # fewest readable scans to measure at all

# --- searching for the pitch -----------------------------------------------
MIN_PITCH = 80              # --min-pitch, px
MAX_PITCH = 1200            # --max-pitch, px; never above a quarter of the
                            # scan's width
BASELINE_WIDTH = 61         # columns in the local baseline taken off each
                            # scan's evidence, so only thin peaks remain
GRID_TOL = 2                # a grid line takes the strongest column within
                            # this many px
MIN_GRID_LINES = 4          # a phase needs at least this many grid lines
FUNDAMENTAL_SHARE = 0.9     # the smallest pitch scoring this share of the
                            # best is the fundamental
MULTIPLE_TOL = 3            # px per multiple: pitches this close to a
                            # multiple of the chosen one are the same grid
REFINE_RADIUS = 6           # how far the phase may move onto the seams, px
REFINE_REL = 0.35           # a seam counts for that move at this share of
REFINE_PERCENTILE = 99      # the strongest column, and never below this
                            # percentile of all columns
MIN_SIGMA = 4.0             # the chosen pitch must stand this many standard
MIN_MARGIN = 1.15           # deviations clear, and this many times the next
                            # candidate that is not a multiple of it
TOP_SHOWN = 8               # candidates listed


# --------------------------------------------------------------------------
# the evidence of one scan
# --------------------------------------------------------------------------

def seam_profile(a):
    """Per-column seam votes of one scan, less a local baseline, scaled so
    the strongest column is 1. Brightness and busy texture raise a column
    and its neighbours alike, so the baseline removes them and only thin
    isolated peaks remain."""
    v, _ = seam_evidence(a)
    k = BASELINE_WIDTH
    w = v - np.convolve(v, np.ones(k) / k, mode="same")
    w[w < 0] = 0.0
    m = w.max()
    return w / m if m > 0 else w


def _job(path):
    """(name, profile, error) of one scan. Top-level so it can be pickled.

    The scan is opened inside the worker and only its profile comes back: a
    few thousand floats, against the tens of megabytes of the scan."""
    name = os.path.basename(path)
    try:
        a, _ = read_scan(path)
        return name, seam_profile(a), None
    except Exception as exc:            # a bad file must not stop the run
        return name, None, f"{type(exc).__name__}: {exc}"


def accumulate(profiles):
    """The mean profile on a common column axis; a narrower scan counts over
    its own width only."""
    n = max(len(p) for p in profiles)
    acc = np.zeros(n)
    cnt = np.zeros(n)
    for p in profiles:
        acc[:len(p)] += p
        cnt[:len(p)] += 1.0
    cnt[cnt == 0] = 1.0
    return acc / cnt


# --------------------------------------------------------------------------
# searching for the pitch
# --------------------------------------------------------------------------

def score_pitches(acc, lo, hi):
    """{pitch: (score, phase)} for every pitch lo..hi.

    For each phase, the mean of the evidence on the grid is taken as an
    EXCESS over the mean column, so a grid landing on noise scores about
    zero, and multiplied by sqrt(number of grid lines), so a pitch that
    explains the whole width beats a wide one with few lines to fit. The
    best phase gives the pitch its score."""
    n = len(acc)
    base = float(acc.mean())
    pad = np.full(GRID_TOL, -np.inf)
    near = sliding_window_view(np.concatenate([pad, acc, pad]),
                               2 * GRID_TOL + 1).max(axis=1)
    out = {}
    for P in range(lo, hi + 1):
        rows = -(-n // P)
        buf = np.zeros(rows * P)
        buf[:n] = near
        lines = np.full(P, rows)
        lines[n - (rows - 1) * P:] -= 1       # phases without a last line
        ok = lines >= MIN_GRID_LINES
        if not ok.any():
            out[P] = (-1e9, 0)
            continue
        mean = buf.reshape(rows, P).sum(axis=0) / np.maximum(lines, 1)
        exc = np.where(ok, (mean - base) * np.sqrt(lines), -np.inf)
        ph = int(np.argmax(exc))
        out[P] = (float(exc[ph]), ph)
    return out


def _is_multiple(p, of):
    m = int(round(p / float(of)))
    return m >= 1 and abs(p - m * of) <= MULTIPLE_TOL * m


def choose(acc, scores):
    """(pitch, phase, score, sigma, margin) from the scores.

    The pitch is the fundamental: the smallest scoring FUNDAMENTAL_SHARE of
    the best. sigma is how far its score stands above all pitches, margin
    how many times it beats the best pitch that is not a multiple of it,
    None when no such pitch scores above zero."""
    best = max(e for e, _ in scores.values())
    pitch = min(p for p, (e, _) in scores.items()
                if e >= FUNDAMENTAL_SHARE * best)
    exc, ph = scores[pitch]
    off = weighted_median_offset(
        acc, ph, pitch, REFINE_RADIUS,
        max(REFINE_REL * float(acc.max()),
            float(np.percentile(acc, REFINE_PERCENTILE))))
    phase = (ph + (off or 0)) % pitch

    all_exc = np.array([e for e, _ in scores.values()])
    sigma = (exc - all_exc.mean()) / (all_exc.std() or 1.0)
    runner_up = max((e for p, (e, _) in scores.items()
                     if not _is_multiple(p, pitch)), default=0.0)
    margin = exc / runner_up if runner_up > 0 else None
    return pitch, phase, exc, sigma, margin


# --------------------------------------------------------------------------
# the step
# --------------------------------------------------------------------------

def run(root, args) -> int:
    t0 = time.time()
    exts = parse_exts(args.exts)

    phase(1, 3, "the scans")
    files = list_scans(root, exts)
    use = files[:args.max_images]
    info(f"{len(use)} of {count(len(files), 'scan')} measured")
    if len(use) < FEW_SCANS:
        warn(f"    ! only {count(len(use), 'scan')}: this is a collective "
             f"measurement, and with few scans each photo's own texture can "
             f"hide the grid")
    tasks = [os.path.join(root, f) for f in use]
    workers = 1
    if parallel is not None and args.workers != 1:
        per = parallel.estimate_bytes_per_worker(tasks[0],
                                                 multiple=WORKER_MEM_MULTIPLE)
        workers = parallel.choose_workers(len(tasks), per, args.workers)
    info(count(workers, "worker"))

    bar = Progress(len(tasks), "    measuring")
    if workers <= 1 or len(tasks) <= 1:
        results = []
        for t in tasks:
            results.append(_job(t))
            bar.update()
    else:
        # a worker that dies (out of memory on a huge scan) must cost its
        # own scan, not the evidence of every other one
        results = parallel.run(
            _job, tasks, workers, on_result=bar.update,
            on_error=lambda t, exc: (os.path.basename(t), None,
                                     parallel.describe_error(exc)))
    bar.close()

    profiles, skipped = [], []
    for name, prof, err in results:
        if err:
            skipped.append(f"{name}: {err}")
        else:
            profiles.append(prof)
    if skipped:
        warn_list(f"    ! {count(len(skipped), 'scan')} could not be read "
                  f"and {'is' if len(skipped) == 1 else 'are'} left out:",
                  skipped)
    if len(profiles) < MIN_READABLE:
        error(f"only {count(len(profiles), 'readable scan')}; at least "
              f"{MIN_READABLE} are needed")
        return 1

    phase(2, 3, "the pitch and the phase")
    acc = accumulate(profiles)
    cols = len(acc)
    hi = min(args.max_pitch, cols // 4)
    if hi < args.min_pitch:
        error(f"the scans are too narrow for a pitch of {args.min_pitch} px "
              f"or more")
        return 1
    info(f"pitches {args.min_pitch}..{hi} px, over {cols} columns")
    scores = score_pitches(acc, args.min_pitch, hi)
    pitch, grid_phase, exc, sigma, margin = choose(acc, scores)

    info()
    info("the best candidates:")
    for p, (e, q) in sorted(scores.items(), key=lambda kv: -kv[1][0])[
            :TOP_SHOWN]:
        mark = "   <= chosen (fundamental)" if p == pitch else ""
        print(paint(f"        pitch={p:5d}  phase={q:4d}  score={e:6.2f}",
                    "90") + mark)
    rival = (f"{margin:.2f}x the next candidate" if margin is not None
             else "no other candidate scores")
    info()
    print(paint(f"    pitch = {pitch} px   phase = {grid_phase}   "
                f"({sigma:.1f} sigma, {rival})", "1"))

    phase(3, 3, "the profile")
    if sigma < MIN_SIGMA or (margin is not None and margin < MIN_MARGIN):
        error(f"not writing a profile: this result is not clear enough (it "
              f"needs {MIN_SIGMA:g} sigma and {MIN_MARGIN:g}x)")
        print(paint(f"\nit came from {count(len(profiles), 'scan')}; "
                    f"measure more:\n"
                    f"    python {SCRIPT}{root_arg(root)} --max-images 200\n"
                    f"or, if you know the pitch, skip this step:\n"
                    f"    python detect_seams.py{root_arg(root)} --pitch N",
                    "1"))
        return 1

    dst = stage_json(root, STAGE_CALIBRATE)
    write_json(dst, {
        "version": SCHEMA_VERSION,
        "source_dir": str(root),
        "tool": SCRIPT,
        "note": "pitch and phase of the scanner's seam grid, in columns of "
                "the STORED raster: its seams lie at phase + k * pitch, each "
                "between that column and the next. They belong to the "
                "machine, not to this batch. confidence_sigma is how far the "
                "pitch stands above all pitches tried, margin_over_next how "
                "many times it beats the best one that is not a multiple of "
                "it (null: no such pitch scores).",
        "pitch": int(pitch), "phase": int(grid_phase), "columns": int(cols),
        "images_used": len(profiles),
        "confidence_sigma": round(float(sigma), 2),
        "margin_over_next": (round(float(margin), 3) if margin is not None
                             else None)})
    print(paint(f"\nwrote {dst}  (from {count(len(profiles), 'scan')})",
                "32"))
    print(paint(f"    done in {elapsed(t0)}", "32"))
    print(paint(f"\nnext:  python detect_seams.py{root_arg(root)}", "1"))
    return 2 if skipped else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Stage 1 step 1 of 3 — measure pitch and phase")
    ap.add_argument("root", nargs="?", default=".",
                    help="project root: the folder holding the scans "
                         "(default: this folder)")
    ap.add_argument("--max-images", type=int, default=MAX_IMAGES,
                    help=f"how many scans to measure (default: {MAX_IMAGES}); "
                         "more is better")
    ap.add_argument("--min-pitch", type=int, default=MIN_PITCH,
                    help=f"narrowest pitch to consider, px "
                         f"(default: {MIN_PITCH})")
    ap.add_argument("--max-pitch", type=int, default=MAX_PITCH,
                    help=f"widest pitch to consider, px "
                         f"(default: {MAX_PITCH})")
    ap.add_argument("--exts", default=",".join(SCAN_EXTS),
                    help="which files count as scans")
    if parallel is not None:
        parallel.add_workers_arg(ap)
    else:
        ap.add_argument("--workers", type=int, default=1,
                        help="ignored: parallel.py not found")
    args = ap.parse_args(argv)
    console_setup()

    # exit code 1, not argparse's 2: 2 means "measured, some scans left out"
    if args.max_images < MIN_READABLE:
        raise SystemExit(f"--max-images must be {MIN_READABLE} or more, not "
                         f"{args.max_images}")
    if args.min_pitch < 2:
        raise SystemExit(f"--min-pitch must be 2 px or more, not "
                         f"{args.min_pitch}")
    if args.max_pitch < args.min_pitch:
        raise SystemExit(f"--max-pitch must not be below --min-pitch "
                         f"({args.min_pitch}), not {args.max_pitch}")

    root = pipeline_root(args.root)
    check_scans_present(root, parse_exts(args.exts))
    heading("STAGE 1 step 1  -  measure pitch and phase")
    return run(root, args)


if __name__ == "__main__":
    raise SystemExit(main())
