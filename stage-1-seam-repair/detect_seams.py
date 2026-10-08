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
# detect_seams.py  --  STAGE 1, step 2 of 3: locate the seams in each scan
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
#                 # pillow for --preview and for scans that are not TIFF
#     common.py beside this file; parallel.py too, for --workers
#
# This step only measures: what it writes are column numbers.
#
# THE GRID IS RIGID. The seams come from the sensor's chip layout, so every
# chip boundary is marked at the exact pitch, including the ones that fall on
# smooth or dark content and cannot be seen at all. No single grid line is
# ever moved onto nearby texture; only the phase of the whole grid is
# measured.
#
# THE PHASE. With the pitch and phase from step 1 (or --phase), the phase is
# used exactly as given. Without one, each scan locks it on its own: every
# candidate phase is scored by the SUM OF SQUARES of the seam votes its grid
# lines land on, so a few real seams outweigh the many weak texture peaks of
# a busy photo, and the grid is then moved onto the seams it finds. A scan
# that is mostly blank has too little signal to lock alone, so once the
# folder agrees on a phase, that phase is imposed on every scan: the grid is
# the same in all of them (--no-consensus turns that off).
#
# Each scan is labelled by how many of its seams are clearly VISIBLE: HIGH
# for two or more, MEDIUM for one, LOW for none. With a known or agreed
# phase the label says nothing about whether the positions are right, so
# only LOW scans whose phase rests on themselves alone are warned about.
#
# It reads json files/boxes_calibrate_scanner.json, step 1's pitch and
# phase, unless --pitch is given. Two fixed outputs, both derived from the
# project root -- the folder holding the scans, the one path argument this
# step takes:
#
#     json files/boxes_detect_seams.json   always: the seam columns step 3
#                                          repairs
#     seam_previews/<name>_seam.png        only with --preview: an 8-bit copy
#                                          with the seams marked in red, for
#                                          the eye
#
# Usage:
#     python detect_seams.py [ROOT] [options]
#
#     ROOT              project root: the folder holding the scans
#                       (default: this folder)
#     --pitch PX        seam pitch (default: from step 1)
#     --phase COL       use this seam column as the phase (default: from
#                       step 1, else locked in each scan)
#     --preview         also write a marked PNG per scan, to check by eye
#     --no-consensus    let every scan keep the phase it locked on its own
#     --exts .tif,...   which files count as scans
#     --workers N       parallel workers (default: auto; 1 disables the pool)
#
#   detection tuning:
#     --bands K         fewest voting bands per scan (default 20)
#     --lock PX         how far from a grid line a seam still counts, when
#                       the phase is scored and when seams are counted as
#                       clearly visible (default 3)
#
#     python detect_seams.py
#     python detect_seams.py D:\archive\scans --preview
# ==========================================================================
from __future__ import annotations

import argparse
import json
import os
import sys
import time

try:
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view
except ModuleNotFoundError as exc:
    sys.exit(f"missing package: {exc.name} (see Requirements above)")

try:
    from common import (MIN_BANDS, PREVIEW_DIR, SCAN_EXTS, SCHEMA_VERSION,
                        STAGE_CALIBRATE, STAGE_DETECT, Image, Progress,
                        check_scans_present, console_setup, count,
                        display_rgb, elapsed, error, full_scale, heading,
                        info, list_scans, paint, parse_exts, phase,
                        pipeline_root, read_scan, root_arg, seam_evidence,
                        stage_json, warn, warn_list, weighted_median_offset,
                        write_json)
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
SCRIPT = "detect_seams.py"
WORKER_MEM_MULTIPLE = 3.0   # peak RAM of one worker per byte of the scan
                            # file: the scan, one band at a time, a preview

# --- the grid --------------------------------------------------------------
DEFAULT_LOCK = 3            # --lock, px

# --- moving the grid onto the seams (phase not given) ----------------------
REFINE_RADIUS = 6           # how far the grid may move onto the seams, px
REFINE_REL = 0.35           # a seam counts for that move at this share of
REFINE_ABS = 0.25           # the strongest vote, and never below this vote

# --- how visible the seams are (the HIGH / MEDIUM / LOW label) -------------
HIGH_MIN_CLEAR = 2          # clearly visible seams for HIGH; one is MEDIUM,
                            # none LOW
CLEAR_REL = 0.6             # a seam is "clearly visible" when its vote
CLEAR_ABS = 0.5             # reaches this share of the strongest (and this
CLEAR_LEVELS = 4.0          # vote) and its step this many grey levels
                            # (8-bit scale)

# --- the folder phase ------------------------------------------------------
CONSENSUS_TOL = 6           # phases this close vote together, px (or --lock
                            # if more)
CONSENSUS_MIN_SCANS = 3     # fewest scans, and fewest agreeing scans, for a
                            # folder phase
CONSENSUS_MIN_SHARE = 0.4   # share of the scans that must agree

# --- previews --------------------------------------------------------------
PREVIEW_MIN_WIDTH = 500     # narrower previews are enlarged to at least this
PREVIEW_MARK = (255, 0, 0)  # colour of a seam in the preview
PREVIEW_PNG_LEVEL = 1       # PNG compression of a preview: fast, as it is
                            # only looked at


# --------------------------------------------------------------------------
# locking the grid
# --------------------------------------------------------------------------

def grid_columns(n, phase, pitch):
    """The rigid grid: every column phase + k*pitch inside 0..n-1, n being
    the number of column boundaries (the scan's width less one)."""
    return list(range(int(phase) % pitch, n, pitch))


def _phase_scores(v, pitch, lock):
    """Score of every phase 0..pitch-1: the sum, over its grid lines, of the
    squared strongest vote within `lock` of the line."""
    n = len(v)
    pad = np.full(lock, -np.inf)
    near = sliding_window_view(np.concatenate([pad, v, pad]),
                               2 * lock + 1).max(axis=1)
    rows = -(-n // pitch)
    buf = np.zeros(rows * pitch)
    buf[:n] = near ** 2
    return buf.reshape(rows, pitch).sum(axis=0)


def count_clear(v, step, seams, scale, lock):
    """How many of `seams` are clearly visible: within `lock` of the line,
    a vote of CLEAR_REL of the strongest (and at least CLEAR_ABS) and a
    column step of CLEAR_LEVELS grey levels."""
    n = len(v)
    vote_min = max(CLEAR_REL * float(v.max()), CLEAR_ABS)
    step_min = CLEAR_LEVELS * scale / 255.0
    clear = 0
    for c in seams:
        lo, hi = max(0, c - lock), min(n, c + lock + 1)
        if v[lo:hi].max() >= vote_min and step[lo:hi].max() >= step_min:
            clear += 1
    return clear


def label(clear, n_seams, source):
    """(level, note): the level from how many seams are clearly visible;
    the note also says where the phase came from."""
    if not n_seams:
        return "LOW", "no grid line falls inside this scan"
    level = ("HIGH" if clear >= HIGH_MIN_CLEAR
             else "MEDIUM" if clear else "LOW")
    return level, (f"{source}; {clear} of {count(n_seams, 'seam')} clearly "
                   f"visible")


def lock_grid(v, step, pitch, scale, anchor=None, source="phase given",
              lock=DEFAULT_LOCK):
    """The seam columns of one scan and how visible they are.

    v and step are the scan's seam_evidence. With `anchor` the phase is
    anchor % pitch, exactly, and `source` names where it came from. Without
    it the phase is scored from the votes alone -- a strong colour step is
    mostly texture, while a thin peak in band after band is what marks a
    seam -- and the grid is then moved onto the seams it finds.

    Returns (seams, confidence dict)."""
    if anchor is not None:
        phase = int(anchor) % pitch
    else:
        phase = int(np.argmax(_phase_scores(v, pitch, lock)))
        off = weighted_median_offset(v, phase, pitch, REFINE_RADIUS,
                                     max(REFINE_REL * float(v.max()),
                                         REFINE_ABS))
        if off is not None:
            phase += off
        source = "phase locked in this scan"
    seams = grid_columns(len(v), phase, pitch)
    clear = count_clear(v, step, seams, scale, lock)
    level, note = label(clear, len(seams), source)
    return seams, {"level": level, "clear": clear, "note": note,
                   "phase": (phase % pitch) if seams else None}


# --------------------------------------------------------------------------
# the folder-wide phase
# --------------------------------------------------------------------------

def _circ(a, b, P):
    d = abs(a - b) % P
    return min(d, P - d)


def folder_consensus_phase(records, P, group_tol):
    """The phase the folder agrees on: the largest group of scan phases
    within `group_tol` of one candidate, refined to the median of that
    group.

    Returns (phase, votes, total), with phase None when no group is large
    enough to impose."""
    phases = [r["phase"] for r in records if r.get("phase") is not None]
    if len(phases) < CONSENSUS_MIN_SCANS:
        return None, 0, len(phases)
    best = (0, None)
    for cand in sorted(set(phases)):
        votes = sum(1 for p in phases if _circ(p, cand, P) <= group_tol)
        if votes > best[0]:
            best = (votes, cand)
    votes, cand = best
    if cand is None or votes < max(CONSENSUS_MIN_SCANS,
                                   CONSENSUS_MIN_SHARE * len(phases)):
        return None, votes, len(phases)
    # unwrap the agreeing phases around the candidate before the median
    agree = [cand + ((p - cand + P // 2) % P - P // 2) for p in phases
             if _circ(p, cand, P) <= group_tol]
    return int(round(np.median(agree))) % P, votes, len(phases)


# --------------------------------------------------------------------------
# preview (8-bit, to look at only)
# --------------------------------------------------------------------------

def make_preview(a, seams):
    """An 8-bit copy with each seam marked on the column right of it."""
    disp = display_rgb(a).copy()
    h, w = disp.shape[:2]
    for c in seams:
        disp[:, min(c + 1, w - 1), :] = PREVIEW_MARK
    im = Image.fromarray(disp)
    up = max(1, PREVIEW_MIN_WIDTH // max(w, 1))
    if up > 1:
        im = im.resize((w * up, h * up), Image.NEAREST)
    return im


def _preview_path(prev_dir, name):
    return os.path.join(prev_dir, f"{os.path.splitext(name)[0]}_seam.png")


# --------------------------------------------------------------------------
# one scan
# --------------------------------------------------------------------------

def _job(task):
    """Locate the seams of one scan. Top-level so it can be pickled.

    The worker writes its own preview, so the scan never crosses a pipe;
    only a record of a few hundred bytes comes back."""
    path, pitch, phase, source, bands, lock, prev_path = task
    name = os.path.basename(path)
    try:
        a, _ = read_scan(path)
        v, step = seam_evidence(a, bands)
        seams, conf = lock_grid(v, step, pitch, full_scale(a), anchor=phase,
                                source=source, lock=lock)
        if prev_path:
            make_preview(a, seams).save(prev_path,
                                         compress_level=PREVIEW_PNG_LEVEL)
    except Exception as exc:            # a bad file must not stop the run
        return {"file": name, "error": f"{type(exc).__name__}: {exc}"}
    return {"file": name, "width": int(a.shape[1]),
            "height": int(a.shape[0]), "dtype": str(a.dtype),
            "seams": seams, "conf": conf, "phase": conf["phase"],
            "corrected": False, "error": None}


def _apply_consensus(records, pitch, lock, bands, root, prev_dir, preview):
    """Put every scan on the folder phase, and measure how visible its seams
    are on the new grid. Returns the consensus dict for the project file, or
    None."""
    tol = max(lock, CONSENSUS_TOL)
    ph, votes, total = folder_consensus_phase(records, pitch, tol)
    if ph is None:
        warn(f"    ! no clear folder phase ({votes} of {total} scans agree); "
             f"every scan keeps the phase it locked on its own")
        return None
    print(paint(f"    folder phase {ph} ({votes} of {total} scans agree)",
                "1"))
    fixed = 0
    for r in records:
        if r["phase"] is None or r["phase"] == ph:
            continue
        fixed += 1
        source = f"phase {r['phase']} -> {ph}, the folder phase"
        r["seams"] = grid_columns(r["width"] - 1, ph, pitch)
        r["phase"], r["corrected"] = ph, True
        try:
            a, _ = read_scan(os.path.join(root, r["file"]))
            v, step = seam_evidence(a, bands)
            clear = count_clear(v, step, r["seams"], full_scale(a), lock)
            level, note = label(clear, len(r["seams"]), source)
            if preview:               # phase 2's preview shows the old grid
                make_preview(a, r["seams"]).save(
                    _preview_path(prev_dir, r["file"]),
                    compress_level=PREVIEW_PNG_LEVEL)
        except Exception as exc:
            clear, level = 0, "LOW"
            note = f"{source}; not measured again ({type(exc).__name__})"
            warn(f"    ! {r['file']}: not measured again on the folder "
                 f"phase: {exc}")
        r["conf"] = {"level": level, "clear": clear, "note": note,
                     "phase": ph}
        info(f"{r['file']}: re-gridded to the folder phase [{level}]")
        print(paint(f"        seams={r['seams']}", "90"))
    if not fixed:
        print(paint("    every scan already had it", "90"))
    return {"phase": int(ph), "votes": int(votes), "images": int(total)}


# --------------------------------------------------------------------------
# the step
# --------------------------------------------------------------------------

def run(root, args) -> int:
    t0 = time.time()
    exts = parse_exts(args.exts)

    phase(1, 4, "the pitch and the phase")
    pitch, given = args.pitch, args.phase
    source = "phase from --phase"
    if pitch is None:
        prof_path = stage_json(root, STAGE_CALIBRATE)
        if not os.path.isfile(prof_path):
            error(f"no pitch given and no {os.path.basename(prof_path)} in "
                  f"{os.path.dirname(prof_path)}")
            print(paint(f"\nmeasure the scanner first, or give the pitch "
                        f"yourself:\n"
                        f"    python calibrate_scanner.py{root_arg(root)}\n"
                        f"    python {SCRIPT}{root_arg(root)} --pitch N",
                        "1"))
            return 1
        with open(prof_path, "r", encoding="utf-8") as fh:
            prof = json.load(fh)
        pitch = int(prof["pitch"])
        if given is None and prof.get("phase") is not None:
            given = int(prof["phase"])
            source = "phase from step 1"
        name = os.path.basename(prof_path)
        if source == "phase from step 1":
            info(f"pitch {pitch} and phase {given}, from {name}")
        elif given is not None:
            info(f"pitch {pitch} from {name}; phase {given} from --phase")
        else:
            info(f"pitch {pitch} from {name}; no phase there, so each scan "
                 f"locks its own")
    else:
        info(f"pitch {pitch} from --pitch; "
             + (f"phase {given} from --phase" if given is not None
                else "each scan locks its own phase"))
    if pitch < 2:
        error(f"the pitch must be 2 px or more, not {pitch}")
        return 1

    phase(2, 4, "the seams in each scan")
    files = list_scans(root, exts)
    prev_dir = os.path.join(root, PREVIEW_DIR)
    if args.preview:
        os.makedirs(prev_dir, exist_ok=True)
    info(count(len(files), "scan"))
    tasks = [(os.path.join(root, f), pitch, given, source, args.bands,
              args.lock, _preview_path(prev_dir, f) if args.preview else None)
             for f in files]
    workers = 1
    if parallel is not None and args.workers != 1:
        per = parallel.estimate_bytes_per_worker(tasks[0][0],
                                                 multiple=WORKER_MEM_MULTIPLE)
        workers = parallel.choose_workers(len(tasks), per, args.workers)
    info(count(workers, "worker"))

    # collect first, print in INPUT order, so two runs of the same folder
    # can be compared line by line
    bar = Progress(len(tasks), "    locating")
    if workers <= 1 or len(tasks) <= 1:
        results = []
        for t in tasks:
            results.append(_job(t))
            bar.update()
    else:
        # a worker that dies (out of memory on a huge scan) must cost its
        # own scans, not the seams already found for every other one
        results = parallel.run(
            _job, tasks, workers, on_result=bar.update,
            on_error=lambda t, exc: {
                "file": os.path.basename(t[0]),
                "error": parallel.describe_error(exc)})
    bar.close()

    records, failed = [], []
    for r in results:
        if r.get("error"):
            error(f"    {r['file']}: FAILED {r['error']}")
            failed.append(r["file"])
            continue
        records.append(r)
        c = r["conf"]
        info(f"{r['file']}: {r['width']}x{r['height']} {r['dtype']}, "
             f"{count(len(r['seams']), 'seam')} [{c['level']}]")
        print(paint(f"        seams={r['seams']}", "90"))
        print(paint(f"        phase={r['phase']} clear={c['clear']}/"
                    f"{len(r['seams'])}", "90"))
    if not records:
        error("no scan could be read; nothing written")
        return 1

    phase(3, 4, "the folder phase")
    cons = None
    if given is not None:
        info(f"not needed: {source}, the same for every scan")
    elif args.no_consensus:
        info("off (--no-consensus): every scan keeps the phase it locked")
    elif len(records) < CONSENSUS_MIN_SCANS:
        info(f"not sought: it needs {CONSENSUS_MIN_SCANS} scans or more")
    else:
        cons = _apply_consensus(records, pitch, args.lock, args.bands, root,
                                prev_dir, args.preview)

    phase(4, 4, "the project file")
    items = {}
    for r in records:
        c = r["conf"]
        items[r["file"]] = {
            "width": r["width"], "height": r["height"], "dtype": r["dtype"],
            "seams": [int(x) for x in r["seams"]],
            "confidence": c["level"], "clear_seams": c["clear"],
            "phase": r["phase"],
            "phase_corrected": bool(r["corrected"]), "note": c["note"]}
    doc = {
        "version": SCHEMA_VERSION,
        "source_dir": str(root),
        "tool": SCRIPT,
        "note": "seams are column numbers of the STORED raster: a seam at c "
                "lies between columns c and c+1, and step 3 inserts one "
                "column there. Every seam of a scan sits at phase + k * "
                "pitch. confidence says how many seams are clearly VISIBLE "
                "in that scan (clear_seams): HIGH for two or more, MEDIUM "
                "for one, LOW for none. It says nothing about whether their "
                "positions are right: with a phase from step 1, from "
                "--phase or from the folder consensus they are right "
                "whatever the label. "
                "phase_corrected marks a scan re-gridded to the folder "
                "phase.",
        "pitch": pitch, "phase": given, "consensus": cons,
        "items": items,
    }
    dst = stage_json(root, STAGE_DETECT)
    write_json(dst, doc)

    # with a given or agreed phase a LOW label only says the seams are faint
    low = ([n for n, it in items.items() if it["confidence"] == "LOW"]
           if given is None and cons is None else [])
    if low:
        warn_list(f"    ! {count(len(low), 'scan')} locked "
                  f"{'its' if len(low) == 1 else 'their'} phase alone on "
                  f"little seam evidence; check them with --preview, or give "
                  f"--phase:", low)
    if failed:
        warn_list(f"    ! {count(len(failed), 'scan')} could not be read "
                  f"and {'is' if len(failed) == 1 else 'are'} left out:",
                  failed)
    n = sum(len(it["seams"]) for it in items.values())
    print(paint(f"\nwrote {dst}  ({count(len(items), 'scan')}, "
                f"{count(n, 'seam')})", "32"))
    if args.preview:
        print(paint(f"    previews in {prev_dir}", "32"))
    print(paint(f"    done in {elapsed(t0)}", "32"))
    print(paint(f"\nnext:  python repair_seams.py{root_arg(root)}", "1"))
    return 2 if failed else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Stage 1 step 2 of 3 — locate the seams in each scan")
    ap.add_argument("root", nargs="?", default=".",
                    help="project root: the folder holding the scans "
                         "(default: this folder)")
    ap.add_argument("--pitch", type=int, default=None,
                    help="seam pitch in px (default: from step 1)")
    ap.add_argument("--phase", type=int, default=None,
                    help="use this seam column as the phase (default: from "
                         "step 1, else locked in each scan)")
    ap.add_argument("--preview", action="store_true",
                    help=f"also write a marked PNG per scan into "
                         f"{PREVIEW_DIR}/, for visual checking")
    ap.add_argument("--no-consensus", action="store_true",
                    help="let every scan keep the phase it locked on its own")
    ap.add_argument("--exts", default=",".join(SCAN_EXTS),
                    help="which files count as scans")
    if parallel is not None:
        parallel.add_workers_arg(ap)
    else:
        ap.add_argument("--workers", type=int, default=1,
                        help="ignored: parallel.py not found")
    tune = ap.add_argument_group("detection tuning")
    tune.add_argument("--bands", type=int, default=MIN_BANDS,
                      help=f"fewest voting bands per scan "
                           f"(default: {MIN_BANDS})")
    tune.add_argument("--lock", type=int, default=DEFAULT_LOCK,
                      help="how far from a grid line a seam still counts, "
                           "when the phase is scored and when seams are "
                           f"counted as clearly visible, px "
                           f"(default: {DEFAULT_LOCK})")
    args = ap.parse_args(argv)
    console_setup()

    # exit code 1, not argparse's 2: 2 means "some scans left out"
    if args.lock < 0:
        raise SystemExit(f"--lock must be 0 or more, not {args.lock}")
    if args.bands < 1:
        raise SystemExit(f"--bands must be 1 or more, not {args.bands}")

    root = pipeline_root(args.root)
    check_scans_present(root, parse_exts(args.exts))
    if Image is None and args.preview:
        raise SystemExit("--preview needs pillow:\n    pip install pillow")
    heading("STAGE 1 step 2  -  locate the seams in each scan")
    return run(root, args)


if __name__ == "__main__":
    raise SystemExit(main())
