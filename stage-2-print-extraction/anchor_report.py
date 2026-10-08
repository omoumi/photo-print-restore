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
# anchor_report.py  --  STAGE 2: per-side diagnostic for step 1
#                       (not a step)
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
#     the same as detect_borders.py
#     detect_borders.py and common.py beside this file
#
# Explains why step 1 put each side of each print where it did. The scan is
# prepared by detect_borders.prepare_scan, the function step 1 itself uses,
# so this report and the detector always look at the identical data. For
# every side of every detected print it reports, for the FIRST tracing pass
# (no margin hint yet):
#
#     anchors   how many scan lines produced a print-edge anchor
#     a-spread  how far those anchors scatter (px @ BASE). A healthy side is
#               a couple of px; tens of px means some lines anchored on
#               picture content instead of the print edge.
#     pts       how many scan lines produced an image-edge sample. Far below
#               the line count means most of the side was never measured.
#     p-spread  scatter of those samples around their own robust line
#     fit       whether a side line could be fitted, and if not, why
#     inset     the margin between image edge and outline, and whether it
#               was accepted
#
# and then the final quad with the pass that placed each side.
#
# With --draw it also writes, per scan, a picture of the FINAL state of
# every side, drawn on top of the red frame step 1 would place:
#
#     circle  = a sample point the final side line was fitted on
#     cross   = a print-edge anchor point
#     no circles on a side = that side fell back to a prior (anchor + margin
#     or the rough box): nothing was measured on it
#     back mode: the frame is the OUTWARD ENVELOPE of the crosses (tangent
#     to the card edge); there are no circles on a back
#
# The red frame must pass through the circles. The pictures go to
# detect_review/<name>_points.jpg beside the scan, the folder of step 1's
# own previews. None of this applies to step 1 --ring, which works another
# way.
#
# Usage:
#     python anchor_report.py SCAN [SCAN ...] [options]
#
#     --back            the scans show the backs of prints (as step 1 --back)
#     --draw            also write the picture described above
#
#     python anchor_report.py raw0016.tif
#     python anchor_report.py raw0016.tif raw0020.tif --draw
# ==========================================================================
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError as exc:
    sys.exit(f"missing package: {exc.name} (see Requirements above)")

try:
    import detect_borders as D
    from common import REVIEW_DIR, console_setup, error, heading, paint
except ModuleNotFoundError as exc:
    sys.exit(f"{exc.name}.py must sit beside this script"
             if exc.name in ("detect_borders", "common")
             else f"missing package: {exc.name} (see Requirements above)")

# --------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------
# --- the report ------------------------------------------------------------
SIDES = ("top", "bot", "lef", "rig")
THIN_SIDE_FRAC = 0.6    # flag a side measured on fewer lines than this share
SCATTER_WARN = 8.0      # flag anchors scattering more than this (px @ BASE)


# --------------------------------------------------------------------------
# measurements
# --------------------------------------------------------------------------

def _split(pts, kind):
    """(position along the side, offset across it)."""
    horiz = kind in ("top", "bot")
    t = np.array([p[0] if horiz else p[1] for p in pts], float)
    off = np.array([p[1] if horiz else p[0] for p in pts], float)
    return t, off


def _n_lines(bb, kind):
    """Scan lines trace_side offers on this side (a short side yields fewer
    than SAMPLES_PER_SIDE)."""
    bx, by, bw, bh = bb
    if kind in ("top", "bot"):
        s0 = bx + int(D.EDGE_SKIP_FRAC * bw)
        s1 = bx + bw - int(D.EDGE_SKIP_FRAC * bw)
    else:
        s0 = by + int(D.EDGE_SKIP_FRAC * bh)
        s1 = by + bh - int(D.EDGE_SKIP_FRAC * bh)
    step = max(1, (s1 - s0) // D.SAMPLES_PER_SIDE)
    return len(range(s0, s1 + 1, step))


def fit_diagnosis(pts, kind, tol):
    """(verdict, (slope, offset) or None): the robust fit of
    detect_borders._fit_line_robust, step by step, saying what stopped it."""
    if len(pts) < D.MIN_POINTS:
        return f"NO ({len(pts)} pts < MIN_POINTS={D.MIN_POINTS})", None
    t, off = _split(pts, kind)
    n = len(t)
    if np.unique(t).size < D.MIN_POINTS:
        return "NO (samples not spread along the side)", None
    i, j = np.triu_indices(n, 1)
    dt = t[j] - t[i]
    ok = np.abs(dt) > 1e-6
    slope = float(np.median((off[j] - off[i])[ok] / dt[ok]))
    b = float(np.median(off - slope * t))
    need = max(D.MIN_POINTS, int(round(D.MIN_INLIER_FRAC * n)))
    best = 0
    for _ in range(3):
        inl = np.abs(off - (slope * t + b)) < tol
        best = max(best, int(inl.sum()))
        if inl.sum() < need:
            return (f"NO (only {best}/{n} samples within {tol:.0f}px of a "
                    f"straight line; {need} needed)"), None
        A = np.stack([t[inl], np.ones(int(inl.sum()))], axis=1)
        (slope, b), *_ = np.linalg.lstsq(A, off[inl], rcond=None)
    inl = np.abs(off - (slope * t + b)) < tol
    if inl.sum() < need:
        return (f"NO (final rescore: only {int(inl.sum())}/{n} inliers; "
                f"{need} needed)"), None
    tilt = abs(np.degrees(np.arctan(slope)))
    if tilt > D.MAX_TILT_DEG:
        return (f"NO (tilt {tilt:.2f} deg > MAX_TILT_DEG={D.MAX_TILT_DEG})",
                None)
    return (f"yes ({int(inl.sum())}/{n} inliers, tilt {tilt:.2f} deg)",
            (slope, b))


def spread(pts, kind, s, fit=None):
    """p90 |residual| around the fitted line (or the median), px @ BASE."""
    if not pts:
        return float("nan")
    t, off = _split(pts, kind)
    ref = (fit[0] * t + fit[1]) if fit else np.median(off)
    return float(np.percentile(np.abs(off - ref), 90)) / s


def _inset_verdict(line, aline, kind, s, lim, n_anchors):
    """One line on the margin between image edge and outline, with the same
    limits detect_quad's side_inset applies."""
    if aline is None:
        return (f"inset=NONE (no anchor line could be fitted from "
                f"{n_anchors} anchors)")
    if line is None:
        return "inset=NONE (no image-edge line)"
    tt = line[0] if kind in ("top", "bot") else line[1]
    off_l = line[1] if kind in ("top", "bot") else line[0]
    off_a = D.line_off_at(aline, kind, tt)
    d = (off_l - off_a) if kind in ("top", "lef") else (off_a - off_l)
    why = ("OK" if 2 * s < d < lim else
           "REJECTED: anchor is INSIDE the image edge" if d < -0.05 else
           "REJECTED: no margin, the anchor is on the image edge"
           if d <= 0.05 else
           f"REJECTED: below the {2 * s:.0f}px floor" if d <= 2 * s else
           f"REJECTED: above the {lim:.0f}px limit")
    return f"raw inset={d:+7.1f}px ({d / s:+.1f}px@BASE)  -> {why}"


def _cluster_note(pts, kind, slope, tol, gdir, s):
    """Where the densest consistent cluster of an unfitted side lies, and
    whether it sits on a real edge. Report only."""
    cand, cl = D.cluster_at_angle(pts, kind, slope, tol)
    if not cl:
        return None
    a = np.asarray(cl)
    off = a[:, 1] if kind in ("top", "bot") else a[:, 0]
    w = max(1, int(round(1.5 * s)))
    vals = []
    for sx, sy in cl:
        x, y = int(round(sx)), int(round(sy))
        if kind in ("top", "bot"):
            seg = gdir[max(0, y - w):y + w + 1, x:x + 1]
        else:
            seg = gdir[y:y + 1, max(0, x - w):x + w + 1]
        if seg.size:
            vals.append(float(seg.max()))
    gmed = float(np.median(vals)) if vals else float("nan")
    usable = cand is not None and gmed >= D.BAND_EDGE_MIN
    return (f"cluster: {len(cl)}/{len(pts)} samples, offset "
            f"{off.min():.0f}..{off.max():.0f}, median gradient {gmed:.0f} "
            f"(needs >= {D.BAND_EDGE_MIN} to be believed) -> "
            f"{'USABLE' if usable else 'rejected'}")


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------

def _draw(path, img, sc, found):
    """Write the --draw picture: the red frame of every box, then the
    points each side was fitted on, on top. Returns the path, or None."""
    H, W = img.shape[:2]
    cl, ct = sc["off"]
    quads, items = [], []
    for q, dbg in found:
        q = q + np.array([cl, ct], np.int32)
        q[:, 0] = np.clip(q[:, 0], 0, W - 1)
        q[:, 1] = np.clip(q[:, 1], 0, H - 1)
        quads.append(q)
        items.extend(dbg or [])
    out = D.draw_quads(img, quads)
    D._draw_dbg(out, items, D.scale_of(sc["gray"]), (cl, ct))
    folder = Path(path).resolve().parent / REVIEW_DIR
    os.makedirs(folder, exist_ok=True)
    dst = folder / f"{Path(path).stem}_points.jpg"
    return dst if D._imwrite(dst, out) else None


def report(path, back=False, draw=False):
    """Print the report for one scan. False when it could not be read."""
    img = D.read_image(path)
    if img is None:
        return False
    sc = D.prepare_scan(img, back)
    if sc is None:
        error(f"    ! {Path(path).name}: too small to process")
        return False
    gray, paper = sc["gray"], sc["paper"]
    H, W = img.shape[:2]
    s = D.scale_of(gray)
    tol = D.FIT_TOL * s
    cl, ct = sc["off"]
    print(f"\n=== {Path(path).name}  {W}x{H}  scale={s:.2f}  paper={paper:.0f}"
          f"  inlier tol={tol:.0f}px"
          + ("  [back mode]" if sc["back"] else ""))

    gdir = {"top": sc["gy"], "bot": sc["gy"], "lef": sc["gx"],
            "rig": sc["gx"]}
    galt = {"top": sc["gy_alt"], "bot": sc["gy_alt"], "lef": sc["gx_alt"],
            "rig": sc["gx_alt"]}
    found = []                           # (quad, debug items) per box
    for bi, bb in enumerate(sc["boxes"]):
        res = D.measure_box(sc, bb, want_dbg=draw)
        if res is None:
            continue                     # no real edge in this box
        q, dbg, prov = res
        found.append((q, dbg))
        bx, by, bw, bh = bb
        print(f"\n  box {bi}: x={bx} y={by} w={bw} h={bh}   "
              f"(add x+{cl}, y+{ct} for scan coordinates)")
        first = {k: D.trace_side(gray, gdir[k], paper, bb, k, None,
                                 gdir_alt=galt[k]) for k in SIDES}
        th0 = D._consensus_tilt(*(first[k][0] for k in SIDES))
        lim = D.SIDE_INSET_LIM * max(bw, bh)
        for k in SIDES:
            line, aline, pts, anchors = first[k]
            n_s = _n_lines(bb, k)
            a_sp = spread(anchors, k, s) if anchors else float("nan")
            verdict, fit = fit_diagnosis(pts, k, tol)
            p_sp = spread(pts, k, s, fit)
            slope = (float(np.tan(th0)) if k in ("top", "bot")
                     else float(-np.tan(th0)))
            print(f"    {k}: anchors={len(anchors):3d}/{n_s}  "
                  f"a-spread={a_sp:5.1f}px  "
                  f"pts={len(pts):3d}/{n_s}  p-spread={p_sp:5.1f}px")
            print(f"         fit={verdict}")
            print("         " + _inset_verdict(line, aline, k, s, lim,
                                                len(anchors)))
            if line is None and pts:
                note = _cluster_note(pts, k, slope, tol, gdir[k], s)
                if note:
                    print("         " + note)
            if line is None:
                r = D._refit_at_angle(pts, k, slope, tol)
                if r is None:
                    print("         refit at consensus angle: FAILED")
                else:
                    print("         refit at consensus angle: recovered by "
                          "detect_quad")
            if len(pts) < THIN_SIDE_FRAC * n_s:
                print("         ^ most of this side was never measured")
            if a_sp > SCATTER_WARN:
                print("         ^ anchors scattered: lines are latching onto "
                      "content, not the print edge")

        m = prov.get("m_est")
        print(f"\n    final quad: tilt={prov['tilt_deg']:+.2f} deg   "
              f"m_est={'-' if m is None else f'{m:.0f}px'}"
              f" ({'-' if m is None else f'{m / s:.1f}px@BASE'})")
        for k in SIDES:
            ins = prov["inset"][k]
            print(f"      {k}: from {prov['source'][k]:<28s} "
                  f"pts={prov['n_pts'][k]:3d}  "
                  f"inset={'-' if ins is None else f'{ins:.0f}px'}")
        # The fallbacks: anchor + margin on a front, the rough box on any
        # print. On a back the anchors ARE the measurement.
        weak = [k for k in SIDES
                if prov["source"][k].endswith("(anchor + m_est)")
                or prov["source"][k].startswith(("prior", "rough box"))]
        if weak:
            print(f"      ^ {', '.join(weak)} placed by a prior (anchor + "
                  "margin, or the rough box), not by their own samples")
    if draw:
        dst = _draw(path, img, sc, found)
        if dst is None:
            error(f"    ! {Path(path).name}: could not write the picture")
        else:
            print(paint(f"\n    picture: {dst}", "32"))
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Stage 2 — per-side diagnostic for step 1 "
                    "(detect_borders.py)")
    ap.add_argument("images", nargs="+", help="scan file(s) to inspect")
    ap.add_argument("--back", action="store_true",
                    help="the scans show the backs of prints")
    ap.add_argument("--draw", action="store_true",
                    help="also write detect_review/<name>_points.jpg: the "
                         "red frame with the points each side was fitted on")
    args = ap.parse_args(argv)
    console_setup()
    heading("STAGE 2  -  anchor report: why step 1 placed each side")
    ok = [report(p, back=args.back, draw=args.draw) for p in args.images]
    return 0 if all(ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
