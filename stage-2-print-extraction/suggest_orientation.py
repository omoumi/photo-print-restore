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
# suggest_orientation.py  --  STAGE 2, step 2 of 4: suggest the way up
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
#     pip install opencv-python numpy tifffile onnxruntime [imagecodecs]
#                 [psutil]
#     common.py beside this file; parallel.py too, for --workers
#
# NOTHING HERE IS A DECISION. This step writes one file of its own,
#
#     json files/orientation_suggestions.json
#
# and no other: the three project files are left as they are. review.py reads
# the suggestions and SHOWS a confident one: a scan you have not confirmed yet
# opens already turned the suggested way, and Enter confirms it like any other
# turn. Until you press Enter nothing is stored, so step 4 never applies a
# suggestion you have not seen. Run review.py with --no-suggestions (which is
# what run_pipeline.bat does when you answer n) or delete the file, and no
# suggestion is shown.
#
# HOW. The print (the largest box of the latest project file) is looked at
# in all four quarter turns with SCRFD-10G, the face detector of InsightFace
# (run through onnxruntime), which also finds the eyes and the mouth
# corners. A face counts only in the turn where the eyes are level and sit
# above the mouth. The turn with the most such faces wins; it is "high"
# confidence when it clearly beats every other turn. The model (about
# 17 MB) is downloaded once into models/ beside this script and checked
# against its SHA-256.
#
#     LICENCE: InsightFace releases its trained models for non-commercial
#     research use only; not for a commercial product.
#
# A print without a face gets NO suggestion: its way up is decided by hand in
# review.py, exactly as without this step.
#
# THE ARCHIVE FLIP. review.py's flip mirrors a scan top to bottom, which no
# quarter turn can undo. A mirrored print is upright after the turn
# (180 - rot) mod 360, where rot is the turn for the unmirrored scan; both are
# stored, so V and --flip keep working with a suggestion in force.
#
# Usage:
#     python suggest_orientation.py [ROOT] [options]
#
#     ROOT              project root: the folder holding the scans
#                       (default: this folder)
#     --workers N       parallel workers (default: auto; 1 disables the pool)
# ==========================================================================
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import urllib.request
import zipfile

try:
    import cv2
    import numpy as np
except ModuleNotFoundError as exc:
    sys.exit(f"missing package: {exc.name} (see Requirements above)")

try:
    from common import (SCHEMA_VERSION, SUGGEST_NAME, Progress,
                        check_scans_present, console_setup, count, elapsed,
                        error, heading, info, json_folder, load_project,
                        orient_image, paint, phase, pipeline_root, read_page,
                        root_arg, to_display_rgb, warn_list, write_json)
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
SCRIPT = "suggest_orientation.py"
WORKER_MEM_MULTIPLE = 4.0       # peak RAM of one worker per byte of the scan
                                # file: the decoded scan, the model and its
                                # buffers ...
WORKER_MEM_FLOOR = 1024 ** 3    # ... and never less than this, bytes

# --- the face model --------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")   # downloaded once, beside the
                                           # scripts
SCRFD_FILE = "det_10g.onnx"      # SCRFD-10G, InsightFace's buffalo_l detector
SCRFD_SHA256 = ("5838f7fe053675b1c7a08b633df49e7a"
                "f5495cee0493c7dcf6697200b85b5b91")
# tried in order; every download is checked against SCRFD_SHA256, so a wrong
# or changed mirror is simply rejected
SCRFD_URL = ("https://huggingface.co/public-data/insightface/resolve/main/"
             "models/buffalo_l/" + SCRFD_FILE)
SCRFD_ZIP = ("https://github.com/deepinsight/insightface/releases/download/"
             "v0.7/buffalo_l.zip")         # official, ~275 MB: last resort
DOWNLOAD_TIMEOUT = 60           # seconds a stalled download may wait per read

# --- finding upright faces -------------------------------------------------
FACE_SIDE = 1024                # the print is searched at this long side
FACE_MIN = 14                   # px at FACE_SIDE: smaller "faces" are noise
FACE_SCORE = 0.60               # SCRFD confidence a face must reach
NMS_IOU = 0.4                   # overlap above which two boxes are one face
HIGH_SCORE = 0.85               # summed face score the winning turn needs ...
HIGH_RATIO = 0.35               # ... while no other turn reaches this share
FACE_LEVEL_TOL = 0.6            # eyes level: height difference under this
                                # share of their distance
FACE_MOUTH_DROP = 0.2           # mouth below the eyes by this share of the
                                # face height
TURNS = (0, 90, 180, 270)       # the quarter turns tried, clockwise


# --------------------------------------------------------------------------
# faces
# --------------------------------------------------------------------------

def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url, dst, label):
    """Stream url into dst, with a timeout per read so a stalled connection
    cannot hang the step."""
    with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT) as r, \
            open(dst, "wb") as fh:
        total = int(r.headers.get("Content-Length") or 0)
        got, shown = 0, -1
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            got += len(chunk)
            if total and sys.stdout.isatty():
                pct = int(100 * got / total)
                if pct != shown:
                    sys.stdout.write(f"\r    downloading {label}: {pct:3d}%")
                    sys.stdout.flush()
                    shown = pct
    if total and sys.stdout.isatty():
        print()


def scrfd_model():
    """(path to the SCRFD-10G model, None) -- downloading it once -- or
    (None, why it is unavailable)."""
    import importlib.util
    if importlib.util.find_spec("onnxruntime") is None:
        return None, "onnxruntime is not installed:  pip install onnxruntime"
    path = os.path.join(MODEL_DIR, SCRFD_FILE)
    if os.path.isfile(path):
        if _sha256(path) == SCRFD_SHA256:
            return path, None
        os.remove(path)                   # a broken earlier download
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp, errors = path + ".part", []
    try:
        # 1) the model file on its own, about 17 MB
        try:
            print("    fetching the face model once (about 17 MB)")
            _download(SCRFD_URL, tmp, SCRFD_FILE)
            if _sha256(tmp) == SCRFD_SHA256:
                os.replace(tmp, path)
                return path, None
            errors.append("mirror: not the expected file")
        except Exception as exc:
            errors.append(f"mirror: {type(exc).__name__}: {exc}")
        # 2) InsightFace's own release, which holds it among other models
        zpath = path + ".zip.part"
        try:
            print("    that failed; fetching InsightFace's official package "
                  "instead (about 275 MB, once)")
            _download(SCRFD_ZIP, zpath, "buffalo_l.zip")
            with zipfile.ZipFile(zpath) as z:
                member = next(n for n in z.namelist()
                              if n.rsplit("/", 1)[-1] == SCRFD_FILE)
                with z.open(member) as src, open(tmp, "wb") as dst:
                    dst.write(src.read())
            if _sha256(tmp) == SCRFD_SHA256:
                os.replace(tmp, path)
                return path, None
            errors.append("official package: not the expected file")
        except Exception as exc:
            errors.append(f"official package: {type(exc).__name__}: {exc}")
        finally:
            if os.path.exists(zpath):
                os.remove(zpath)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
    return None, ("the face model could not be downloaded ("
                  + "; ".join(errors) + ").\n      Download "
                  f"{SCRFD_ZIP}\n      and put {SCRFD_FILE} from it into "
                  f"{os.path.dirname(path)}")


_SESSION = {}


def _session(model, threads):
    sess = _SESSION.get(model)
    if sess is None:
        import onnxruntime as ort
        opt = ort.SessionOptions()
        # the model is declared for 640x640 and warns on every other size;
        # any size is fine for SCRFD, so keep the console readable
        opt.log_severity_level = 3
        if threads:
            opt.intra_op_num_threads = threads
            opt.inter_op_num_threads = 1
        sess = ort.InferenceSession(model, sess_options=opt,
                                    providers=["CPUExecutionProvider"])
        _SESSION[model] = sess
    return sess


def _nms(boxes, scores):
    """Indices kept by plain non-maximum suppression."""
    x1, y1, x2, y2 = boxes.T
    area = (x2 - x1) * (y2 - y1)
    order, keep = scores.argsort()[::-1], []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (area[i] + area[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= NMS_IOU]
    return keep


def detect_faces(bgr, model, threads=0):
    """SCRFD-10G on one image: (boxes Nx4, scores N, landmarks Nx5x2).

    Landmarks, in image terms: eye on the left, eye on the right, nose,
    mouth corner on the left, mouth corner on the right. The image goes in at
    its own size (the model takes any size), padded to a multiple of 32, so
    small faces are not shrunk away."""
    sess = _session(model, threads)
    h, w = bgr.shape[:2]
    H, W = -(-h // 32) * 32, -(-w // 32) * 32
    canvas = np.zeros((H, W, 3), np.uint8)
    canvas[:h, :w] = bgr
    blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128, (W, H),
                                 (127.5, 127.5, 127.5), swapRB=True)
    outs = sess.run(None, {sess.get_inputs()[0].name: blob})
    outs = [o[0] if o.ndim == 3 else o for o in outs]      # drop a batch axis
    all_b, all_s, all_k = [], [], []
    for k, stride in enumerate((8, 16, 32)):
        sc, bb, kp = outs[k][:, 0], outs[k + 3] * stride, outs[k + 6] * stride
        gh, gw = H // stride, W // stride
        ys, xs = np.mgrid[:gh, :gw]
        centers = np.stack([xs, ys], -1).reshape(-1, 2).astype(np.float32)
        centers = np.repeat(centers * stride, 2, axis=0)   # 2 anchors / cell
        sel = sc >= FACE_SCORE
        if not sel.any():
            continue
        c = centers[sel]
        d = bb[sel]
        all_b.append(np.stack([c[:, 0] - d[:, 0], c[:, 1] - d[:, 1],
                               c[:, 0] + d[:, 2], c[:, 1] + d[:, 3]], -1))
        all_s.append(sc[sel])
        all_k.append((kp[sel].reshape(-1, 5, 2) + c[:, None, :]))
    if not all_s:
        return (np.zeros((0, 4)), np.zeros(0), np.zeros((0, 5, 2)))
    b, s_, k_ = np.concatenate(all_b), np.concatenate(all_s), \
        np.concatenate(all_k)
    keep = _nms(b, s_)
    return b[keep], s_[keep], k_[keep]


def _faces_upright(bgr, model, threads=0):
    """Summed score and count of UPRIGHT faces: eyes level, mouth below."""
    boxes, scores, kps = detect_faces(bgr, model, threads)
    score, n = 0.0, 0
    for (x1, y1, x2, y2), sc, k in zip(boxes, scores, kps):
        fw, fh = x2 - x1, y2 - y1
        if min(fw, fh) < FACE_MIN:
            continue
        (lex, ley), (rex, rey) = k[0], k[1]
        mouth_y = (k[3][1] + k[4][1]) / 2.0
        eye_y = (ley + rey) / 2.0
        level = abs(rey - ley) < FACE_LEVEL_TOL * abs(rex - lex) + 1e-6
        # a face turned upside down keeps its eyes level; the mouth test
        # catches it
        if level and mouth_y - eye_y > FACE_MOUTH_DROP * fh:
            score += float(sc)
            n += 1
    return score, n


def _shrink(img, side):
    h, w = img.shape[:2]
    s = side / max(h, w)
    if s >= 1.0:
        return img
    size = (max(1, int(round(w * s))), max(1, int(round(h * s))))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def _job(task):
    """One scan: read, cut out the print, look for faces in all four turns.
    Returns (scores, counts, error)."""
    path, box, model, threads = task
    cv2.setNumThreads(1)
    try:
        arr, _meta = read_page(path)
        x0, y0, x1, y1 = box
        # cut the print out first, so only it is converted to 8 bits
        rgb = to_display_rgb(
            arr[max(0, y0):max(y0 + 1, y1), max(0, x0):max(x0 + 1, x1)])
        if rgb.ndim == 2:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
        else:
            bgr = cv2.cvtColor(np.ascontiguousarray(rgb[..., :3]),
                               cv2.COLOR_RGB2BGR)
        small = _shrink(bgr, FACE_SIDE)
        scores, counts = [], []
        for r in TURNS:
            view = orient_image(small, False, r)
            s, n = _faces_upright(view, model, threads)
            scores.append(s)
            counts.append(n)
        return scores, counts, None
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def face_verdict(scores, counts):
    """(rot, confidence, why) from the per-turn face scores, or None."""
    order = sorted(range(4), key=lambda k: scores[k], reverse=True)
    best, second = order[0], order[1]
    if counts[best] == 0:
        return None
    n = counts[best]
    why = f"{n} face{'s' if n != 1 else ''}"
    clear = scores[second] <= HIGH_RATIO * scores[best]
    if clear and scores[best] >= HIGH_SCORE:
        conf = "high"
    elif clear or scores[best] > scores[second]:
        conf = "medium"
    else:
        return None                      # two turns tie: no suggestion
    return TURNS[best], conf, why


# --------------------------------------------------------------------------
# the step
# --------------------------------------------------------------------------

def run(root, args) -> int:
    t0 = time.time()

    phase(1, 4, "the prints")
    proj = load_project(root)
    if proj is None:
        error(f"no project file in {json_folder(root)}")
        print(paint(f"\nrun step 1 first:\n    python detect_borders.py"
                    f"{root_arg(root)}", "1"))
        return 1
    if proj.added:
        warn_list(f"    ! {count(len(proj.added), 'scan')} measured by "
                  f"step 1 after the review began "
                  f"{'is' if len(proj.added) == 1 else 'are'} added, not yet "
                  f"confirmed:", proj.added)
    tasks, names = [], []
    for name in proj.names():
        it = proj.items[name]
        if not it.boxes:
            continue
        b = max(it.boxes, key=lambda b: b.width * b.height)
        path = os.path.join(root, name)
        if os.path.isfile(path):
            tasks.append((path, (b.x0, b.y0, b.x1, b.y1)))
            names.append(name)
    if not tasks:
        error("no scan with a box to look at")
        return 1
    info(f"{count(len(tasks), 'print')} to look at (the largest box of "
         f"each scan)")

    phase(2, 4, "the face model")
    model, why = scrfd_model()
    if not model:
        error("no suggestions: " + why)
        return 1
    info(f"SCRFD-10G ({os.path.relpath(model, HERE)})")

    phase(3, 4, "the faces in each print")
    workers = 1
    if parallel is not None and args.workers != 1:
        per = parallel.estimate_bytes_per_worker(
            tasks[0][0], multiple=WORKER_MEM_MULTIPLE, floor=WORKER_MEM_FLOOR)
        workers = parallel.choose_workers(len(tasks), per, args.workers)
    info(count(workers, "worker"))
    # one thread per worker when there are several, so they do not fight
    threads = 1 if workers > 1 and len(tasks) > 1 else 0
    tasks = [(p, b, model, threads) for p, b in tasks]
    bar = Progress(len(tasks), "    looking")
    results = [None] * len(tasks)

    def report(i, r):
        results[i] = r
        if r[2]:
            bar.message(paint(f"    ! {names[i]}: {r[2]}", "1;31"))
        bar.update()

    if workers > 1 and len(tasks) > 1:
        # a worker that dies (out of memory on a huge scan) costs its own
        # scans only; every suggestion already made is still written
        parallel.run(_job, tasks, workers, on_result=report,
                     on_error=lambda t, exc: (
                         None, None, parallel.describe_error(exc)))
    else:
        for i, t in enumerate(tasks):
            report(i, _job(t))
    bar.close()

    phase(4, 4, "the suggestions")
    items, no_face, failed = {}, 0, []
    for name, (scores, counts, err) in zip(names, results):
        if err:
            failed.append(name)
            continue
        if max(counts) == 0:
            no_face += 1                 # decided by hand in review.py
            continue
        v = face_verdict(scores, counts)
        if v:
            items[name] = v

    if len(failed) == len(names):
        warn_list("    ! no scan could be processed; nothing written:",
                  failed)
        return 1

    doc = {
        "version": SCHEMA_VERSION,
        "tool": SCRIPT,
        "note": "Suggestions only; review.py shows them and Enter confirms. "
                "rot = quarter turns clockwise, as in the project files, for "
                "the scan as it sits on disk; rot_flipped = the same for a "
                "scan the archive flip XOR V mirrors top to bottom.",
        "faces": "scrfd_10g",
        "items": {n: {"rot": r, "rot_flipped": (180 - r) % 360,
                      "confidence": c, "why": w}
                  for n, (r, c, w) in sorted(items.items())},
    }
    dst = os.path.join(json_folder(root), SUGGEST_NAME)
    write_json(dst, doc)

    high = sum(1 for v in items.values() if v[1] == "high")
    turned = sum(1 for v in items.values() if v[1] == "high" and v[0])
    undecided = len(names) - len(items) - no_face - len(failed)
    info(f"high confidence  {high:>5}   ({turned} of them will open turned "
         f"in review.py)")
    info(f"medium           {len(items) - high:>5}   (shown, not turned)")
    info(f"no suggestion    {len(names) - len(items):>5}   "
         f"({no_face} without a face, {undecided} undecided between turns"
         + (f", {len(failed)} unreadable" if failed else "") + ")")
    info("Scans without a suggestion keep their way up; set it by hand in "
         "review.py.")
    if failed:
        warn_list(f"    ! {count(len(failed), 'scan')} could not be "
                  f"processed and got no suggestion:", failed)
    print(paint(f"\nwrote {dst}", "32"))
    print(paint(f"    done in {elapsed(t0)}", "32"))
    print(paint(f"\nnext:  python review.py{root_arg(root)}", "1"))
    return 2 if failed else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Stage 2 step 2 of 4 — suggest the way up")
    ap.add_argument("root", nargs="?", default=".",
                    help="project root: the folder holding the scans "
                         "(default: this folder)")
    if parallel is not None:
        parallel.add_workers_arg(ap)
    else:
        ap.add_argument("--workers", type=int, default=1,
                        help="ignored: parallel.py not found")
    args = ap.parse_args(argv)
    console_setup()

    root = pipeline_root(args.root)
    check_scans_present(root)
    heading("STAGE 2 step 2  -  suggest the way up")
    return run(root, args)


if __name__ == "__main__":
    raise SystemExit(main())
