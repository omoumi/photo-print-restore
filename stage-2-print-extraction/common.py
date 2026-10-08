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
# common.py  --  STAGE 2: code shared by every step (not a step)
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
#     pip install numpy tifffile [opencv-python] [imagecodecs]
#
# Holds the project files (boxes_<stage>.json), image reading with its
# metadata, the TIFF writer and the console helpers every step prints with
# -- the general part, the same in every stage -- and, for Stage 2 only,
# the boxes and the Project that reads and writes them, the orientation
# arithmetic that converts coordinates between the file on disk and the
# turned or mirrored view shown in review.py, and the one tilt measurement
# that review.py displays and deskew_crop.py straightens by.
#
# THE FOLDER CONTRACT. One project root -- the folder holding the scans, the
# only argument any step needs -- contains these fixed names:
#
#     json files/                   every project file lives here
#         boxes_detect_borders.json     written by step 1
#         orientation_suggestions.json  written by step 2: suggested ways
#                                       up, read by step 3, never a decision
#         boxes_review.json             written by step 3: the decisions
#         boxes_deskew_crop.json        written by step 4: the manifest
#     detect_review/                step-1 previews, only with --review
#     deskewed-cropped/             step-4 output: straightened and cropped
#                                   (deskew_crop.py --out DIR writes the
#                                   crops anywhere else)
#
# The scans themselves sit in the root. Steps 2 to 4 read the file of the
# LATEST step that exists: boxes_review.json, else boxes_detect_borders.json,
# so a fresh archive works before anything has been reviewed and a
# half-reviewed one resumes where it was.
#
# BOX COORDINATES ALWAYS DESCRIBE THE FILE AS IT SITS ON DISK, never the
# view. Flipping or turning the view in step 3 changes what you see, not
# what is stored, so a box placed by hand survives any number of
# orientation changes.
#
# Every orientation decision lives in the project file -- the archive-wide
# flip at the top level, the per-scan flip and quarter turn beside each box.
# No step takes a flip on the command line except review.py, which stores
# it, so what deskew_crop.py writes is by construction what review.py
# showed.
#
# Step 4's --verify pass re-implements the orientation arithmetic and the
# TIFF reader rather than calling into this module, so a mistake here cannot
# hide behind a checker that shares it.
# ==========================================================================
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import tifffile

# --------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------
# --- general: the same in every stage --------------------------------------
SCHEMA_VERSION = 1               # "version" written into every project file
TIFF_EXT = (".tif", ".tiff", ".TIF", ".TIFF")
JSON_DIR = "json files"          # every project file lives in this folder
LIST_MAX = 8                     # file names listed in one warning, at most

# TIFF codecs that would re-quantise the data if written back out:
# 6/7/34712/34892 the JPEG family, 34934/50001 WEBP, 34935/50002 JPEG XL.
# ZSTD (34933) is lossless, so "keep" keeps it.
LOSSY_CODECS = {6, 7, 34712, 34892, 34934, 50001, 34935, 50002}
CODEC = {"keep": None, "none": 1, "lzw": 5, "deflate": 8}
DEFLATE = 8                      # written instead of a lossy source codec

# --- Stage 2: the folder contract ------------------------------------------
SCAN_EXTS = TIFF_EXT + (".jpg", ".jpeg", ".png", ".bmp", ".webp")  # step 1
STAGE_DETECT = "detect_borders"  # boxes_detect_borders.json, step 1
STAGE_REVIEW = "review"          # boxes_review.json, step 3
STAGE_DESKEW_CROP = "deskew_crop"  # boxes_deskew_crop.json, step 4
SUGGEST_NAME = "orientation_suggestions.json"   # step 2
REVIEW_DIR = "detect_review"     # step-1 previews (--review only)
CROPPED_DIR = "deskewed-cropped"  # step-4 output, unless --out

# --- Stage 2: the tilt -----------------------------------------------------
# Below this tilt (degrees) step 4 does not rotate at all: it only slices,
# so the crop keeps every sample exactly. review.py warns from the same
# number, so the warning appears exactly for the prints step 4 resamples.
STRAIGHTEN_MIN_DEG = 0.2
TILT_COARSE_DEG = 0.25           # tilt search: first pass step ...
TILT_FINE_DEG = 0.01             # ... then this step around the best


# --------------------------------------------------------------------------
# the folder contract (general)
# --------------------------------------------------------------------------

def pipeline_root(path: str = ".") -> str:
    """The project root, from whatever the user handed the step.

    Accepts the root itself, the `json files` folder, a project file inside
    it, or a single scan file, so `python <step>.py`,
    `python <step>.py D:\\archive\\scans` and
    `python <step>.py "json files/<file>.json"` all land in the same
    place."""
    p = resolve(path)
    if os.path.isfile(p):
        d = os.path.dirname(p)
        return os.path.dirname(d) if os.path.basename(d) == JSON_DIR else d
    if os.path.isdir(p):
        return os.path.dirname(p) if os.path.basename(p) == JSON_DIR else p
    return os.path.dirname(p)


def json_folder(project_root: str) -> str:
    """The shared folder every project file lives in."""
    return os.path.join(resolve(project_root), JSON_DIR)


def stage_json(project_root: str, stage: str) -> str:
    """The project file of one stage: boxes_<stage>.json, always inside the
    shared `json files` folder of the root."""
    return os.path.join(json_folder(project_root), f"boxes_{stage}.json")


def write_json(path: str, doc: dict) -> None:
    """Write a project file atomically: a crash never leaves it truncated."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def resolve(path: str) -> str:
    """Expand ~ and %VARS%/$VARS ourselves.

    cmd.exe and PowerShell do not expand ~, so a path copied out of a
    Unix-style example arrives here as the literal string '~/scans'."""
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))


# --------------------------------------------------------------------------
# image reading (general)
# --------------------------------------------------------------------------

def read_page(path: str):
    """Return (array, metadata dict) for page 0.

    TIFF goes through tifffile, which hands back the raster exactly as
    stored -- no Orientation tag applied -- with the metadata write_tiff
    carries across. Any other format decodes through OpenCV, 8-bit, with the
    Orientation tag ignored too, so coordinates match the file on disk; meta
    is empty then."""
    if not str(path).lower().endswith(TIFF_EXT):
        try:
            import cv2
        except ImportError:
            raise SystemExit(f"{path}: only TIFF reads natively here; "
                             "install opencv-python to preview other formats")
        try:
            buf = np.fromfile(str(path), dtype=np.uint8)
        except OSError:
            raise SystemExit(f"could not read: {path}")
        a = cv2.imdecode(buf,
                         cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if a is None:
            raise SystemExit(f"could not read: {path}")
        return np.ascontiguousarray(a[:, :, ::-1]), {}     # BGR -> RGB

    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        arr = page.asarray()
        # planar storage arrives as (C, H, W); every consumer here slices and
        # thinks in (H, W, C), so interleave it once, here
        if arr.ndim == 3 and getattr(page, "planarconfig", 1) == 2 and \
                arr.shape[0] <= 4 < arr.shape[1]:
            arr = np.ascontiguousarray(np.transpose(arr, (1, 2, 0)))
        meta = _page_metadata(page)
    return arr, meta


def _tag(page, name, default=None):
    t = page.tags.get(name)
    return t.value if t is not None else default


def _page_metadata(page) -> dict[str, Any]:
    """Everything needed to write the image back out unchanged."""
    xres = _tag(page, "XResolution")
    yres = _tag(page, "YResolution")

    def rat(v):
        if v is None:
            return None
        if isinstance(v, (tuple, list)) and len(v) == 2 and v[1]:
            return v[0] / v[1]
        return float(v)

    meta = {
        "photometric": page.photometric,
        "compression": page.compression,
        "planarconfig": page.planarconfig,
        "predictor": _tag(page, "Predictor", 1),
        "extrasamples": _tag(page, "ExtraSamples"),
        "bitspersample": page.bitspersample,
        "resolution": (rat(xres), rat(yres)) if xres and yres else None,
        "resolutionunit": _tag(page, "ResolutionUnit"),
        "iccprofile": _tag(page, "InterColorProfile"),
        "description": _tag(page, "ImageDescription"),
        "datetime": _tag(page, "DateTime"),
        "software": _tag(page, "Software"),
        "colormap": _tag(page, "ColorMap"),
        # copied verbatim through extratags
        "make": _tag(page, "Make"),
        "model": _tag(page, "Model"),
        "artist": _tag(page, "Artist"),
        "copyright": _tag(page, "Copyright"),
        "hostcomputer": _tag(page, "HostComputer"),
        "pagename": _tag(page, "PageName"),   # VueScan writes it
    }
    return meta


# --------------------------------------------------------------------------
# preview conversion (general; for display ONLY -- never for output)
# --------------------------------------------------------------------------

def to_display_rgb(arr: np.ndarray) -> np.ndarray:
    """uint8 RGB (or 2-D grayscale) for on-screen display."""
    a = arr
    if a.dtype == np.uint16:
        a = (a >> 8).astype(np.uint8)       # top byte: fast, display only
    elif a.dtype == bool:
        a = (a.astype(np.uint8)) * 255
    elif a.dtype != np.uint8:
        a = a.astype(np.float64)
        lo, hi = float(a.min()), float(a.max())
        a = ((a - lo) / (hi - lo) * 255).astype(np.uint8) if hi > lo \
            else np.zeros_like(a, np.uint8)
    if a.ndim == 3 and a.shape[2] > 3:
        a = a[..., :3]
    return np.ascontiguousarray(a)


# --------------------------------------------------------------------------
# TIFF writing (general)
# --------------------------------------------------------------------------

def _ascii_extratags(meta: dict) -> list:
    """ASCII tags tifffile has no dedicated keyword for."""
    tags = []
    for code, key in ((271, "make"), (272, "model"), (315, "artist"),
                      (33432, "copyright"), (316, "hostcomputer"),
                      (285, "pagename")):
        v = meta.get(key)
        if v:
            if isinstance(v, str):
                v = v.encode("utf-8")
            tags.append((code, 2, len(v) + 1, v, True))
    return tags


def write_tiff(dst: str, arr, meta: dict, compress: str = "keep",
               extratags=None) -> None:
    """Write one page, carrying every tag worth keeping from the source.

    metadata=None is not optional. Without it tifffile appends its own
    {"shape": ...} JSON as a SECOND ImageDescription tag, which is invalid
    TIFF -- a tag may appear once per IFD -- and which then shadows the
    scanner's own description for every stage downstream.

    Orientation (274) is not taken from meta. A step that has turned the
    pixels must not carry it, or every viewer would turn the image a second
    time; a step that leaves the pixels where they were passes it, with any
    other tag it carries verbatim, in `extratags`.

    A planar source (PlanarConfiguration 2) is written planar again; the
    array, interleaved by read_page, goes back to (C, H, W) for that."""
    comp = meta.get("compression")
    comp_code = int(getattr(comp, "value", comp) or 1)
    forced = CODEC.get(compress)
    if forced is not None:
        comp_code = forced
    if comp_code in LOSSY_CODECS:
        warn(f"    ! source uses a lossy codec ({comp}); writing deflate "
             f"instead")
        comp_code = DEFLATE

    arr = np.asarray(arr)
    kw = dict(photometric=meta["photometric"], compression=comp_code,
              metadata=None)
    if meta.get("planarconfig") is not None:
        kw["planarconfig"] = meta["planarconfig"]
        if int(meta["planarconfig"]) == 2 and arr.ndim == 3:
            arr = np.transpose(arr, (2, 0, 1))
    pred = int(meta.get("predictor") or 1)
    if comp_code in (5, 8) and compress != "keep":
        pred = 2                       # horizontal differencing: lossless, and
    if pred > 1 and comp_code in (5, 8):    # much smaller on photographic data
        kw["predictor"] = pred
    for key in ("extrasamples", "resolution", "resolutionunit", "iccprofile",
                "description", "datetime", "software", "colormap"):
        if meta.get(key) is not None:
            kw[key] = meta[key]

    extra = _ascii_extratags(meta) + list(extratags or ())
    if extra:
        kw["extratags"] = extra

    tifffile.imwrite(dst, np.ascontiguousarray(arr), **kw)


# --------------------------------------------------------------------------
# console (general): colours change how the log looks, never what it says.
# Plain text when the output is not a console, when the console cannot show
# colours, or when NO_COLOR is set; stdout and stderr are decided apart. One
# meaning per colour in every step (run_pipeline.bat uses the same):
#
#     1;31  bold red     error: a file or the run failed      error()
#     31    red          detail line under an error
#     33    yellow       warning: the run goes on             warn()
#     32    green        success: a file written, a check passed
#     1     bold         headings, the summary, the next step
#     90    grey         details and counters
# --------------------------------------------------------------------------
_ANSI = {"out": False, "err": False}
_VT_PROCESSING = 0x0004          # ENABLE_VIRTUAL_TERMINAL_PROCESSING


def _colour_console(stream) -> bool:
    """Can this stream show ANSI colours? On Windows this also switches the
    console into the mode that interprets them (Windows 10 and later); a
    console that refuses stays plain rather than printing raw escapes."""
    try:
        if not stream.isatty():
            return False
    except (AttributeError, ValueError):
        return False
    if os.name != "nt":
        return True
    try:
        import ctypes
        import msvcrt
        kernel32 = ctypes.windll.kernel32
        handle = msvcrt.get_osfhandle(stream.fileno())
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        if mode.value & _VT_PROCESSING:
            return True
        return bool(kernel32.SetConsoleMode(handle,
                                            mode.value | _VT_PROCESSING))
    except (AttributeError, OSError, ValueError):
        return False


def console_setup() -> None:
    """Decide, once per run, whether stdout and stderr get colours."""
    off = bool(os.environ.get("NO_COLOR"))
    _ANSI["out"] = not off and _colour_console(sys.stdout)
    _ANSI["err"] = not off and _colour_console(sys.stderr)


def paint(text, code, stream: str = "out") -> str:
    """text in colour `code` (see the table above) for stdout, or for
    stderr with stream="err"; unchanged when that stream is plain."""
    return f"\033[{code}m{text}\033[0m" if _ANSI[stream] else text


def warn(text) -> None:
    """A warning, in yellow, on stderr."""
    print(paint(text, "33", "err"), file=sys.stderr)


def error(text) -> None:
    """An error, in bold red, on stderr."""
    print(paint(text, "1;31", "err"), file=sys.stderr)


def warn_list(head, names) -> None:
    """A warning followed by at most LIST_MAX file names, one per line."""
    warn(head)
    for n in names[:LIST_MAX]:
        warn(f"    {n}")
    if len(names) > LIST_MAX:
        warn(f"    ... and {len(names) - LIST_MAX} more")


def error_list(head, names) -> None:
    """An error followed by at most LIST_MAX lines, one per line, in red."""
    error(head)
    for n in names[:LIST_MAX]:
        print(paint(f"    {n}", "31", "err"), file=sys.stderr)
    if len(names) > LIST_MAX:
        print(paint(f"    ... and {len(names) - LIST_MAX} more", "31",
                    "err"), file=sys.stderr)


def _fmt_time(sec) -> str:
    sec = int(round(sec))
    return f"{sec // 60}m{sec % 60:02d}s" if sec >= 60 else f"{sec}s"


class Progress:
    """A bar redrawn in place on a console; on a log file, one line per
    quarter instead."""
    WIDTH = 28

    def __init__(self, total, label):
        self.total, self.label, self.n = max(0, total), label, 0
        self.t0, self.tty = time.time(), sys.stdout.isatty()
        self.last, self.shown = 0, -1
        self._draw()

    def _line(self):
        frac = self.n / self.total if self.total else 1.0
        fill = int(round(frac * self.WIDTH))
        el = time.time() - self.t0
        tail = (f"eta {_fmt_time(el / self.n * (self.total - self.n))}"
                if 0 < self.n < self.total else _fmt_time(el))
        return (f"{self.label} [{'#' * fill}{'.' * (self.WIDTH - fill)}] "
                f"{self.n:>{len(str(self.total))}}/{self.total}  {tail}")

    def _draw(self):
        line = self._line()
        if self.tty:
            sys.stdout.write("\r" + line.ljust(self.last))
            sys.stdout.flush()
            self.last = len(line)
        else:
            q = int(4 * self.n / self.total) if self.total else 4
            if q != self.shown:
                print(line)
                self.shown = q

    def update(self, *_):
        """One more item done (extra arguments are ignored, so this can be
        passed straight to parallel.run as on_result)."""
        self.n = min(self.total, self.n + 1)
        self._draw()

    def message(self, text):
        """Print a line above the bar without breaking it."""
        if self.tty:
            sys.stdout.write("\r" + " " * self.last + "\r")
            print(text)
            self.last = 0
            self._draw()
        else:
            print(text)

    def close(self):
        if self.tty:
            print()


# --------------------------------------------------------------------------
# console layout (general): every step prints the same shape of log -- one
# heading, then a "[phase n of m]" heading for each part of the step with
# indented lines under it; counts come from count(), times from elapsed().
# --------------------------------------------------------------------------

def heading(title) -> None:
    """The heading of a step: 'STAGE n step k  -  what it does'."""
    print(paint("=" * 66 + f"\n  {title}\n" + "=" * 66, "1"))


def phase(n, total, title) -> None:
    """The heading of one phase of a step: '[phase n of total]  title'."""
    print()
    print(paint(f"[phase {n} of {total}]  {title}", "1"))
    print(paint("-" * 66, "90"))


def info(msg="") -> None:
    """A line under a phase, indented; an empty line without msg."""
    print(f"    {msg}" if msg else "")


def count(n, word) -> str:
    """'1 scan', '3 scans'."""
    return f"{n} {word}" + ("" if n == 1 else "s")


def elapsed(t0) -> str:
    """The time since t0, as '42s' or '3m07s'."""
    return _fmt_time(time.time() - t0)


# --------------------------------------------------------------------------
# Stage 2: the folder contract
# --------------------------------------------------------------------------

def review_dir(project_root: str) -> str:
    return os.path.join(resolve(project_root), REVIEW_DIR)


def cropped_dir(project_root: str) -> str:
    return os.path.join(resolve(project_root), CROPPED_DIR)


def check_scans_present(project_root: str) -> None:
    """Fail early, with the one message that helps, when the root holds no
    scans. The root is the only place the steps look for them."""
    d = resolve(project_root)
    if os.path.isdir(d) and any(
            f.lower().endswith(SCAN_EXTS) for f in os.listdir(d)):
        return
    raise SystemExit(
        f"no scans in {d}\n"
        "the project root is the folder holding the scans; every step\n"
        f"reads and writes fixed names beside them ({JSON_DIR}/, "
        f"{REVIEW_DIR}/, {CROPPED_DIR}/)")


def same_dir(a, b) -> bool:
    return (os.path.normcase(os.path.abspath(a))
            == os.path.normcase(os.path.abspath(b)))


def root_arg(root) -> str:
    return "" if same_dir(root, os.getcwd()) else f' "{root}"'


# --------------------------------------------------------------------------
# Stage 2: project files (boxes_<stage>.json)
# --------------------------------------------------------------------------

@dataclass
class Box:
    x0: int
    y0: int
    x1: int          # exclusive, numpy-slice convention
    y1: int          # exclusive
    # The print's tilt in the FILE frame, degrees, positive = turned
    # clockwise on screen, always within +/-45: as step 1 measured it from
    # the bottom edge, or, for a box edited in review.py, from the corners
    # (tilt_from_corners). Informational only: step 4 never reads it -- it
    # measures the tilt from the corners itself, after the scan has been
    # turned the way up you approved.
    angle: float = 0.0
    # The four corners of the print, in order around it (which corner comes
    # first depends on how the box was drawn). THESE ARE THE MEASUREMENT; the
    # axis-aligned box above is derived from them and is only a convenience
    # for drawing and zooming. Step 4 works from the corners alone. Every
    # stored box has them (Project.load refuses a file whose boxes do not);
    # only the boxes the view helpers below compute go without.
    corners: list = None

    def clamp(self, h: int, w: int) -> "Box":
        self.x0 = int(max(0, min(self.x0, w - 1)))
        self.y0 = int(max(0, min(self.y0, h - 1)))
        self.x1 = int(max(self.x0 + 1, min(self.x1, w)))
        self.y1 = int(max(self.y0 + 1, min(self.y1, h)))
        return self

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


@dataclass
class Item:
    size: list           # [height, width]
    boxes: list          # list[Box]
    status: str = "auto"  # auto | confirmed | skipped
    flip: bool = False   # turn this one scan upside down when writing it
    rot: int = 0         # quarter turns clockwise: 0, 90, 180, 270
    orientation: int = 1     # TIFF Orientation tag, reported by step 1
    # Step 1's paper level, for diagnostics only. It is measured on step 1's
    # 8-bit detection copy, not on the scan's own samples, so no later step
    # may use it as a tonal reference.
    paper_level: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["boxes"] = [asdict(b) if isinstance(b, Box) else dict(b)
                      for b in self.boxes]
        return d

    @staticmethod
    def from_dict(d: dict) -> "Item":
        return Item(
            size=list(d["size"]),
            boxes=[Box(**b) for b in d["boxes"]],
            status=d.get("status", "auto"),
            flip=bool(d.get("flip", False)),
            rot=int(d.get("rot", 0)) % 360,
            orientation=int(d.get("orientation", 1)),
            paper_level=float(d.get("paper_level", 0.0)),
        )


class Project:
    """Thin wrapper around a boxes_<stage>.json file.

    source_dir is provenance only -- the root the step ran on. Nothing looks
    scans up through it: the folder contract fixes where everything lives.

    A step can LOAD one file and SAVE another -- review.py reads
    boxes_detect_borders.json and saves boxes_review.json -- which is what
    out_path is for. Without it, saves go to the loaded file."""

    def __init__(self, path: str, source_dir: str = "",
                 out_path: str | None = None):
        self.path = path
        self.out_path = out_path or path
        self.source_dir = source_dir
        self.flip = False    # archive-wide flip decision, set in review.py
        self.items: dict[str, Item] = {}
        self.added: list[str] = []   # scans load_project added from step 1

    # -- io ----------------------------------------------------------------
    @staticmethod
    def load(path: str) -> "Project":
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if raw.get("detector") == "detect_borders.py":
            return Project._from_detect_borders(path, raw)
        p = Project(path, raw.get("source_dir", ""))
        p.flip = bool(raw.get("flip", False))
        p.items = {k: Item.from_dict(v)
                   for k, v in raw.get("items", {}).items()}
        p._check_corners()
        return p

    def _check_corners(self) -> None:
        """Refuse a file with a box that carries no four corners: every step
        works from the corners."""
        for name, it in self.items.items():
            if any(not b.corners or len(b.corners) != 4 for b in it.boxes):
                raise SystemExit(
                    f"{self.path}: {name} has a box without its four "
                    f"corners.\nThe steps cannot use this file: delete it "
                    f"and run the steps again from detect_borders.py.")

    @staticmethod
    def _from_detect_borders(path: str, raw: dict) -> "Project":
        """Read step 1's project file.

        Step 1 reports four corners per print. Their bounding box becomes
        the Box, for drawing and zooming, and the corners are carried along
        beside it as the measurement."""
        p = Project(path, raw.get("source_dir", ""))
        p.flip = bool(raw.get("flip", False))
        for name, it in raw.get("items", {}).items():
            boxes = []
            for ph in it.get("photos", []):
                x0, y0, x1, y1 = ph["bbox"]
                boxes.append(Box(int(x0), int(y0), int(x1), int(y1),
                                 float(ph.get("tilt_deg", 0.0)),
                                 ph.get("corners")))
            p.items[name] = Item(
                size=list(it["size"]), boxes=boxes,
                status=it.get("status", "auto"),
                flip=bool(it.get("flip", False)),
                rot=int(it.get("rot", 0)) % 360,
                orientation=int(it.get("orientation", 1)),
                paper_level=float(it.get("paper_level", 0.0)))
        p._check_corners()
        return p

    def save(self) -> None:
        target = self.out_path or self.path
        base = os.path.dirname(os.path.abspath(target))
        # keep the project portable: record the root RELATIVE to the project
        # file whenever that stays inside the project tree, so the whole
        # folder can be moved or renamed and the provenance stays readable
        src = self.source_dir
        try:
            rel = os.path.relpath(resolve(src), base)
            near = rel == "." or (not rel.startswith(
                os.pardir + os.sep + os.pardir) and not os.path.isabs(rel))
            if near:
                src = rel.replace(os.sep, "/")
        except Exception:
            pass                       # across drives and the like: absolute
        raw = {
            "version": SCHEMA_VERSION,
            "source_dir": src,
            "flip": self.flip,
            "items": {k: v.to_dict() for k, v in self.items.items()},
        }
        write_json(target, raw)

    # -- queries -----------------------------------------------------------
    def names(self) -> list[str]:
        return sorted(self.items.keys())

    def counts(self) -> dict[str, int]:
        out = {"auto": 0, "confirmed": 0, "skipped": 0}
        for it in self.items.values():
            out[it.status] = out.get(it.status, 0) + 1
        return out

    def flipped(self, item: "Item") -> bool:
        """The flip that will actually be written for this scan: the
        archive-wide decision XOR the per-scan V decision. Every stage that
        writes or checks pixels asks this one question, so the crop always
        matches what review.py showed."""
        return bool(self.flip) != bool(item.flip)


def load_project(project_root: str) -> "Project | None":
    """The project steps 2 to 4 work on: boxes_review.json when it exists,
    else boxes_detect_borders.json; None when neither does.

    A scan step 1 measured after the review began -- one added to the
    folder later -- is not in boxes_review.json yet. It is taken over from
    boxes_detect_borders.json as it stands, unconfirmed, and named in
    `added`, so no scan is ever silently left out."""
    review_path = stage_json(project_root, STAGE_REVIEW)
    detect_path = stage_json(project_root, STAGE_DETECT)
    if not os.path.isfile(review_path):
        return Project.load(detect_path) if os.path.isfile(detect_path) \
            else None
    proj = Project.load(review_path)
    if os.path.isfile(detect_path):
        det = Project.load(detect_path)
        for name in det.names():
            if name not in proj.items:
                proj.items[name] = det.items[name]
                proj.added.append(name)
    return proj


# --------------------------------------------------------------------------
# Stage 2: orientation
# --------------------------------------------------------------------------
# Orientation is an optional vertical flip followed by optional quarter turns
# clockwise. Both only move pixels, never compute new ones. Box coordinates
# are always stored against the file as it sits on disk; these helpers
# convert to and from the orientation the user is looking at. Step 4
# measures the tilt in that viewed orientation, straightens and crops in the
# scan's own frame, and orients the finished crop last.


def orient_image(arr, flip: bool, rot: int):
    """Apply the stored orientation to an array."""
    if flip:
        arr = arr[::-1]
    k = (int(rot) // 90) % 4
    if k:
        arr = np.rot90(arr, -k)           # negative k = clockwise
    return np.ascontiguousarray(arr)


def _flip_box(b: "Box", h: int, w: int) -> "Box":
    return Box(b.x0, h - b.y1, b.x1, h - b.y0, b.angle)


def _cw_box(b: "Box", h: int, w: int) -> "Box":
    """One quarter turn clockwise; the image becomes w x h."""
    return Box(h - b.y1, b.x0, h - b.y0, b.x1, b.angle)


def _ccw_box(b: "Box", h: int, w: int) -> "Box":
    """Inverse of _cw_box, given the dimensions AFTER the clockwise turn."""
    return Box(b.y0, w - b.x1, b.y1, w - b.x0, b.angle)


def view_shape(shape, rot: int):
    h, w = shape[0], shape[1]
    return (w, h) if (int(rot) // 90) % 4 in (1, 3) else (h, w)


def box_to_view(b: "Box", shape, flip: bool, rot: int) -> "Box":
    """File coordinates -> view coordinates. b.corners is NOT transformed;
    review.py maps the corner points itself through the same arithmetic."""
    h, w = shape[0], shape[1]
    if flip:
        b = _flip_box(b, h, w)
    for _ in range((int(rot) // 90) % 4):
        b = _cw_box(b, h, w)
        h, w = w, h
    return b


def box_from_view(b: "Box", shape, flip: bool, rot: int) -> "Box":
    """View coordinates -> file coordinates; shape is the ORIGINAL file shape.
    b.corners is NOT transformed, for the same reason as box_to_view."""
    h, w = view_shape(shape, rot)
    for _ in range((int(rot) // 90) % 4):
        b = _ccw_box(b, h, w)
        h, w = w, h
    if flip:
        b = _flip_box(b, shape[0], shape[1])
    return b


def orient_points(pts, shape, flip: bool, rot: int):
    """Map (x, y) PIXEL INDICES through the same flip and quarter turns
    orient_image() applies to an array.

    The box helpers above work in the exclusive-end slice convention, which is
    right for rectangles and wrong for points: a point at row y lands on row
    h-1-y after a flip, not h-y."""
    p = np.asarray(pts, float).reshape(-1, 2)
    h, w = int(shape[0]), int(shape[1])
    if flip:
        p = np.column_stack([p[:, 0], (h - 1) - p[:, 1]])
    for _ in range((int(rot) // 90) % 4):
        p = np.column_stack([(h - 1) - p[:, 1], p[:, 0]])
        h, w = w, h
    return p


def swap_resolution(meta: dict, rot: int) -> dict:
    """Exchange X and Y resolution after a quarter turn.

    A quarter turn makes what was the horizontal axis vertical, so the dots
    per inch recorded for each axis have to follow. With a flatbed set to the
    same value on both axes this changes nothing, but scanners do exist that
    sample the two axes differently, and a wrong figure here would make the
    print come out the wrong physical size years later."""
    if (int(rot) // 90) % 2 == 0:
        return meta
    res = meta.get("resolution")
    if not res or res[0] is None or res[1] is None:
        return meta
    out = dict(meta)
    out["resolution"] = (res[1], res[0])
    return out


# --------------------------------------------------------------------------
# Stage 2: the tilt, one measurement for review.py and step 4
# --------------------------------------------------------------------------

def _rot_pts(p, deg):
    """Turn points about the origin, positive = clockwise on screen
    (y down)."""
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.asarray(p, float) @ np.array([[c, -s], [s, c]]).T


def _inner_rect_area(p):
    """Area of the largest axis-aligned rectangle inside a near-rectangular
    quad: inside both left corners, both right, both top, both bottom."""
    xs = np.sort(p[:, 0])
    ys = np.sort(p[:, 1])
    return max(0.0, xs[2] - xs[1]) * max(0.0, ys[2] - ys[1])


def _quad_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def tilt_from_corners(corners, coarse=TILT_COARSE_DEG,
                      fine=TILT_FINE_DEG):
    """(tilt_deg, kept_fraction) of a print from its four corners.

    tilt_deg is how far the print is turned, positive = clockwise on screen;
    turning it back by the same amount squares it up. It is the turn that
    leaves the most of the print inside an axis-aligned rectangle, not the
    angle of any one side: a detected print is never a perfect rectangle --
    paper shrinks, edges are trimmed by hand, the detector rounds -- so no
    single angle squares all four sides at once, and maximising the
    surviving rectangle makes that compromise explicit.

    The search covers -45..+45 degrees and nothing more. A larger turn would
    change which side of the print is up, and that is decided only by the
    quarter turn and flip approved in review.py -- never by this
    measurement. The corners must therefore be given in the frame whose way
    up is meant: the view for review.py, the turned scan for step 4.
    kept_fraction is that rectangle's share of the quad's area."""
    p = np.asarray(corners, float).reshape(-1, 2)
    p = p - p.mean(0)
    total = _quad_area(p)
    if total <= 0:
        return 0.0, 0.0
    best_a, best_area = 0.0, -1.0
    for a in np.arange(-45.0, 45.0 + 1e-9, coarse):
        ar = _inner_rect_area(_rot_pts(p, a))
        if ar > best_area:
            best_a, best_area = float(a), ar
    for a in np.arange(best_a - coarse, best_a + coarse + 1e-9, fine):
        ar = _inner_rect_area(_rot_pts(p, a))
        if ar > best_area:
            best_a, best_area = float(a), ar
    # best_a is the clockwise turn that squares the print, so the print
    # itself is turned the other way
    return round(-best_a, 4) + 0.0, best_area / total
