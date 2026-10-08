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
# common.py  --  STAGE 4: code shared by every step (not a step)
#
# STAGE 4 corrects the colour of the prints: a faded colour print gets its
# colour back, a B&W print (toned, yellowed or not) becomes neutral grey:
#
#     VueScan                  step 1 of 2   levels and curves
#     restore_colors.py        step 2 of 2   colour and B&W correction
#     common.py                shared code, not a step
#     parallel.py              shared worker-count policy, not a step
#
# Requirements:
#     pip install numpy tifffile [opencv-python] [imagecodecs]
#
# Holds the project files, image reading with its metadata, the TIFF writer
# and the console helpers every step prints with -- the general part, the
# same in every stage -- and, for Stage 4 only, its folder contract and
# small picture helpers.
#
# THE FOLDER CONTRACT. One project root -- the folder of TIFFs step 1
# (VueScan) writes, or any folder of prints; the only argument the step
# needs; Stage 4 reads nothing of any other stage -- holds the prints and
# these fixed names:
#
#     restore_overrides.txt         your corrections (you write it)
#     restore_review/               the plan, the contact sheets, the log of
#                                   --apply and the measuring cache
#     restored/                     the corrected prints, under their own
#                                   names, written by --apply (--out DIR
#                                   puts them anywhere else)
# ==========================================================================
from __future__ import annotations

import json
import os
import sys
import time
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

# --- Stage 4: the folder contract ------------------------------------------
SCAN_EXTS = TIFF_EXT             # the prints are TIFFs
REVIEW_DIR = "restore_review"    # the plan, the sheets, the log, the cache
RESTORED_DIR = "restored"        # the corrected prints, unless --out


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
# Stage 4: the folder contract
# --------------------------------------------------------------------------

def review_dir(project_root: str) -> str:
    return os.path.join(resolve(project_root), REVIEW_DIR)


def restored_dir(project_root: str) -> str:
    return os.path.join(resolve(project_root), RESTORED_DIR)


def list_prints(project_root: str) -> list:
    """The prints in the root, by name."""
    d = resolve(project_root)
    return sorted(f for f in os.listdir(d)
                  if f.lower().endswith(SCAN_EXTS)
                  and os.path.isfile(os.path.join(d, f)))


def check_scans_present(project_root: str) -> None:
    """Fail early, with the one message that helps, when the root holds no
    prints. The root is the only place the step looks for them."""
    d = resolve(project_root)
    if os.path.isdir(d) and any(
            f.lower().endswith(SCAN_EXTS) for f in os.listdir(d)):
        return
    raise SystemExit(
        f"no prints in {d}\n"
        "the project root is the folder of the prints; the step reads and\n"
        f"writes fixed names beside them ({REVIEW_DIR}/, {RESTORED_DIR}/)")


def same_dir(a, b) -> bool:
    return (os.path.normcase(os.path.abspath(a))
            == os.path.normcase(os.path.abspath(b)))


def root_arg(root) -> str:
    return "" if same_dir(root, os.getcwd()) else f' "{root}"'


# --------------------------------------------------------------------------
# Stage 4: pictures
# --------------------------------------------------------------------------

def unit_range(dtype) -> float:
    """The value that stands for full white in an integer image."""
    return float(np.iinfo(dtype).max) if np.issubdtype(dtype, np.integer) \
        else 1.0


def colour_part(arr: np.ndarray) -> np.ndarray:
    """The picture channels: a 2-D grey image as it is, the first three
    channels of anything with more (an alpha channel is not picture)."""
    return arr if arr.ndim == 2 else arr[..., :3]
