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
# common.py  --  STAGE 1: code shared by every step (not a step)
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
#     pip install numpy tifffile [pillow] [imagecodecs]
#                 # pillow only for scans that are not TIFF
#
# Holds the project files (boxes_<stage>.json), image reading with its
# metadata, the TIFF writer and the console helpers every step prints with
# -- the general part, the same in every stage -- and, for Stage 1 only,
# which files are scans, the per-column seam evidence that steps 1 and 2
# both measure, and the weighted fit that moves a grid onto the seams.
#
# THE FOLDER CONTRACT. One project root -- the folder holding the scans, the
# only argument any step needs -- contains these fixed names:
#
#     json files/                       every project file lives here
#         boxes_calibrate_scanner.json      step 1: pitch and phase
#         boxes_detect_seams.json           step 2: the seam columns
#         boxes_repair_seams.json           step 3: the manifest
#     seam_previews/                    step-2 previews, only with --preview
#     seam-repaired/                    step-3 output, the input of Stage 2
#
# WHAT A SEAM IS. A contact image sensor is built from several chips laid end
# to end. At each junction one column of the image is dropped, leaving a
# faint, one-pixel-wide, full-height step. The junctions are physical, so the
# seams repeat at a fixed PITCH and sit at a fixed PHASE in every scan the
# machine makes. A seam at column c lies between columns c and c+1.
# ==========================================================================
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

import numpy as np
import tifffile

try:
    from PIL import Image                 # only for scans that are not TIFF
except ModuleNotFoundError:
    Image = None

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

# --- Stage 1: the folder contract ------------------------------------------
SCAN_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")
NOT_SCANS = ("_marked.png",)   # pixel_ruler.py's pictures, beside the scans
STAGE_CALIBRATE = "calibrate_scanner"      # boxes_calibrate_scanner.json
STAGE_DETECT = "detect_seams"              # boxes_detect_seams.json
STAGE_REPAIR = "repair_seams"              # boxes_repair_seams.json
PREVIEW_DIR = "seam_previews"              # step-2 previews (--preview only)
REPAIRED_DIR = "seam-repaired"             # step-3 output, unless --out

# --- Stage 1: the seam evidence --------------------------------------------
MIN_BANDS = 20            # fewest horizontal voting bands per scan
MAX_BAND_ROWS = 20        # tallest band, rows: a taller one averages a
                          # 3-5 grey-level seam away
BRIGHT_FULL_VOTE = 0.35   # band brightness (share of full scale) at and
                          # above which a band votes with full weight
GIANT_LEVELS = 20.0       # a column step above this many grey levels (8-bit
                          # scale) is an edge, not a seam ...
GIANT_MEDIANS = 4.0       # ... or above this many times the median step
GIANT_GUARD = 16          # columns ignored on each side of such an edge

# --- Stage 1: the tags carried from a scan ---------------------------------
# Tags the TIFF writer produces itself, or that cannot be copied as they
# are: structural ones, the ones write_tiff takes from meta, and pointers to
# other IFDs (Exif, GPS), whose stored value is a file offset. Every other
# tag of a scan is carried verbatim.
NOT_CARRIED = {
    "NewSubfileType", "SubfileType", "ImageWidth", "ImageLength",
    "BitsPerSample", "Compression", "PhotometricInterpretation", "FillOrder",
    "StripOffsets", "SamplesPerPixel", "RowsPerStrip", "StripByteCounts",
    "MinSampleValue", "MaxSampleValue", "SMinSampleValue", "SMaxSampleValue",
    "XResolution", "YResolution", "ResolutionUnit", "PlanarConfiguration",
    "Predictor", "ColorMap", "TileWidth", "TileLength", "TileOffsets",
    "TileByteCounts", "ExtraSamples", "SampleFormat", "JPEGTables",
    "YCbCrCoefficients", "YCbCrSubSampling", "YCbCrPositioning",
    "ReferenceBlackWhite", "InterColorProfile", "ImageDescription",
    "DateTime", "Software", "Make", "Model", "Artist", "Copyright",
    "HostComputer", "PageName", "ExifTag", "GPSTag", "InteroperabilityTag",
    "SubIFDs",
}


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
# Stage 1: the folder contract
# --------------------------------------------------------------------------

def repaired_dir(project_root: str) -> str:
    return os.path.join(resolve(project_root), REPAIRED_DIR)


def same_dir(a, b) -> bool:
    return (os.path.normcase(os.path.abspath(a))
            == os.path.normcase(os.path.abspath(b)))


def root_arg(root) -> str:
    return "" if same_dir(root, os.getcwd()) else f' "{root}"'


def check_scans_present(project_root: str, exts=SCAN_EXTS) -> None:
    """Fail early, with the one message that helps, when the root holds no
    scans. The root is the only place the steps look for them."""
    d = resolve(project_root)
    if os.path.isdir(d) and list_scans(d, exts):
        return
    raise SystemExit(
        f"no scans in {d}\n"
        "the project root is the folder holding the scans; every step\n"
        f"reads and writes fixed names beside them ({JSON_DIR}/, "
        f"{PREVIEW_DIR}/, {REPAIRED_DIR}/)")


# --------------------------------------------------------------------------
# Stage 1: which files are scans, and reading them
# --------------------------------------------------------------------------

def parse_exts(text: str) -> tuple:
    """'.tif,.png' -> ('.tif', '.png'), lower case."""
    return tuple(e.strip().lower() for e in text.split(",") if e.strip())


def list_scans(root: str, exts=SCAN_EXTS) -> list[str]:
    """The scans in the root, sorted. The pictures pixel_ruler.py saves
    beside the scans (NOT_SCANS) are left out; the other outputs are in
    folders of their own."""
    exts = tuple(e.lower() for e in exts)
    return sorted(f for f in os.listdir(root)
                  if f.lower().endswith(exts)
                  and not f.lower().endswith(NOT_SCANS)
                  and os.path.isfile(os.path.join(root, f)))


def carried_tags(path: str) -> list:
    """Every tag of a TIFF scan outside NOT_CARRIED, as tifffile extratags,
    so it is written back exactly as it was stored (Orientation, XMP,
    DocumentName, private tags, ...)."""
    with tifffile.TiffFile(path) as tf:
        return [(t.code, int(t.dtype), t.count, t.value, True)
                for t in tf.pages[0].tags.values()
                if t.name not in NOT_CARRIED]


def read_scan(path: str):
    """(array, meta) in the scan's own dtype and shape.

    TIFF goes through read_page, with every other tag in meta["extratags"].
    Other formats go through Pillow, which keeps 16-bit PNG at 16 bits;
    meta then holds what a TIFF writer needs, the ICC profile and the dpi."""
    if path.lower().endswith(TIFF_EXT):
        arr, meta = read_page(path)
        meta["extratags"] = carried_tags(path)
        return arr, meta
    if Image is None:
        raise RuntimeError("pillow is needed for scans that are not TIFF: "
                           "pip install pillow")
    with Image.open(path) as im:
        if im.mode not in ("L", "LA", "RGB", "RGBA", "I", "I;16",
                           "I;16B", "I;16L"):
            raise RuntimeError(f"image mode {im.mode} is not supported: "
                               "convert the scan to TIFF first")
        arr = np.array(im)
        channels = 1 if arr.ndim == 2 else arr.shape[2]
        meta = {"photometric": "rgb" if channels >= 3 else "minisblack",
                "iccprofile": im.info.get("icc_profile"), "extratags": []}
        dpi = im.info.get("dpi")
        if dpi:
            meta["resolution"] = (float(dpi[0]), float(dpi[1]))
            meta["resolutionunit"] = 2                     # inch
    return arr, meta


def as_hwc(a: np.ndarray) -> np.ndarray:
    """The array as H x W x C: a grayscale scan gets a channel axis (a view,
    not a copy)."""
    return a[:, :, None] if a.ndim == 2 else a


def full_scale(a: np.ndarray) -> float:
    """The largest value the scan's dtype can hold: 255 for 8-bit, 65535 for
    16-bit. Float data is taken as 0..1."""
    if np.issubdtype(a.dtype, np.integer):
        return float(np.iinfo(a.dtype).max)
    return 1.0


def display_rgb(a: np.ndarray) -> np.ndarray:
    """8-bit H x W x 3 for display: to_display_rgb, with grey spread to three
    channels and a grey+alpha scan shown by its grey."""
    d = as_hwc(to_display_rgb(a))
    if d.shape[2] < 3:
        d = np.repeat(d[:, :, :1], 3, axis=2)
    return d


# --------------------------------------------------------------------------
# Stage 1: the seam evidence
# --------------------------------------------------------------------------

def _exclude_giants(score, step, scale):
    """Zero `score` around columns whose mean step is an edge, not a seam."""
    thr = max(GIANT_LEVELS * scale / 255.0, GIANT_MEDIANS * np.median(step))
    for gc in np.where(step > thr)[0]:
        score[max(0, gc - GIANT_GUARD): gc + GIANT_GUARD + 1] = 0.0
    return score


def seam_evidence(a: np.ndarray, min_bands: int = MIN_BANDS):
    """(votes, step): two per-column measures of how much each column
    boundary looks like a seam. Column c describes the boundary between
    columns c and c+1, so both have width-1 entries.

    votes: the scan is cut into horizontal bands no taller than
    MAX_BAND_ROWS, and each band votes for the columns that are a sharp
    one-column peak of its horizontal step, weighted by the band's
    brightness -- where a scan is nearly black no seam can be seen. A seam is
    thin and full-height, so it collects votes from band after band; texture
    does not.

    step: the mean horizontal step of each column over the whole height.

    Both are zeroed around strong edges (GIANT_*). Only colour channels are
    used; alpha is ignored. The scan is worked through one band at a time, so
    a worker holds little beyond the scan itself."""
    a = as_hwc(a)
    a = a[:, :, :3] if a.shape[2] >= 3 else a[:, :, :1]
    h, w, _ = a.shape
    scale = full_scale(a)
    k = min(max(int(min_bands), int(np.ceil(h / float(MAX_BAND_ROWS)))), h)
    edges = np.linspace(0, h, k + 1).astype(int)
    n = w - 1
    votes = np.zeros(n)
    step_sum = np.zeros(n)
    wsum = 0.0
    for i in range(k):
        r0, r1 = edges[i], edges[i + 1]
        band = a[r0:r1].astype(np.float64)
        g = np.abs(band[:, 1:, :] - band[:, :-1, :]).mean(axis=2)
        for row in g:                   # row by row, as a sum over the height
            step_sum += row
        gb = g.mean(0)
        wgt = float(np.clip(band.mean(axis=2).mean()
                            / (BRIGHT_FULL_VOTE * scale), 0.0, 1.0))
        peak = np.zeros(n)
        peak[1:-1] = ((gb[1:-1] > gb[:-2]) & (gb[1:-1] > gb[2:])
                      & (gb[1:-1] > np.median(gb))).astype(float)
        votes += wgt * peak
        wsum += wgt
    votes /= (wsum if wsum > 0 else k)
    step = step_sum / h
    return (_exclude_giants(votes, step, scale),
            _exclude_giants(step.copy(), step, scale))


def weighted_median_offset(profile, start, pitch, radius, threshold):
    """How far a grid of `pitch` starting at `start` must move to sit on the
    seams, or None when no grid line finds a seam.

    Each grid line looks for the strongest column within `radius`; lines
    that find one at or above `threshold` vote with its offset, weighted by
    the square of its strength, so a few strong seams outweigh weak ones and
    one off-grid outlier cannot drag the result."""
    n = len(profile)
    offs, wts = [], []
    for c in range(start, n, pitch):
        lo, hi = max(0, c - radius), min(n, c + radius + 1)
        if hi <= lo:
            continue
        j = lo + int(np.argmax(profile[lo:hi]))
        if profile[j] >= threshold:
            offs.append(j - c)
            wts.append(float(profile[j]) ** 2)
    if not offs:
        return None
    order = np.argsort(offs)
    ov = np.array(offs, dtype=np.float64)[order]
    cw = np.cumsum(np.array(wts, dtype=np.float64)[order])
    return int(ov[int(np.searchsorted(cw, cw[-1] / 2.0))])
