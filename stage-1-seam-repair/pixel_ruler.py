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
# pixel_ruler.py  --  STAGE 1: read a seam's column by eye, for step 2
#                     (not a step)
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
#     pip install numpy tifffile matplotlib [pillow] [imagecodecs]
#                 # pillow only for scans that are not TIFF
#     common.py beside this file
#
# Opens one scan and shows the COLUMN under the pointer, so a seam can be
# found or checked without counting pixels. The number it reads off is what
# detect_seams.py --phase expects: a seam at column c lies between columns c
# and c+1.
#
# The scan is shown at its true size, so the numbers are the file's own.
# S saves a picture of the window beside the scan, as <scan>_marked.png;
# the steps do not take it for a scan.
#
# The magnifier and pan tools release themselves when the mouse comes up,
# so the next click marks a column instead of starting another zoom. The
# live readout repaints only the overlay over a cached image of the scan,
# and a pan drag shows a coarse copy until it ends, so the window stays
# quick on a large scan.
#
# Usage:
#     python pixel_ruler.py SCAN
#
#     move the mouse    the readout under the image shows col / row
#     left click        mark that column and print it
#     magnifier / pan   drag to zoom or pan; the tool releases itself
#     z / x             zoom in / out around the pointer
#     c / u             clear all marks / undo the last one
#     s                 save the window as <scan>_marked.png
#
#     python pixel_ruler.py raw0001.tif
# ==========================================================================
from __future__ import annotations

import argparse
import os
import sys

try:
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    sys.exit(f"missing package: {exc.name} (see Requirements above)")

try:
    from common import console_setup, display_rgb, heading, info, read_scan
except ModuleNotFoundError as exc:
    sys.exit("common.py must sit beside this script" if exc.name == "common"
             else f"missing package: {exc.name} (see Requirements above)")

# --------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------
# --- the window ------------------------------------------------------------
FIG_PX_PER_INCH = 90        # initial window size: scan pixels per inch ...
FIG_MIN = (6, 4)            # ... kept within these sizes, inches (w, h)
FIG_MAX = (16, 10)
PROXY_PIXELS = 1.5e6        # the coarse copy shown during a pan drag has
                            # about this many pixels
ZOOM_IN = 0.5               # z: view size multiplied by this
ZOOM_OUT = 2.0              # x: view size multiplied by this
MARK_COLOUR = "red"
SAVE_DPI = 150              # resolution of the saved picture of the window
FREED_KEYS = (("c", "keymap.back"), ("s", "keymap.save"))   # see free_keys


# --------------------------------------------------------------------------
# the window
# --------------------------------------------------------------------------

def free_keys():
    """matplotlib binds 'c' (back) and 's' (save dialog) itself, so a press
    would fire two actions at once. Only these two are freed."""
    for key, param in FREED_KEYS:
        try:
            matplotlib.rcParams[param] = [k for k in matplotlib.rcParams[param]
                                          if k != key]
        except Exception:
            pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Stage 1 — read a seam's column by eye, for step 2 "
                    "(detect_seams.py --phase)")
    ap.add_argument("scan", help="the scan to open")
    args = ap.parse_args(argv)
    console_setup()
    path = args.scan
    if not os.path.isfile(path):
        raise SystemExit(f"not a file: {path}")
    heading("STAGE 1  -  pixel ruler: read a seam's column by eye")
    try:
        a, _ = read_scan(path)
    except Exception as exc:
        raise SystemExit(f"could not read: {path} ({exc})")
    img = display_rgb(a)
    H, W = img.shape[:2]

    free_keys()
    fig, ax = plt.subplots(
        figsize=(min(FIG_MAX[0], max(FIG_MIN[0], W / FIG_PX_PER_INCH)),
                 min(FIG_MAX[1], max(FIG_MIN[1], H / FIG_PX_PER_INCH))))
    im = ax.imshow(img, interpolation="nearest")

    # the coarse copy for pan drags; the extent stays pinned to the full
    # pixel grid, so coordinates never change
    extent = (-0.5, W - 0.5, H - 0.5, -0.5)
    im.set_extent(extent)
    pstep = max(1, int(np.ceil(np.sqrt(H * W / PROXY_PIXELS))))
    proxy = img[::pstep, ::pstep] if pstep > 1 else None
    showing_proxy = {"on": False}

    def use_proxy(on):
        if proxy is None or showing_proxy["on"] == on:
            return
        im.set_data(proxy if on else img)
        im.set_extent(extent)
        showing_proxy["on"] = on

    ax.set_title(f"{os.path.basename(path)}   ({W} x {H})")
    # the readout has its own text box: rewriting the title on every move
    # fights the toolbar for the status area
    readout = fig.text(0.01, 0.01, "move the mouse over the image",
                       fontsize=9, family="monospace")

    marks = []          # (column, line artist, label artist)

    def tool_active():
        """True while the magnifier or pan tool is selected, so the press
        that starts a drag does not also mark a column."""
        tb = getattr(fig.canvas, "toolbar", None)
        return bool(str(getattr(tb, "mode", "")).strip())

    # -- the overlay: readout and marks over a cached image of the scan ----
    bg = {"cache": None}

    def drop_background(_evt=None):
        bg["cache"] = None           # rebuilt on the next repaint

    def refresh_background():
        """Cache the figure without the overlay."""
        readout.set_visible(False)
        for _, line, label in marks:
            line.set_visible(False)
            label.set_visible(False)
        fig.canvas.draw()
        bg["cache"] = fig.canvas.copy_from_bbox(fig.bbox)
        readout.set_visible(True)
        for _, line, label in marks:
            line.set_visible(True)
            label.set_visible(True)

    def blit_overlay():
        """Repaint only the readout and the marks over the cache."""
        try:
            if bg["cache"] is None:
                refresh_background()
            fig.canvas.restore_region(bg["cache"])
            for _, line, label in marks:
                ax.draw_artist(line)
                ax.draw_artist(label)
            fig.draw_artist(readout)
            fig.canvas.blit(fig.bbox)
        except Exception:
            fig.canvas.draw_idle()   # a backend that cannot blit

    def on_move(event):
        if showing_proxy["on"] and not tool_active():
            use_proxy(False)             # a drag ended without a release
            drop_background()
            fig.canvas.draw_idle()
        if event.inaxes is ax and event.xdata is not None:
            col, row = int(round(event.xdata)), int(round(event.ydata))
            if 0 <= col < W and 0 <= row < H:
                txt = (f"col = {col:5d}   row = {row:5d}"
                       f"   (a seam here lies between col {col} and "
                       f"{col + 1})")
            else:
                txt = "outside the image"
        else:
            txt = ""
        if txt != readout.get_text():
            readout.set_text(txt)
            blit_overlay()

    def add_mark(col):
        line = ax.axvline(col + 0.5, color=MARK_COLOUR, linewidth=1)
        label = ax.annotate(str(col), xy=(col, 0), xytext=(0, 4),
                            textcoords="offset points", color=MARK_COLOUR,
                            fontsize=8, ha="center", va="bottom",
                            annotation_clip=False)
        marks.append((col, line, label))
        info(f"marked column {col}   (seam between col {col} and "
             f"{col + 1})")
        blit_overlay()

    def on_click(event):
        if tool_active():
            use_proxy(True)              # a drag begins
            return
        if event.button != 1 or event.inaxes is not ax or event.xdata is None:
            return
        col = int(round(event.xdata))
        if 0 <= col < W:
            add_mark(col)

    def on_release(event):
        """Release the magnifier or pan tool as its drag ends."""
        tb = getattr(fig.canvas, "toolbar", None)
        mode = str(getattr(tb, "mode", "")).strip()
        if not mode or tb is None:
            if showing_proxy["on"]:
                use_proxy(False)
                drop_background()
                fig.canvas.draw_idle()
            return
        use_proxy(False)
        try:
            # pan first: its mode string is "pan/zoom", which also contains
            # "zoom", so testing zoom first would leave pan switched on
            if "pan" in mode.lower():
                tb.pan()
            elif "zoom" in mode.lower():
                tb.zoom()
        except Exception:
            pass                         # an unknown backend: leave it be
        drop_background()                # the view moved

    def zoom_at(event, factor):
        if event.xdata is None or event.ydata is None:
            return
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        cx, cy = event.xdata, event.ydata
        ax.set_xlim(cx - (cx - x0) * factor, cx + (x1 - cx) * factor)
        ax.set_ylim(cy - (cy - y0) * factor, cy + (y1 - cy) * factor)
        drop_background()
        fig.canvas.draw_idle()

    def on_key(event):
        k = event.key
        if k == "c":
            for _, line, label in marks:
                line.remove()
                label.remove()
            marks.clear()
            info("cleared all marks")
            blit_overlay()
        elif k == "u":
            if marks:
                col, line, label = marks.pop()
                line.remove()
                label.remove()
                info(f"removed mark {col}")
                blit_overlay()
        elif k == "z":
            zoom_at(event, ZOOM_IN)
        elif k == "x":
            zoom_at(event, ZOOM_OUT)
        elif k == "s":
            out = os.path.splitext(path)[0] + "_marked.png"
            fig.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
            cols = sorted(set(c for c, _, _ in marks))
            info(f"saved the window -> {out}   (columns: {cols})")

    fig.canvas.mpl_connect("draw_event", drop_background)
    fig.canvas.mpl_connect("resize_event", drop_background)
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)

    info(f"{path}  ({W} x {H})")
    info("click a seam to read its column; z/x zoom, c clear, u undo, "
         "s save")
    info()
    plt.show()

    if marks:
        info()
        info(f"all marked columns: {sorted(set(c for c, _, _ in marks))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
