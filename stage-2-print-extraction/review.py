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
# review.py  --  STAGE 2, step 3 of 4: approve the box, turn and way up
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
#     pip install "pyqtgraph>=0.12" PyQt5 numpy tifffile [imagecodecs]
#                 [opencv-python]      # opencv only for scans not in TIFF
#     common.py beside this file
#
# This step edits coordinates and decisions, not images. One project
# file in, one out, both inside the shared `json files/` folder of the
# project root -- the one path argument this step takes:
#
#     reads   json files/boxes_review.json            to resume a session,
#             else json files/boxes_detect_borders.json  the fresh detections
#     writes  json files/boxes_review.json            on every confirm, skip,
#                                                     move to another scan,
#                                                     turn, flip, Ctrl+S and
#                                                     quit
#
# This step never detects. You can quit at any point; the next run opens at
# the first scan you have not confirmed.
#
# AN UNTOUCHED BOX IS NEVER REWRITTEN. pyqtgraph can only draw a
# right-angled rectangle, but the quad step 1 measures rarely is one: paper
# shrinks unevenly, edges were trimmed by hand, the detector rounds.
# Rebuilding a rectangle from such a quad squares it off and moves each
# corner by several pixels. So each rectangle remembers the corners it was
# drawn from, and on the way out its current corners are compared against
# them. Unchanged, and the original box goes back into the project
# untouched, byte for byte. Changed, and your version wins.
#
# Turning, flipping and rotating are recorded as DECISIONS, not applied. The
# stored box always refers to the file as it sits on disk, so changing the
# view never disturbs a box you have already placed by hand. A box you turn
# is stored as its four corners. Step 4 measures the tilt from those corners
# in the way up you approved here and straightens by it -- so the way up is
# decided only by the flip and the quarter turns you confirm, never by the
# tilt.
#
# HANDLES: the four CORNERS rotate, the four EDGE MIDPOINTS resize. Rotation
# pivots about the middle and never changes either side length, so a box
# fitted to a print keeps its proportions however far you turn it.
#
# SUGGESTED WAY UP. If step 2 (suggest_orientation.py) has run, its file
# json files/orientation_suggestions.json is read at start, unless
# --no-suggestions is given. A scan you have not confirmed yet, and have not
# turned yourself, then OPENS turned the way a high-confidence suggestion
# says, marked [auto: turned · 2 faces] in the status bar. That turn is only
# in the view: nothing is stored until you press Enter, which confirms it
# like any turn of your own. . and , turn on from what you see; R goes back
# to step 1's record and shows the suggestion again. A medium-confidence
# suggestion is only named, [suggest: ROT 90? ...], and the view is left
# alone.
#
# The archive-wide flip is stored in the project file beside the per-scan V
# decisions, and every later step reads it from there: step 4 writes exactly
# what this window showed, with no flag passed to it. What actually gets
# applied is the archive flip XOR this scan's V, so the two are independent
# and V keeps working whichever way the archive is set.
#
# Usage:
#     python review.py [ROOT] [options]
#
#     ROOT              project root: the folder holding the scans
#                       (default: this folder)
#     --flip            flip the WHOLE archive. A STATE, not a toggle: with
#                       the flag the archive flip is on, without it, off, so
#                       resuming a half-finished session never depends on
#                       remembering how the last run left it. Stored in the
#                       project file, so deskew_crop.py picks it up. V still
#                       flips one scan at a time, on top of it.
#     --no-suggestions  ignore json files/orientation_suggestions.json this
#                       time: no scan opens turned by a suggestion
#
#     python review.py
#     python review.py D:\archive\scans --flip
#
# Keys:
#     Enter / Space   confirm this scan and go to the next (with the turn you
#                     see, including one the suggestion made)
#     Ctrl + Enter    confirm THIS scan and every later one not yet decided,
#                     each exactly as it would open (a high-confidence
#                     suggested turn included), then save and close so the
#                     next step can start. Asks first, with the counts.
#                     Scans already confirmed or skipped are left alone; a
#                     scan without a box is marked skipped, as Enter does.
#     N / P           next / previous without confirming
#     arrows          move the selected box by 1 px  (Shift = 10 px)
#     Ctrl + arrows   move the box's right/bottom edge by 1 px (Shift = 10)
#     Q / E           turn the box by 0.1 deg  (Ctrl = 1 deg); 0 squares it
#                     again. Dragging any CORNER handle does the same by hand,
#                     pivoting about the middle, at constant proportions;
#                     hold Ctrl while dragging to snap to 15 deg.
#     V               flip this scan upside down
#     =               point at an edge, then press = to move THAT edge until
#                     the box is square. The opposite edge is pinned, so the
#                     other three sides and the angle do not move. Use it
#                     when a print really was square -- a contact print from
#                     a 6x6 negative is 56 x 56 mm -- and one edge was
#                     detected short. On a print trimmed by hand it would
#                     take off a real edge, so nothing does this by default.
#     . / ,           rotate 90 degrees clockwise / anticlockwise
#     A / D           add a box / delete the selected box (click it first)
#     Tab             select the next box (only matters with more than one);
#                     a click on a box selects it too
#     R               RESET this scan: discard every manual edit and restore
#                     the boxes, the flip and the turn exactly as
#                     boxes_detect_borders.json recorded them
#     S               skip this scan entirely
#     F / Z           fit to window / zoom to the selected box
#     Ctrl+S          save            Esc   save and quit
#
# The status bar shows the box in view coordinates, its size, its tilt as
# you see it (measured from its corners by the same common.tilt_from_corners
# step 4 uses, positive = clockwise), whether the box is still as detected or
# has been edited, and [FLIPPED] / [ROT n] markers for the decisions in
# force. From a tilt of common.STRAIGHTEN_MIN_DEG up, it warns that step 4
# will straighten this print, which recomputes its samples; below that,
# step 4 cuts the print out untouched. Mouse: wheel zooms, dragging the
# middle moves, edge handles resize, corner handles turn.
# ==========================================================================
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor

try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
except ModuleNotFoundError as exc:
    sys.exit(f"missing package: {exc.name} (see Requirements above)")
except Exception:                         # pyqtgraph found no Qt binding
    sys.exit("missing package: PyQt5, or another Qt binding for pyqtgraph "
             "(see Requirements above)")

try:
    from common import (JSON_DIR, STAGE_DETECT, STAGE_REVIEW,
                        STRAIGHTEN_MIN_DEG, SUGGEST_NAME, Box, Project,
                        box_from_view, box_to_view, check_scans_present,
                        console_setup, count, error, heading, info,
                        json_folder, load_project, orient_image, paint,
                        phase, pipeline_root, read_page, root_arg,
                        stage_json, tilt_from_corners, to_display_rgb,
                        view_shape, warn, warn_list)
except ModuleNotFoundError as exc:
    sys.exit("common.py must sit beside this script" if exc.name == "common"
             else f"missing package: {exc.name} (see Requirements above)")

# --------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------
# --- general ---------------------------------------------------------------
SCRIPT = "review.py"

# --- the window ------------------------------------------------------------
WINDOW_SIZE = (1200, 850)   # initial window size, px
ROT_STEP = 0.1          # degrees per Q/E press; Ctrl multiplies by ten
CACHE_BEHIND = 1        # decoded scans kept in memory behind the current one
CACHE_AHEAD = 2         # ... and ahead of it (the next one is prefetched)
ZOOM_MARGIN = 10        # Z leaves 1/this of the box width around it ...
ZOOM_MARGIN_MIN = 20    # ... but at least this many px
MSG_SHORT_MS = 1500     # status-bar notes: short ...
MSG_LONG_MS = 2500      # ... longer ...
MSG_ERROR_MS = 4000     # ... and for a failed reset

# --- the box and its handles -----------------------------------------------
HANDLE_SIZE = 13        # handle size on screen, px
# Edge midpoints resize; corners turn. Two handles on one spot is a coin
# toss, so the two jobs never share a position.
SCALE_HANDLES = [([0.5, 0], [0.5, 1]), ([0.5, 1], [0.5, 0]),
                 ([0, 0.5], [1, 0.5]), ([1, 0.5], [0, 0.5])]
ROTATE_HANDLES = [[0, 0], [1, 0], [1, 1], [0, 1]]

PEN_IDLE = pg.mkPen((235, 60, 45), width=2)
PEN_SEL = pg.mkPen((255, 0, 0), width=3)
# black outline with a white core, so the handles stay visible against both a
# dark photo and the near-white scanner lid
HANDLE_PEN = pg.mkPen((0, 0, 0), width=3)
HANDLE_HOVER_PEN = pg.mkPen((255, 255, 255), width=3)
# the turning handles read differently at a glance, so the hand knows which
# it is reaching for before it arrives
ROT_HANDLE_PEN = pg.mkPen((0, 90, 200), width=3)


# --------------------------------------------------------------------------
# the window
# --------------------------------------------------------------------------

class Reviewer(QtWidgets.QMainWindow):
    def __init__(self, proj, src_dir: str, reset_path: str,
                 suggest: dict | None = None):
        super().__init__()
        self.proj = proj
        self.suggest = suggest or {}   # name -> suggest_orientation.py entry
        self._declined: set = set()    # turned by hand this session
        self.src_dir = src_dir
        self.reset_path = reset_path   # boxes_detect_borders.json, for R
        self.orig_shape = (1, 1)
        self.names = proj.names()
        self.idx = self._first_unconfirmed()
        self.rois: list = []
        self.sel = 0
        self.shape = (1, 1)
        self._cache: dict = {}
        self._loading = False     # re-entrancy guard; see load()
        self._broken = None       # why the scan on screen cannot be read
        self._pool = ThreadPoolExecutor(max_workers=1)

        pg.setConfigOptions(imageAxisOrder="row-major", antialias=False)
        self.glw = pg.GraphicsLayoutWidget()
        self.vb = self.glw.addViewBox()
        self.vb.setAspectLocked(True)
        self.vb.invertY(True)
        # autoDownsample draws a zoomed-out large scan from fewer pixels,
        # so panning and zooming stay smooth; the data is not changed
        self.img = pg.ImageItem(autoDownsample=True)
        self.vb.addItem(self.img)

        self.status = self.statusBar()
        self.setCentralWidget(self.glw)
        QtWidgets.QApplication.instance().installEventFilter(self)
        self.resize(*WINDOW_SIZE)
        self.load(self.idx)

    # -- navigation --------------------------------------------------------
    def _first_unconfirmed(self) -> int:
        for i, n in enumerate(self.names):
            if self.proj.items[n].status == "auto":
                return i
        return 0

    def _pending(self):
        """The turn a high-confidence suggestion puts in the VIEW of this
        scan, or None. See _pending_for."""
        return self._pending_for(self.names[self.idx])

    def _pending_for(self, name):
        """The turn a high-confidence suggestion puts in the view of scan
        `name`, or None. Only for a scan still unconfirmed whose stored turn
        is untouched and which was not turned by hand this session. The
        value follows the flip in force (a mirrored scan needs another turn).
        Nothing is stored until the scan is confirmed."""
        it = self.proj.items[name]
        s = self.suggest.get(name)
        if (not s or s.get("confidence") != "high" or it.status != "auto"
                or it.rot or name in self._declined):
            return None
        key = "rot_flipped" if self.proj.flipped(it) else "rot"
        return int(s.get(key, 0)) % 360

    def _orient(self) -> tuple:
        """(flip, rot) exactly as step 4 will apply them once confirmed: the
        archive-wide decision XOR this scan's V decision, and the stored turn
        -- or, while one is pending, the suggested turn (see _pending)."""
        it = self.proj.items[self.names[self.idx]]
        p = self._pending()
        return (self.proj.flipped(it), it.rot if p is None else p)

    def _suggest_mark(self) -> str:
        """Status-bar note on a suggestion for this scan, if any."""
        name = self.names[self.idx]
        s = self.suggest.get(name)
        if not s or self.proj.items[name].status != "auto":
            return ""
        if self._pending() is not None:
            turn = "as it is" if self._orient()[1] == 0 else "turned"
            return f"   [auto: {turn} · {s.get('why', '')}]"
        if s.get("confidence") == "medium" and name not in self._declined:
            key = "rot_flipped" if self.proj.flipped(self.proj.items[name]) \
                else "rot"
            return (f"   [suggest: ROT {int(s.get(key, 0)) % 360}? · "
                    f"{s.get('why', '')}]")
        return ""

    def _from_view(self, b):
        f, rot = self._orient()
        return box_from_view(b, self.orig_shape, f, rot)

    def _pt_to_view(self, x, y):
        """One file-frame point -> view coordinates, through the same
        unit-square round trip _commit uses for corners, so load and commit
        are exact inverses of each other and a box survives any number of
        navigation cycles unchanged."""
        f, rot = self._orient()
        b = box_to_view(Box(int(round(x)), int(round(y)),
                            int(round(x)) + 1, int(round(y)) + 1),
                        self.orig_shape, f, rot)
        return (float(b.x0), float(b.y0))

    def _read(self, name: str):
        if name not in self._cache:
            arr, _ = read_page(os.path.join(self.src_dir, name))
            self._cache[name] = (to_display_rgb(arr), arr.shape[:2])
        return self._cache[name]

    def _prefetch(self, i: int) -> None:
        if 0 <= i < len(self.names):
            n = self.names[i]
            if n not in self._cache:
                self._pool.submit(self._read, n)

    def load(self, i: int) -> None:
        """Show one scan. Not re-entrant, and careful about the order.

        It calls processEvents() to get the "loading" title on screen, which
        can deliver another key press in the middle of this method (hold N
        down on 600 dpi scans). Keys are therefore ignored while a load is in
        flight, and nothing that _commit() reads -- self.idx, self.rois,
        self.orig_shape -- moves until the new image is in hand, so a commit
        can never write one scan's rectangles onto another scan."""
        if self._loading:
            return
        self._loading = True
        try:
            idx = max(0, min(i, len(self.names) - 1))
            name = self.names[idx]
            self.setWindowTitle(f"loading {name} …")
            QtWidgets.QApplication.processEvents()

            # read first: until this returns, self.idx and self.rois still
            # agree with each other and with what the operator can see. A
            # scan that cannot be read (deleted, moved or damaged since
            # step 1) is shown empty, with its boxes left as they are.
            try:
                disp, orig_shape = self._read(name)
                broken = None
            except Exception as exc:
                disp, broken = None, f"{type(exc).__name__}: {exc}"

            self._clear_rois()          # the old boxes go with the old index
            self.idx = idx
            self._broken = broken
            self.sel = 0
            if broken is None:
                self.orig_shape = orig_shape
                f, rot = self._orient()
                disp = orient_image(disp, f, rot)
                self.shape = view_shape(self.orig_shape, rot)
                self.img.setImage(disp, autoLevels=False, levels=(0, 255))
                self.vb.autoRange()
                self._draw_boxes(self.proj.items[name].boxes)
            else:
                self.orig_shape = self.shape = (1, 1)
                self.img.clear()
        finally:
            self._loading = False
        self._restyle()
        self._refresh_title()
        near = {self.names[j]
                for j in range(self.idx - CACHE_BEHIND,
                               self.idx + CACHE_AHEAD + 1)
                if 0 <= j < len(self.names)}
        for k in list(self._cache):                 # keep memory bounded
            if k not in near:
                self._cache.pop(k, None)
        self._prefetch(self.idx + 1)

    def _draw_boxes(self, boxes) -> None:
        """One ROI per stored box, each remembering where it came from. The
        corners fully specify the drawn rectangle; rebuilding it from the
        axis-aligned envelope instead would inflate every tilted box."""
        for b in boxes:
            self._add_roi_from_corners(b.corners, src=b)

    def _refresh_title(self) -> None:
        name = self.names[self.idx]
        it = self.proj.items[self.names[self.idx]]
        c = self.proj.counts()
        rot = self._orient()[1]
        self.setWindowTitle(f"[{self.idx + 1}/{len(self.names)}] {name}  —  "
                            f"{it.status}"
                            f"{'  [FLIPPED]' if self.proj.flipped(it) else ''}"
                            f"{('  [ROT %d]' % rot) if rot else ''}"
                            f"   (confirmed {c.get('confirmed', 0)}, "
                            f"skipped {c.get('skipped', 0)})")
        self._refresh_status()

    def _refresh_status(self) -> None:
        if self._broken:
            self.status.showMessage(f"this scan cannot be read "
                                    f"({self._broken}) — S or Enter skips "
                                    f"it, N / P move on")
            return
        if not self.rois:
            self.status.showMessage("no box — press A to add one")
            return
        roi = self.rois[self.sel]
        b = self._roi_box(roi)
        ang, _keep = tilt_from_corners(self._roi_corners(roi))
        warn = ("   ⚠ tilt: step 4 will straighten this print, which "
                "recomputes its samples (not bit-exact)"
                if abs(ang) >= STRAIGHTEN_MIN_DEG else "")
        fl, rot = self._orient()
        marks = (("   [FLIPPED]" if fl else "")
                 + (f"   [ROT {rot}]" if rot else ""))
        marks += "   [edited]" if self._is_edited(roi) else "   as detected"
        marks += self._suggest_mark()
        self.status.showMessage(
            f"box {self.sel + 1}/{len(self.rois)}   "
            f"x {b.x0}–{b.x1}   y {b.y0}–{b.y1}   "
            f"{b.width} × {b.height} px   tilt {ang:+.2f}°{marks}{warn}")

    # -- ROI plumbing ------------------------------------------------------
    def _clear_rois(self) -> None:
        for r in self.rois:
            self.vb.removeItem(r)
        self.rois = []

    def _add_roi_geometry(self, x, y, w, h, angle: float = 0.0, src=None):
        """Build one RectROI at exact (possibly fractional) geometry.

        `src` is the project's own Box this rectangle was drawn from, or None
        for a box you have just added. It is kept, with a snapshot of the
        corners as first drawn, so _commit can tell an untouched rectangle
        from an edited one and leave the untouched one's measurement alone."""
        # No maxBounds argument on purpose: Qt's bound check treats a turned
        # rectangle by its upright envelope, so a box that fits the frame
        # perfectly refuses to rotate the moment its corners reach the edge.
        # Step 4 clips anyway, and _roi_box clamps what gets stored.
        roi = pg.RectROI([x, y], [w, h], pen=PEN_IDLE,
                         movable=True, rotatable=True, resizable=True)
        roi.handleSize = HANDLE_SIZE
        roi.handlePen = HANDLE_PEN
        roi.handleHoverPen = HANDLE_HOVER_PEN
        for hd in list(roi.handles):
            item = hd["item"]
            roi.removeHandle(item)
            # removeHandle only takes it out of the SCENE, and only if the
            # ROI is already in one -- it is not, at this point. The handle
            # is still a child item of the ROI, so it goes on painting itself
            # at whatever position it last had, no longer following the box.
            item.setParentItem(None)
        for pos, center in SCALE_HANDLES:
            roi.addScaleHandle(pos, center)
        # Corners turn about the middle, so the box pivots in place and
        # neither side length changes. Rotation is free, not snapped; hold
        # Ctrl while dragging to snap to 15 degrees. addRotateHandle keeps
        # each handle anchored to its corner (addRotateFreeHandle would
        # rewrite the anchor at the end of a drag and throw the handle off
        # the box).
        for pos in ROTATE_HANDLES:
            hnd = roi.addRotateHandle(pos, [0.5, 0.5])
            try:
                hnd.pen = hnd.currentPen = ROT_HANDLE_PEN
                hnd.update()
            except Exception:                 # older pyqtgraph: colour is
                pass                          # cosmetic, position is not
        # integer-only geometry: the crop must land on whole pixels
        roi.snapSize = 1.0
        roi.translateSnap = True
        roi.scaleSnap = True
        if abs(angle) > 1e-9:
            roi.setAngle(angle, centerLocal=[w / 2.0, h / 2.0])
        roi.sigRegionChanged.connect(lambda *_: self._refresh_status())
        # pyqtgraph ROIs ignore clicks unless told otherwise, and then
        # sigClicked never fires: without this a click never selects a box,
        # and the keys that act on the selected box act on another one.
        # Dragging and the handles are not affected.
        roi.setAcceptedMouseButtons(QtCore.Qt.LeftButton)
        roi.sigClicked.connect(lambda r, *_: self._select(self.rois.index(r)))
        self.vb.addItem(roi)
        # snapshot AFTER every geometry call above, so it describes the
        # rectangle as the operator first sees it
        roi._src_box = src
        roi._built_corners = self._corner_ints(roi)
        self.rois.append(roi)
        return roi

    def _add_roi(self, b, angle: float = 0.0, src=None):
        return self._add_roi_geometry(b.x0, b.y0, b.width, b.height, angle,
                                      src=src)

    def _add_roi_from_corners(self, corners, src=None):
        """Rebuild the tilted rectangle its four corners specify, exactly.

        The corners are the full description of the drawn shape. The stored
        axis-aligned box is only the shape's crop envelope: an envelope is
        the projection of the tilted rectangle, so feeding its width and
        height back as the rectangle's side lengths inflates the frame by
        roughly b*sin(tilt) per axis -- tens of pixels on a real scan, which
        read as the box sitting slightly off the print.

        The angle is used raw, not folded into +/-45 degrees. `a` is measured
        along the first edge, so the angle has to describe that same edge; a
        fold would keep one and turn the other, and the rectangle would come
        back with its sides swapped."""
        pts = [self._pt_to_view(px, py) for px, py in corners]
        cx = sum(p[0] for p in pts) / 4.0
        cy = sum(p[1] for p in pts) / 4.0
        a = math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])   # TL->TR
        b = math.hypot(pts[3][0] - pts[0][0], pts[3][1] - pts[0][1])   # TL->BL
        ang = math.degrees(math.atan2(pts[1][1] - pts[0][1],
                                      pts[1][0] - pts[0][0]))
        return self._add_roi_geometry(cx - a / 2.0, cy - b / 2.0, a, b, ang,
                                      src=src)

    def _select(self, i: int) -> None:
        self.sel = max(0, min(i, len(self.rois) - 1))
        self._restyle()
        self._refresh_status()

    def _restyle(self) -> None:
        for i, r in enumerate(self.rois):
            r.setPen(PEN_SEL if i == self.sel else PEN_IDLE)

    @staticmethod
    def _angle_of(roi) -> float:
        """ROI rotation in degrees, across pyqtgraph versions that expose it
        as a method or only inside the state dict."""
        a = getattr(roi, "angle", None)
        if callable(a):
            return float(a())
        st = roi.state
        st = st() if callable(st) else st
        return float(st["angle"])

    def _roi_corners(self, roi):
        """The four corners of the ROI as drawn, turning included.

        pos() and size() describe the rectangle before rotation, so on their
        own they lose any angle the user has dialled in. Mapping the corners
        through the item's transform keeps it."""
        s = roi.size()
        pts = [QtCore.QPointF(0, 0), QtCore.QPointF(s.x(), 0),
               QtCore.QPointF(s.x(), s.y()), QtCore.QPointF(0, s.y())]
        out = []
        for p in pts:
            m = roi.mapToParent(p)
            out.append([float(m.x()), float(m.y())])
        return out

    def _corner_ints(self, roi):
        """The ROI's corners rounded exactly as they would be stored.

        Comparison happens on these, not on the floats, because rounding is
        what actually reaches the project file: a rectangle whose corners
        round to the same integers writes the same box, and is not an edit."""
        return [(int(round(p[0])), int(round(p[1])))
                for p in self._roi_corners(roi)]

    def _is_edited(self, roi) -> bool:
        """Has this rectangle moved since it was drawn? A box you have just
        added always counts as edited; every other one is decided by
        comparing what is on screen with what was put there."""
        src = getattr(roi, "_src_box", None)
        if src is None:
            return True
        return self._corner_ints(roi) != getattr(roi, "_built_corners", None)

    def _roi_box(self, roi):
        """The corners this rectangle stands for, plus its upright envelope.

        The CORNERS are the measurement and the only thing step 4 reads: it
        turns the scan the way up approved here, measures the tilt from the
        corners in that frame, straightens, and cuts out the bounding box of
        the straightened quad. The upright box here is the quad's envelope in
        the UNSTRAIGHTENED frame, which is a different rectangle and is not
        what gets cropped. It is stored because Z zooms to it and later
        steps use it to find the print quickly.

        The angle is the print's tilt measured from the corners as seen
        (common.tilt_from_corners, always within +/-45), converted to the
        file frame: a mirrored view negates a tilt, a quarter turn does not
        change it. It is NOT the rectangle's own rotation, which after a
        quarter turn of the view includes that quarter turn."""
        c = self._roi_corners(roi)
        xs = [p[0] for p in c]
        ys = [p[1] for p in c]
        b = Box(int(round(min(xs))), int(round(min(ys))),
                int(round(max(xs))), int(round(max(ys))))
        b = b.clamp(self.shape[0], self.shape[1])
        seen, _keep = tilt_from_corners(c)
        fl, _rot = self._orient()
        b.angle = round(-seen if fl else seen, 4) + 0.0   # -> file-frame tilt
        b.corners = [[int(round(p[0])), int(round(p[1]))] for p in c]
        return b

    def _commit(self) -> None:
        """Copy the on-screen rectangles back into the project.

        An untouched rectangle puts its ORIGINAL box object back, not a
        rebuilt one. pyqtgraph draws right-angled rectangles and step 1's
        quad usually is not one, so rebuilding squares it off and shifts its
        corners; doing that on every navigation would quietly replace the
        whole archive's measurements with approximations of themselves. The
        list is still rebuilt each time, so boxes can be added and deleted.

        Nothing is committed for a scan that could not be read: its boxes
        stay exactly as stored."""
        if self._broken:
            return
        it = self.proj.items[self.names[self.idx]]
        new = []
        for r in self.rois:
            if not self._is_edited(r):
                new.append(r._src_box)              # untouched: keep as found
                continue
            v = self._roi_box(r)
            b = self._from_view(v)                  # back to file coordinates
            b.angle = v.angle
            if v.corners:
                f, rot = self._orient()
                b.corners = [
                    [c.x0, c.y0] for c in
                    (box_from_view(Box(int(p[0]), int(p[1]),
                                       int(p[0]) + 1, int(p[1]) + 1),
                                   self.orig_shape, f, rot)
                     for p in v.corners)]
            new.append(b)
            # this rectangle is now what the project says, so its own
            # baseline moves with it: committing twice is not two edits
            r._src_box = b
            r._built_corners = self._corner_ints(r)
        it.boxes = new

    # -- actions -----------------------------------------------------------
    def add_box(self) -> None:
        if self._broken:
            return
        h, w = self.shape
        b = Box(w // 4, h // 4, w * 3 // 4, h * 3 // 4)
        self._add_roi(b, src=None)   # src=None: always written on commit
        self._select(len(self.rois) - 1)

    def del_box(self) -> None:
        """Delete the selected box: the one last clicked (or reached with
        Tab), drawn in the selection colour."""
        if not self.rois:
            return
        self.vb.removeItem(self.rois.pop(self.sel))
        self.sel = max(0, self.sel - 1)
        self._restyle()
        self._refresh_status()

    def _pointer_in_view(self):
        """The mouse pointer in view (image) coordinates, or None."""
        try:
            views = self.vb.scene().views()
            if not views:
                return None
            gp = QtGui.QCursor.pos()
            scene_pt = views[0].mapToScene(
                views[0].viewport().mapFromGlobal(gp))
            return self.vb.mapSceneToView(scene_pt)
        except Exception:
            return None

    def turn(self, deg: float) -> None:
        """Turn the selected box about its own middle.

        For a box that is nearly right this beats dragging: a tenth of a degree
        is far below what the hand can do on a corner handle, and the box stays
        where it is instead of drifting."""
        if not self.rois:
            return
        r = self.rois[self.sel]
        s = r.size()
        r.setAngle(self._angle_of(r) + deg,
                   centerLocal=[s.x() / 2.0, s.y() / 2.0])
        self._refresh_status()

    def straighten(self) -> None:
        """Put the box back square to the image."""
        if not self.rois:
            return
        r = self.rois[self.sel]
        s = r.size()
        r.setAngle(0.0, centerLocal=[s.x() / 2.0, s.y() / 2.0])
        self._refresh_status()

    def nudge(self, dx: int, dy: int, edge: bool) -> None:
        if not self.rois:
            return
        r = self.rois[self.sel]
        if edge:
            s = r.size()
            r.setSize([max(1, s.x() + dx), max(1, s.y() + dy)])
        else:
            p = r.pos()
            r.setPos([p.x() + dx, p.y() + dy])
        self._refresh_status()

    def _edge_near_pointer(self, roi):
        """Which of the four edges the pointer is closest to: lef/rig/top/bot.

        The pointer rather than a stored selection, because pyqtgraph has no
        notion of a 'current edge' -- a handle is only current while it is
        being dragged. Comparing distances to the four edge MIDPOINTS, mapped
        out of the ROI's own frame, keeps this right for a turned box, where
        'the top edge' is no longer the topmost one on screen.

        Returns None if the pointer cannot be located, so the caller can do
        nothing rather than guess an edge.
        """
        pt = self._pointer_in_view()
        if pt is None:
            return None
        w = float(roi.size().x())
        h = float(roi.size().y())
        mids = {"lef": (0.0, h / 2.0), "rig": (w, h / 2.0),
                "top": (w / 2.0, 0.0), "bot": (w / 2.0, h)}
        best, best_d = None, None
        for name, (lx, ly) in mids.items():
            p = roi.mapToParent(QtCore.QPointF(lx, ly))
            d = (p.x() - pt.x()) ** 2 + (p.y() - pt.y()) ** 2
            if best_d is None or d < best_d:
                best, best_d = name, d
        return best

    def square_box(self) -> None:
        """Move the edge nearest the pointer until the box is square.

        The edge OPPOSITE the one that moves is pinned, so the three sides
        the operator did not point at stay exactly where they were and the
        angle is untouched: setSize with centerLocal keeps one local point
        fixed in parent coordinates, and a rectangle whose opposite edge is
        pinned cannot rotate.

        Nothing here knows whether the print really was square. A contact
        print from a 6x6 negative is 56 x 56 mm and this squares up a
        detection error; a print trimmed by hand is not square at all and
        this would take a real edge off it. That judgement is the operator's,
        which is why it is a key and not a rule.

        Squaring counts as an edit like any other, so the quad step 1
        measured is replaced by this rectangle for that box.
        """
        if not self.rois:
            return
        roi = self.rois[self.sel]
        w = int(round(float(roi.size().x())))
        h = int(round(float(roi.size().y())))
        if w <= 0 or h <= 0 or w == h:
            self.status.showMessage("already square", MSG_SHORT_MS)
            return
        side = self._edge_near_pointer(roi)
        if side is None:
            self.status.showMessage(
                "point at the edge to move, then press =", MSG_LONG_MS)
            return
        # the pinned point is the midpoint of the opposite edge, in ROI-local
        # coordinates; only the axis being changed matters to setSize
        if side in ("lef", "rig"):
            new, pin = (h, h), ([w, 0.0] if side == "lef" else [0.0, 0.0])
        else:
            new, pin = (w, w), ([0.0, h] if side == "top" else [0.0, 0.0])
        roi.setSize([new[0], new[1]], centerLocal=pin)
        moved = abs(h - w)
        self.status.showMessage(
            f"squared to {new[0]} × {new[1]} px — {side} edge moved "
            f"{moved} px", MSG_LONG_MS)
        self._refresh_status()

    def confirm_next(self) -> None:
        if self._broken:                # nothing on screen to confirm
            self.skip()
            return
        self._commit()
        it = self.proj.items[self.names[self.idx]]
        pending = self._pending()
        if pending is not None and it.boxes:
            it.rot = pending            # the suggested turn you just saw
        it.status = "confirmed" if it.boxes else "skipped"
        self.proj.save()
        if self.idx + 1 < len(self.names):
            self.load(self.idx + 1)
        else:
            self._refresh_title()
            QtWidgets.QMessageBox.information(
                self, "done", "last scan reached — everything is saved.\n\n"
                              + next_step(self.src_dir))

    def confirm_rest(self) -> None:
        """Ctrl+Enter: confirm this scan and every later scan still waiting
        ("auto"), each exactly as it would open, then save and close.

        Nothing is loaded or redrawn for the later scans, so their boxes go
        back untouched, byte for byte, exactly as an untouched box does on
        Enter. Their turn is the stored one, or the high-confidence suggested
        turn they would open with (_pending_for). Confirmed and skipped scans
        are not touched: those are decisions already made."""
        self._commit()
        cur = self.names[self.idx]
        later = [n for n in self.names[self.idx + 1:]
                 if self.proj.items[n].status == "auto"]
        todo = [cur] + later
        turned = sum(1 for n in todo if self.proj.items[n].boxes
                     and self._pending_for(n) is not None)
        empty = sum(1 for n in todo if not self.proj.items[n].boxes)
        earlier = sum(1 for n in self.names[:self.idx]
                      if self.proj.items[n].status == "auto")
        text = (f"Confirm {len(todo)} scan(s), from {cur} to the end, "
                f"each exactly as it would open?\n\n"
                f"{turned} of them take the suggested turn they open with.\n"
                f"{empty} have no box and are marked skipped.\n"
                f"Scans already confirmed or skipped are left as they are.")
        if earlier:
            text += (f"\n\n{earlier} scan(s) BEFORE this one are still not "
                     f"confirmed; step 4 will not crop them.")
        text += "\n\nThe window then saves and closes."
        ans = QtWidgets.QMessageBox.question(
            self, "confirm the rest", text,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No)
        if ans != QtWidgets.QMessageBox.Yes:
            return
        for n in todo:
            it = self.proj.items[n]
            p = self._pending_for(n)
            if p is not None and it.boxes:
                it.rot = p
            unreadable = n == cur and self._broken
            it.status = ("confirmed" if it.boxes and not unreadable
                         else "skipped")
        self.proj.save()
        print(paint(f"    confirmed {count(len(todo) - empty, 'scan')} and "
                    f"skipped {empty} with Ctrl+Enter, from {cur} to the "
                    f"end", "32"))
        self.close()

    def skip(self) -> None:
        self._commit()
        self.proj.items[self.names[self.idx]].status = "skipped"
        self.proj.save()
        if self.idx + 1 < len(self.names):
            self.load(self.idx + 1)
        else:
            self._refresh_title()

    def reset(self) -> None:
        """Discard every manual decision on this scan and restore the boxes,
        the flip, the turn and the status exactly as boxes_detect_borders.json
        recorded them.

        The on-screen edits are deliberately NOT committed first -- throwing
        them away is the whole point of the key. The detect file is re-read
        from disk each time, so a reset always answers to what step 1 last
        wrote, and status falls back to "auto" for the operator to confirm
        again."""
        name = self.names[self.idx]
        if not self.reset_path or not os.path.isfile(self.reset_path):
            self.status.showMessage(
                "reset needs json files/boxes_detect_borders.json",
                MSG_ERROR_MS)
            return
        snap = Project.load(self.reset_path)
        if name not in snap.items:
            self.status.showMessage(
                f"{name}: not in boxes_detect_borders.json", MSG_ERROR_MS)
            return
        self.proj.items[name] = snap.items[name]
        self._declined.discard(name)    # the suggestion comes back with it
        self.proj.save()
        self.load(self.idx)

    def _reorient(self, flip: bool = False, turn: int = 0) -> None:
        """Change how this scan will be written, and redraw to match."""
        if self._broken:
            return
        self._commit()
        it = self.proj.items[self.names[self.idx]]
        if flip:
            it.flip = not it.flip
        if turn:
            # turn on from what is on screen, a pending suggestion included;
            # from here on the turn is the operator's own
            it.rot = (self._orient()[1] + turn) % 360
            self._declined.add(self.names[self.idx])
        self.proj.save()

        disp, self.orig_shape = self._read(self.names[self.idx])
        f, rot = self._orient()
        disp = orient_image(disp, f, rot)
        self.shape = view_shape(self.orig_shape, rot)
        self.img.setImage(disp, autoLevels=False, levels=(0, 255))
        self.vb.autoRange()

        sel = self.sel
        self._clear_rois()
        # redrawn in the NEW view frame, so each rectangle takes a fresh
        # baseline there: turning the view is not an edit to any box
        self._draw_boxes(it.boxes)
        self._select(min(sel, max(0, len(self.rois) - 1)))
        self._refresh_title()

    # -- keyboard ----------------------------------------------------------
    def eventFilter(self, obj, ev):
        """Catch keys for the whole application.

        pyqtgraph's GraphicsView passes key presses straight to its scene and
        they do not reliably reach the window, so binding them to the window
        alone leaves some keys dead depending on what has focus. Filtering at
        the application level sidesteps the focus chain entirely."""
        if ev.type() == QtCore.QEvent.KeyPress:
            if QtWidgets.QApplication.activeModalWidget() is None:
                if self._handle_key(ev.key(), ev.modifiers()):
                    return True
        return False

    def keyPressEvent(self, ev) -> None:
        if not self._handle_key(ev.key(), ev.modifiers()):
            super().keyPressEvent(ev)

    def _handle_key(self, k, mods) -> bool:
        step = 10 if mods & QtCore.Qt.ShiftModifier else 1
        edge = bool(mods & QtCore.Qt.ControlModifier)

        if self._loading:
            # A load is in flight and processEvents() has just handed us this
            # key from the middle of it. Swallow it: acting now would commit
            # the rectangles of one scan against the identity of another.
            return True

        if k in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter) and edge:
            self.confirm_rest()
        elif k in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter,
                   QtCore.Qt.Key_Space):
            self.confirm_next()
        elif k == QtCore.Qt.Key_N:
            self._commit(); self.proj.save(); self.load(self.idx + 1)
        elif k == QtCore.Qt.Key_P:
            self._commit(); self.proj.save(); self.load(self.idx - 1)
        elif k == QtCore.Qt.Key_A:
            self.add_box()
        elif k == QtCore.Qt.Key_D:
            self.del_box()
        elif k == QtCore.Qt.Key_R:
            self.reset()
        elif k == QtCore.Qt.Key_V:
            self._reorient(flip=True)
        elif k == QtCore.Qt.Key_Period:
            self._reorient(turn=90)
        elif k == QtCore.Qt.Key_Comma:
            self._reorient(turn=-90)
        elif k == QtCore.Qt.Key_S and not edge:
            self.skip()
        elif k == QtCore.Qt.Key_S and edge:
            self._commit(); self.proj.save()
            self.status.showMessage("saved", MSG_SHORT_MS)
        elif k == QtCore.Qt.Key_Tab:
            self._select((self.sel + 1) % max(1, len(self.rois)))
        elif k == QtCore.Qt.Key_F:
            self.vb.autoRange()
        elif k == QtCore.Qt.Key_Z and self.rois:
            b = self._roi_box(self.rois[self.sel])
            m = max(ZOOM_MARGIN_MIN, b.width // ZOOM_MARGIN)
            self.vb.setRange(QtCore.QRectF(b.x0 - m, b.y0 - m,
                                           b.width + 2 * m, b.height + 2 * m))
        elif k == QtCore.Qt.Key_Left:
            self.nudge(-step, 0, edge)
        elif k == QtCore.Qt.Key_Right:
            self.nudge(step, 0, edge)
        elif k == QtCore.Qt.Key_Up:
            self.nudge(0, -step, edge)
        elif k == QtCore.Qt.Key_Down:
            self.nudge(0, step, edge)
        elif k in (QtCore.Qt.Key_Q, QtCore.Qt.Key_BraceLeft):
            ctrl = bool(mods & QtCore.Qt.ControlModifier)
            self.turn(-ROT_STEP * (10 if ctrl else 1))
        elif k in (QtCore.Qt.Key_E, QtCore.Qt.Key_BraceRight):
            ctrl = bool(mods & QtCore.Qt.ControlModifier)
            self.turn(ROT_STEP * (10 if ctrl else 1))
        elif k == QtCore.Qt.Key_0:
            self.straighten()
        elif k in (QtCore.Qt.Key_Equal, QtCore.Qt.Key_Plus):
            self.square_box()
        elif k == QtCore.Qt.Key_Escape:
            self.close()
        else:
            return False
        return True

    def closeEvent(self, ev) -> None:
        QtWidgets.QApplication.instance().removeEventFilter(self)
        self._commit()
        self.proj.save()
        self._pool.shutdown(wait=False)
        ev.accept()


# --------------------------------------------------------------------------
# the step
# --------------------------------------------------------------------------

def next_step(root) -> str:
    """The command for step 4, as every step names the next one."""
    return f"\nnext:  python deskew_crop.py{root_arg(root)}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Stage 2 step 3 of 4 — approve the box, turn and "
                    "way up")
    ap.add_argument("root", nargs="?", default=".",
                    help="project root: the folder holding the scans "
                         "(default: this folder)")
    ap.add_argument("--flip", action="store_true",
                    help="flip the WHOLE archive. This is a STATE, not a "
                         "toggle: with the flag the archive flip is on, "
                         "without it, off. Stored in the project file, so "
                         "deskew_crop.py picks it up from there. V still "
                         "flips one scan at a time")
    ap.add_argument("--no-suggestions", action="store_true",
                    help="ignore json files/orientation_suggestions.json "
                         "this time: no scan opens turned by a suggestion")
    args = ap.parse_args(argv)
    console_setup()

    root = pipeline_root(args.root)
    check_scans_present(root)
    heading("STAGE 2 step 3  -  approve the box, turn and way up")

    phase(1, 2, "the project file")
    # resume a half-finished session, else start from the fresh detections;
    # whatever was loaded, the decisions are always saved to boxes_review.
    # The detect file stays reachable for the R (reset) key.
    proj = load_project(root)
    if proj is None:
        error(f"no project file in {os.path.join(root, JSON_DIR)}")
        print(paint(f"\nrun step 1 first:\n    python detect_borders.py"
                    f"{root_arg(root)}", "1"))
        return 1
    proj.out_path = stage_json(root, STAGE_REVIEW)
    if proj.path != proj.out_path:
        info(f"reviewing {os.path.basename(proj.path)} -> "
             f"{os.path.basename(proj.out_path)}")
    else:
        info(f"resuming {os.path.basename(proj.path)}")
    if proj.added:
        warn_list(f"    ! {count(len(proj.added), 'scan')} measured by "
                  f"step 1 after the review began "
                  f"{'is' if len(proj.added) == 1 else 'are'} added, not yet "
                  f"confirmed:", proj.added)
    reset_path = stage_json(root, STAGE_DETECT)
    if not os.path.isfile(reset_path):
        warn("    ! boxes_detect_borders.json is gone: the R (reset) "
             "key has nothing to reset to")

    # --flip states what the archive flip SHOULD BE rather than toggling it,
    # so `review.py --flip` always continues flipped and `review.py` always
    # continues unflipped. V is unaffected and still flips one scan.
    want = bool(args.flip)
    if proj.flip != want:
        done = proj.counts()
        n_done = done.get("confirmed", 0) + done.get("skipped", 0)
        proj.flip = want
        proj.save()
        print(paint(f"    archive flip is now {'ON' if want else 'OFF'} "
                    f"(stored in {proj.out_path})", "1"))
        if n_done:
            warn(f"    ! {count(n_done, 'scan')} "
                 f"{'was' if n_done == 1 else 'were'} already decided under "
                 f"the other setting.\n"
                 f"      Their per-scan V decisions are untouched, but the "
                 f"archive-wide flip has\n"
                 f"      changed for all of them. Run {SCRIPT} "
                 f"{'with --flip' if not want else 'without --flip'} to put "
                 f"it back.")
    else:
        print(paint(f"    archive flip is {'ON' if proj.flip else 'OFF'}",
                    "1"))

    if not proj.items:
        error("the project file is empty")
        return 1
    c = proj.counts()
    info(f"{count(len(proj.items), 'scan')}: {c.get('confirmed', 0)} "
         f"confirmed, {c.get('skipped', 0)} skipped, "
         f"{c.get('auto', 0)} still to decide")

    suggest = {}
    s_path = os.path.join(json_folder(root), SUGGEST_NAME)
    if args.no_suggestions:
        if os.path.isfile(s_path):
            info(f"--no-suggestions: {SUGGEST_NAME} is not used this time")
    elif os.path.isfile(s_path):
        try:
            with open(s_path, encoding="utf-8") as fh:
                suggest = dict(json.load(fh).get("items", {}))
        except (OSError, ValueError) as exc:
            warn(f"    ! {SUGGEST_NAME} unreadable ({exc}); no "
                 f"suggestions")
        n_auto = sum(1 for n, s in suggest.items()
                     if s.get("confidence") == "high" and n in proj.items
                     and proj.items[n].status == "auto"
                     and not proj.items[n].rot
                     and int(s.get("rot_flipped" if proj.flipped(
                         proj.items[n]) else "rot", 0)) % 360)
        opens = "one opens" if n_auto == 1 else "ones open"
        info(f"way-up suggestions: {count(len(suggest), 'scan')}; "
             f"{n_auto} unconfirmed {opens} turned the suggested way")

    phase(2, 2, "the review window")
    info("the window is open; Esc saves and closes it")
    app = QtWidgets.QApplication(sys.argv[:1])
    w = Reviewer(proj, root, reset_path, suggest)
    w.show()
    code = app.exec_() if hasattr(app, "exec_") else app.exec()
    c = proj.counts()
    print(paint(f"\nsaved {proj.out_path}", "32"))
    info(f"{c.get('confirmed', 0)} confirmed, {c.get('skipped', 0)} skipped, "
         f"{c.get('auto', 0)} still to decide")
    print(paint(next_step(root), "1"))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
