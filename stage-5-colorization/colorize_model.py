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
# colorize_model.py  --  STAGE 5, step 1 of 2: colour from the model
#
# STAGE 5 colorizes the B&W prints: a model paints each of them, and only
# the model's colour is kept, laid on the print's own lightness:
#
#     colorize_model.py        step 1 of 2   colour from the model
#     transfer_color.py        step 2 of 2   the colour on the print
#     calibrate_color.py       sets step 2's colour numbers (not a step)
#     common.py                shared code, not a step
#     parallel.py              shared worker-count policy, not a step
#     colorize_by_hand.json    this step's graph, for ComfyUI's own page
#
# Requirements:
#     pip install numpy tifffile opencv-python [imagecodecs]
#     common.py beside this file
#     ComfyUI running on this computer, with the model files and the GGUF
#     custom node below
#
# Works on any folder of prints -- Stage 4's output, say; it reads nothing
# of any other stage or step. Only the B&W prints are colorized; every
# other print is left out and counted. The guides go to a folder of their
# own.
#
# WHICH PRINTS ARE B&W. A print stored with one picture channel (a second
# one is alpha) is B&W; a print with three is left alone, however grey it
# looks. Stage 4 writes every B&W print as one grey channel, and a colour
# print and the back of a print in RGB, so in its output this is exactly
# its own decision: no back and no colour print is ever colorized.
#
# THE MODEL. Qwen-Image-Edit-2511, quantized to GGUF (UNET_FILE), with the
# 8-step Lightning LoRA (LORA_FILE), its text encoder (CLIP_FILE) and VAE
# (VAE_FILE), run by ComfyUI on this computer at COMFY_URL. The step only
# talks to it there, so ComfyUI must be running, in a window of its own,
# and left open -- run_pipeline.bat starts it; by hand:
#
#     C:\venvs\comfy\Scripts\activate
#     cd C:\tools\ComfyUI
#     python main.py
#
# The model files sit in ComfyUI's models\diffusion_models, models\loras,
# models\text_encoders and models\vae; the GGUF loader is the ComfyUI-GGUF
# custom node. The step checks all of this before it starts.
#
# THE GUIDES. Every B&W print goes up to ComfyUI as an 8-bit grey picture
# at its own size, and ComfyUI brings it to MODEL_MP megapixels, whatever
# the print's size (Lanczos, its sides multiples of SIDE_STEP), before the
# model sees it. With it goes PROMPT -- plus the --context sentence, and
# the print's own line from colorize_hints.txt when it has one, after
# PROMPT's first sentence. The model paints from noise, the print being
# its reference. The model's colour picture, the same size, is the print's
# guide:
#
#     colorize_guides/<name>_seed<N>.png
#
# The guide is not the result: the model redraws the whole picture, at low
# resolution, and may change a face or a detail. Step 2 takes only its
# colour. Every print goes up to ComfyUI under one name, so its input
# folder does not fill up; ComfyUI keeps its own copies of the guides in
# its temp folder, which it empties each time it starts.
#
# ONE RUN AT A TIME. As every print goes up under one name, two runs of
# this step on one ComfyUI would hand each other's prints to the model. So
# a run holds a lock (LOCK_FILE, in the system's temp folder, one for each
# COMFY_URL) until it ends, however it ends; a second run stops at once
# and says so. Holding it, the run takes off ComfyUI's queue whatever an
# earlier run left there -- one whose window was closed goes on in ComfyUI
# with its print, and the new run would wait behind it.
#
# A GUIDE COUNTS AS MADE when it is there and newer than its print, so a
# print written again (Stage 4 run again, say) gets a new guide. A guide
# is not made again when the prompt, a hint or a setting changes: --redo
# does that, with --only for one print. The step names the prints with a
# hint whose guide is older than colorize_hints.txt.
#
# THE RESULTS AS THEY COME (--apply, as run_pipeline.bat runs it). Each
# guide, once made, goes at once to step 2 (transfer_color.py beside this
# file), which writes the print's full-size result into colorized/ (or
# the folder --out names) in one background process with one thread, while
# the model works on the next print; so the results appear one by one, and
# a run that is stopped keeps every result of the guides it made. A result
# that fails is named, and step 2 run on its own writes it later.
#
# HINTS (optional). colorize_hints.txt beside the prints gives a print a
# sentence of its own, one print a line; '#' starts a comment:
#
#     raw0012: The car is dark red.
#     raw0047: The walls of the house are white.
#
# STOPPING AND GOING ON. A guide already made is not made again (unless
# --redo), so the step can be stopped at any time -- Ctrl+C, or closing the
# window -- and run again later: it goes on from the first print without a
# guide. On Ctrl+C it also stops ComfyUI's work on the current print; after
# a closed window, the next run does (ONE RUN AT A TIME). A print that
# fails, or takes longer than TIMEOUT_MIN, is reported and the step goes
# on with the next; the next run tries it again. ComfyUI answering too
# slowly (CALL_SEC, CALL_TRIES) fails the print, not the run; only
# ComfyUI not answering at all stops the run.
#
# Usage:
#     python colorize_model.py [ROOT] [options]
#
#     ROOT                  project root: the folder of the prints (default:
#                           this folder)
#     --seed N              the seed of the guides (default: SEED in
#                           common.py); another seed gives other colours
#     --only NAME [NAME ...]
#                           only these prints (the extension may be left out)
#     --context TEXT        a sentence for every print of this run, e.g.
#                           "It was taken in the 1960s at the seaside."
#     --redo                make the guides again, even those already there
#     --apply               write each print's full-size result (step 2) as
#                           soon as its guide is made
#     --out DIR             with --apply: the folder for the results
#                           (default: ROOT\colorized); never ROOT
#     --count               only count the guides still to make: exit code 0
#                           when there are some, 3 when there are none
#                           (run_pipeline.bat starts ComfyUI only for some)
#
#     python colorize_model.py D:\archive\restored
#     python colorize_model.py D:\archive\restored --redo --only raw0012
# ==========================================================================
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import ProcessPoolExecutor

try:
    import cv2
    import numpy as np
except ModuleNotFoundError as exc:
    sys.exit(f"missing package: {exc.name} (see Requirements above)")

try:
    from common import (SEED, Progress, check_scans_present, console_setup,
                        count, elapsed, error, error_list, grey_of, guide_dir,
                        guide_name, heading, info, is_bw, newer, paint, phase,
                        pick_prints, pipeline_root, read_page, resolve,
                        result_dir, root_arg, stem, warn, warn_list)
except ModuleNotFoundError as exc:
    sys.exit("common.py must sit beside this script" if exc.name == "common"
             else f"missing package: {exc.name} (see Requirements above)")

# --------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------
# --- general ---------------------------------------------------------------
SCRIPT = "colorize_model.py"
HINTS_FILE = "colorize_hints.txt"   # beside the prints, optional

# --- ComfyUI ---------------------------------------------------------------
COMFY_URL = "http://127.0.0.1:8188"  # run_pipeline.bat has its own copy
UNET_FILE = "qwen-image-edit-2511-Q4_K_M.gguf"
LORA_FILE = "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors"
CLIP_FILE = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
VAE_FILE = "qwen_image_vae.safetensors"
UPLOAD_NAME = "stage5_print.png"    # every print goes up under this name,
UPLOAD_DIR = "stage5"               # in this subfolder of ComfyUI's input
POLL_SEC = 5            # how often the step asks ComfyUI whether it is done
TIMEOUT_MIN = 60        # a print that takes longer counts as failed
CALL_SEC = 60           # one request to ComfyUI waits this long ...
CALL_TRIES = 3          # ... and a question is asked this many times
LOCK_FILE = "stage5_comfyui_{}.lock"    # in the temp folder; {} from
                                        # COMFY_URL

# --- the model -------------------------------------------------------------
STEPS = 8               # the Lightning LoRA's steps
SHIFT = 3.1             # ModelSamplingAuraFlow, as ComfyUI's own workflow
MODEL_MP = 2.0          # megapixels of the picture the model sees
UPSCALE = "lanczos"     # how ComfyUI brings the print to that size
SIDE_STEP = 16          # the sides of the picture the model sees are
                        # multiples of this
PROMPT = ("Colorize this black-and-white photograph with natural, realistic "
          "and slightly muted colours, like an ordinary colour photograph of "
          "the same time, not vivid or saturated. Where the colour of "
          "something cannot be told from the photograph, keep it neutral "
          "(white or grey) rather than giving it a bright colour. "
          "Keep everything else exactly as it is: the composition, the "
          "people, their faces and expressions, and every detail.")


# --------------------------------------------------------------------------
# the prompt
# --------------------------------------------------------------------------

def read_hints(root: str):
    """({print stem, lower case: sentence}, [lines that could not be
    read])."""
    path = os.path.join(resolve(root), HINTS_FILE)
    hints, bad = {}, []
    if not os.path.isfile(path):
        return hints, bad
    with open(path, encoding="utf-8-sig") as f:
        for n, raw in enumerate(f, 1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            name, sep, text = line.partition(":")
            if not sep or not name.strip() or not text.strip():
                bad.append(f"line {n}: {raw.strip()}")
                continue
            hints[stem(name.strip())] = text.strip()
    return hints, bad


def prompt_for(name: str, context: str, hints: dict) -> str:
    """PROMPT with the run's context and the print's hint after its first
    sentence."""
    parts = [s.strip() for s in
             (context or "", hints.get(stem(name), ""))
             if s and s.strip()]
    if not parts:
        return PROMPT
    extra = " ".join(p if p.endswith((".", "!", "?")) else p + "."
                     for p in parts)
    first, sep, rest = PROMPT.partition(". ")
    return f"{first}. {extra} {rest}" if sep else f"{PROMPT} {extra}"


# --------------------------------------------------------------------------
# ComfyUI
# --------------------------------------------------------------------------

def workflow(image: str, prompt: str, seed: int) -> dict:
    """The graph: the GGUF model with the Lightning LoRA, the print brought
    to MODEL_MP megapixels and used as the reference at the output's own
    size (so nothing is cropped or shifted), euler, STEPS steps, CFG 1."""
    return {
        "1": {"class_type": "UnetLoaderGGUF",
              "inputs": {"unet_name": UNET_FILE}},
        "2": {"class_type": "ModelSamplingAuraFlow",
              "inputs": {"model": ["1", 0], "shift": SHIFT}},
        "3": {"class_type": "CFGNorm",
              "inputs": {"model": ["2", 0], "strength": 1.0}},
        "4": {"class_type": "LoraLoaderModelOnly",
              "inputs": {"model": ["3", 0], "lora_name": LORA_FILE,
                         "strength_model": 1.0}},
        "5": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": CLIP_FILE, "type": "qwen_image",
                         "device": "default"}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_FILE}},
        "7": {"class_type": "LoadImage", "inputs": {"image": image}},
        "8": {"class_type": "ImageScaleToTotalPixels",
              "inputs": {"image": ["7", 0], "upscale_method": UPSCALE,
                         "megapixels": MODEL_MP,
                         "resolution_steps": SIDE_STEP}},
        "9": {"class_type": "TextEncodeQwenImageEditPlus",
              "inputs": {"clip": ["5", 0], "image1": ["8", 0],
                         "prompt": prompt}},
        "10": {"class_type": "TextEncodeQwenImageEditPlus",
               "inputs": {"clip": ["5", 0], "image1": ["8", 0],
                          "prompt": ""}},
        "13": {"class_type": "VAEEncode",
               "inputs": {"pixels": ["8", 0], "vae": ["6", 0]}},
        "17": {"class_type": "ReferenceLatent",
               "inputs": {"conditioning": ["9", 0], "latent": ["13", 0]}},
        "18": {"class_type": "ReferenceLatent",
               "inputs": {"conditioning": ["10", 0], "latent": ["13", 0]}},
        "11": {"class_type": "FluxKontextMultiReferenceLatentMethod",
               "inputs": {"conditioning": ["17", 0],
                          "reference_latents_method": "index_timestep_zero"}},
        "12": {"class_type": "FluxKontextMultiReferenceLatentMethod",
               "inputs": {"conditioning": ["18", 0],
                          "reference_latents_method": "index_timestep_zero"}},
        "14": {"class_type": "KSampler",
               "inputs": {"model": ["4", 0], "positive": ["11", 0],
                          "negative": ["12", 0], "latent_image": ["13", 0],
                          "seed": seed, "steps": STEPS, "cfg": 1.0,
                          "sampler_name": "euler", "scheduler": "simple",
                          "denoise": 1.0}},
        "15": {"class_type": "VAEDecode",
               "inputs": {"samples": ["14", 0], "vae": ["6", 0]}},
        "16": {"class_type": "PreviewImage",
               "inputs": {"images": ["15", 0]}},
    }


NEEDED = {      # node: (an input it must have, the file that input must
                # offer, or None); None: only the node
    "UnetLoaderGGUF": ("unet_name", UNET_FILE),
    "LoraLoaderModelOnly": ("lora_name", LORA_FILE),
    "CLIPLoader": ("clip_name", CLIP_FILE),
    "VAELoader": ("vae_name", VAE_FILE),
    "TextEncodeQwenImageEditPlus": None,
    "ReferenceLatent": None,
    "FluxKontextMultiReferenceLatentMethod": None,
    "CFGNorm": None,
    "ImageScaleToTotalPixels": ("resolution_steps", None),
}


class ComfyError(Exception):
    pass


class ComfyGone(ComfyError):
    """ComfyUI does not answer at all."""


def _slow(exc) -> bool:
    """The request reached ComfyUI, which was too slow to answer."""
    why = getattr(exc, "reason", exc)
    return isinstance(why, (TimeoutError, socket.timeout))


def _call(path: str, data=None, headers=None, timeout=CALL_SEC) -> bytes:
    """One request. A question (no data) is asked again when ComfyUI is too
    slow, CALL_TRIES times in all; a slow answer is a ComfyError (this
    print fails), no answer at all a ComfyGone (the run stops)."""
    tries = 1 if data is not None else CALL_TRIES
    for _ in range(tries):
        req = urllib.request.Request(COMFY_URL + path, data=data,
                                     headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:2000]
            raise ComfyError(f"HTTP {exc.code} on {path}: {body}") from None
        except (urllib.error.URLError, OSError) as exc:
            if not _slow(exc):
                raise ComfyGone(f"no answer from ComfyUI at {COMFY_URL} "
                                f"({getattr(exc, 'reason', exc)})") from None
    raise ComfyError(f"ComfyUI did not answer {path} within {timeout} s"
                     + (f" ({tries} tries)" if tries > 1 else ""))


def _json(path: str, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    hdr = {} if payload is None else {"Content-Type": "application/json"}
    raw = _call(path, data, hdr)
    return json.loads(raw) if raw.strip() else None


def check_comfy():
    """(None, a line on ComfyUI) when it answers and has every node and
    file; otherwise (what is missing, None)."""
    try:
        stats = _json("/system_stats") or {}
    except ComfyError:
        return (f"ComfyUI does not answer at {COMFY_URL}; start it first "
                f"(run_pipeline.bat\ndoes; by hand, see THE MODEL at the top "
                f"of {SCRIPT}), and leave it open"), None
    missing = []
    for node, need in NEEDED.items():
        try:
            spec = (_json("/object_info/" + node) or {}).get(node)
        except ComfyError:
            spec = None
        if not spec:
            missing.append(f"the node {node}" + (" (the ComfyUI-GGUF custom "
                                                 "node)" if "GGUF" in node
                                                 else ""))
            continue
        if need:
            field, fname = need
            inputs = spec.get("input") or {}
            entry = ((inputs.get("required") or {}).get(field)
                     or (inputs.get("optional") or {}).get(field))
            if not entry:
                missing.append(f"the input {field} of {node} (update "
                               f"ComfyUI)")
                continue
            if fname is None:
                continue
            try:
                # [[files]] in older nodes, ["COMBO", {"options": [files]}]
                choices = (entry[1].get("options", [])
                           if entry[0] == "COMBO" else entry[0])
            except (IndexError, KeyError, TypeError, AttributeError):
                choices = None
            if choices is None or not isinstance(choices, (list, tuple)):
                missing.append(f"the file list of {node} (not readable: "
                               f"update ComfyUI)")
            elif fname not in choices:
                missing.append(f"the file {fname}")
    if missing:
        return "ComfyUI is missing " + ", ".join(missing), None
    dev = (stats.get("devices") or [{}])[0]
    ver = stats.get("system", {}).get("comfyui_version", "?")
    return None, f"{dev.get('name', '?')}; ComfyUI {ver}"


def model_input(path: str) -> bytes:
    """The print as an 8-bit grey PNG at its own size."""
    pic = grey_of(read_page(path)[0])
    ok, buf = cv2.imencode(".png", np.round(pic * 255.0).astype(np.uint8))
    if not ok:
        raise OSError("could not encode the model's picture")
    return buf.tobytes()


def upload(png: bytes) -> str:
    """Put the picture in ComfyUI's input folder; its name for LoadImage."""
    bound = uuid.uuid4().hex
    parts = [f"--{bound}\r\nContent-Disposition: form-data; name=\"{k}\""
             f"\r\n\r\n{v}\r\n".encode()
             for k, v in (("overwrite", "true"), ("type", "input"),
                          ("subfolder", UPLOAD_DIR))]
    parts.append(f"--{bound}\r\nContent-Disposition: form-data; "
                 f"name=\"image\"; filename=\"{UPLOAD_NAME}\"\r\n"
                 f"Content-Type: image/png\r\n\r\n".encode() + png + b"\r\n")
    parts.append(f"--{bound}--\r\n".encode())
    got = json.loads(_call("/upload/image", b"".join(parts), {
        "Content-Type": f"multipart/form-data; boundary={bound}"}))
    sub = got.get("subfolder") or ""
    return f"{sub}/{got['name']}" if sub else got["name"]


def stop(pid: str) -> None:
    """Take the print off ComfyUI's queue, and stop it if it is running."""
    for path, payload in (("/queue", {"delete": [pid]}),
                          ("/interrupt", {"prompt_id": pid})):
        try:
            _json(path, payload)
        except ComfyError:
            pass


def hold_lock():
    """The open lock file of this run (keep it open), or None when another
    run holds it. The system lets go of it when the run ends, however it
    ends."""
    port = re.sub(r"\W+", "_", COMFY_URL.split("//", 1)[-1]).strip("_")
    fh = open(os.path.join(tempfile.gettempdir(), LOCK_FILE.format(port)),
              "a+")
    try:
        if os.name == "nt":
            import msvcrt
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    return fh


def clear_leftovers(image: str) -> int:
    """Take off ComfyUI's queue every job that colours `image` -- left by a
    run that was stopped without Ctrl+C; how many."""
    queue = _json("/queue") or {}
    found = 0
    for key in ("queue_running", "queue_pending"):
        for item in queue.get(key) or []:
            try:
                pid, graph = item[1], item[2]
                ours = any(node.get("class_type") == "LoadImage"
                           and node.get("inputs", {}).get("image") == image
                           for node in graph.values())
            except (IndexError, TypeError, AttributeError):
                continue
            if ours:
                stop(pid)
                found += 1
    return found


def run_one(image: str, prompt: str, seed: int,
            client: str) -> bytes:
    """Queue the graph, wait for it, return the guide's PNG bytes."""
    got = _json("/prompt", {"prompt": workflow(image, prompt, seed),
                            "client_id": client}) or {}
    if got.get("node_errors"):
        raise ComfyError(f"ComfyUI refused the graph: {got['node_errors']}")
    pid = got["prompt_id"]
    t0 = time.time()
    try:
        while True:
            hist = (_json("/history/" + pid) or {}).get(pid)
            if hist:
                status = hist.get("status", {})
                if status.get("status_str") == "error":
                    msgs = {m[0]: m[1] for m in status.get("messages", [])
                            if m and len(m) > 1}
                    if "execution_error" in msgs:
                        why = msgs["execution_error"].get(
                            "exception_message", "").strip() or "an error"
                    elif "execution_interrupted" in msgs:
                        why = "stopped in ComfyUI"
                    else:
                        why = "an unknown error"
                    raise ComfyError(f"ComfyUI failed: {why}")
                imgs = (hist.get("outputs", {}).get("16", {})
                        .get("images", []))
                if imgs:
                    q = urllib.parse.urlencode(
                        {"filename": imgs[0]["filename"],
                         "subfolder": imgs[0].get("subfolder", ""),
                         "type": imgs[0].get("type", "temp")})
                    return _call("/view?" + q, timeout=120)
                if status.get("completed"):
                    raise ComfyError("ComfyUI finished without a picture")
            if time.time() - t0 > TIMEOUT_MIN * 60:
                stop(pid)
                raise ComfyError(f"no guide after {TIMEOUT_MIN} min")
            time.sleep(POLL_SEC)
    except KeyboardInterrupt:
        stop(pid)
        raise
    except ComfyGone:
        raise
    except ComfyError:
        stop(pid)                         # too slow to answer: not left there
        raise


# --------------------------------------------------------------------------
# the results as they come (--apply)
# --------------------------------------------------------------------------

class Results:
    """Step 2's full-size results, written in one background process while
    the model works on the next print."""

    def __init__(self, root: str, out_dir: str):
        import transfer_color
        self.step2 = transfer_color
        self.root = resolve(root)
        self.odir = out_dir
        os.makedirs(self.odir, exist_ok=True)
        self.pool = ProcessPoolExecutor(max_workers=1)
        self.pending, self.failed, self.made = [], [], 0

    def add(self, name: str, guide: str) -> None:
        """Write `name`'s result from `guide`, in the background."""
        dst = os.path.join(self.odir, os.path.splitext(name)[0] + ".tif")
        task = (name, os.path.join(self.root, name), guide, dst, True,
                False)                    # False: one thread, the GPU's
        self.pending.append((name, self.pool.submit(self.step2._job, task)))

    def collect(self, bar, wait: bool = False) -> None:
        """Report the results finished (all of them, with `wait`)."""
        for item in list(self.pending):
            name, fut = item
            if not (wait or fut.done()):
                continue
            self.pending.remove(item)
            try:
                _, line, err = fut.result()
            except Exception as exc:     # the process itself died
                line, err = None, f"{type(exc).__name__}: {exc}"
            if err:
                self.failed.append(f"{name}: {err}")
                bar.message(paint(f"    ! {name}: result failed", "33"))
            else:
                self.made += 1
                bar.message(paint(f"    {name}: result written ({line})",
                                  "90"))

    def finish(self, bar) -> None:
        """Wait for the results still being written, and report them."""
        if self.pending:
            bar.message(paint(f"    waiting for "
                              f"{count(len(self.pending), 'result')}...",
                              "90"))
        self.collect(bar, wait=True)
        self.pool.shutdown()

    def drop(self) -> None:
        """Stop at once (Ctrl+C): results not started are not written."""
        self.pool.shutdown(wait=False, cancel_futures=True)


# --------------------------------------------------------------------------
# the step
# --------------------------------------------------------------------------

def run(root, args) -> int:
    t0 = time.time()
    gdir = guide_dir(root)

    phase(1, 3, "the prints")
    names, missing, clashes = pick_prints(root, args.only)
    if missing:
        warn_list(f"    ! {count(len(missing), 'name')} given with --only "
                  f"not found:", missing)
    if clashes:
        warn_list(f"    ! {count(len(clashes), 'print')} left out: the same "
                  f"name but for the extension or the case:", clashes)
    if args.only and not names:
        error("none of the prints given with --only is in the folder")
        return 1
    bw, other, unreadable = [], [], []
    for name in names:
        one, err = is_bw(os.path.join(root, name))
        if err:
            unreadable.append(f"{name}: {err}")
        elif one:
            bw.append(name)
        else:
            other.append(name)

    def guide(n):
        return os.path.join(gdir, guide_name(n, args.seed))

    made_before = [n for n in bw if newer(guide(n), os.path.join(root, n))]
    stale = [n for n in bw if os.path.isfile(guide(n))
             and n not in made_before]
    todo = bw if args.redo else [n for n in bw if n not in made_before]
    info(f"{count(len(bw), 'B&W print')} (one channel); left out: "
         f"{count(len(other), 'print')} in RGB (colour prints, backs)")
    info(f"seed {args.seed}: {count(len(made_before), 'guide')} made "
         f"already, {len(todo)} to make" + (" (--redo: all again)"
                                            if args.redo else ""))
    if stale and not args.redo:
        warn_list(f"    ! {count(len(stale), 'guide')} older than "
                  f"{'its' if len(stale) == 1 else 'their'} print: made "
                  f"again", stale)
    hints, bad = read_hints(root)
    if hints or bad:
        used = sum(1 for n in todo if stem(n) in hints)
        info(f"{HINTS_FILE}: {count(len(hints), 'hint')}, {used} for the "
             f"prints to make")
        if bad:
            warn_list(f"    ! {count(len(bad), 'line')} of {HINTS_FILE} "
                      f"not read (name: sentence):", bad)
        hint_file = os.path.join(resolve(root), HINTS_FILE)
        older = [n for n in made_before if n not in todo
                 and stem(n) in hints and not newer(guide(n), hint_file)]
        if older:
            warn_list(f"    ! {count(len(older), 'print')} with a hint, "
                      f"made before {HINTS_FILE} last changed; if the "
                      f"hint is new, make the guide again with --redo "
                      f"--only NAME:", older)
    if args.context:
        info(f"context: {args.context}")
    if unreadable:
        warn_list(f"    ! {count(len(unreadable), 'print')} could not be "
                  f"read; left out:", unreadable)
    if args.count:
        print(paint(f"\n{count(len(todo), 'guide')} to make", "1"))
        return 0 if todo else 3
    if not todo:
        print(paint(f"\nnothing to make; the guides are in {gdir}", "32"))
        if bw:
            print(paint(f"\nnext:  python transfer_color.py"
                        f"{root_arg(root)}", "1"))
        return 2 if unreadable else 0

    phase(2, 3, "ComfyUI")
    lock = hold_lock()
    if lock is None:
        error(f"another run of this step is using ComfyUI at {COMFY_URL}; "
              f"wait for it to end, or close its window")
        return 1
    problem, line = check_comfy()
    if problem:
        error(problem)
        return 1
    info(line)
    image = f"{UPLOAD_DIR}/{UPLOAD_NAME}"
    try:
        gone = clear_leftovers(image)
    except ComfyError as exc:
        error(f"could not read ComfyUI's queue: {exc}")
        return 1
    if gone:
        info(f"took {count(gone, 'job')} of a stopped run off ComfyUI's "
             f"queue")

    phase(3, 3, "the guides" + (", and their results" if args.apply
                                    else ""))
    os.makedirs(gdir, exist_ok=True)
    results = None
    if args.apply:
        try:
            results = Results(root, args.out_dir)
        except ModuleNotFoundError as exc:
            error(f"--apply needs {exc.name}.py beside this script")
            return 1
    client = uuid.uuid4().hex
    failed, made = [], 0
    bar = Progress(len(todo), "    colorizing")
    try:
        for name in todo:
            t1 = time.time()
            dst = guide(name)
            try:
                png = model_input(os.path.join(root, name))
                if upload(png) != image:
                    raise ComfyError("ComfyUI stored the print under "
                                     "another name")
                png = run_one(image, prompt_for(name, args.context, hints),
                              args.seed, client)
                tmp = dst + ".tmp"
                with open(tmp, "wb") as f:
                    f.write(png)
                os.replace(tmp, dst)
                made += 1
                bar.message(paint(f"    {name}: {elapsed(t1)}", "90"))
                if results:
                    results.add(name, dst)
            except ComfyGone as exc:
                if results:
                    results.finish(bar)
                bar.close()
                error(f"{name}: {exc}\nComfyUI has stopped; start it again "
                      f"and run this step again to go on")
                return 1
            except Exception as exc:
                failed.append(f"{name}: {type(exc).__name__}: {exc}")
                bar.message(paint(f"    ! {name}: failed", "33"))
            if results:
                results.collect(bar)
            bar.update()
        if results:
            results.finish(bar)
    except KeyboardInterrupt:
        if results:
            results.drop()
        bar.close()
        warn(f"\nstopped: {count(made, 'guide')} made this run; run the "
             f"step again to go on")
        return 1
    bar.close()
    lock.close()

    print(paint(f"\nmade {count(made, 'guide')} in {gdir}", "32"))
    if failed:
        error_list(f"    {count(len(failed), 'print')} failed (the next "
                   f"run tries {'it' if len(failed) == 1 else 'them'} "
                   f"again):", failed)
    if results:
        print(paint(f"wrote {count(results.made, 'result')} in "
                    f"{results.odir}", "32"))
        if results.failed:
            error_list(f"    {count(len(results.failed), 'result')} could "
                       f"not be written (step 2 writes "
                       f"{'it' if len(results.failed) == 1 else 'them'} "
                       f"later):", results.failed)
    print(paint(f"    done in {elapsed(t0)}", "32"))
    nxt = ((" --apply" + (f' --out "{results.odir}"' if args.out else ""))
           if results else "")
    print(paint(f"\nnext:  python transfer_color.py{root_arg(root)}{nxt}",
                "1"))
    return 2 if (failed or unreadable or (results and results.failed)) \
        else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Stage 5 step 1 of 2 — colour from the model")
    ap.add_argument("root", nargs="?", default=".",
                    help="project root: the folder of the prints (default: "
                         "this folder)")
    ap.add_argument("--seed", type=int, default=SEED,
                    help=f"the seed of the guides (default {SEED}); "
                         f"another seed gives other colours")
    ap.add_argument("--only", nargs="+", metavar="NAME",
                    help="only these prints (the extension may be left out)")
    ap.add_argument("--context", default="",
                    help="a sentence for every print of this run")
    ap.add_argument("--redo", action="store_true",
                    help="make the guides again, even those already there")
    ap.add_argument("--apply", action="store_true",
                    help="write each print's full-size result (step 2) as "
                         "soon as its guide is made")
    ap.add_argument("--out", default=None, metavar="DIR",
                    help="with --apply: the folder for the results "
                         "(default: ROOT/colorized)")
    ap.add_argument("--count", action="store_true",
                    help="only count the guides still to make: exit code 0 "
                         "when there are some, 3 when there are none")
    args = ap.parse_args(argv)
    console_setup()
    if args.seed < 0:
        raise SystemExit(f"--seed must be 0 or more, not {args.seed}")
    if args.out and not args.apply:
        raise SystemExit("--out goes with --apply")

    root = pipeline_root(args.root)
    check_scans_present(root)
    args.out_dir = result_dir(root, args.out) if args.apply else None
    heading("STAGE 5 step 1  -  colour from the model")
    return run(root, args)


if __name__ == "__main__":
    raise SystemExit(main())
