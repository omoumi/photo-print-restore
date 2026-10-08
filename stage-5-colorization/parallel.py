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
# parallel.py  --  STAGE 5: how a step decides on a worker count
#                  (not a step)
#
# STAGE 5 colorizes the B&W prints: a model paints each of them, and only
# the model's colour is kept, laid on the print's own lightness:
#
#     colorize_model.py        step 1 of 2   colour from the model
#     transfer_color.py        step 2 of 2   the colour on the print
#     calibrate_color.py       sets step 2's colour numbers (not a step)
#     common.py                shared code, not a step
#     parallel.py              shared worker-count policy, not a step
#     colorize_by_hand.json    step 1's graph, for ComfyUI's own page
#
# Requirements:
#     pip install [psutil]        # optional; without it, free RAM is guessed
#
# Each scan is processed independently, so the work spreads across CPU
# cores. The worker count is the smaller of two limits, so a run never
# thrashes memory on large 16-bit scans:
#
#     cores  : os.cpu_count() - RESERVE_CORES   (one left for the OS and UI)
#     memory : MEM_FRACTION of free RAM / estimated peak RAM per worker
#
# Callers pass a `multiple`: how much memory their worker needs per byte of
# the scan file (see WORKER_MEM_MULTIPLE in each step).
#
# Results come back in INPUT ORDER whatever finishes first. A count of 1 runs
# everything in-process with no pool, which keeps small jobs simple and
# tracebacks readable. Every step accepts --workers N to override.
#
# Usage (from another script):
#     import parallel
#     parallel.add_workers_arg(ap)                       # adds --workers
#     per = parallel.estimate_bytes_per_worker(path, multiple=6.0)
#     n   = parallel.choose_workers(len(tasks), per, args.workers)
#     out = parallel.run(worker_fn, tasks, n, on_result=report,
#                        on_error=lambda task, exc: <a failed result>)
#
# worker_fn must be a top-level function so it can be pickled.
# ==========================================================================
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    import psutil
except ImportError:
    psutil = None

# --------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------
RESERVE_CORES = 1                     # cores left for the OS and the UI
MEM_FRACTION = 0.70                   # share of free RAM the workers may use
FALLBACK_FREE_RAM = 4 * 1024 ** 3     # assumed free RAM without psutil
DEFAULT_MULTIPLE = 8.0                # peak RAM per byte of the scan file
MIN_BYTES_PER_WORKER = 256 * 1024 ** 2   # never estimate below this


# --------------------------------------------------------------------------
# worker count
# --------------------------------------------------------------------------

def _free_ram_bytes():
    if psutil is not None:
        try:
            return int(psutil.virtual_memory().available)
        except Exception:
            pass
    return FALLBACK_FREE_RAM


def estimate_bytes_per_worker(sample_path, multiple=DEFAULT_MULTIPLE,
                              floor=MIN_BYTES_PER_WORKER):
    """Estimate the peak RAM one worker needs for one scan, from the size of
    a sample file.

    The estimate starts from the size ON DISK, so a strongly compressed
    source (LZW, deflate, ZSTD) decodes to several times that and the caller
    should raise `multiple` accordingly; `floor` is a sane minimum."""
    try:
        sz = os.path.getsize(sample_path)
    except Exception:
        return floor
    return max(floor, int(sz * multiple))


def choose_workers(n_items, bytes_per_item=None, requested=None,
                   reserve_cores=RESERVE_CORES):
    """Pick a worker count from cores AND memory, whichever limits more;
    `requested` (--workers) overrides both."""
    cores = max(1, (os.cpu_count() or 1) - reserve_cores)
    limit = cores
    if bytes_per_item and bytes_per_item > 0:
        mem_workers = max(1, int(MEM_FRACTION * _free_ram_bytes()
                                 / bytes_per_item))
        limit = min(limit, mem_workers)
    limit = min(limit, max(1, n_items))
    if requested is not None and requested > 0:
        limit = min(requested, max(1, n_items))
    return max(1, limit)


# --------------------------------------------------------------------------
# running the work
# --------------------------------------------------------------------------

def run(worker_fn, items, workers, on_result=None, on_error=None):
    """Map worker_fn over items and return results in INPUT ORDER.

    worker_fn(item) must be a top-level (picklable) function.  on_result(i, r)
    is called as each result arrives -- in COMPLETION order, but always with
    the item's own index i -- so a callback that prints should print i to stay
    readable; only the returned list is guaranteed to be in input order.

    on_error(item, exc) turns a failure that escaped worker_fn into a result
    of the same shape, which then goes through on_result like any other. The
    case it exists for is a worker process that dies outright -- killed for
    running out of memory on a huge scan, say. The pool is then broken and
    every item still waiting fails with it. Without on_error the first
    failure is raised and the results already finished are lost."""
    n = len(items)
    results = [None] * n

    def _fail(i, exc):
        if on_error is None:
            raise exc
        return on_error(items[i], exc)

    if workers <= 1 or n <= 1:
        for i, it in enumerate(items):
            try:
                r = worker_fn(it)
            except Exception as exc:
                r = _fail(i, exc)
            results[i] = r
            if on_result:
                on_result(i, r)
        return results

    with ProcessPoolExecutor(max_workers=workers) as ex:
        fut_to_idx = {ex.submit(worker_fn, it): i
                      for i, it in enumerate(items)}
        for fut in as_completed(fut_to_idx):
            i = fut_to_idx[fut]
            try:
                r = fut.result()
            except Exception as exc:        # BrokenProcessPool included
                r = _fail(i, exc)
            results[i] = r
            if on_result:
                on_result(i, r)
    return results


def describe_error(exc):
    """One line for a failure, with a hint when the pool itself broke."""
    if type(exc).__name__ == "BrokenProcessPool":
        return ("a worker process died (most often: out of memory) -- "
                "run again with fewer --workers, or --workers 1")
    return f"{type(exc).__name__}: {exc}"


def add_workers_arg(ap):
    """Add the standard --workers flag to an argparse parser."""
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel workers (default: auto, from the cores "
                         "and the free RAM; 1 disables the pool)")
