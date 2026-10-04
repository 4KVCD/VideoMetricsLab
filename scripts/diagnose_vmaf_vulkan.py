"""Finds where a GPU's Vulkan VMAF first differs from the known-good sums,
for a GPU whose driver fails the probe ("calculates VMAF wrongly") and
cannot be debugged where the app is developed (AMD's, say).

    .venv\\Scripts\\python.exe scripts\\diagnose_vmaf_vulkan.py [--device N]

Scores the probe's frames (vmaf_vulkan.probe_frames) at 8 and 10 bits:
1. All three frames: every sum that differs from the reference, by name.
2. The first frame again, stopping after 1, 2, 3 ... of the shader passes:
   after each, the sums and the working buffers are compared with the
   reference's. The first pass after which something differs is the shader
   that goes wrong; its differing 256-word blocks are counted and the first
   of them printed, to look up against the reference where it was made.

The reference (scripts/vmaf_vulkan_reference.json) is written with
--write-reference on a GPU that passes the probe. It prints a report to
send back; nothing else is written.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vmaf_app.core import vmaf_vulkan

REFERENCE = Path(__file__).with_name("vmaf_vulkan_reference.json")
BLOCK_WORDS = 256
WIDTH, HEIGHT = vmaf_vulkan._PROBE_SIZE
#: vv_read_buffer's buffers and their sizes in bytes for a WIDTH x HEIGHT picture.
_W1, _H1 = (WIDTH + 1) // 2, (HEIGHT + 1) // 2
_W2, _H2 = (_W1 + 1) // 2, (_H1 + 1) // 2
BUFFERS = {"admR": _W1 * _H1 * 16, "admA": _W1 * _H1 * 16, "admF": _W1 * _H1 * 16,
           "bandsRef0": _W1 * _H1 * 16, "bandsDis0": _W1 * _H1 * 16,
           "bandsRef1": _W2 * _H2 * 16, "bandsDis1": _W2 * _H2 * 16, "vifTmp": WIDTH * HEIGHT * 32}
#: The scored passes in the order vmaf_vulkan.cpp's build_passes adds them.
PASSES = ([f"vif_{kind} scale {scale}" for scale in range(4) for kind in ("vert", "hori")]
          + [f"adm_{kind} scale {scale}" for scale in range(4)
             for kind in ("dwt", "csf_den", "decouple (gain limit 100)", "cm (gain limit 100)",
                          "decouple (gain limit 1)", "cm (gain limit 1)")])
_VIF = ("x", "num_x", "den_log", "num_non_log", "den_non_log", "x2 (limit 100)", "num_log (limit 100)",
        "x2 (limit 1)", "num_log (limit 1)")


def slot_name(slot: int) -> str:
    if slot == 0:
        return "motion: SAD"
    if slot < 37:
        return f"VIF scale {(slot - 1) // 9}: {_VIF[(slot - 1) % 9]}"
    if slot < 49:
        return f"ADM csf_den scale {(slot - 37) // 3} band {(slot - 37) % 3}"
    rest = slot - 49
    return f"ADM cm gain limit {(100, 1)[rest // 12]} scale {rest % 12 // 3} band {rest % 3}"


def read_buffer(scorer, which: int, size: int) -> np.ndarray:
    lib = scorer._lib
    lib.vv_read_buffer.restype = ctypes.c_int
    lib.vv_read_buffer.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint64]
    out = np.zeros(size // 4, dtype="<u4")
    if lib.vv_read_buffer(scorer._context, which, out.ctypes.data, size) != 0:
        raise vmaf_vulkan.VmafVulkanError("reading a GPU buffer failed")
    return out


def blocks(words: np.ndarray) -> list[str]:
    return [hashlib.sha256(words[start:start + BLOCK_WORDS].tobytes()).hexdigest()[:10]
            for start in range(0, len(words), BLOCK_WORDS)]


def full_sums(device: int, bits: int) -> list[list[int]]:
    reference, distorted = vmaf_vulkan.probe_frames(bits)
    scorer = vmaf_vulkan.VulkanScorer(WIDTH, HEIGHT, bits, {}, device=device)
    try:
        for ref, dis in zip(reference, distorted, strict=True):
            scorer.add(ref, dis)
        scorer.features()
        return [[int(value) for value in scorer.sums(frame)] for frame in range(len(reference))]
    finally:
        scorer.close()


def after_passes(device: int, bits: int, count: int) -> tuple[list[int], dict[str, np.ndarray]]:
    """The first frame's sums and the working buffers after `count` scored passes."""
    reference, distorted = vmaf_vulkan.probe_frames(bits, 1)
    scorer = vmaf_vulkan.VulkanScorer(WIDTH, HEIGHT, bits, {}, device=device, pass_limit=count)
    try:
        scorer.add(reference[0], distorted[0])
        scorer.features()
        sums = [int(value) for value in scorer.sums(0)]
        return sums, {name: read_buffer(scorer, which, size) for which, (name, size) in enumerate(BUFFERS.items())}
    finally:
        scorer.close()


def write_reference(device: int) -> None:
    document = {}
    for bits in (8, 10):
        steps = []
        previous: dict[str, str] = {}
        for count in range(1, len(PASSES) + 1):
            sums, buffers = after_passes(device, bits, count)
            changed = {}
            for name, words in buffers.items():
                digest = hashlib.sha256(words.tobytes()).hexdigest()
                if previous.get(name) != digest:  # only what this pass wrote
                    changed[name] = {"sha256": digest, "blocks": blocks(words)}
                    previous[name] = digest
            steps.append({"sums": sums, "buffers": changed})
        document[str(bits)] = {"frames": full_sums(device, bits), "passes": steps}
    REFERENCE.write_text(json.dumps(document, separators=(",", ":")), encoding="utf-8")
    print(f"reference written: {REFERENCE} ({REFERENCE.stat().st_size // 1024} KB)")


def diagnose(device: int, name: str) -> bool:
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    print(f"Vulkan VMAF diagnosis on GPU {device}: {name}")
    all_same = True
    for bits in (8, 10):
        want = reference[str(bits)]
        print(f"== {bits}-bit")
        got = full_sums(device, bits)
        differing = [(frame, slot) for frame in range(len(got)) for slot in range(len(got[frame]))
                     if got[frame][slot] != want["frames"][frame][slot]]
        digest = vmaf_vulkan.probe_sums(device, bits)
        expected = vmaf_vulkan._PROBE_SUMS[bits == 10]
        print(f"1. three frames: probe sums {'MATCH' if digest == expected else 'DIFFER'}; "
              f"{len(differing)} of {sum(len(frame) for frame in got)} sums differ")
        for frame, slot in differing[:60]:
            print(f"   frame {frame} {slot_name(slot)}: got {got[frame][slot]}, expected {want['frames'][frame][slot]}")
        if len(differing) > 60:
            print(f"   ... and {len(differing) - 60} more")
        all_same &= not differing
        print("2. the first frame, pass by pass")
        known: dict[str, dict] = {}
        first_bad = None
        for count, step in enumerate(want["passes"], start=1):
            known.update(step["buffers"])
            sums, buffers = after_passes(device, bits, count)
            bad_sums = [slot for slot in range(len(sums)) if sums[slot] != step["sums"][slot]]
            bad_buffers = [label for label, words in buffers.items()
                           if hashlib.sha256(words.tobytes()).hexdigest() != known[label]["sha256"]]
            verdict = "same" if not bad_sums and not bad_buffers else "DIFFERENT"
            print(f"   after {count:2d} passes (last: {PASSES[count - 1]}): {verdict}")
            if verdict == "same":
                continue
            all_same = False
            for slot in bad_sums:
                print(f"      sum {slot_name(slot)}: got {sums[slot]}, expected {step['sums'][slot]}")
            for label in bad_buffers:
                mine, theirs = blocks(buffers[label]), known[label]["blocks"]
                bad = [index for index in range(len(mine)) if mine[index] != theirs[index]]
                print(f"      buffer {label}: {len(bad)} of {len(mine)} blocks of {BLOCK_WORDS} words differ; "
                      f"blocks {bad[:24]}{' ...' if len(bad) > 24 else ''}")
                if first_bad is None and bad:
                    start = bad[0] * BLOCK_WORDS
                    words = buffers[label][start:start + BLOCK_WORDS]
                    print(f"      {label} words {start}..{start + len(words) - 1} as this GPU wrote them (hex):")
                    for row in range(0, len(words), 16):
                        print("        " + " ".join(f"{int(word):08x}" for word in words[row:row + 16]))
            if first_bad is None:
                first_bad = count
                print(f"   => the first difference is in pass {count}: {PASSES[count - 1]}")
            if count >= first_bad + 2:
                print("   (stopping two passes after the first difference)")
                break
    print("RESULT: " + ("every sum and buffer is the reference's" if all_same else "this GPU differs: see above"))
    return all_same


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", type=int, help="Vulkan's number of the GPU (default: the one the app would use)")
    parser.add_argument("--write-reference", action="store_true")
    args = parser.parse_args()
    found = vmaf_vulkan.devices()
    for device in found:
        print(f"Vulkan GPU {device.index}: {device.name}{'' if device.usable else ' (not usable: no 64-bit integers)'}")
    chosen = (next((d for d in found if d.index == args.device), None) if args.device is not None
              else vmaf_vulkan.best_device(found))
    if chosen is None:
        sys.exit("no such GPU")
    if args.write_reference:
        for bits, expected in zip((8, 10), vmaf_vulkan._PROBE_SUMS, strict=True):
            if vmaf_vulkan.probe_sums(chosen.index, bits) != expected:
                sys.exit(f"{chosen.name} fails the probe: the reference must come from a GPU that passes it")
        write_reference(chosen.index)
        return 0
    return 0 if diagnose(chosen.index, chosen.name) else 1


if __name__ == "__main__":
    sys.exit(main())
