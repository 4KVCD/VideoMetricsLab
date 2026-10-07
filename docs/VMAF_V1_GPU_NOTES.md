# VMAF v1 on the GPU (Vulkan): work notes

Notes for whoever integrates this work (another Claude session or a person).
They cover the experimental branch `VMAF-v1-full-Vulkan` in **both**
repositories:

| Repository | Branch | Base | Head |
|---|---|---|---|
| VideoMetricsLab | `VMAF-v1-full-Vulkan` | `origin/release/v1.5` (2c223d8) | c66f3b2 |
| libvmaf-fast | `VMAF-v1-full-Vulkan` | `fast` (afe877a5) | 88a0dd36 |

The VMAF version is **VMAF v1** (the `vmaf_v1.0.16` models), with libvmaf 3.2.0.
Every change is **bit-exact** against libvmaf's CPU code: the same double for
every frame, not just the same mean. The speed-ups never change a score.

All work and measurements were done on an AMD laptop: a Radeon 780M iGPU
(gfx1103, driver 32.0.31041.1004) with a Ryzen 7 8845HS. The NVIDIA paths
compile and share code, but they were **not run here**. See "For the main PC"
below.

---

## 1. What the branch does

VMAF v1 = ADM3 + motion3 + CAMBI + SpEED chroma, then the model. Before
this branch, only ADM3 and motion3 ran on the GPU (Vulkan), and CAMBI and
SpEED ran in libvmaf on the CPU. Now:

- **Everything runs on the GPU (`vmaf_vulkan.dll`), bit for bit libvmaf's:**
  - ADM3 and motion3;
  - **CAMBI**, whose top-k pooling uses a radix select with exact 64-bit
    sums. When a sum would not be exact in a double, the engine falls back
    to the CPU's pooling (see §6);
  - **SpEED chroma**. Its `est_params` least-squares fit stays on CPU
    worker threads, off the submitting thread.
- **AMD decoders hand their pictures to the engine on the GPU, with no CPU
  copy:**
  - **Windows' own Media Foundation decoder** (`native/mf_frames.cpp`, new)
    is tried first on AMD (`gpu_frames.MEDIA_FOUNDATION`, "amd-mf").
  - **AMF's Direct3D 11 decoder** (`native/amf_frames.cpp`) is the
    fallback.
  - The shared hand-over code is `native/amf_handover.h` and
    `native/amf_direct.slang`.
- **Picture mode (engine flag bit 2, `vv_pictures`).** The engine reads the
  decoder's own D3D11 textures (imported into Vulkan by KMT handle), so no
  copy of the picture is made at all.
  - Python gives each slot back only once the GPU has read it
    (`V1Scorer.release_later` / `_release_done`). The reference's slot is
    kept one frame longer, because the next frame's motion reads it.
  - The decoder pool is 6 in picture mode (4 otherwise): two pictures being
    matched, three frames in flight, and the previous reference.
  - Since c66f3b2, picture mode is used for **every** pair, H.264 included.
    Windows' H.264 decoder outputs layers of a 9-picture texture array, so
    each H.264 picture is still copied once by D3D11 into a slot texture of
    its own.

### Engine API added (libvmaf-fast `fast/vulkan/vmaf_vulkan.cpp`)

| Export / flag | Purpose |
|---|---|
| `vv_v1_cambi`, `vv_v1_cambi_scores` | CAMBI on the GPU |
| `vv_v1_speed`, `vv_submit_v1`, `vv_chroma_staging`, `vv_shared_chroma` | SpEED chroma, chroma planes in |
| `vv_import_timeline`, `vv_commit_after` | a decoder's timeline semaphore; its copies are waited for on the GPU |
| flag bit 19 (`SHARED_FLAG`) + `vv_export` / `vv_shared_next` / `vv_shared_device` | slots in GPU memory that a decoder imports |
| flag bit 2 + `vv_pictures`, `vv_pictures_mode` | picture mode: the decoder's textures read in place |
| flag bits 8-15 | frames in flight (3) |

Misuse in picture mode is refused with an error (88a0dd36). That covers a
commit without `vv_pictures`, a picture not inside its texture, and
`vv_test_fill_slot`. In picture mode the slots' staging buffers are only
4 KB, so these must never write into them.

---

## 2. Results (Radeon 780M, warm, scores identical throughout)

### Speed

| | CPU-only VMAF v1 | Now (GPU) |
|---|---|---|
| 4K, scoring only (start-up excluded) | 26.5 fps | about 95 fps warm, up to 110 cool (about 3.6x) |
| 1080p, scoring only | 99 fps | about 212-250 fps (about 2.1-2.5x) |
| Whole 120 s run, start-up included | | 4K about 70 fps, 1080p about 157 fps (true baseline at the start of this work: 16.8 / 24.6) |

- **Engine alone at 4K:** about 21 fps at the start of this work, 140 fps
  now (`fast/tests/bench_vmaf_v1.py ... --shared`).
- **Thermal drift.** This laptop's numbers move by up to 30% with
  temperature. Every comparison here was an A/B run alternately, several
  rounds; never compare absolute numbers across sessions.

### GPU memory (peak "Total Committed" of the scoring process, GPU Process Memory counters)

| | Before | After |
|---|---|---|
| 4K (HEVC 10-bit pair) | 1,940 MB | **986 MB** |
| 4K shared memory | 803 MB | 43 MB |
| 1080p (H.264 pair) | 414 MB | **291 MB** |

At 4K, 819 MB of what is left is Windows' decoders' own picture pools
(about 405 MB per 4K HEVC decoder), which no attribute changes. The engine
is about 86 MB:

| Engine buffers at 4K | Size |
|---|---|
| ADM band images | 2 x 7.9 + 2 x 2.0 MB |
| CAMBI scale images | about 31 MB |
| CAMBI c-values kept for CPU pooling (3 slots, host memory) | 3 x 10.5 MB |

Speed with the memory work against before it (bf16c50 → 8c5eb4e): 4K
73.9 → 77.3 fps (+4.5%), 1080p unchanged.

---

## 3. Commits

### VideoMetricsLab (`origin/release/v1.5..VMAF-v1-full-Vulkan`)

| Commit | What |
|---|---|
| 7afd854 | AMD hand-over: pictures collected on a thread of their own, two shared textures in turn |
| 6223a29 | hand-over copies into other memory on the GPU's copy queue |
| 1627c90 | CAMBI and SpEED chroma on the GPU (engine from libvmaf-fast's branch) |
| a25b43b | chroma planes copied by the GPU into the engine's memory, not downloaded |
| 0c0bb7f | engine d411e56b (compute-only queue) |
| 5aeb4cc | hand-over reads AMF's own pictures where they are (no D3D11 copy) |
| 49703c3 | AMD's copies into the engine's memory waited for by the GPU (timeline semaphore) |
| 4545854 | engine d83654ab |
| 47d9e90 | engine c49a68b0; one GPU copy a picture (merged copy_planes); NVIDIA chroma fix |
| 4463ffc | Windows' Media Foundation decoder for AMD, tried before AMF (+13% at 4K) |
| 895f3a3 | picture mode: the engine reads the decoder's textures (+8% at 4K in-process) |
| bf16c50 | engine e7ca8273 (CAMBI band heights) |
| 571a601 | GPU memory: no slot buffers for decoders that hold their pictures; 1,933 → 1,027 MB at 4K |
| 8c5eb4e | GPU memory: held pictures copied straight to their destination (no scratch buffer), 986 MB at 4K; **race fix** (below) |
| c66f3b2 | H.264 pairs in picture mode: 342 → 291 MB at 1080p, same speed |

### libvmaf-fast (`fast..VMAF-v1-full-Vulkan`, 45 commits)

These are the engine's optimisations, each measured and bit-exact (the
subjects carry the gains). The milestones:

- **CAMBI and SpEED:**
  - f46e56fb and 346cce46: CAMBI and SpEED chroma on the GPU, bit for bit.
  - 71edd9f0: CAMBI c-values from sliding column counts, 13.4 → 2.65
    ms/frame at 4K.
- **ADM:**
  - 287982b9: ADM after the DWT in one pass per scale, 15.1 → 4.5 ms/frame.
  - a5915575, 0f71d21f and d144caba: the DWTs fused into the ADM passes.
- **Motion:** 64653421 and a17b5b92, 1.7 → 0.9 ms/frame and less.
- **Engine-wide:**
  - d411e56b: a compute-only queue.
  - ea0d8057: wave32 for CAMBI c-values and fused ADM.
  - fc85b0a2: passes recorded in turns.
- **SpEED's CPU part:** 370e7653, 8349812a, 37dd8c2f and c49a68b0.
- **Decoder integration:**
  - a351fb4f, f0212b89 and 1adf9c79: reading from a decoder's slots.
  - 1e1a12eb: the timeline semaphore.
  - 4d7a6ad1: picture mode.
- **Memory and safety:**
  - 71df1364: less GPU memory. Staging is 4 KB in picture mode, there are no
    host chroma buffers, and ADM's first band image is 1/16 the size.
  - 88a0dd36: misuse refused.

The engine DLL committed in VideoMetricsLab
(`vmaf_app/tools/vmaf_vulkan/vmaf_vulkan.dll`, SHA-256 `2c5df0c1...`) is a
**local build of libvmaf-fast 88a0dd36**
(`fast/scripts/build_vmaf_vulkan.ps1`), not a release. See §7.

---

## 4. Bugs found and fixed along the way (worth knowing)

- **Race in the AMD hand-over (8c5eb4e):**
  - What happened: `sources` (decoder textures imported into Vulkan) was a
    `std::vector` grown by the decoding thread while a copy on another
    thread held a reference into it. A reallocation then left that
    reference dangling.
  - Symptom: an access violation, or a wrong picture, in about 1 in 12
    parallel runs of the GPU tests.
  - Fix: a fixed `std::array` with an atomic count. Afterwards: 0 failures
    in 30 parallel runs, and the full suite 5 times clean.
  - This also probably explains an earlier "lazy slot buffer" design that
    failed intermittently for no visible reason.
- **Descriptor-pool allocations** from two threads are now made under a lock
  (`descriptor_lock`).
- **Media Foundation:**
  - P010 is not offered until the stream changes, so a provisional output
    type is set.
  - The decoder blocks inside `ProcessOutput` when the app holds too many of
    its samples (about 9 at 4K, 7 at 640x360), whatever pool is asked for.
    Hence `kMostHeld = 6`: past it, pictures are copied into the slot's own
    texture. A sweeper thread gives back deferred samples.
- **AMF's own Vulkan decoder** decodes HEVC's last B-pictures before a
  closed-GOP IDR wrongly. The hand-over uses AMF's D3D11 path; a test covers
  it.

---

## 5. How it was verified

- **Bit-exactness:**
  - libvmaf-fast `fast/tests/compare_vmaf_v1.py --matrix`: 73 cases, all
    identical.
  - Full-app per-frame comparisons against CPU VMAF v1 at 4K (1,440 frames)
    and 1080p (241 frames).
  - Codec clips: HEVC 8-bit open-GOP, HEVC 10-bit, AV1 8/10-bit, H.264
    B-pyramid MP4.
  - 8K 10-bit.
  - Frame-rate mismatches: a 48 fps and a 30 fps distorted video against a
    24 fps reference.
- **Picture mode vs copy mode**, frame for frame, in one process:
  - with random delays in the loop;
  - with cancel-and-rerun;
  - with 5 runs in one process, where GPU memory returns to 8 MB after each,
    so there is no leak.
  - Controls showing the check is sensitive: releasing pictures at once
    changes 1,123 of 1,440 frames. Releasing one frame early is masked by
    the decoder's reuse distance.
- **Buffer overruns:** a debug engine gives every buffer 64 KB of guard
  space and checks it at the end. It caught a deliberate overrun, then found
  none:
  - across the matrix;
  - at 8K, 5K and odd sizes (3842x2162, 1922x1082);
  - in full-app 8K runs.
- **Tests:** VideoMetricsLab's full suite passes (1732 passed, 67 skipped:
  the skips are NVIDIA and Intel). `tests/test_frame_handover.py` checks
  that picture mode scores the same as copying, for HEVC and H.264 pairs,
  and checks the deferred-release bookkeeping.

---

## 6. Hardware and driver facts learned (Radeon 780M)

- **Windows' decoder ignores `MF_SA_MINIMUM_OUTPUT_SAMPLE_COUNT`.** Its
  memory and hold cap are the same for requests of 0 to 32.
- **Windows' H.264 decoder** always outputs a 9-layer texture array.
  - Vulkan *can* import the whole array and view each plane as a 2D array.
    The old crash was per-layer views only.
  - But the decoder stalls once 5 of its pictures are held, so reading H.264
    in place isn't possible with picture mode's 6 held pictures.
- **AMF uses more memory than Windows' decoder:** 561 MB against 405 MB per
  4K HEVC decoder.
- **At 4K the app is bound by the video decode hardware.** Two 4K HEVC
  decoders alone give about 120 pairs/s; with the engine nearly idle the
  app gains only about 6%.
- **At 1080p the engine isn't the limit either.** The app gives about 335
  pairs/s with scoring nearly off, against 389 for the decoders alone.
- **CAMBI's top-k at 8K** overflows an exact 2^53-unit sum on every frame
  (scale 0's sum is about 3.5e16 units), so it is pooled on the CPU from the
  kept c-values. At 1080p and 4K on the test clips this never happened, but
  4K is close to the limit, so the kept buffers are needed.
- **CPU use** is under 1 core at 4K and at 1080p.

---

## 7. To integrate

1. **Merge the branches.** Both remote base branches have only README
   commits made on GitHub since these branches started:
   - libvmaf-fast `VMAF-v1-full-Vulkan` into `fast`;
   - VideoMetricsLab `VMAF-v1-full-Vulkan` into `release/v1.5`.
2. **Release libvmaf-fast with this engine** (`fast/scripts/package.ps1`),
   then update `scripts/fetch_libvmaf_fast.ps1`'s pinned version and SHA-256
   and re-fetch. The branch's committed `vmaf_vulkan.dll` is a local build;
   a release should replace it. `libvmaf.dll` did not change.
3. **Native decoders.** `mf_frames.dll` is new; it and `amf_frames.dll`
   include `amf_handover.h`, and `nvdec_frames.cpp` changed too (47d9e90).
   Build them with `scripts/build_gpu_frames.ps1` (git-ignored, built in
   releases).
4. **Changelog / README** are not updated on this branch; that text needs
   the owner's approval.

### For the main PC (NVIDIA): re-check before release

- `fast/tests/compare_vmaf_v1.py --matrix` in libvmaf-fast. The engine's
  shared descriptor, recording and band-height code runs on every GPU.
- A VMAF v1 job in the app with NVIDIA decoding, and the full test suite.
- NVIDIA never uses picture mode (`hands_over_textures` is AMD-only), but
  47d9e90 changed NVIDIA's chroma path (merged copy, a chroma fix) in
  `nvdec_frames.cpp`.
- Tuning was done on the 780M (tile sizes, rows per workgroup, wave32).
  Results are correct everywhere but may not be the fastest elsewhere.

---

## 8. Tried and rejected (do not redo without a new reason)

About 70 ideas were measured over the whole effort; about two thirds were
kept.

- **Engine:**
  - packed 16-bit CAMBI input;
  - c-value layouts (column-interleaved counters, reading only gated
    counts);
  - taller tiles and taller c-values bands (−16%, −32%);
  - row bands for small scales;
  - two "last workgroup finishes" pass merges;
  - a division table;
  - tiled DWT and tiled mode filter;
  - two GPU queues (−17% 4K, −26% 1080p);
  - wave32 for every pass;
  - eight covariances per SpEED sweep;
  - other ADM band heights.
- **Pipeline:**
  - engine with 2 frames in flight (−5% then, −1% now, saves only 10 MB);
  - decoder pools of 3/5/7/8 (no gain);
  - Media Foundation copying every picture (−16%);
  - three AMF decoder settings;
  - a shorter Python GIL switch interval (no gain).
- **Memory:**
  - smaller MF pools (ignored by the decoder);
  - AMF instead of MF (more memory);
  - reading H.264 arrays in place (decoder stalls, §6);
  - narrower CAMBI images (about 13 MB at 4K for a large shader rewrite:
    not done).

---

## 9. Tools

- **libvmaf-fast `fast/tests/`:**
  - `compare_vmaf_v1.py` (bit-exactness, `--matrix`);
  - `bench_vmaf_v1.py` (engine speed with a per-pass GPU profile, `--shared`
    = the app's path);
  - `diagnose_vmaf_vulkan.py`.
- **VideoMetricsLab:** `.local-dev/bench_cpu.py` (full-app run to JSON and
  per-frame scores). It is local only (git-excluded) on the AMD laptop.
- The stress and memory scripts (picture-mode stress, GPU-memory sampler,
  guard-engine patch) were session scratch files and are not committed.
  The methods are described in §5.
