# neeha-public

## Environment

The notebooks/scripts in this repo were developed and run against the `event` conda
environment (Python 3.9.18). Key dependencies:

| Package | Version |
|---|---|
| numpy | 1.26.4 |
| scipy | 1.13.1 |
| torch / torchvision / torchaudio | 2.6.0 / 0.21.0 / 2.6.0 (CUDA 12.4) |
| opencv-python-headless | 4.8.1.78 |
| openslide-python / openslide-bin | 1.4.1 / 4.0.0.6 |
| scikit-image / scikit-learn | 0.24.0 / 1.6.1 |
| pillow | 11.1.0 |
| pyfftw | 0.14.0 |
| imageio | 2.37.0 |
| tqdm | 4.67.1 |
| jupyter / jupyterlab | 1.1.1 / 4.4.2 |
| matplotlib | 3.9.4 |
| numba | 0.60.0 |
| metavision_core / metavision_hal / metavision_sdk_* | (Prophesee Metavision SDK / OpenEB) |

The TV-L1 reconstruction step uses `torch.cuda` and was run on a machine with multiple
NVIDIA GPUs (device `cuda:1` in the notebook). A CUDA-capable GPU is required for that
step; everything else runs on CPU.

The Metavision packages are **not** available on PyPI — they come from Prophesee's
[OpenEB](https://github.com/prophesee-ai/openeb) SDK, which has to be built/installed
separately (e.g. into the same conda env's `site-packages`) following OpenEB's own
build instructions. They are only required for the raw-scan merging pipeline in
[Raw Event Scan Merging](#raw-event-scan-merging) below; the simulation section does
not need them.

Recreate the environment with conda:

```bash
conda create -n event python=3.9
conda activate event
pip install numpy scipy torch torchvision torchaudio opencv-python-headless \
    openslide-python openslide-bin scikit-image scikit-learn pillow pyfftw \
    imageio tqdm jupyterlab matplotlib numba
# then separately build/install Prophesee OpenEB (metavision_core, metavision_hal, ...)
```

## Simulation: Event Scan and Reconstruction

[`src/scan_simulation.ipynb`](src/scan_simulation.ipynb) simulates an event-camera-style
line scan over a static image and reconstructs the original image from the simulated
events.

**Input.** A cropped, grayscale whole-slide-image patch
(`data/slide_image_level_0_cropped.png`), converted to linear intensity via a gamma
(2.2) decode.

**Scanning simulation.** A sliding window ("kernel", default `H x W = 1000 x 40`) sweeps
across the image in both the X and Y directions (`scan_image_x` / `scan_image_y` in the
notebook). At each 1-pixel step:
1. The window's brightness is compared, pixel-by-pixel, against a buffered reference
   brightness (log-encoded via `lin_log` in [`src/utils.py`](src/utils.py), mimicking a
   DVS-style event sensor's log response).
2. Gaussian noise (`noise_level`) is optionally added to the brightness difference.
3. Wherever the absolute difference exceeds a per-pixel threshold (linearly ramped
   across the kernel), an event `(t, x, y, p)` is fired — timestamp, pixel location, and
   polarity (brightness increased or decreased) — and the reference buffer is updated at
   those pixels.

This produces two raw event streams, one for horizontal scanning and one for vertical
scanning, which are then merged back into dense per-pixel gradient images
(`merge_events_x`, `merge_events_y`) representing accumulated horizontal and vertical
intensity gradients.

**Reconstruction.** The two gradient fields are treated as the x- and y-gradients of the
original image and fed into a TV-L1 Poisson reconstruction solver
([`src/tvl1_poisson_solver.py`](src/tvl1_poisson_solver.py)), a Chambolle-Pock
primal-dual scheme implemented in PyTorch and run on GPU. This inverts the gradient
field back into an intensity image without directly having access to the original
pixel values, using only the simulated event stream.

An FFT/DCT-based direct Poisson solver
([`src/fft_poisson_solver.py`](src/fft_poisson_solver.py),
[`src/fft_poisson_solver_opt.py`](src/fft_poisson_solver_opt.py)) is also included as a
faster, non-iterative alternative for gradient-domain reconstruction.

**Post-processing.** The raw reconstruction is normalized, exponentiated (to undo the
log-domain scanning), gamma-corrected, and passed through a histogram-based auto-HDR
exposure adjustment (`auto_hdr` in `src/utils.py`) before being converted to an 8-bit
image for display/export.

## Raw Event Scan Merging

Unlike the section above, [`src/merging_events_x.ipynb`](src/merging_events_x.ipynb)
and [`src/merging_events_y.ipynb`](src/merging_events_y.ipynb) operate on **real**
recordings from a Prophesee event camera scanning a physical grayscale slide (not a
simulated event stream), and stitch the individual scan lines into one large merged
gradient image.

**Data.** The two raw recordings (`x.raw`, `y.raw`, one per scan direction) are several
GB each and are excluded from git (see `.gitignore`). Download them from
[this Google Drive folder](https://drive.google.com/drive/folders/1UcL-1i2yeQuqwwhz2gfKMWoPpf0kqWEy?usp=sharing)
and place them at:

```
data/grayscale_example/x.raw
data/grayscale_example/y.raw
```

`merging_events_x.ipynb` and `merging_events_y.ipynb` are run independently of each
other — each reads only its corresponding `.raw` file and writes its own outputs into
`results/grayscale_example/`.

**Pipeline (per direction).**
1. **Load events + triggers.** `RawReader` (from Prophesee's `metavision_core`) streams
   the raw event file and separately extracts the external hardware trigger events
   (`get_ext_trigger_events`), which mark the start/end of each physical scan line.
2. **Derive scan geometry.** Known stage parameters (`pixel_per_mm`, `mm_per_move`,
   sensor width, triggers per line) are used to compute how many scan lines were
   captured and the expected trigger spacing, with an `assert` sanity check against the
   observed trigger count.
3. **Split into per-line frames.** `process_line_by_line` (Numba-JIT-parallelized, in
   [`src/parallel_merge.py`](src/parallel_merge.py)) bins every event into its scan line
   and into an `(x, y)` pixel position within that line by linearly interpolating the
   event timestamp between the two bracketing trigger times, producing one small dense
   event image per line.
4. **Estimate inter-line shift.** `phase_cross_correlation` on the overlapping edge
   strips of consecutive lines gives a coarse odd/even row shift and refined overlap
   width, used to correct a systematic bidirectional-scan offset (`np.roll`).
5. **Stitch lines together.** Lines are merged left-to-right: for each new line, a
   refined sub-pixel shift (`phase_cross_correlation`, upsampled) plus a dense optical
   flow field (`cv2.calcOpticalFlowFarneback`, seeded with that shift) estimate the
   local row-wise misalignment against the growing merged image; the new line is warped
   (`skimage.transform.warp`) with a per-column blend from full correction (at the seam)
   to none (past the overlap) before being concatenated on.
6. **Outputs**, written to `results/grayscale_example/`:
   - `all_events_{x,y}.png`, `exapmle_events_{x,y}.png` — debug views of the raw per-line
     frames.
   - `merged_shifted_final_{x,y}.npy` / `.png` — the final stitched gradient image for
     that scan direction.

Although the multi-GB raw `.raw` recordings only live on Google Drive, these merged
`.npy` outputs are kept locally in `results/grayscale_example/` (not regenerated each
time) so that downstream steps — the gradient registration step below, and eventually
Poisson/TV-L1 reconstruction — can load them directly without re-running the raw-event
merge. Note `results/` is git-ignored, so these intermediate files need to be produced
locally (by running the two notebooks above) before any downstream step that depends
on them.

## Gradient Registration

The x- and y-direction scans are two independent physical passes of the scanner, so the
merged gradient fields they produce (`merged_shifted_final_x.npy`,
`merged_shifted_final_y.npy`) are not pixel-aligned with each other — before they can be
used as the `∂I/∂x` and `∂I/∂y` inputs to a Poisson/TV-L1 reconstruction, the y-gradient
map has to be registered onto the x-gradient map's coordinate frame. This is done in
[`src/registration.ipynb`](src/registration.ipynb).

1. **Load & clean.** Loads `merged_shifted_final_x.npy` (transposed) and
   `merged_shifted_final_y.npy` from `results/grayscale_example/`, inspects the
   non-zero gradient value distribution, and clips outliers to `[-35, 35]`.
2. **Crop a working region.** A `4000x4000` window centered at `(3000, 3000)` is cropped
   out of both fields — registration is tuned/run on this region rather than the full
   image.
3. **Build edge masks.** Both crops are converted to thresholded absolute-value maps
   (small values zeroed out) so that only strong gradient "edges" drive registration,
   which is more robust than using the raw signed gradient values directly.
4. **Coarse alignment.** `phase_cross_correlation` on the edge masks gives an integer
   pixel shift, which is applied to re-crop the y-gradient field into rough alignment
   with the x-gradient field.
5. **Fine alignment.** Dense optical flow (`cv2.calcOpticalFlowFarneback`) between the
   two edge maps estimates a per-pixel `(u, v)` displacement field, which is used to
   `warp` the coarsely-aligned y-gradient crop onto the x-gradient crop's pixel grid —
   correcting local/non-rigid misalignment that a single global shift can't.
6. **Outputs**, written to `results/grayscale_example/image_registration/`: diagnostic
   plots (`test_shift_x.png/.svg`, `test_shift_y_warp.png`, `test_shift_norm_u_v.png`,
   a histogram SVG of gradient values) plus the registered gradient pair itself —
   `test_shift_x.npy` and `test_shift_y_warp.npy` — normalized to `[-1, 1]`, ready to be
   fed together into a Poisson/TV-L1 solver the same way `events_merged_x_cropped` /
   `events_merged_y_cropped` are in the simulation section.
