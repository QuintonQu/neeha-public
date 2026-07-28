# neeha-public

The full data and results (including files too large for git) are hosted on Google
Drive:
- [Raw event scan data](https://drive.google.com/drive/folders/1UcL-1i2yeQuqwwhz2gfKMWoPpf0kqWEy?usp=drive_link)
- [Segmentation data and results](https://drive.google.com/drive/folders/1GE5i_M013WhExqis7vumaZY_hKbx43nP?usp=sharing)

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

[`src/stardist_segmentation_my_scan.ipynb`](src/stardist_segmentation_my_scan.ipynb) (see
[Segmentation](#segmentation) below) depends on
[StarDist](https://github.com/stardist/stardist)/CSBDeep and TensorFlow, which pin an
older NumPy than the `event` env above — it was run in a separate `stardist` conda
environment (Python 3.9):

| Package | Version |
|---|---|
| stardist | 0.9.2 |
| csbdeep | 0.8.2 |
| tensorflow | 2.10.1 |
| numpy | 1.23.5 |

```bash
conda create -n stardist python=3.9
conda activate stardist
pip install stardist csbdeep "tensorflow==2.10.1" "numpy<2"
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

**Focal stack scans.** A focal stack recording captures multiple focal-depth "layers" in
a single raw scan, with each layer occupying its own band of sensor rows. Use
[`src/merging_events_focal_stack.ipynb`](src/merging_events_focal_stack.ipynb) instead of
`merging_events_x.ipynb` / `merging_events_y.ipynb` for this data — the only difference
is that the temporal-to-spatial mapping (`process_line_by_line_n_deg`, the focal-stack
counterpart of `process_line_by_line`) is done separately for each layer's group of
lines, by filtering events to that layer's row band (`y_min`/`y_max`, set via the
`layer` variable) before merging. Everything downstream — gradient registration,
Poisson/TV-L1 reconstruction — is identical to the grayscale pipeline, just run
per-layer against `results/focal_stack/<layer>/`.

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

## RGB Reconstruction

For RGB (color) reconstruction, use
[`src/registration_to_green_x.ipynb`](src/registration_to_green_x.ipynb) to register all
RGB+XY gradient data (6 files: x- and y-gradients for each of the R, G, B channels) onto
the green channel's x-gradient (`green_x`), which serves as the common reference frame —
the same coarse (`phase_cross_correlation`) + fine (optical-flow `warp`) alignment
approach as [Gradient Registration](#gradient-registration) above, generalized across
channels. Set `folder_name`/`target_dir`/`axis` at the top of the notebook and rerun it
once per non-reference file (`r_x`, `r_y`, `g_y`, `b_x`, `b_y`) — `target_dir` stays
pointed at the green channel's folder throughout, while `folder_name` points at the
channel being registered; `green_x` itself needs no registration and is used as-is.

The registered RGB gradient data is available in the `results` folder on Google Drive
(linked at the top of this README).

## Real-World Grayscale Reconstruction

[`src/reconstruction.ipynb`](src/reconstruction.ipynb) is the final step of the real-data
pipeline: it takes the registered x/y gradient pair produced above and reconstructs the
grayscale slide image, the real-data counterpart of the *Reconstruction* step in the
simulation section above.

**Input.** `test_shift_x.npy` and `test_shift_y_warp.npy` from
`results/grayscale_example/image_registration/`.

**Solving.** The gradient pair is fed into the same TV-L1 Poisson solver used in the
simulation pipeline (`tvl1_poisson_solver.tv_l1_reconstruction_cuda`, run under
`torch.no_grad()` on GPU `cuda:1`, `lambda_tv=1.0`, `n_iters=5000`) to invert the
gradients back into an intensity image. The FFT-based direct solver
(`fft_poisson_solver.poisson_solver_with_brightness`) is included as a commented-out,
faster alternative.

**Post-processing.** As in the simulation section, the raw reconstruction is
min-max normalized, exponentiated (to undo the log-domain scanning), and gamma-corrected
(`gamma=2.2`), then passed through the same histogram-based auto-HDR exposure adjustment
to produce `test_shift_reconstructed.png` / `test_shift_reconstructed_hdr.png`.

**Patch-wise auto-HDR (optimized).** Because slide-scale images have spatially varying
dynamic range, a second, GPU-vectorized auto-HDR pass (`auto_hdr_vectorized`) is applied
per-patch instead of globally: overlapping `100x100` patches (stride 2) are batched
(`batch_size=1024`), each patch's own histogram-based low/high exposure points are
computed in parallel via `torch.searchsorted`, and the per-patch results are blended back
together with an overlap-count accumulator. Patches with very low dynamic range that are
already bright (`range < 0.15` and `min > 0.5`) are saturated to white instead of being
over-stretched. The result is written to
`results/grayscale_example/reconstruction_l1/test_shift_reconstructed_auto_hdr_optimized.png`.

## Segmentation

[`src/stardist_segmentation_my_scan.ipynb`](src/stardist_segmentation_my_scan.ipynb) runs
nucleus/cell instance segmentation directly on the reconstructed gradient images using
[StarDist2D](https://github.com/stardist/stardist), a CSBDeep/TensorFlow-based model
originally pretrained for H&E-stained histology.

**Models.** Two finetuned StarDist2D checkpoints are provided under `model/`:
- `model/finetuned_he_experiment` — finetuned using only the x-gradient channel.
- `model/finetuned_he_experiment_xy` — finetuned using both x- and y-gradient channels.

The notebook loads a model by folder name (`StarDist2D(None, "finetuned_he_experiment",
"../model/")` in cell 2); swap in `"finetuned_he_experiment_xy"` to use the xy-trained
checkpoint instead (the commented-out `_xy` loading code adjusts the corresponding
image-stacking logic in the segmentation loop below it).

**Example data**, under `data/segmentation_example/`:
- `my_scan_cropped_from_leica_events/` — a small **synthetic** example (simulated event
  gradients cropped from a Leica-scanned slide image), committed directly in git.
- `my_scan_events/` — a **real-world** event-camera scan (`test_shift_x.npy` /
  `test_shift_y_warp.npy`, ~128MB each). These exceed GitHub's size limits and are
  excluded via `.gitignore`; download them from the
  [segmentation Google Drive folder](https://drive.google.com/drive/folders/1GE5i_M013WhExqis7vumaZY_hKbx43nP?usp=sharing)
  and place them at `data/segmentation_example/my_scan_events/<slice_id>/`.

**Pipeline.** For each slice-id subfolder under the chosen input folder:
1. Load the `*_x.npy` gradient array and rescale it to `[0, 1]`
   (`slice_x / abs(slice_x).max() / 2 + 0.5`), then replicate it across 3 channels to
   match the model's RGB-style input (the y-gradient and gradient-magnitude channels are
   available but commented out, for use with the `_xy` model).
2. Run `model.predict_instances` to get per-pixel instance labels. Real-world scans need
   `scale`/`prob_thresh` tuned for good results (the notebook uses `scale=0.5777**2,
   prob_thresh=0.1`); synthetic data works with the model's default parameters.
3. Extract instance boundaries (`skimage.segmentation.find_boundaries`), dilate them for
   visibility (`binary_dilation`), and overlay them in red on the rescaled input image.
4. **Outputs**, written to `results/segmentation_example/<slice_id>/`:
   `<n>_leica_outline.png` — the input image with segmented cell/nucleus boundaries
   drawn in red.
