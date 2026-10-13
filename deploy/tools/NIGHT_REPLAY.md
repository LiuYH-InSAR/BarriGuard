# Restored night_batch_v2_20 method

The archived final.json identifies regional_two_state_v1 with yellow support.
The method is restored with the archived thresholds: original warmup median background,
protected EMA updates and ON_DELTA=40 for discovery; 5x5 regional BGR means
converted to Gray; two-Gaussian OFF/ON states; yellow increases acceptance by
lowering the separation requirement from 2.5 to 2.0 when yellow_score >=0.5.
Other fit gates: mean gap >=12, each weight >=0.05, BIC gain >=10, samples >=60.
The onset minimum is now 20 per user request; original autocorrelation checks remain unchanged.
Each clip also exports duty.npy, duty_map.jpg (fixed 0..100% scale) and
duty_candidate_mask.jpg (duty floor plus opening, before final classification).
The batch copies both JPGs into duty_maps. Detection thresholds otherwise retain v2 values.
Low-state discovery backgrounds and short-pulse classification are disabled.
Pulse config fields remain compatibility-only and do not affect this method.

## Replay three videos

Run from the repository root in PowerShell. A separate output preserves archives.

```powershell
.\.venv\Scripts\python.exe deploy\tools\run_night_batch.py --limit 3 --out output\night_v2_restored_3
```

Each clip is replayed independently and continuously. final_images collects
final_night.jpg and gray_plots collects selected Gray plots. summary.csv and
per-video run.log record status. The batch default is five videos.

## Candidate diagnostics

intermediate/series.npz contains tracked regional gray_raw/gray_corrected,
pixel_gray_corrected, on_state, actual time_s, sample_index, exposure_offset,
xy, row_id and tracking_start_index. Unobserved samples are NaN. Candidates
include accepted/rejected reasons, fitted means, weights, separation, BIC,
yellow score, channel changes and periodicity details in final.json.
Each candidate has CSV and SVG files; candidate_positions.jpg marks positions.
SVG layout: full sequence plus first/middle/last 10 seconds, X ticks 2 seconds,
Y ticks 20 Gray. Short sequences may have overlapping detail windows.
Regional Gray here uses float BGR weighting; the pixel matrix below uses
OpenCV uint8 Gray rounding, so a small rounding difference is expected.

## Pixel Gray matrix query (retained)

Each video writes gray_matrix/gray_raw.npy as uint8, axes row/column/sample.
The saved rectangle bounds the expanded ROI. valid_mask.npy excludes invalid
pixels; manifest.json gives its original-video origin. time_s.npy records actual
algorithm sample times (including warmup); exposure_offset.npy reconstructs
corrected Gray without a second large matrix. These are sampled frames, not
every video frame. Conversion briefly uses twice the matrix disk size.

```powershell
.\.venv\Scripts\python.exe deploy\tools\inspect_gray.py --matrix output\night_v2_restored_5\1002490_20260918_1900\gray_matrix --row 634 --col 1421
```

Coordinates are zero-based original-video coordinates: row=y, col=x.
The query works for any saved valid pixel, not only selected candidates.
Output: positions/row_<row>_col_<col>/series.csv, series.npz, pixel_gray.svg,
region_gray.svg, position.jpg and position.json. Data includes raw/corrected
pixel and 5x5 Gray, exposure offset, pixel changes and regional high-state labels.
These state labels are diagnostics, not a final lamp verdict with yellow support.
Outside-ROI coordinates are rejected; existing candidate-only archives cannot
supply the spatial matrix and require replay. --no-save-intermediate disables
all intermediate saving in the single-video script.

Coordinate plots include red ON-mean and blue OFF-mean dashed lines, fitted separately
for pixel and regional corrected Gray over the full sequence. The same means are
shown in all four panels and stored in position.json. These are fitted state means,
not averages of pulse maxima. Fewer than 60 valid samples or internal gaps leave
means unavailable. Diagnostic fits alone do not imply a periodic lamp verdict.

## Disturbance replay acceleration

Recognition rules are unchanged. Valid ROI pixels are processed in batches of
4096; each pixel retains its own temporal mask. Unmasked pixels reuse the original
online count, peak and final background. Masked pixels replay the same background
rules with Numba (`fastmath=False`), or NumPy when Numba is unavailable.
Install the optional accelerator with:
` .\.venv\Scripts\python.exe -m pip install -r deploy\tools\requirements-acceleration.txt `.
The first compiled call includes compilation overhead; subsequent calls reuse it.

Each gray_matrix folder stores disturbance_mask_packed.npy, masked_statistics.npz,
masked_cache.json and masked_performance.json. Replay cache identity includes input
metadata, hashes of small input arrays, run parameters and relevant source code.
An unchanged existing matrix can reuse the mask and statistics. Re-running a video
rewrites the input matrix and normally invalidates this cache. Performance JSON
records backend, elapsed seconds, cache use, reused pixels and replayed pixels.
Gray queries and plots continue to use the saved masks and historical thresholds.
