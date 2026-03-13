# CFR Experiment README

This README explains the two CFR summary folders under `xr6`:

- `F:\databackup\xr6\output\CFR_Necessity\Summaries`
- `F:\databackup\xr6\output\CFR_EXIF_Full\Summaries`

They answer different questions and should not be mixed in the paper.

## 1. What each folder proves

### `CFR_Necessity\Summaries`

This folder is for the necessity study of CFR itself.

It compares three settings:

- `fit`
- `exif_only`
- `raw_direct`

It is used to answer:

- whether cross-FoV / cross-sensor input can be reconstructed directly;
- whether EXIF-only alignment is enough for stable SfM and downstream training;
- whether the proposed `fit` stage is necessary.

Interpretation:

- `raw_direct` failure is part of the conclusion, not a bug;
- `exif_only` means EXIF alignment only, without CFR `fit`;
- `fit` means the CFR-aligned input used by the main method.

### `CFR_EXIF_Full\Summaries`

This folder is for the final quality comparison between:

- `fit_full`
- `exif_full`

It is used to answer:

- if both settings run the full GTGS pipeline, does `fit` still improve the final outputs.

This folder now contains three levels of evidence:

- step-2 crop/alignment metrics;
- RGB-stage quality metrics;
- T-stage quality metrics;
- full fusion sweep values for all `strategy + alpha` combinations.

Interpretation:

- `fit_full` is the main-method full result reused from the main campaign;
- `exif_full` is the full pipeline rerun on EXIF-only aligned data;
- fusion results are reported as a sweep, not as one automatically selected best fused model.

## 2. File descriptions

### `CFR_Necessity\Summaries`

- `CFR_Source.csv`
  - Raw per-dataset records.
  - Contains registration statistics, sparse points, reprojection error, smoke-train completeness, failure stage, and step-2 crop metrics.

- `CFR_SfM.xlsx`
  - SfM-focused table.
  - Use this for registration success, registered images, sparse points, and reprojection error.

- `CFR_SmokeTrain.xlsx`
  - Smoke-train table.
  - Use this for whether the pipeline can continue after SfM.

- `CFR_Step2.xlsx`
  - Step-2 crop/alignment comparison table.
  - Contains `mi`, `nmi`, `grad_ncc`, `edge_f1`, and `grad_ssim` for `fit` and `exif`.

- `CFR_AllInOne.xlsx`
  - Combined workbook for quick inspection.

- `CFR_QA.json`
  - Completeness check.

### `CFR_EXIF_Full\Summaries`

- `CFR_FinalQuality_Source.csv`
  - Raw per-dataset records for `fit_full` and `exif_full`.
  - Contains step-2 metrics, RGB-stage metrics, T-stage metrics, time fields, and output paths.

- `CFR_FinalQuality_T.xlsx`
  - Main per-dataset summary table.
  - Contains step-2, RGB, and T metrics.

- `CFR_FinalQuality_F.xlsx`
  - Fusion sweep table.
  - Contains all evaluated `strategy + alpha` rows.

- `CFR_FinalQuality_FusionSweep.csv`
  - Raw CSV version of the full fusion sweep.

- `CFR_FinalQuality_AllInOne.xlsx`
  - Combined workbook with `Main`, `FusionSweep`, and `QA` sheets.

- `CFR_FinalQuality_QA.json`
  - Completeness check.

## 3. Recommended writing logic

Use the two folders in this order:

1. `CFR_Necessity`
   - show that `raw_direct` is unreliable or fails under cross-FoV / cross-sensor input;
   - show that `exif_only` is a reasonable baseline but does not replace CFR;
   - conclude that CFR is necessary.

2. `CFR_EXIF_Full`
   - compare `fit_full` and `exif_full` after the entire pipeline is finished;
   - discuss the difference at step-2, RGB-stage, and T-stage;
   - inspect the fusion sweep without forcing one universal best alpha.

## 4. Notes for the writing assistant

- Do not merge `CFR_Necessity` and `CFR_EXIF_Full` into one undifferentiated table.
- Do not describe `raw_direct` failure as an implementation error.
- Do not describe `fit_full` as a separately invented baseline; it is the full main-method result reused for CFR comparison.
- Prefer the `.csv` files for exact numeric extraction and the `.xlsx` files for presentation.
