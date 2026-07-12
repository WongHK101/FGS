# GPT Review Brief For Reference-Depth Protocol

This note is for reviewing the proposed **no-retrain reference-depth geometry evaluation** before implementation.

Please review the attached protocol file:

- `DEPTH_REFERENCE_PROTOCOL.md`

## Review Goal

Please judge whether this protocol is scientifically clean and paper-appropriate enough to justify implementation as a new geometry-evidence experiment.

## Intended Use

The protocol is meant to support the claim that our method has **cleaner and more stable geometry** than competing methods, especially in the failure mode where roofs or facades grow outward toward the camera.

It is explicitly meant to avoid the cost of retraining all baselines under new odd/even seeds.

## Key Fixed Design Choices

- training-only RGB views are used to construct external reference geometry
- held-out views are used only as evaluation cameras
- reference depth is rendered from a training-only reference mesh
- each trained method renders thermal held-out depth in the exact same evaluation cameras
- the main metric is `FrontIntrusionRate@delta`, not just generic depth MAE

## What Needs GPT Judgment

Please focus on the following:

1. Is this protocol scientifically acceptable for a top-tier vision paper if it is clearly described as **reference-depth-based geometric evaluation** rather than absolute GT depth accuracy?

2. Do you agree with the choice:
   - `training-only RGB MVS -> fused dense result -> reference mesh -> held-out D_ref`
   instead of:
   - direct dense-point-cloud vs Gaussian-center comparison
   - or a full retraining-based repeatability study for every baseline?

3. Is `FrontIntrusionRate@delta` a strong primary metric for the observed failure mode, or should another depth-derived metric take priority?

4. Is the current threshold sweep:
   - `0.10 / 0.25 / 0.50 / 1.00 m`
   scientifically reasonable for large outdoor scenes?

5. Do you see any protocol blocker that should be fixed before implementation?

6. If you approve implementation, should the first pilot remain:
   - `Building`
   - 5 methods
   - thermal `T` models only

## Non-Negotiable Wording Constraint

If approved, the experiment will be described as:

- `reference-depth-based geometric evaluation`
- `held-out geometry consistency against a training-only MVS reference`

It will **not** be described as:

- absolute ground-truth depth accuracy
- geometric ground-truth evaluation

