# Current Depth-Reference Assistant Package

Use this package for downstream paper tables, plots, and assistant analysis:

`G:\ACMMM26\GeometricRepeatability\AssistantPack_DepthReference_MaskNoMask_5scene8method_nativealign_20260508_v1`

Zip:

`G:\ACMMM26\GeometricRepeatability\AssistantPack_DepthReference_MaskNoMask_5scene8method_nativealign_20260508_v1.zip`

Model bundle root:

`G:\ACMMM26\GeometricRepeatability\DepthReference\Formal_5scene_8method_9thr_nativealign_20260508_v1`

PVpanel, Road, and TransmissionTower reference root:

`G:\ACMMM26\GeometricRepeatability\DepthReference\MeshFix_ROICrop_failed3_20260506_v1`

Native-align guard:

- Every consumed model bundle must have `camera_frame_mode=probe_manifest_native_align`.
- Every consumed model bundle must include `strict_to_native_alignment`.
- Every consumed view must include `native_camera_to_world`.
- Do not use `AssistantPack_DepthReference_MaskNoMask_5scene8method_20260507_v2` or `_v2_LIGHT`; both are deprecated.

Validation command:

```powershell
python tools\geometric_repeatability\validate_depth_reference_nativealign_package.py
```

Optional zip-integrity check:

```powershell
python tools\geometric_repeatability\validate_depth_reference_nativealign_package.py --check_zip
```

Regeneration command:

```powershell
python tools\geometric_repeatability\package_depth_reference_mask_nomask_v2.py `
  --formal_root G:\ACMMM26\GeometricRepeatability\DepthReference\Formal_5scene_8method_9thr_nativealign_20260508_v1 `
  --meshfix_root G:\ACMMM26\GeometricRepeatability\DepthReference\MeshFix_ROICrop_failed3_20260506_v1 `
  --out G:\ACMMM26\GeometricRepeatability\AssistantPack_DepthReference_MaskNoMask_5scene8method_nativealign_20260508_v1 `
  --recompute_metrics `
  --require_native_align
```
