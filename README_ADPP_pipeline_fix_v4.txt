ADPP pipeline consolidated fix (v4)

Files:
- run_gtgs_full_pipeline.py

What it fixes:
1) Restores full CLI (e.g., --train_preset, --blend_endpoint_mode, --rgb_adp_*, --adpp_*, --adpp_disable_edge_loss)
2) Adds --save_iterations / --test_iterations to the internal train.py commands so the final point_cloud is saved at:
   - RGB: 7000 and rgb_iter
   - Thermal: t_iter
3) If the requested thermal point_cloud at t_iter is missing, the pipeline will automatically fall back to the latest available
   point_cloud iteration (and print a WARN), so blending can proceed without crashing.

How to install:
- Copy run_gtgs_full_pipeline.py into your repo root (same folder as train.py), overwriting the old one.

Quick verification:
- python run_gtgs_full_pipeline.py -h   (you should see --train_preset and --blend_endpoint_mode in help)

