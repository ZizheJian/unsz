# Current coordinate-transform results

This file records coordinate-transform comparisons that use the current formal RBF configuration. Keep both explicit `off` and explicit `on` rows: after the automatic decision rule is fixed, plotting can select the corresponding rows without rerunning RD experiments.

Current formal spatial parameters:

- RBF kernel: `exponential`
- RBF shape: `0.01`
- RBF nugget: `1e-6`
- prediction references: `10/10`
- ordering: `global`,one-point MSD,16 candidates
- EB schedule: `inverse_rank_log2`,base `0.8`,shift `0.5`,maximum exponent `7`
- coordinate-transform training: first-sample ratio `0.3`,second-sample count `3000`,sample-file count `3`

## Reusable complete curves

| Dataset | Fields | Result file | Explicit task names | RD | Throughput |
|---|---|---|---|---|---|
| Tsunami | all six fields | `tsunami_geometry_screen.csv` | `tsunami_coordinate_off` / `tsunami_coordinate_on` | ready,5 REL points per mode | ready in `tsunami_off_on_timing_median.csv`,using the median of 7 allocated-node repetitions |
| E3SM-h0 | CLDTOT,FLNS,PRECC,PS,TAUX,TREFHT,U10 | `../remaining_adjust_coord_screen/pure_spatial_curves.csv` | `e3sm_h0_single_set_auto_screen_curve_off` / `e3sm_h0_single_set_auto_screen_curve_on` | ready,5 REL points per mode | available in the same rows |
| ADCIRC Haiyan | depth | `../remaining_adjust_coord_screen/pure_spatial_curves.csv` | `haiyan_depth_auto_screen_curve_off` / `haiyan_depth_auto_screen_curve_on` | ready,5 REL points per mode | available in the same rows |
| Ocean | all nine fields using the selected flow recorded below | `../remaining_staged_flow_screen/ocean.csv` / `../remaining_adjust_coord_screen/temporal_forced.csv` | `ocean_value_flow_screen` / `ocean_adjust_on` | ready,5 REL points per mode; the auto rows are known to be off because every recorded gain is below 0.2 | available in the same rows |
| Tpost20 | scalar,linear-interpolation spatial-residual flow | `../remaining_staged_flow_screen/tpost20.csv` / `../remaining_staged_flow_screen/coordinate_mode_rescreen.csv` | `tpost20_value_flow_screen` / `tpost20_flow_off` | ready,4 REL points per mode; the auto rows are known to be on from gain 0.376398 | available in the same rows |
| Katrina | attr0 | `katrina_current_off_on.csv` | `katrina_coordinate_off` / `katrina_coordinate_on` | ready,5 point-set-config points per mode | ready,measured on the same allocated node |

The CSV rows retain compression time,decompression time,and both preprocessing times,so the same experiments can feed RD and throughput plots. Coordinate-transform model files are reusable metadata and are not counted toward compressed bytes.

Tsunami timing repetitions are preserved in `../current/formal_runs/tsunami_coordinate_timing_21067449.log` and `../current/formal_runs/tsunami_coordinate_timing_21067589.log`. `summarize_tsunami_timing.py` deterministically regenerates the median timing CSV.

## Results that must not define the new thresholds

- `e3sm.csv`,`haiyan.csv`,`katrina.csv`,`les.csv`,and`tsunami.csv` predate the current formal RBF/configuration update. They are historical results only.
- `remaining_adjust_coord_screen/taps.csv` and `taps_opposite.csv` contain only one REL point per mode. They are useful screens but are not complete RD curves.
- LES,E3SM-h4,andHaiyan-zeta do not yet have a clean current opposite-mode comparison for the selected final flow.
- An `auto` curve is never treated as an explicit `off` or `on` baseline unless the recorded transform decision is also available.

