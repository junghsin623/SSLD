# SSLD

SSLD is a navigation-guided lane-search system designed primarily for NuScenes front-camera data. It supports two operating modes:

- `gt`: uses ground-truth GPS and heading data to generate reference pseudo-labels.
- `osm`: uses an OSM navigation route, vehicle speed, and yaw rate for localization and lane search.

The project can generate CULane-style `.lines.txt` labels, matched BEV edge points, per-frame statistics, and debugging visualizations.

## Project structure

```text
codes/
├── ssld.py                 # Core SSLD localization and lane-search algorithm
├── bulk_inference.py       # GT-mode batch label generator
├── check_algorithm.py      # OSM-mode batch label generator
├── eval_algo.py            # Geometric comparison of GT and OSM outputs
├── eval_lane_level.py      # Lane IoU, precision, recall, and F1 evaluation
├── visualizer.py           # Image, vehicle-BEV, and map-BEV visualization
├── utils.py                # Route processing and ego-lane selection utilities
└── exp.py                  # Scene exclusion lists
```

## Requirements

Python 3.9 or newer is recommended. Install the main third-party dependencies with:

```bash
pip install numpy torch opencv-python scipy pandas joblib tqdm openpyxl
```

`openpyxl` is only required when `eval_algo.py` exports an Excel report.

The repository also depends on an external `tools` package that is not included in this directory.

[utils.py](codes/utils.py) imports shared route-processing utilities:

```python
from tools.utils import *
```

The package must provide functions used by SSLD, including `match_route`, `clip_en_route`, `map_en_route_to_veh_route`, `compute_walk_point`, `subdivide_route`, and related helpers.

[visualizer.py](codes/visualizer.py) also imports shared visualization helpers:

```python
from tools.visualizer import draw_grid_lines, truncate_route, combine_custom_dashboard
```

The external module must therefore provide `draw_grid_lines`, `truncate_route`, and `combine_custom_dashboard`.

Make sure the required `tools` package is available through `PYTHONPATH` before running any scripts.

## Dataset layout

The entry-point scripts currently use the following NuScenes paths by default:

```text
/data/NuScene/v1.0-trainval/
/data/NuScene/v1.0-trainval/sweeps/CAM_FRONT/split/
```

Each scene JSON file is expected at:

```text
<split_root>/<scene_id>/<scene_id>_culane.json
```

Each scene must contain camera calibration, frame data, and the route required by the selected mode:

- The GT generator reads `route`.
- The OSM generator reads `route_osm`.
- Frame data is expected to provide the image filename and relevant GPS, heading, speed, yaw-rate, and time-step values.

If your dataset or output paths differ, update the `Config` and artifact-root values in `bulk_inference.py` and `check_algorithm.py`, or pass a different `--split_root` to the evaluation scripts.

## Generate GT-mode labels

Run commands from the source directory:

```bash
cd codes
```

Process and visualize one scene:

```bash
python bulk_inference.py --scene 0001 --show
```

Generate labels and visualizations for the validation split:

```bash
python bulk_inference.py --val --label --show
```

Available split flags are `--train`, `--val`, and `--test`. Use `--scene <scene_id>` for single-scene processing.

The default output roots are defined in `bulk_inference.py`:

```text
/artifact/ssld/ssld_gt_label/v4_latest/
/artifact/ssld/ssld_gt_img/v4_latest/
```

## Generate OSM-mode labels

Process one scene in debug mode:

```bash
python check_algorithm.py --scene 0001 --show
```

Generate labels and visualizations for the validation split:

```bash
python check_algorithm.py --val --label --show
```

The default output roots are defined in `check_algorithm.py`:

```text
/artifact/ssld/ssld_osm_label/v4_latest_2/
/artifact/ssld/ssld_osm_img/v4_latest_2/
```

Batch processing is resumable: scenes with existing output are skipped. Scenes listed by `exp.py` may also be excluded according to the current configuration.

## Compare GT and OSM outputs

`eval_algo.py` compares two SSLD output sets using image-space lane lines and matched BEV points:

```bash
python eval_algo.py \
  --gt_dir /path/to/gt_labels \
  --osm_dir /path/to/osm_labels \
  --output_excel evaluation_results.xlsx
```

Useful options include:

- `--scene 0001`: evaluate one scene only.
- `--line_thresh_px 100`: image-space line-hit threshold.
- `--point_thresh_m 0.5`: BEV point-hit threshold.
- `--workers 8`: number of multiprocessing workers.
- `--split_root <path>`: scene-JSON root directory.

## Evaluate external lane predictions

`eval_lane_level.py` rasterizes lane lines and reports TP, FP, FN, precision, recall, and F1 based on mask IoU:

```bash
python eval_lane_level.py \
  --gt_dir /path/to/generated_labels \
  --pred_dir /path/to/prediction_txt \
  --line_width 150 \
  --iou_thresh 0.5 \
  --output_txt evaluation.txt \
  --output_scene_csv per_scene.csv
```

For models that predict several non-ego lanes, keep only the nearest left and right ego-lane boundaries before evaluation:

```bash
python eval_lane_level.py \
  --gt_dir /path/to/generated_labels \
  --pred_dir /path/to/prediction_txt \
  --pred_ego_only \
  --save_ego_pred_dir /path/to/filtered_predictions
```

Unreliable scenes can be excluded with `--exclude_scenes`, `--exclude_scene_file`, or the `EXCLUSIVE_SCENES` list in `eval_lane_level.py`.

## Output formats

### Lane labels

Each image produces one `.lines.txt` file. Every line represents one lane boundary as image coordinates:

```text
x1 y1 x2 y2 x3 y3 ...
```

OSM mode may produce an empty label when the frame was processed successfully but no lane boundary passed the quality checks. Such a file can be used as a negative sample.

### Matched points

Each `.matched.txt` file contains raw edge points selected in BEV space, measured in meters:

```text
x1 y1 x2 y2 x3 y3 ...
```

### Visualizations

The visualization pipeline can render:

- The source camera image with pseudo-labels.
- Matched edge points.
- Vehicle-centric BEV.
- Map-centric BEV with pose history.
- Optional 3D route-carpet debugging infrastructure.

## Notes

- Several dataset and artifact paths are currently hard-coded and must be updated when moving to another environment.
- `exp.py` currently focuses on excluding scenes without a usable OSM route.
- Lane widths, search thresholds, turn protection, and quality filters can be tuned in `ssld.py`.
- Saving visualizations significantly increases runtime and disk usage.
- Validate calibration and input fields with `--scene` before processing a complete split.
