# filename: check_algorithm.py
# -*- coding: utf-8 -*-
"""
[OSM Generator] Batch inference in OSM mode.
Generates OSM-guided pseudo-labels and visualizations.
Outputs:
    1. Label: /artifact/ssld_osm_label/[split]_label/*.txt
    2. Image: /artifact/ssld_osm_img/[split]_img/[scene_id]/*.png
"""
import os
import random
import torch
import numpy as np
import argparse
import shutil
import traceback
import cv2
cv2.setNumThreads(0)
import gc
from joblib import Parallel, delayed

# Core modules
import ssld
import utils
import visualizer as vis
import exp

def parse_args():
    parser = argparse.ArgumentParser(description="SSLD Algorithm Check (OSM Mode)")
    parser.add_argument('--train', action='store_true', help='Run on Train split')
    parser.add_argument('--val', action='store_true', help='Run on Validation split')
    parser.add_argument('--test', action='store_true', help='Run on Test split')
    parser.add_argument('--scene', type=str, default=None, help='Run specific scene ID (debug mode)')
    
    # Feature switches aligned with bulk_inference.py.
    parser.add_argument('--label', action='store_true', help='Generate label txt files')
    parser.add_argument('--show', action='store_true', help='Generate visualization images')
    
    return parser.parse_args()

class Config:
    def __init__(self, label_root, viz_root):
        self.img_root = "/data/NuScene/v1.0-trainval/"
        self.split_root = "/data/NuScene/v1.0-trainval/sweeps/CAM_FRONT/split"
        
        # Output settings
        self.label_dir = label_root
        self.viz_dir = viz_root
        
        # Pseudo-label sampling aligned with GT mode.
        self.sample_y_start = 890
        self.sample_y_end = 500
        self.sample_step = 10
        
        # Visualization settings
        self.bin_deg = 15
        self.arrows_per_frame = 220  # Retain arrows for grid-search debugging.
        self.out_dir = viz_root  # Compatibility with the legacy visualizer interface.
        
        if self.label_dir: utils.ensure_dir(self.label_dir)
        if self.viz_dir: utils.ensure_dir(self.viz_dir)

def save_pseudo_label(cfg, Hpx, ssld_cfg, filename, lane_results, allow_empty=True):
    """
    Generate a .lines.txt file.
    Write lane coordinates when reliable lanes exist, otherwise an empty negative sample.
    """
    if not cfg.label_dir:
        return False

    basename = os.path.basename(filename)
    txt_name = os.path.splitext(basename)[0] + ".lines.txt"
    txt_path = os.path.join(cfg.label_dir, txt_name)

    # Missing lane_results indicates failure or incomplete initialization, not a negative sample.
    if not lane_results:
        return False

    y_anchors = np.arange(cfg.sample_y_start, cfg.sample_y_end, -cfg.sample_step)
    img_w = 1600
    lines_to_write = []

    for key in ['best_left_lane', 'best_right_lane']:
        bev_lane = lane_results.get(key)
        if bev_lane is None or len(bev_lane) < 2:
            continue

        try:
            img_pts = vis.project_bev_poly_to_image(bev_lane, Hpx, ssld_cfg)
        except Exception:
            continue

        if len(img_pts) < 2:
            continue

        sort_idx = np.argsort(img_pts[:, 1])[::-1]
        img_pts_sorted = img_pts[sort_idx]

        xs, ys = img_pts_sorted[:, 0], img_pts_sorted[:, 1]

        ys_increasing = ys[::-1]
        xs_corresponding = xs[::-1]

        interp_xs = np.interp(
            y_anchors,
            ys_increasing,
            xs_corresponding,
            left=-9999,
            right=-9999
        )

        line_coords = []
        for x, y in zip(interp_xs, y_anchors):
            if x != -9999 and 0 <= x < img_w:
                line_coords.append(f"{x:.2f} {y:.2f}")

        if len(line_coords) >= 2:
            lines_to_write.append(" ".join(line_coords))

    # Write detected lanes.
    if lines_to_write:
        with open(txt_path, "w") as f:
            for line_str in lines_to_write:
                f.write(line_str + "\n")
        return True

    # Write an empty negative label when no reliable lane remains.
    if allow_empty:
        open(txt_path, "w").close()
        return True

    return False

def save_matched_points(cfg, filename, lane_results):
    """
    Save selected raw BEV edge points in meter coordinates.
    """
    if not lane_results or not cfg.label_dir: return False

    # Store matched points beside the .lines.txt file.
    basename = os.path.basename(filename)
    txt_name = os.path.splitext(basename)[0] + ".matched.txt"
    txt_path = os.path.join(cfg.label_dir, txt_name)

    all_pts_str = []
    
    # Collect matched BEV points from both sides.
    for key in ['matched_left_pts', 'matched_right_pts']:
        pts = lane_results.get(key)
        if pts is not None and len(pts) > 0:
            # Serialize each [x, y] coordinate.
            for p in pts:
                all_pts_str.append(f"{p[0]:.4f} {p[1]:.4f}")

    # Write the output file.
    if all_pts_str:
        with open(txt_path, "w") as f:
            f.write("\n".join(all_pts_str))
        return True
    return False

def save_empty_side_by_side_frame(scene_viz_dir, scene_id, idx, img_path, speed=None):
    """
    Save frames rejected during initialization or early filtering as an unannotated side-by-side image.
    """
    img = cv2.imread(img_path)
    if img is None:
        return False

    left_img = img.copy()
    right_img = img.copy()

    # Annotate that the frame produced no lane.
    text = "No lane output"
    if speed is not None:
        text += f" | speed={speed:.2f}"

    cv2.putText(
        right_img, text, (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9, (0, 0, 0), 4, cv2.LINE_AA
    )
    cv2.putText(
        right_img, text, (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9, (255, 255, 255), 2, cv2.LINE_AA
    )

    side_by_side = cv2.hconcat([left_img, right_img])

    utils.ensure_dir(scene_viz_dir)

    sid = utils.norm_scene_id(scene_id)
    out_path = os.path.join(
        scene_viz_dir,
        f"e_side_by_side_fit_{sid}_{idx:04d}.png"
    )

    cv2.imwrite(out_path, side_by_side)
    return True

def process_scene(scene_id, cfg, args):
    """
    Process one scene with isolated output directories for parallel safety.
    """
    positive_labels = 0
    empty_labels = 0

    # Track image paths for this scene.
    valid_image_paths = []
    
    # Report progress independently for each worker.
    print(f"[Worker] Starting Scene {scene_id} ...", end="\n", flush=True)
    
    ssld_instance = ssld.SSLD(mode='osm')
    
    try:
        data = utils.load_scene_json(scene_id, cfg.split_root)
    except Exception as e:
        return valid_image_paths 

    frames_all = data.get("frame_data", [])

    raw_route_data = data.get("route_osm")
    if not (isinstance(raw_route_data, list) and len(raw_route_data) >= 2):
        return valid_image_paths

    raw_route_tensor = torch.tensor(raw_route_data, dtype=torch.float64)
    en_route_en = utils.subdivide_route(raw_route_tensor, step=1.0, smooth_iters=10)

    homography = ssld.compute_inverse_perspective_homography(
        torch.tensor(data["ego2cam_rot"], dtype=torch.float64),
        torch.tensor(data["ego2cam_trans"], dtype=torch.float64),
        torch.tensor(data["calib_mat"], dtype=torch.float64)
    )
    Hpx, PPM = ssld.img2bev_Hpx(homography, ssld_instance)
    
    calib_data = {
        'rot': torch.tensor(data["ego2cam_rot"], dtype=torch.float64),
        'trans': torch.tensor(data["ego2cam_trans"], dtype=torch.float64),
        'mat': torch.tensor(data["calib_mat"], dtype=torch.float64)
    }

    # ==========================================
    # Delay creation of the isolated scene output directory.
    # ==========================================
    scene_viz_dir = None
    if args.show and cfg.viz_dir:
        # Define the path without creating it yet.
        scene_viz_dir = os.path.join(cfg.viz_dir, utils.norm_scene_id(scene_id))
        
    class TempVizCfg:
        def __init__(self, out_dir, s_id, bin_deg): 
            self.out_dir = out_dir
            self.scene_id = s_id
            self.bin_deg = bin_deg

    total_frames = len(frames_all)
    
    # Track directory creation and the first successful frame.
    viz_dir_created = False
    first_success_found = False 

    for i, fr in enumerate(frames_all):
        if not isinstance(fr, dict): continue

        img_path = os.path.join(cfg.img_root, fr["filename"])
        fr["gps_to_use"] = fr.get("gps_perturbed") or fr.get("gps_gt")

        results = ssld_instance.process_one_frame(
            img_path, homography, fr, i,
            en_route_en, scene_id=scene_id
        )

        # Skip frames rejected during initialization or without lane results.
        if not results:
            if args.show and scene_viz_dir:
                if not viz_dir_created:
                    utils.ensure_dir(scene_viz_dir)
                    viz_dir_created = True

                save_empty_side_by_side_frame(
                    scene_viz_dir=scene_viz_dir,
                    scene_id=scene_id,
                    idx=i,
                    img_path=img_path,
                    speed=fr.get("speed", None)
                )

            continue
        
        # ==========================================
        # Keep visualization enabled even when labels are not requested.
        # ==========================================
        success = True 
        
        if args.label:
            success = save_pseudo_label(
                cfg, Hpx, ssld_instance,
                fr["filename"],
                results.get("lane_search_results"),
                allow_empty=True
            )

            save_matched_points(cfg, fr["filename"], results.get("lane_search_results"))

            if success:
                full_path = os.path.abspath(img_path)
                valid_image_paths.append(full_path)

                label_path = os.path.join(
                    cfg.label_dir,
                    os.path.splitext(os.path.basename(fr["filename"]))[0] + ".lines.txt"
                )

                if os.path.getsize(label_path) == 0:
                    empty_labels += 1
                else:
                    positive_labels += 1

        # ==========================================
        # Visualize the first success and subsequent ten-frame intervals.
        # ==========================================
        is_first_success = not first_success_found
        if success and is_first_success:
            first_success_found = True

        # Force visualization for the first active frame or every tenth frame.
        need_viz = args.show# and (is_first_success or (i % 10 == 0))

        if need_viz and scene_viz_dir and success:
            
            # Create the directory only when an image will be saved.
            if not viz_dir_created:
                utils.ensure_dir(scene_viz_dir)
                viz_dir_created = True
                
            # Build an isolated visualization configuration for every frame.
            safe_viz_settings = TempVizCfg(scene_viz_dir, scene_id, cfg.bin_deg)
            
            ssld.visualize_frame_results(
                cfg=ssld_instance, Hpx=Hpx, idx=i, results=results,
                vis_settings=safe_viz_settings, 
                arrows_per_frame=cfg.arrows_per_frame,
                prev_poses=ssld_instance.prev_poses,  
                en_route_en=en_route_en,
                calib_data=calib_data
            )

    stats_out_dir = cfg.label_dir if cfg.label_dir else cfg.viz_dir
    if stats_out_dir:
        parent_dir = os.path.dirname(stats_out_dir)
        ssld_instance.export_statistics(out_dir=parent_dir)

    print(
        f"[Worker] Finished Scene {scene_id}. "
        f"Positive labels: {positive_labels}, Empty labels: {empty_labels}, "
        f"Total used: {len(valid_image_paths)}",
        flush=True
    )

    del ssld_instance
    gc.collect()

    return valid_image_paths

def main():
    args = parse_args()

    # Validate requested actions.
    if not (args.label or args.show):
        print("Please specify action: --label (generate txt) or --show (generate viz) or both.")
        # Default to visualization when no action flags are supplied.
        if args.scene:
            print("Debug mode detected: enabling --show by default.")
            args.show = True
        else:
            return

    # OSM-specific output paths.
    base_label_artifact = "/artifact/ssld/ssld_osm_label/v4_latest_2"
    base_img_artifact = "/artifact/ssld/ssld_osm_img/v4_latest_2"

    ex_train, ex_val = exp.get_exclusive_sets()

    # Resolve the scene list.
    splits = utils.create_splits_scenes()
    tasks = []
    
    if args.scene:
        # Single-scene debug mode.
        sid = utils.norm_scene_id(args.scene)
        # Keep debug output in output_osm for easier inspection.
        debug_out = f"output_osm_new_vis/scene-{sid}"
        tasks.append(('debug', [f"scene-{sid}"], debug_out, debug_out))
    else:
        # Batch mode.
        if args.train: tasks.append(('train', splits['train'], 
                                     os.path.join(base_label_artifact, "train_label"), 
                                     os.path.join(base_img_artifact, "train_img")))
        if args.val:   tasks.append(('val',   splits['val'],
                                     os.path.join(base_label_artifact, "val_label"),
                                     os.path.join(base_img_artifact, "val_img")))
        if args.test:  tasks.append(('test',  splits['test'],
                                     os.path.join(base_label_artifact, "test_label"),
                                     os.path.join(base_img_artifact, "test_img")))

    if not tasks:
        print("Please specify a target: --train, --val, --test, or --scene [ID]")
        return

    # Run the selected tasks.
    for split_name, scenes_list, label_path, img_path in tasks:
        # Enable output paths according to feature flags.
        target_label_dir = label_path if args.label else None
        target_viz_dir = img_path if args.show else None
        
        print(f"\n>>> Processing Split: [{split_name}] (Mode: OSM)")
        if args.label: 
            print(f"    Label Output: {target_label_dir}")
            # Preserve existing output for resumable processing.
            utils.ensure_dir(target_label_dir) 
        if args.show:  
            print(f"    Image Output: {target_viz_dir}")
            # Preserve existing output for resumable processing.
            utils.ensure_dir(target_viz_dir)

        cfg = Config(target_label_dir, target_viz_dir)
        all_valid_image_paths = []

        # Prepare scene exclusions.
        exclusion_set = set()
        if split_name == 'train':
            exclusion_set = ex_train
        elif split_name == 'val':
            exclusion_set = ex_val
            
        # Filter scenes for resumable processing.
        scenes_to_process = []
        for sid in scenes_list:
            if sid in exclusion_set:
                print(f"Skipping {sid} (in exclusive list)")
                continue

            # Skip scenes with existing outputs.
            if target_viz_dir:
                scene_viz_dir = os.path.join(target_viz_dir, utils.norm_scene_id(sid))
                if os.path.exists(scene_viz_dir) and len(os.listdir(scene_viz_dir)) > 0:
                    print(f"[{sid}] [ALREADY PROCESSED - SKIPPED]")
                    continue
            scenes_to_process.append(sid)

        print(f"Total scenes to process in {split_name}: {len(scenes_to_process)}")

        # Process scenes in parallel; keep n_jobs conservative to limit memory use.
        num_jobs = 12 
        print(f"Starting parallel processing with {num_jobs} jobs...")
        
        results_list = Parallel(n_jobs=num_jobs)(
            delayed(process_scene)(sid, cfg, args) for sid in scenes_to_process
        )
        
        # Collect successful image paths.
        for paths in results_list:
            all_valid_image_paths.extend(paths)

        # Write the image-list file.
        if args.label and target_label_dir and all_valid_image_paths:
            print(f"\n    Writing list file for [{split_name}]...")
            list_dir = os.path.join(target_label_dir, "list")
            utils.ensure_dir(list_dir)
            # Append without replacing previous entries.
            list_file_path = os.path.join(list_dir, "train.txt")
            
            with open(list_file_path, "a") as f:
                for p in all_valid_image_paths:
                    f.write(p + "\n")
            print(f"    -> Done. Total {len(all_valid_image_paths)} frames labeled appended.")

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()
