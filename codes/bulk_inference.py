# filename: bulk_inference.py
# -*- coding: utf-8 -*-
"""
[GT Generator] Batch inference for ground-truth pseudo-label generation.
Runs in 'gt' mode and stores flat labels plus scene-grouped visualizations.
"""
import os
import random
import torch
import numpy as np
import argparse
import shutil
import traceback
import cv2
from joblib import Parallel, delayed
import gc

# Core modules
import ssld
import utils
import visualizer as vis
import exp

def parse_args():
    parser = argparse.ArgumentParser(description="SSLD GT Generation")
    
    # Dataset split
    parser.add_argument('--train', action='store_true', help='Process Train split')
    parser.add_argument('--val', action='store_true', help='Process Validation split')
    parser.add_argument('--test', action='store_true', help='Process Test split')
    
    # Optional single-scene selection
    parser.add_argument('--scene', type=str, default=None, help='Run specific scene ID (e.g. 0001)')

    # Feature switches
    parser.add_argument('--label', action='store_true', help='Generate label txt files')
    parser.add_argument('--show', action='store_true', help='Generate visualization images')
    
    return parser.parse_args()

class Config:
    def __init__(self, label_root, viz_root):
        self.img_root = "/data/NuScene/v1.0-trainval/"
        self.split_root = "/data/NuScene/v1.0-trainval/sweeps/CAM_FRONT/split"
        
        # Separate label and visualization outputs.
        self.label_dir = label_root  # e.g., "/artifact/ssld_gt_label/val_label"
        self.viz_dir = viz_root      # e.g., "/artifact/ssld_gt_img/val_img"
        
        # Pseudo-label sampling
        self.sample_y_start = 890
        self.sample_y_end = 500
        self.sample_step = 10
        
        # Visualization
        self.bin_deg = 15
        self.arrows_per_frame = 0  # Do not draw random points in GT mode.
        
        # Create only enabled output directories.
        if self.label_dir:
            utils.ensure_dir(self.label_dir)
        if self.viz_dir:
            utils.ensure_dir(self.viz_dir)

def save_pseudo_label(cfg, Hpx, ssld_cfg, filename, lane_results):
    """
    Generate a .lines.txt file in label_dir.
    """
    if not lane_results or not cfg.label_dir: return False

    # Keep the label directory flat.
    basename = os.path.basename(filename)
    txt_name = os.path.splitext(basename)[0] + ".lines.txt"
    txt_path = os.path.join(cfg.label_dir, txt_name)

    y_anchors = np.arange(cfg.sample_y_start, cfg.sample_y_end, -cfg.sample_step)
    img_w = 1600
    lines_to_write = []

    for key in ['best_left_lane', 'best_right_lane']:
        bev_lane = lane_results.get(key)
        if bev_lane is None or len(bev_lane) < 2: continue
            
        try:
            img_pts = vis.project_bev_poly_to_image(bev_lane, Hpx, ssld_cfg)
        except Exception: continue
            
        if len(img_pts) < 2: continue

        # Sort by image y from bottom to top.
        sort_idx = np.argsort(img_pts[:, 1])[::-1] 
        img_pts_sorted = img_pts[sort_idx]
        
        xs, ys = img_pts_sorted[:, 0], img_pts_sorted[:, 1]
        
        # Interpolate samples.
        ys_increasing = ys[::-1]
        xs_corresponding = xs[::-1]
        
        interp_xs = np.interp(y_anchors, ys_increasing, xs_corresponding, left=-9999, right=-9999)
        
        line_coords = []
        for x, y in zip(interp_xs, y_anchors):
            if x != -9999 and 0 <= x < img_w:
                line_coords.append(f"{x:.2f} {y:.2f}")
        
        if len(line_coords) >= 2:
            lines_to_write.append(" ".join(line_coords))

    if lines_to_write:
        with open(txt_path, "w") as f:
            for line_str in lines_to_write:
                f.write(line_str + "\n")
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
    
    # Collect matched points from both sides.
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

def process_scene(scene_id, cfg, args, split_name):
    valid_image_paths = []
    print(f"[Worker] Starting Scene {scene_id} ...", end="\n", flush=True)
    
    ssld_instance = ssld.SSLD(mode='gt')
    
    try:
        data = utils.load_scene_json(scene_id, cfg.split_root)
    except Exception:
        print(f"[Skip] Load failed: {scene_id}")
        return valid_image_paths 

    frames_all = data.get("frame_data", [])
    raw_route_data = data.get("route")
    if not (isinstance(raw_route_data, list) and len(raw_route_data) >= 2):
        return valid_image_paths
        
    en_route_en = torch.tensor(raw_route_data, dtype=torch.float64)

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

    scene_viz_dir = None
    if args.show and cfg.viz_dir:
        scene_viz_dir = os.path.join(cfg.viz_dir, utils.norm_scene_id(scene_id))
        
    class TempVizCfg:
        def __init__(self, out_dir, s_id, bin_deg): 
            self.out_dir = out_dir; self.scene_id = s_id; self.bin_deg = bin_deg

    viz_dir_created = False
    
    for i, fr in enumerate(frames_all):
        if not isinstance(fr, dict): continue
        
        is_debug_mode = (split_name == 'debug')
        need_viz = args.show and (is_debug_mode or (i % 10 == 0))
        
        if args.show and not args.label and not need_viz:
            continue

        fr["gps_to_use"] = fr.get("gps_gt")
        fr["heading_gt"] = fr.get("heading_gt")
        img_path = os.path.join(cfg.img_root, fr["filename"])

        results = ssld_instance.process_one_frame(
            img_path, homography, fr, i, en_route_en
        )

        if not results:
            continue
        
        success = False
        if args.label:
            success = save_pseudo_label(cfg, Hpx, ssld_instance, fr["filename"], results.get("lane_search_results"))
            save_matched_points(cfg, fr["filename"], results.get("lane_search_results"))
            if success:
                valid_image_paths.append(os.path.abspath(img_path))

        if need_viz and scene_viz_dir and success:
            if not viz_dir_created:
                utils.ensure_dir(scene_viz_dir)
                viz_dir_created = True
                
            temp_viz_cfg = TempVizCfg(scene_viz_dir, scene_id, cfg.bin_deg)
            
            ssld.visualize_frame_results(
                cfg=ssld_instance, Hpx=Hpx, idx=i, results=results,
                vis_settings=temp_viz_cfg,
                arrows_per_frame=0,
                prev_poses=ssld_instance.prev_poses, 
                en_route_en=en_route_en,
                calib_data=calib_data
            )

    stats_out_dir = cfg.label_dir if cfg.label_dir else cfg.viz_dir
    if stats_out_dir:
        parent_dir = os.path.dirname(stats_out_dir)
        ssld_instance.export_statistics(out_dir=parent_dir)

    print(f"[Worker] Finished Scene {scene_id}. Extracted {len(valid_image_paths)} labels.", flush=True)
    
    del ssld_instance
    gc.collect()
    return valid_image_paths


def process_split(split_name, scenes_list, args):
    base_label_artifact = "/artifact/ssld/ssld_gt_label/v4_latest"
    base_img_artifact = "/artifact/ssld/ssld_gt_img/v4_latest"
    
    label_root = os.path.join(base_label_artifact, f"{split_name}_label") if args.label else None
    viz_root = os.path.join(base_img_artifact, f"{split_name}_img") if args.show else None
    
    print(f"\n>>> Processing Split: [{split_name}] (Mode: GT)")
    if args.label: 
        print(f"    Label Output: {label_root}")
        utils.ensure_dir(label_root) 
    if args.show:  
        print(f"    Image Output: {viz_root}")
        utils.ensure_dir(viz_root)
    
    cfg = Config(label_root, viz_root)
    ex_train, ex_val = exp.get_exclusive_sets()
    
    exclusion_set = set()
    if split_name == 'train':
        exclusion_set = ex_train
    elif split_name == 'val':
        exclusion_set = ex_val
    
    scenes_to_process = []
    for scene_id in scenes_list:
        if scene_id in exclusion_set:
            print(f"Skipping {scene_id} (in exclusive list)")
            continue

        # Resume from existing outputs.
        if viz_root:
            scene_viz_dir = os.path.join(viz_root, utils.norm_scene_id(scene_id))
            if os.path.exists(scene_viz_dir) and len(os.listdir(scene_viz_dir)) > 0:
                print(f"[{scene_id}] [ALREADY PROCESSED - SKIPPED]")
                continue
        scenes_to_process.append(scene_id)

    print(f"Total scenes to process in {split_name}: {len(scenes_to_process)}")

    num_jobs = 12 
    print(f"Starting parallel processing with {num_jobs} jobs...")
    
    results_list = Parallel(n_jobs=num_jobs)(
        delayed(process_scene)(sid, cfg, args, split_name) for sid in scenes_to_process
    )
    
    all_valid_image_paths = []
    for paths in results_list:
        all_valid_image_paths.extend(paths)

    if args.label and label_root and all_valid_image_paths:
        print(f"\n    Writing list file for [{split_name}]...")
        list_dir = os.path.join(label_root, "list")
        utils.ensure_dir(list_dir)
        
        list_file_path = os.path.join(list_dir, "train.txt")
        
        # Append without replacing previous entries.
        with open(list_file_path, "a") as f:
            for p in all_valid_image_paths:
                f.write(p + "\n")
                
        print(f"    -> Done. Total {len(all_valid_image_paths)} frames labeled appended.")

def main():
    args = parse_args()
    
    # Default to visualization for a single scene when no action is selected.
    if args.scene and not (args.label or args.show):
        print("Debug mode detected: Enabling --show by default.")
        args.show = True

    if not (args.label or args.show):
        print("Please specify action: --label (generate txt) or --show (generate viz) or both.")
        return

    splits = utils.create_splits_scenes()
    tasks = []

    # Single-scene mode takes precedence.
    if args.scene:
        sid = utils.norm_scene_id(args.scene)
        target_scene = f"scene-{sid}"
        # Name single-scene output 'debug'.
        tasks.append(('debug', [target_scene]))
    else:
        # Otherwise run batch tasks.
        if args.train: tasks.append(('train', splits['train']))
        if args.val:   tasks.append(('val',   splits['val']))
        if args.test:  tasks.append(('test',  splits['test']))
    
    if not tasks:
        print("Please specify a target: --train, --val, --test, or --scene [ID]")
        return

    for split_name, scenes in tasks:
        try:
            process_split(split_name, scenes, args)
        except KeyboardInterrupt:
            print("\n[Stopped] User interrupted.")
            break
        except Exception:
            traceback.print_exc()

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()
