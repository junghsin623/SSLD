import os
import numpy as np
import argparse
from tqdm import tqdm
import json
from scipy.spatial import cKDTree
import pandas as pd  
# Multiprocessing support
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SSLD Precision (Multi-processing)")
    parser.add_argument('--gt_dir', type=str, required=True, help='Path to GT labels')
    parser.add_argument('--osm_dir', type=str, required=True, help='Path to OSM labels')
    parser.add_argument('--scene', type=str, default=None, help='Specific scene ID (e.g., 0007)')
    parser.add_argument('--split_root', type=str, default="/data/NuScene/v1.0-trainval/sweeps/CAM_FRONT/split")
    parser.add_argument('--line_thresh_px', type=float, default=100.0, help='Line hit threshold in pixels')
    parser.add_argument('--point_thresh_m', type=float, default=0.5, help='Point hit threshold in meters')
    parser.add_argument('--output_excel', type=str, default="evaluation_results.xlsx", help='Path to output Excel file')
    # Default to 80% of available CPUs to keep the system responsive.
    parser.add_argument('--workers', type=int, default=max(1, int(multiprocessing.cpu_count() * 0.8)), help='Number of CPU workers')
    return parser.parse_args()

def load_lane_lines(path):
    lanes = []
    if not os.path.exists(path): return lanes
    with open(path, 'r') as f:
        for line in f:
            pts = np.array(line.strip().split(), dtype=float).reshape(-1, 2)
            lanes.append(pts[np.argsort(pts[:, 1])])
    return lanes

def load_matched_points(path):
    pts = []
    if not os.path.exists(path): return np.empty((0, 2))
    with open(path, 'r') as f:
        for line in f:
            data = line.strip().split()
            if len(data) == 2:
                pts.append([float(data[0]), float(data[1])])
    return np.array(pts)

def classify_left_right(lanes):
    if len(lanes) == 0:
        return None, None
    elif len(lanes) == 1:
        avg_x = np.mean(lanes[0][:, 0])
        return (lanes[0], None) if avg_x < 800 else (None, lanes[0])
    else:
        avg_x0 = np.mean(lanes[0][:, 0])
        avg_x1 = np.mean(lanes[1][:, 0])
        return (lanes[0], lanes[1]) if avg_x0 < avg_x1 else (lanes[1], lanes[0])

def _compare_single_line(gt_line, osm_line, threshold_px):
    if osm_line is None or gt_line is None:
        return 0, 0

    y_min = max(osm_line[:, 1].min(), 450)
    y_max = min(osm_line[:, 1].max(), 890)
    if y_max <= y_min: return 0, 0
    
    osm_y_samples = np.arange(y_min, y_max, 5)
    total = len(osm_y_samples)

    overlap_y_min = max(gt_line[:, 1].min(), y_min)
    overlap_y_max = min(gt_line[:, 1].max(), y_max)
    
    hits = 0
    if overlap_y_max > overlap_y_min:
        overlap_samples = np.arange(overlap_y_min, overlap_y_max, 5)
        gt_x = np.interp(overlap_samples, gt_line[:, 1], gt_line[:, 0])
        osm_x = np.interp(overlap_samples, osm_line[:, 1], osm_line[:, 0])
        diff = np.abs(gt_x - osm_x)
        hits = np.sum(diff < threshold_px)

    return hits, total

def get_line_hit_rate(gt_lanes, osm_lanes, threshold_px):
    gt_l, gt_r = classify_left_right(gt_lanes)
    osm_l, osm_r = classify_left_right(osm_lanes)
    
    h_l, t_l = _compare_single_line(gt_l, osm_l, threshold_px)
    h_r, t_r = _compare_single_line(gt_r, osm_r, threshold_px)
    
    return h_l + h_r, t_l + t_r

def get_point_hit_rate(gt_pts, osm_pts, threshold_m):
    if len(osm_pts) == 0 or len(gt_pts) == 0:
        return 0, 0
    
    tree = cKDTree(gt_pts)
    dist, _ = tree.query(osm_pts, k=1)
    hits = np.sum(dist < threshold_m)
    return int(hits), len(osm_pts)

# =========================================================================
# Keep the per-frame worker at module scope so multiprocessing can pickle it.
# =========================================================================
def process_single_frame(base, current_sid, gt_dir, osm_dir, line_thresh, point_thresh):
    # Load lane lines.
    gt_l = load_lane_lines(os.path.join(gt_dir, base + ".lines.txt"))
    osm_l = load_lane_lines(os.path.join(osm_dir, base + ".lines.txt"))
    l_h, l_t = get_line_hit_rate(gt_l, osm_l, line_thresh)
    
    # Load matched points.
    gt_p = load_matched_points(os.path.join(gt_dir, base + ".matched.txt"))
    osm_p = load_matched_points(os.path.join(osm_dir, base + ".matched.txt"))
    p_h, p_t = get_point_hit_rate(gt_p, osm_p, point_thresh)
    
    # Return frame statistics.
    return {
        'sid': current_sid,
        'l_h': l_h,
        'l_t': l_t,
        'p_h': p_h,
        'p_t': p_t
    }

def main():
    args = parse_args()
    
    EXCLUDED_SCENES = [
        # '0017', '0039', '0097', '0101', '0103', '0268', '0271', '0273', '0276', '0277', '0346', '0524', '0557', '0638', '0783', '0795', '0909', '0930', '0931', '0962', '0966', '0969', '1065', '1072'
    ]
    
    all_excluded_scenes = {f"scene-{s}" if not s.startswith("scene-") else s for s in EXCLUDED_SCENES}

    excluded_basenames = set()
    if all_excluded_scenes:
        for scene_id in all_excluded_scenes:
            sid = scene_id.replace("scene-", "")
            json_p = os.path.join(args.split_root, sid, f"{sid}_culane.json")
            if os.path.exists(json_p):
                with open(json_p, 'r') as f:
                    data = json.load(f)
                    for fr in data.get('frame_data', []):
                        base = os.path.basename(fr['filename']).split('.')[0]
                        excluded_basenames.add(base)

    frame_to_scene = {}
    if os.path.exists(args.split_root):
        for sid in os.listdir(args.split_root):
            json_p = os.path.join(args.split_root, sid, f"{sid}_culane.json")
            if os.path.exists(json_p):
                with open(json_p, 'r') as f:
                    data = json.load(f)
                    for fr in data.get('frame_data', []):
                        base = os.path.basename(fr['filename']).split('.')[0]
                        frame_to_scene[base] = sid

    if args.scene:
        sid = args.scene.replace("scene-", "")
        target_scene = f"scene-{sid}"
        
        if target_scene in all_excluded_scenes:
            print(f"[Warning] The requested scene {args.scene} is currently excluded.")
            
        json_p = os.path.join(args.split_root, sid, f"{sid}_culane.json")
        base_names = []
        if os.path.exists(json_p):
            with open(json_p, 'r') as f:
                data = json.load(f)
                for fr in data['frame_data']:
                    base = os.path.basename(fr['filename']).split('.')[0]
                    if os.path.exists(os.path.join(args.osm_dir, base + ".lines.txt")):
                        if base not in excluded_basenames:
                            base_names.append(base)
        print(f">>> Evaluating Scene: {args.scene} ({len(base_names)} valid OSM frames found)")
    else:
        raw_base_names = [f.replace('.lines.txt', '') for f in os.listdir(args.osm_dir) if f.endswith('.lines.txt')]
        base_names = [b for b in raw_base_names if b not in excluded_basenames]
        
        excluded_count = len(raw_base_names) - len(base_names)
        print(f">>> Evaluating Bulk Mode")
        print(f"    - Found OSM frames: {len(raw_base_names)}")
        print(f"    - Excluded frames:  {excluded_count}")
        print(f"    - Valid frames:     {len(base_names)}")

    line_stats = {'hits': 0, 'total': 0}
    point_stats = {'hits': 0, 'total': 0}
    valid_frames = 0
    scene_stats = {}

    print(f"\n[Info] Starting multiprocessing with {args.workers} CPU workers.")
    
    # =========================================================================
    # Evaluate frames concurrently with ProcessPoolExecutor.
    # =========================================================================
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Dispatch all frame tasks.
        futures = []
        for base in base_names:
            current_sid = frame_to_scene.get(base, "Unknown")
            # Initialize scene statistics in deterministic order.
            if current_sid not in scene_stats:
                scene_stats[current_sid] = {
                    'line_hits': 0, 'line_total': 0, 
                    'point_hits': 0, 'point_total': 0, 
                    'valid_frames': 0
                }
                
            future = executor.submit(
                process_single_frame, 
                base, current_sid, 
                args.gt_dir, args.osm_dir, 
                args.line_thresh_px, args.point_thresh_m
            )
            futures.append(future)

        # Collect and aggregate results with a progress bar.
        for future in tqdm(as_completed(futures), total=len(futures)):
            res = future.result()
            
            sid = res['sid']
            l_h, l_t = res['l_h'], res['l_t']
            p_h, p_t = res['p_h'], res['p_t']
            
            # Aggregate global statistics.
            line_stats['hits'] += l_h
            line_stats['total'] += l_t
            point_stats['hits'] += p_h
            point_stats['total'] += p_t
            
            # Aggregate per-scene statistics.
            scene_stats[sid]['line_hits'] += l_h
            scene_stats[sid]['line_total'] += l_t
            scene_stats[sid]['point_hits'] += p_h
            scene_stats[sid]['point_total'] += p_t
            
            # Count valid frames.
            if l_t > 0 or p_t > 0: 
                valid_frames += 1
                scene_stats[sid]['valid_frames'] += 1

    # =========================================================================
    # Print the evaluation summary.
    # =========================================================================
    print(f"\n{'='*40}")
    print(f"{'Precision Evaluation (Filtered)':^40}")
    print(f"{'='*40}")
    print(f"Frames Evaluated: {valid_frames}")
    
    if line_stats['total'] > 0:
        l_rate = (line_stats['hits'] / line_stats['total']) * 100
        print(f"Line Hit Rate:  {l_rate:>7.2f}%  (Thresh: {args.line_thresh_px}px)")
    
    if point_stats['total'] > 0:
        p_rate = (point_stats['hits'] / point_stats['total']) * 100
        print(f"Point Hit Rate: {p_rate:>7.2f}%  (Thresh: {args.point_thresh_m}m)")
    print(f"{'='*40}")

    # Generate the Excel report.
    if scene_stats:
        excel_records = []
        for sid, stats in scene_stats.items():
            s_l_rate = (stats['line_hits'] / stats['line_total'] * 100) if stats['line_total'] > 0 else 0
            s_p_rate = (stats['point_hits'] / stats['point_total'] * 100) if stats['point_total'] > 0 else 0
            
            excel_records.append({
                'Scene ID': sid,
                'Valid Frames': stats['valid_frames'],
                f'Line Hit Rate (Thresh: {args.line_thresh_px}px) %': round(s_l_rate, 2),
                f'Point Hit Rate (Thresh: {args.point_thresh_m}m) %': round(s_p_rate, 2)
            })
            
        df = pd.DataFrame(excel_records)
        df = df.sort_values(by='Scene ID').reset_index(drop=True)
        
        try:
            df.to_excel(args.output_excel, index=False)
            print(f"\n[Success] Per-scene results exported to Excel: {args.output_excel}")
        except Exception as e:
            print(f"\n[Error] Excel export failed. Ensure openpyxl is installed (pip install openpyxl): {e}")

if __name__ == "__main__":
    main()
