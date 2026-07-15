# filename: eval_lane_level.py
import os
import cv2
import json
import shutil
import numpy as np
import argparse
from tqdm import tqdm
from collections import defaultdict
import torch

import utils


# ======================================================================
# Exclusive scene list
# ======================================================================
# Put scenes with unreliable pseudo labels here.
# Accepted formats:
#   "0345", "scene-0345"
#
EXCLUSIVE_SCENES = [
    # TODO: Add your excluded scene IDs here.
    "0016", "0036", "0039", "0097", "0101", "0103", "0106",
    "0268", "0273", "0276", "0277", "0524", "0559", "0562",
    "0625", "0637", "0638", "0782", "0783", "0795", "0904",
    "0905", "0906", "0908", "0924", "0926", "0929", "0963",
    "0966", "0967", "1062", "1064", "1065", "1068", "1069",
    "1070", "1072", "scene-0914", "scene-0093", "scene-0345", "scene-0012", "scene-0962", "scene-0919", "scene-0928"
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Lane Detection Performance (IoU / F1)")
    parser.add_argument('--gt_dir', type=str, required=True, help='Path to generated labels')
    parser.add_argument('--pred_dir', type=str, required=True, help='Path to CondLaneNet prediction (txt)')
    parser.add_argument('--scene', type=str, default=None, help='Evaluate specific scene ID')
    parser.add_argument('--split_root', type=str, default="/data/NuScene/v1.0-trainval/sweeps/CAM_FRONT/split")
    parser.add_argument('--img_w', type=int, default=1600)
    parser.add_argument('--img_h', type=int, default=900)

    # Pass --line_width 50 to use the alternative evaluation setting.
    parser.add_argument('--line_width', type=int, default=150, help='Line drawing width for IoU mask')
    parser.add_argument('--iou_thresh', type=float, default=0.5, help='IoU threshold for TP')

    parser.add_argument('--output_txt', type=str, default=None, help='Path to save evaluation results')
    parser.add_argument('--output_scene_csv', type=str, default=None, help='Path to save per-scene evaluation CSV')

    # Scene exclusion settings.
    # You can either edit EXCLUSIVE_SCENES above, pass scenes from command line,
    # or provide a txt file with one or more scene IDs per line.
    parser.add_argument(
        '--exclude_scenes',
        nargs='*',
        default=[],
        help='Scene IDs to exclude, e.g. --exclude_scenes 0345 0268 or --exclude_scenes scene-0345,scene-0268'
    )
    parser.add_argument(
        '--exclude_scene_file',
        type=str,
        default=None,
        help='Optional txt file containing scene IDs to exclude. Supports one ID per line, spaces, commas, and # comments.'
    )

    # Ego-lane filtering for CULane pretrained model / multi-lane predictions
    parser.add_argument(
        '--pred_ego_only',
        action='store_true',
        help='Use utils.find_ego_lanes to keep only predicted ego lanes before evaluation'
    )
    parser.add_argument(
        '--save_ego_pred_dir',
        type=str,
        default=None,
        help='Optional directory to save prediction txt after ego-lane filtering'
    )

    # Demo export arguments
    parser.add_argument(
        '--img_dir',
        type=str,
        default=None,
        help='Path to source demo images. Original image and GT image are expected to be in this same directory.'
    )

    parser.add_argument(
        '--demo_root',
        type=str,
        default="/artifact/ssld/demo/High_TP",
        help='Root directory to save demo frames when evaluating a single scene'
    )

    return parser.parse_args()


# load all lane lines from txt file
def load_lines(txt_path):
    lines = []

    if not os.path.exists(txt_path):
        return lines

    with open(txt_path, 'r') as f:
        for line_str in f:
            line_str = line_str.strip()

            # skip blank line
            if line_str == "":
                continue

            coords = list(map(float, line_str.split()))

            # need at least two points to form a lane
            if len(coords) < 4:
                continue

            # coordinates must be even
            if len(coords) % 2 != 0:
                continue

            pts = np.array(coords).reshape(-1, 2)
            lines.append(pts.astype(np.int32))

    return lines


def save_lines(lines, txt_path):
    """
    Save lane lines to CULane-style txt.
    This is mainly for debugging --save_ego_pred_dir.
    """
    out_dir = os.path.dirname(txt_path)
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)

    with open(txt_path, "w") as f:
        for line in lines:
            if line is None or len(line) < 2:
                continue

            # Keep the common label order: image y from bottom to top.
            line = np.asarray(line, dtype=np.float32)
            line = line[np.argsort(line[:, 1])[::-1]]

            coords = []
            for x, y in line:
                coords.append(f"{float(x):.2f} {float(y):.2f}")

            if len(coords) >= 2:
                f.write(" ".join(coords) + "\n")


# draw one lane line as a binary mask
def draw_line_mask(line, h, w, thickness):
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(line) >= 2:
        cv2.polylines(mask, [line], isClosed=False, color=255, thickness=thickness)
    return mask


# compute IoU between two(GT/Pred) binary masks
def compute_iou(gt_mask, pred_mask):
    intersection = np.logical_and(gt_mask > 0, pred_mask > 0).sum()
    union = np.logical_or(gt_mask > 0, pred_mask > 0).sum()

    if union == 0:
        return 0.0

    return intersection / union


# build pairwise IoU matrix for all GT and Pred lanes
def build_iou_matrix(gt_lines, pred_lines, h, w, thickness):
    num_gt = len(gt_lines)
    num_pred = len(pred_lines)

    iou_matrix = np.zeros((num_gt, num_pred), dtype=np.float32)

    # Pre-render masks so every pair can reuse the same lines.
    gt_masks = [draw_line_mask(gt_line, h, w, thickness) for gt_line in gt_lines]
    pred_masks = [draw_line_mask(pred_line, h, w, thickness) for pred_line in pred_lines]

    for i, gt_mask in enumerate(gt_masks):
        for j, pred_mask in enumerate(pred_masks):
            iou_matrix[i, j] = compute_iou(gt_mask, pred_mask)

    return iou_matrix


# greedy matching between GT lanes and Pred lanes
def greedy_match(iou_matrix, iou_thresh):
    matched_gt = set()
    matched_pred = set()
    matched_pairs = []    # matched lane pairs [(index_gt, index_pred, iou)]

    while True:
        max_iou = -1
        best_i = -1
        best_j = -1

        # search best unmatched lane pair
        for i in range(iou_matrix.shape[0]):
            if i in matched_gt:
                continue

            for j in range(iou_matrix.shape[1]):
                if j in matched_pred:
                    continue

                if iou_matrix[i, j] > max_iou:
                    max_iou = iou_matrix[i, j]
                    best_i = i
                    best_j = j

        # stop if best IoU is below threshold
        if max_iou < iou_thresh:
            break

        matched_gt.add(best_i)
        matched_pred.add(best_j)
        matched_pairs.append((best_i, best_j, max_iou))

    return matched_pairs, matched_gt, matched_pred


# ======================================================================
# Ego-lane filtering utilities
# ======================================================================

def compute_inverse_perspective_homography_from_scene(scene_data):
    """
    Build the same image-to-BEV homography used by OLRA / SSLD.
    Input image point:  [u, v, 1]
    Output BEV point:   [x_bev, y_bev]
    """
    rot = np.asarray(scene_data["ego2cam_rot"], dtype=np.float64)
    trans = np.asarray(scene_data["ego2cam_trans"], dtype=np.float64).reshape(3)
    calib = np.asarray(scene_data["calib_mat"], dtype=np.float64)

    bev_to_img_extrinsic = np.ones((3, 4), dtype=np.float64)
    bev_to_img_extrinsic[:3, :3] = rot
    bev_to_img_extrinsic[:3, 3] = trans

    bev_to_img_xform = calib @ bev_to_img_extrinsic

    # Use BEV x, BEV y, and translation columns.
    # This follows OLRA/SSLD convention: column 0, column 2, column 3.
    bev_to_img_xform_del = np.column_stack((
        bev_to_img_xform[:, 0],
        bev_to_img_xform[:, 2],
        bev_to_img_xform[:, 3]
    ))

    img_to_bev_homography = np.linalg.inv(bev_to_img_xform_del)
    return img_to_bev_homography


def get_scene_homography(scene_id, args, homography_cache):
    """
    Load and cache image-to-BEV homography for a scene.
    """
    if scene_id is None or scene_id == "Unknown":
        return None

    if scene_id in homography_cache:
        return homography_cache[scene_id]

    try:
        data = utils.load_scene_json(scene_id, args.split_root)
        H_img_to_bev = compute_inverse_perspective_homography_from_scene(data)
        homography_cache[scene_id] = H_img_to_bev
        return H_img_to_bev
    except Exception as e:
        print(f"[Warning] Failed to build homography for {scene_id}: {e}")
        homography_cache[scene_id] = None
        return None


def image_lines_to_bev_tensors(img_lines, H_img_to_bev):
    """
    Convert image-space lane lines to BEV-space lane tensors for utils.find_ego_lanes.

    Important:
        utils.find_ego_lanes uses lane[-1] as the nearest point to the vehicle.
        In BEV, y is forward distance, so the nearest point should have smaller y.
        Therefore each lane is sorted by y descending: far -> near.
    """
    bev_lanes = []

    for img_line in img_lines:
        if img_line is None or len(img_line) < 2:
            continue

        pts = np.asarray(img_line, dtype=np.float64)
        pts_h = np.column_stack((pts[:, 0], pts[:, 1], np.ones(len(pts), dtype=np.float64)))

        bev_h = (H_img_to_bev @ pts_h.T).T
        denom = bev_h[:, 2:3]

        valid = np.abs(denom[:, 0]) > 1e-9
        if not np.any(valid):
            continue

        bev_xy = bev_h[valid, :2] / denom[valid]
        bev_xy = bev_xy[np.isfinite(bev_xy).all(axis=1)]

        if len(bev_xy) < 2:
            continue

        # far -> near, so lane[-1] is nearest to ego vehicle.
        bev_xy = bev_xy[np.argsort(bev_xy[:, 1])[::-1]]
        bev_lanes.append(torch.tensor(bev_xy, dtype=torch.float64))

    return bev_lanes


def bev_lanes_to_image_lines(bev_lanes, H_img_to_bev, img_w, img_h):
    """
    Project ego lanes selected in BEV back to image space for the original IoU evaluation.
    """
    if len(bev_lanes) == 0:
        return []

    H_bev_to_img = np.linalg.inv(H_img_to_bev)
    img_lines = []

    for bev_lane in bev_lanes:
        if bev_lane is None or len(bev_lane) < 2:
            continue

        if isinstance(bev_lane, torch.Tensor):
            bev_xy = bev_lane.detach().cpu().numpy().astype(np.float64)
        else:
            bev_xy = np.asarray(bev_lane, dtype=np.float64)

        if len(bev_xy) < 2:
            continue

        pts_h = np.column_stack((bev_xy[:, 0], bev_xy[:, 1], np.ones(len(bev_xy), dtype=np.float64)))
        img_homo = (H_bev_to_img @ pts_h.T).T
        denom = img_homo[:, 2:3]

        valid = np.abs(denom[:, 0]) > 1e-9
        if not np.any(valid):
            continue

        img_xy = img_homo[valid, :2] / denom[valid]
        img_xy = img_xy[np.isfinite(img_xy).all(axis=1)]

        if len(img_xy) < 2:
            continue

        # Remove extremely far out-of-image points, but keep slightly outside points
        # because cv2.polylines can still draw clipped lines correctly.
        margin = 200
        keep = (
            (img_xy[:, 0] >= -margin) & (img_xy[:, 0] < img_w + margin) &
            (img_xy[:, 1] >= -margin) & (img_xy[:, 1] < img_h + margin)
        )
        img_xy = img_xy[keep]

        if len(img_xy) < 2:
            continue

        # For consistency with label txt format: bottom -> top.
        img_xy = img_xy[np.argsort(img_xy[:, 1])[::-1]]
        img_lines.append(np.round(img_xy).astype(np.int32))

    return img_lines


def filter_pred_lines_to_ego(raw_pred_lines, scene_id, args, homography_cache):
    """
    Keep only ego lanes from CondLaneNet / CULane predictions.

    Only prediction lanes are filtered. GT lanes are not changed.
    If homography is unavailable, this function falls back to raw predictions
    to avoid silently dropping predictions from evaluation.
    """
    info = {
        "raw_pred_lanes": len(raw_pred_lines),
        "eval_pred_lanes": len(raw_pred_lines),
        "raw_pred_gt2": 1 if len(raw_pred_lines) > 2 else 0,
        "ego_filter_changed": 0,
        "ego_filter_failed": 0,
    }

    if len(raw_pred_lines) == 0:
        info["eval_pred_lanes"] = 0
        return [], info

    H_img_to_bev = get_scene_homography(scene_id, args, homography_cache)

    if H_img_to_bev is None:
        info["ego_filter_failed"] = 1
        return raw_pred_lines, info

    try:
        bev_lanes = image_lines_to_bev_tensors(raw_pred_lines, H_img_to_bev)

        if len(bev_lanes) == 0:
            info["ego_filter_changed"] = 1 if len(raw_pred_lines) > 0 else 0
            info["eval_pred_lanes"] = 0
            return [], info

        ego_lanes_bev = utils.find_ego_lanes(bev_lanes)
        ego_pred_lines = bev_lanes_to_image_lines(
            ego_lanes_bev,
            H_img_to_bev,
            args.img_w,
            args.img_h
        )

        info["eval_pred_lanes"] = len(ego_pred_lines)

        if len(ego_pred_lines) != len(raw_pred_lines) or len(raw_pred_lines) > 2:
            info["ego_filter_changed"] = 1

        return ego_pred_lines, info

    except Exception as e:
        print(f"[Warning] Ego-lane filtering failed for scene={scene_id}: {e}")
        info["ego_filter_failed"] = 1
        return raw_pred_lines, info


# ======================================================================
# Evaluation statistics
# ======================================================================

def init_stat():
    return {
        "frames": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "iou_sum": 0.0,
        "matched_lane_count": 0,
        "total_gt_lanes": 0,
        "total_pred_lanes": 0,        # evaluated prediction lanes
        "total_raw_pred_lanes": 0,    # raw prediction lanes before ego filtering
        "empty_gt_correct_frames": 0,
        "empty_gt_error_frames": 0,
        "frames_raw_pred_gt2": 0,
        "frames_ego_filter_changed": 0,
        "ego_filter_failed_frames": 0,
    }


def update_stat(stat, frame_result):
    stat["frames"] += 1
    stat["tp"] += frame_result["tp"]
    stat["fp"] += frame_result["fp"]
    stat["fn"] += frame_result["fn"]
    stat["iou_sum"] += frame_result["iou_sum"]
    stat["matched_lane_count"] += frame_result["matched_lane_count"]
    stat["total_gt_lanes"] += frame_result["gt_lanes"]
    stat["total_pred_lanes"] += frame_result["pred_lanes"]
    stat["total_raw_pred_lanes"] += frame_result["raw_pred_lanes"]
    stat["empty_gt_correct_frames"] += frame_result["empty_gt_correct"]
    stat["empty_gt_error_frames"] += frame_result["empty_gt_error"]
    stat["frames_raw_pred_gt2"] += frame_result["raw_pred_gt2"]
    stat["frames_ego_filter_changed"] += frame_result["ego_filter_changed"]
    stat["ego_filter_failed_frames"] += frame_result["ego_filter_failed"]


def compute_metrics(stat):
    tp = stat["tp"]
    fp = stat["fp"]
    fn = stat["fn"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    m_iou = stat["iou_sum"] / stat["matched_lane_count"] if stat["matched_lane_count"] > 0 else 0.0

    return precision, recall, f1, m_iou


def evaluate_one_frame(gt_lines, pred_lines, args, ego_info=None):
    if ego_info is None:
        ego_info = {
            "raw_pred_lanes": len(pred_lines),
            "eval_pred_lanes": len(pred_lines),
            "raw_pred_gt2": 1 if len(pred_lines) > 2 else 0,
            "ego_filter_changed": 0,
            "ego_filter_failed": 0,
        }

    frame_result = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "iou_sum": 0.0,
        "matched_lane_count": 0,
        "gt_lanes": len(gt_lines),
        "pred_lanes": len(pred_lines),
        "raw_pred_lanes": ego_info["raw_pred_lanes"],
        "raw_pred_gt2": ego_info["raw_pred_gt2"],
        "ego_filter_changed": ego_info["ego_filter_changed"],
        "ego_filter_failed": ego_info["ego_filter_failed"],
        "empty_gt_correct": 0,
        "empty_gt_error": 0,
    }

    has_gt = len(gt_lines) > 0
    has_pred = len(pred_lines) > 0

    # case A: both GT and prediction have lanes
    if has_gt and has_pred:
        iou_matrix = build_iou_matrix(
            gt_lines,
            pred_lines,
            args.img_h,
            args.img_w,
            args.line_width
        )

        matched_pairs, matched_gt, matched_pred = greedy_match(iou_matrix, args.iou_thresh)

        tp_count = len(matched_pairs)
        fn_count = len(gt_lines) - len(matched_gt)
        fp_count = len(pred_lines) - len(matched_pred)

        frame_result["tp"] = tp_count
        frame_result["fn"] = fn_count
        frame_result["fp"] = fp_count

        for _, _, matched_iou in matched_pairs:
            frame_result["iou_sum"] += matched_iou

        frame_result["matched_lane_count"] = len(matched_pairs)

    # case B: GT has lanes but prediction has no lane
    elif has_gt and not has_pred:
        frame_result["fn"] = len(gt_lines)

    # case C: GT has no lane but prediction has lanes
    elif not has_gt and has_pred:
        frame_result["fp"] = len(pred_lines)
        frame_result["empty_gt_error"] = 1

    # case D: GT has no lane and prediction has no lane
    else:
        frame_result["empty_gt_correct"] = 1

    return frame_result


def filename_to_lines_name(filename):
    base = os.path.basename(filename)
    stem, _ = os.path.splitext(base)
    return stem + ".lines.txt"


def normalize_exclusive_scene_id(scene_id):
    """
    Normalize scene ID to the same format used in evaluation: scene-XXXX.

    Accepted inputs:
        0345
        "0345"
        "scene-0345"
    """
    if scene_id is None:
        return None

    sid = str(scene_id).strip()

    if sid == "" or sid.startswith("#"):
        return None

    if sid.startswith("scene-"):
        sid = sid.replace("scene-", "", 1)

    # Keep nuScenes four-digit style when the input is numeric.
    if sid.isdigit():
        sid = sid.zfill(4)

    return f"scene-{sid}"


def split_scene_tokens(items):
    """
    Split scene tokens from command line or text file.

    Supports:
        ["0345", "0268"]
        ["0345,0268"]
        ["scene-0345 scene-0268"]
    """
    tokens = []

    for item in items:
        if item is None:
            continue

        # Remove inline comments before tokenizing.
        item = str(item).split("#", 1)[0]
        item = item.replace(",", " ")

        for token in item.split():
            token = token.strip()
            if token != "":
                tokens.append(token)

    return tokens


def load_exclusive_scenes(args):
    """
    Build final exclusive scene set from:
        1. EXCLUSIVE_SCENES in this file
        2. --exclude_scenes from command line
        3. --exclude_scene_file from txt file
    """
    exclusive_scenes = set()

    # 1. Hard-coded list in this file
    for sid in split_scene_tokens(EXCLUSIVE_SCENES):
        norm_sid = normalize_exclusive_scene_id(sid)
        if norm_sid is not None:
            exclusive_scenes.add(norm_sid)

    # 2. Command line list
    for sid in split_scene_tokens(args.exclude_scenes):
        norm_sid = normalize_exclusive_scene_id(sid)
        if norm_sid is not None:
            exclusive_scenes.add(norm_sid)

    # 3. Text file list
    if args.exclude_scene_file is not None:
        if not os.path.exists(args.exclude_scene_file):
            print(f"[Warning] exclude_scene_file not found: {args.exclude_scene_file}")
        else:
            with open(args.exclude_scene_file, "r") as f:
                file_lines = f.readlines()

            for sid in split_scene_tokens(file_lines):
                norm_sid = normalize_exclusive_scene_id(sid)
                if norm_sid is not None:
                    exclusive_scenes.add(norm_sid)

    return exclusive_scenes


def filter_target_files_by_exclusive_scenes(target_files, frame_to_scene, exclusive_scenes):
    """
    Remove frames whose scene ID is in exclusive_scenes.
    Returns:
        kept_files
        excluded_frame_counts: dict scene_id -> skipped frame count
    """
    excluded_frame_counts = defaultdict(int)

    if not exclusive_scenes:
        return target_files, excluded_frame_counts

    kept_files = []

    for filename in target_files:
        scene_id = frame_to_scene.get(filename, "Unknown")

        if scene_id in exclusive_scenes:
            excluded_frame_counts[scene_id] += 1
            continue

        kept_files.append(filename)

    return kept_files, excluded_frame_counts


def format_exclusion_block(exclusive_scenes, excluded_frame_counts):
    if not exclusive_scenes:
        return ""

    excluded_frames = sum(excluded_frame_counts.values())

    lines = [
        "--- Exclusive Scene Filter ---\n",
        f"Configured Exclusive Scenes: {len(exclusive_scenes)}\n",
        f"Excluded Frames: {excluded_frames}\n",
    ]

    if len(excluded_frame_counts) > 0:
        lines.append("\nExcluded Scene Frame Counts:\n")
        for scene_id in sorted(excluded_frame_counts.keys()):
            lines.append(f"{scene_id}: {excluded_frame_counts[scene_id]} frames\n")
    else:
        lines.append("\nNo frames were excluded. Please check whether the scene IDs exist in the selected split / target files.\n")

    lines.append("\n")
    return "".join(lines)


def build_demo_image_name(frame_filename):
    """
    Convert frame filename in json to demo image filename.

    Example:
        sweeps/CAM_FRONT/xxx.jpg

    ->  0_data.NuScene.v1.0-trainval.sweeps.CAM_FRONT.xxx.jpg
    """
    base = os.path.basename(frame_filename)
    return f"0_data.NuScene.v1.0-trainval.sweeps.CAM_FRONT.{base}"


def build_demo_gt_name(frame_filename):
    """
    Convert frame filename in json to demo GT image filename.

    Example:
        sweeps/CAM_FRONT/xxx.jpg

    ->  0_data.NuScene.v1.0-trainval.sweeps.CAM_FRONT.xxx.jpg.gt.jpg
    """
    return build_demo_image_name(frame_filename) + ".gt.jpg"


def export_scene_demo_frames(args):
    """
    When evaluating one scene, copy all original frames and GT images to:
        /artifact/ssld/demo/<scene_id>/

    Source original image filename format:
        0_data.NuScene.v1.0-trainval.sweeps.CAM_FRONT.<original basename>

    Source GT image filename format:
        0_data.NuScene.v1.0-trainval.sweeps.CAM_FRONT.<original basename>.gt.jpg

    Original and GT images are expected to be in args.img_dir.
    """
    if args.scene is None:
        return

    if args.img_dir is None:
        print("[Demo Export] Skip: --img_dir is not provided.")
        return

    sid = utils.norm_scene_id(args.scene)
    out_dir = os.path.join(args.demo_root, sid)
    os.makedirs(out_dir, exist_ok=True)

    data = utils.load_scene_json(args.scene, args.split_root)
    frames = data.get("frame_data", [])

    copied_img = 0
    missing_img = 0

    copied_gt = 0
    missing_gt = 0

    for fr in frames:
        if "filename" not in fr:
            continue

        # -----------------------------
        # Copy original image
        # -----------------------------
        demo_img_name = build_demo_image_name(fr["filename"])
        src_img_path = os.path.join(args.img_dir, demo_img_name)
        dst_img_path = os.path.join(out_dir, demo_img_name)

        if os.path.exists(src_img_path):
            shutil.copy2(src_img_path, dst_img_path)
            copied_img += 1
        else:
            missing_img += 1
            print(f"[Demo Export][Missing Image] {src_img_path}")

        # -----------------------------
        # Copy GT image
        # -----------------------------
        demo_gt_name = build_demo_gt_name(fr["filename"])
        src_gt_path = os.path.join(args.img_dir, demo_gt_name)
        dst_gt_path = os.path.join(out_dir, demo_gt_name)

        if os.path.exists(src_gt_path):
            shutil.copy2(src_gt_path, dst_gt_path)
            copied_gt += 1
        else:
            missing_gt += 1
            print(f"[Demo Export][Missing GT] {src_gt_path}")

    print(
        f"[Demo Export] Scene {sid}: "
        f"images copied {copied_img}, images missing {missing_img}, "
        f"GT copied {copied_gt}, GT missing {missing_gt}, "
        f"saved to: {out_dir}"
    )


def build_frame_to_scene_map(split_root):
    """
    Scan all scene json files under split_root and build:
        xxx.lines.txt -> scene-0001
    """
    frame_to_scene = {}

    if not os.path.exists(split_root):
        print(f"[Warning] split_root not found: {split_root}")
        return frame_to_scene

    for sid in sorted(os.listdir(split_root)):
        scene_dir = os.path.join(split_root, sid)

        if not os.path.isdir(scene_dir):
            continue

        json_path = os.path.join(scene_dir, f"{sid}_culane.json")

        if not os.path.exists(json_path):
            continue

        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            scene_name = f"scene-{sid}"

            for fr in data.get("frame_data", []):
                if "filename" not in fr:
                    continue

                lines_name = filename_to_lines_name(fr["filename"])
                frame_to_scene[lines_name] = scene_name

        except Exception as e:
            print(f"[Warning] Failed to load scene json: {json_path}, error={e}")

    return frame_to_scene


def get_target_files(args, frame_to_scene):
    """
    If --scene is specified:
        use frame_data from the scene json.

    If --scene is not specified:
        use union of gt_dir and pred_dir.
        This is safer than only using gt_dir because pred-only frames can still be counted as FP.
    """
    if args.scene:
        data = utils.load_scene_json(args.scene, args.split_root)

        target_files = [
            filename_to_lines_name(fr["filename"])
            for fr in data.get("frame_data", [])
            if "filename" in fr
        ]

        sid = utils.norm_scene_id(args.scene)
        scene_name = f"scene-{sid}"

        # make sure this scene's frames are mapped
        for f in target_files:
            frame_to_scene[f] = scene_name

        return sorted(target_files)

    gt_files = set(f for f in os.listdir(args.gt_dir) if f.endswith(".lines.txt"))
    pred_files = set(f for f in os.listdir(args.pred_dir) if f.endswith(".lines.txt"))

    target_files = sorted(gt_files | pred_files)
    return target_files


def format_summary_block(title, stat, iou_thresh):
    precision, recall, f1, m_iou = compute_metrics(stat)

    return (
        f"--- {title} (IoU Thresh: {iou_thresh}) ---\n"
        f"Total Frames: {stat['frames']}\n"
        f"Total GT Lanes: {stat['total_gt_lanes']}\n"
        f"Total Pred Lanes (Evaluated): {stat['total_pred_lanes']}\n"
        f"Total Raw Pred Lanes: {stat['total_raw_pred_lanes']}\n"

        f"\nMatched Lane Pairs: {stat['matched_lane_count']}\n"
        f"TP: {stat['tp']}, FP: {stat['fp']}, FN: {stat['fn']}\n"

        f"\nPrecision: {precision:.4f}\n"
        f"Recall:    {recall:.4f}\n"
        f"F1 Score:  {f1:.4f}\n"
        f"Mean IoU (matched lanes only): {m_iou:.4f}\n"

        f"\nGT=0, Pred=0 Correct Frames: {stat['empty_gt_correct_frames']}\n"
        f"GT=0, Pred>0 Error Frames:   {stat['empty_gt_error_frames']}\n"

        f"\nFrames with Raw Pred > 2 lanes: {stat['frames_raw_pred_gt2']}\n"
        f"Frames Changed by Ego Filtering: {stat['frames_ego_filter_changed']}\n"
        f"Ego Filtering Failed Frames: {stat['ego_filter_failed_frames']}\n"
    )


def format_scene_table(scene_stats):
    header = (
        "\n--- Per-scene Lane-level Results ---\n"
        f"{'Scene':<12} {'Frames':>6} {'GT':>6} {'Pred':>6} {'RawPred':>8} "
        f"{'TP':>6} {'FP':>6} {'FN':>6} "
        f"{'Prec':>8} {'Recall':>8} {'F1':>8} {'mIoU':>8} "
        f"{'GT0_OK':>8} {'GT0_FP':>8} {'Raw>2':>7} {'EgoChg':>7} {'EgoFail':>8}\n"
    )

    lines = [header]

    for scene_id in sorted(scene_stats.keys()):
        stat = scene_stats[scene_id]
        precision, recall, f1, m_iou = compute_metrics(stat)

        lines.append(
            f"{scene_id:<12} "
            f"{stat['frames']:>6} "
            f"{stat['total_gt_lanes']:>6} "
            f"{stat['total_pred_lanes']:>6} "
            f"{stat['total_raw_pred_lanes']:>8} "
            f"{stat['tp']:>6} "
            f"{stat['fp']:>6} "
            f"{stat['fn']:>6} "
            f"{precision:>8.4f} "
            f"{recall:>8.4f} "
            f"{f1:>8.4f} "
            f"{m_iou:>8.4f} "
            f"{stat['empty_gt_correct_frames']:>8} "
            f"{stat['empty_gt_error_frames']:>8} "
            f"{stat['frames_raw_pred_gt2']:>7} "
            f"{stat['frames_ego_filter_changed']:>7} "
            f"{stat['ego_filter_failed_frames']:>8}\n"
        )

    return "".join(lines)


def save_scene_csv(scene_stats, output_scene_csv):
    output_dir = os.path.dirname(output_scene_csv)

    if output_dir != "":
        os.makedirs(output_dir, exist_ok=True)

    with open(output_scene_csv, "w") as f:
        f.write(
            "Scene ID,Frames,Total GT Lanes,Total Pred Lanes Evaluated,Total Raw Pred Lanes,"
            "TP,FP,FN,Precision,Recall,F1,Mean IoU,"
            "GT=0 Pred=0 Correct Frames,GT=0 Pred>0 Error Frames,"
            "Frames Raw Pred > 2,Frames Changed by Ego Filtering,Ego Filtering Failed Frames\n"
        )

        for scene_id in sorted(scene_stats.keys()):
            stat = scene_stats[scene_id]
            precision, recall, f1, m_iou = compute_metrics(stat)

            f.write(
                f"{scene_id},"
                f"{stat['frames']},"
                f"{stat['total_gt_lanes']},"
                f"{stat['total_pred_lanes']},"
                f"{stat['total_raw_pred_lanes']},"
                f"{stat['tp']},"
                f"{stat['fp']},"
                f"{stat['fn']},"
                f"{precision:.6f},"
                f"{recall:.6f},"
                f"{f1:.6f},"
                f"{m_iou:.6f},"
                f"{stat['empty_gt_correct_frames']},"
                f"{stat['empty_gt_error_frames']},"
                f"{stat['frames_raw_pred_gt2']},"
                f"{stat['frames_ego_filter_changed']},"
                f"{stat['ego_filter_failed_frames']}\n"
            )


def main():
    args = parse_args()

    # Build exclusive scene set.
    exclusive_scenes = load_exclusive_scenes(args)

    if len(exclusive_scenes) > 0:
        print("[Info] Exclusive scene filtering is ENABLED.")
        print(f"[Info] Exclusive scenes ({len(exclusive_scenes)}): {', '.join(sorted(exclusive_scenes))}")
    else:
        print("[Info] Exclusive scene filtering is DISABLED.")

    # If evaluating a single scene, optionally export original frames and GT images.
    export_scene_demo_frames(args)

    # 1. Build filename -> scene id mapping
    frame_to_scene = build_frame_to_scene_map(args.split_root)

    # 2. Decide target files
    target_files = get_target_files(args, frame_to_scene)

    # 2.5 Filter out exclusive scenes before evaluation
    total_files_before_exclusion = len(target_files)
    target_files, excluded_frame_counts = filter_target_files_by_exclusive_scenes(
        target_files,
        frame_to_scene,
        exclusive_scenes
    )

    if len(exclusive_scenes) > 0:
        print(
            f"[Info] Target frames before exclusion: {total_files_before_exclusion}, "
            f"after exclusion: {len(target_files)}, "
            f"excluded: {total_files_before_exclusion - len(target_files)}"
        )

    # 3. Initialize statistics
    total_stat = init_stat()
    scene_stats = defaultdict(init_stat)

    unknown_scene_frames = 0
    homography_cache = {}

    if args.pred_ego_only:
        print("[Info] Prediction ego-lane filtering is ENABLED: utils.find_ego_lanes will be applied to prediction lanes only.")
        if args.save_ego_pred_dir is not None:
            print(f"[Info] Filtered ego predictions will be saved to: {args.save_ego_pred_dir}")

    # 4. Evaluation loop
    for filename in tqdm(target_files):
        gt_p = os.path.join(args.gt_dir, filename)
        pred_p = os.path.join(args.pred_dir, filename)

        gt_lines = load_lines(gt_p)
        raw_pred_lines = load_lines(pred_p)

        scene_id = frame_to_scene.get(filename, "Unknown")

        if scene_id == "Unknown":
            unknown_scene_frames += 1

        if args.pred_ego_only:
            pred_lines, ego_info = filter_pred_lines_to_ego(
                raw_pred_lines,
                scene_id,
                args,
                homography_cache
            )

            if args.save_ego_pred_dir is not None:
                save_lines(pred_lines, os.path.join(args.save_ego_pred_dir, filename))
        else:
            pred_lines = raw_pred_lines
            ego_info = {
                "raw_pred_lanes": len(raw_pred_lines),
                "eval_pred_lanes": len(pred_lines),
                "raw_pred_gt2": 1 if len(raw_pred_lines) > 2 else 0,
                "ego_filter_changed": 0,
                "ego_filter_failed": 0,
            }

        frame_result = evaluate_one_frame(gt_lines, pred_lines, args, ego_info=ego_info)

        # global statistics
        update_stat(total_stat, frame_result)

        # scene-level statistics
        update_stat(scene_stats[scene_id], frame_result)

    # 5. Final metrics
    eval_title = "Lane-level Evaluation Results"
    if args.pred_ego_only:
        eval_title += " (Pred Ego Lanes Only)"

    exclusion_str = format_exclusion_block(exclusive_scenes, excluded_frame_counts)

    result_str = exclusion_str + format_summary_block(
        title=eval_title,
        stat=total_stat,
        iou_thresh=args.iou_thresh
    )

    scene_table_str = format_scene_table(scene_stats)
    result_str = result_str + scene_table_str

    print("\n" + result_str)

    print(f"Check GT = TP + FN: {total_stat['total_gt_lanes']} = {total_stat['tp'] + total_stat['fn']}")
    print(f"Check Eval Pred = TP + FP: {total_stat['total_pred_lanes']} = {total_stat['tp'] + total_stat['fp']}")

    if args.pred_ego_only:
        print(
            f"Raw Pred Lanes before ego filtering: {total_stat['total_raw_pred_lanes']} "
            f"-> Evaluated Pred Lanes: {total_stat['total_pred_lanes']}"
        )

    if unknown_scene_frames > 0:
        print(
            f"[Warning] {unknown_scene_frames} frames could not be mapped to a scene. "
            f"They are counted under 'Unknown'. "
            f"If --pred_ego_only is enabled, those frames cannot be ego-filtered and will use raw predictions."
        )

    # default output path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../.."))

    default_output_path = os.path.join(
        project_root,
        "condlanenet",
        "work_dirs",
        "exps",
        "nuscene",
        "large",
        "ssld_eval.txt"
    )

    output_path = args.output_txt if args.output_txt is not None else default_output_path

    output_dir = os.path.dirname(output_path)

    if output_dir != "":
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(result_str)

    print(f"Results saved to: {output_path}")

    # save per-scene CSV
    if args.output_scene_csv is not None:
        save_scene_csv(scene_stats, args.output_scene_csv)
        print(f"Per-scene CSV saved to: {args.output_scene_csv}")


if __name__ == "__main__":
    main()
