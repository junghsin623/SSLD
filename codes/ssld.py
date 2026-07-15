# filename: ssld.py
# -*- coding: utf-8 -*-
"""
SSLD algorithm library.
Supports both 'gt' (Ground Truth) and 'osm' modes, including asymmetric lane search.
"""
import os
from joblib import Parallel, delayed
import math
import numpy as np
import cv2
import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Dict, Any
import copy
from scipy.spatial import cKDTree
import time 
import pandas as pd

# Local modules
import utils
import visualizer as vis
from utils import match_route, clip_en_route, map_en_route_to_veh_route, clip_veh_route

class SSLD:
    def __init__(self, mode='osm'):
        self.mode = mode # 'gt' or 'osm'
        # print(f"[SSLD] Initialized in mode: {self.mode}")

        # Debug options
        self.show_debug_carpet = True

        self.prev_poses = []
        self.prev_index = 0

        # ====================================================
        # Delay initialization until the vehicle starts moving.
        # ====================================================
        self.is_started = False
        self.consecutive_moving_frames = 0
        
        # These thresholds may be tuned for the target scene.
        self.START_SPEED_THRESHOLD = 6.0
        self.REQUIRED_MOVING_FRAMES = 3
        # ====================================================

        # Heading stabilization
        self.heading_lookahead_points = 5 

        # BEV canvas and projection
        self.forward_m = 60.0
        self.backward_m = 20.0
        self.lr_m = 30.0
        self.bev_w = 600
        self.bev_h = 800
        
        # Lookahead distances
        self.lookahead_long_m = 40.0
        self.lookahead_short_m = 10.0
        self.curv_threshold = 0.05    

        # Reprojection ROI
        self.roi_y_start_m = 3.0
        self.roi_half_width_m = 10.0

        # Longitudinal search distance
        self.fwd_search_m = 40.0

        # ==========================================
        # Turn / Intersection Protection
        # ==========================================
        self.turn_protection_enabled = True

        # Near- and far-range curvature windows in vehicle-BEV meters.
        self.turn_near_y0_m = 3.0
        self.turn_near_y1_m = 12.0
        self.turn_far_y0_m = 12.0
        self.turn_far_y1_m = 35.0

        # Conservative curvature thresholds; tune them from logs if needed.
        self.turn_near_kappa_thr = 0.060
        self.turn_far_kappa_thr = 0.050
        self.turn_diff_kappa_thr = 0.030

        # ==========================================
        # Turn Protection Hysteresis
        # ==========================================
        self.in_turn_protection = False

        # Consecutive raw turn-risk counters.
        self.turn_on_count = 0
        self.turn_off_count = 0

        # Enter protection as soon as a turn is detected.
        self.turn_enter_frames = 1

        # Require several clear frames before leaving protection.
        self.turn_exit_frames = 4

        # Minimum number of frames to remain protected.
        self.turn_min_hold_frames = 5

        # Force coarse-to-fine recovery after too many skipped frames.
        self.turn_skip_force_coarse_frames = 4

        # Search-skip statistics for monitoring turn protection.
        self.turn_skip_count = 0

        # Request broad relocalization after skips during initialization or low confidence.
        self.need_coarse_reinit = False

        # Whether lane-search initialization has succeeded at least once.
        self.has_search_initialized = False

        # ==========================================
        # Coarse Search Budget
        # ==========================================
        # Maximum broad searches: one initialization plus one intersection recovery.
        self.max_coarse_search_count = 2
        self.coarse_search_count = 0

        # Image processing
        self.gauss_ksize = (5, 5)
        self.gauss_sigma = 1.0
        self.canny_low = 30
        self.canny_high = 100
        self.roi_y_min_ratio = 0.45
        self.roi_x_min_ratio = 0.15
        self.roi_x_max_ratio = 0.85
        
        # Navigation-direction filtering
        
        # Lane-search parameters
        self.subroute_length = 80
        
        # Select the lane-width search range by mode.
        if self.mode == 'gt':
            # GT mode permits larger offsets for asymmetric search.
            self.lane_width_min_m = 1.5 
            self.lane_width_max_m = 4.5 
        else:
            # OSM mode uses tighter symmetric constraints to reject road noise.
            self.lane_width_min_m = 2.5
            self.lane_width_max_m = 6

        # ==========================================
        # Tracking state and dynamic search range
        # ==========================================
        self.is_tracking_stable = False  
        self.stable_score_threshold = 120 

        # Coarse search grid
        dx_coarse = np.arange(-3.0, 3.0 + 0.01, 0.25).round(2) 
        dy_coarse = np.arange(-3.0, 3.0 + 0.01, 0.25).round(2)
        dh_coarse = np.arange(-24.0, 24.0 + 0.01, 3.0).round(2) 
                
        self.coarse_offsets_cfg = [(x, y, h) for x in dx_coarse for y in dy_coarse for h in dh_coarse]

        # Search tolerances
        self.lane_width_step_m = 0.2
        self.coarse_match_tolerance_m = 0.30
        self.shift_match_tolerance_m = 0.15
        self.exclusion_radius_m = 0.8
        
        # Beam-search parameters
        self.top_k_coarse = 3

        # ==========================================
        # Segment-coverage scoring parameters
        # ==========================================
        self.segment_count = 10
        self.min_pts_per_segment = 10

        # Cap each segment so dense noise cannot dominate the score.
        self.points_cap_per_segment = 12
        
        # ==========================================
        # Lane-width history for smoothing and outlier rejection.
        # ==========================================
        self.width_history = [] 
        self.max_history_len = 5

        # Width-filter watchdog counter.
        self.consecutive_width_rejections = 0

        self.stats_log = []

    def generate_fine_grid(self, center_dx, center_dy, center_dh):
        """Generate a fine search grid around the given center offset."""
        fine_cfg = []
        
        # Read the lateral tolerance dynamically, defaulting to 0.3 m.
        tol_x = getattr(self, 'shift_match_tolerance_m', 0.3) 
        
        # Keep longitudinal offsets small to prevent drift and reduce computation.
        tol_y = 0.1 
        
        # Search heading offsets within +/-1.5 degrees.
        
        for dx in np.arange(-tol_x, tol_x + 0.01, 0.1).round(2):
            for dy in np.arange(-tol_y, tol_y + 0.01, 0.1).round(2):
                for dh in np.arange(-1.5, 1.5 + 0.01, 0.5).round(2):
                    fine_cfg.append((
                        round(center_dx + dx, 2), 
                        round(center_dy + dy, 2), 
                        round(center_dh + dh, 2)
                    ))
                    
        return fine_cfg

    def nav_path_score(self, nav_poly: np.ndarray, bev_subset: np.ndarray, fast_mode: bool = False) -> Tuple[float, dict]:
        # Accept an already filtered BEV subset.
        results = find_lanes_by_symmetric_expansion(nav_poly, bev_subset, self, fast_mode)
        score = results.get("best_score", -1) if results else -1
        return score, results
    
    def _score_one_pose(self, pose_offset_cfg: Tuple, initial_pose: 'BEVPose', en_route: torch.Tensor, prev_index: int, bev_subset: np.ndarray, FWD_internal: float, fast_mode: bool = False) -> Tuple:
        dx, dy, d_deg = pose_offset_cfg
        heading_offset = math.radians(d_deg)
        candidate_pose = BEVPose(initial_pose.position_x + dx, initial_pose.position_y + dy, initial_pose.heading + heading_offset, requires_grad=False)
        # TODO: sub_en, match_idx = clip_en_route(en_route, candidate_pose, prev_index, self.subroute_length)
        # Preserve the caller's route index.
        cand_gps = torch.tensor([candidate_pose.position_x.item(), candidate_pose.position_y.item()], dtype=torch.float64)
        sub_en, _ = utils.clip_en_route(en_route, cand_gps, prev_index, self.subroute_length)
        veh_route = map_en_route_to_veh_route(sub_en, candidate_pose)
        nav_poly = clip_veh_route(veh_route, walk_dist=FWD_internal).detach().cpu().numpy()

        if len(nav_poly) < 2: return (pose_offset_cfg, nav_poly, -1, {}) 
        score, results = self.nav_path_score(nav_poly, bev_subset, fast_mode)

        # ====================================================
        # Heading-offset penalty
        # ====================================================
        if score > 0:
            penalty_factor = 1.0

            abs_d_deg = abs(d_deg)
            # Allow up to three degrees before applying a penalty.
            if abs_d_deg > 3.0:
                # Deduct 10% per extra degree, retaining at least 30%.
                penalty_factor = max(0.3, 1.0 - (abs_d_deg - 3.0) * 0.10)

            score = score * penalty_factor
        
        return (pose_offset_cfg, nav_poly, score, results)

    def nav_path_refine(self, initial_pose: 'BEVPose', en_route: torch.Tensor, prev_index: int, bev_subset: np.ndarray, FWD_internal: float, fast_mode: bool = False) -> List[Tuple]:
        """
        Run the grid search serially and return valid results by descending score.
        """
        results_list = []
        for offset_cfg in self.pose_candidate_offsets_cfg:
            res = self._score_one_pose(offset_cfg, initial_pose, en_route, prev_index, bev_subset, FWD_internal, fast_mode)
            results_list.append(res)
        
        # Remove invalid results and sort by descending score.
        valid_results = [r for r in results_list if r[2] >= 0]
        valid_results.sort(key=lambda x: x[2], reverse=True)
        
        return valid_results
    def _calculate_segment_score(self, dists: np.ndarray, tolerance: float) -> float:
        """Compute capped per-segment scores using vectorized operations."""
        matched_mask = dists < tolerance
        if not np.any(matched_mask):
            return 0.0
            
        n = len(matched_mask)
        seg_count = self.segment_count
        
        # Pad so the array divides evenly into segments.
        pad_size = (seg_count - (n % seg_count)) % seg_count
        if pad_size > 0:
            matched_mask = np.pad(matched_mask, (0, pad_size), mode='constant', constant_values=False)
            
        # Reshape by segment and count matches in each row.
        reshaped = matched_mask.reshape(seg_count, -1)
        counts = reshaped.sum(axis=1)
        
        # Apply the per-segment cap and sum.
        capped_counts = np.minimum(counts, self.points_cap_per_segment)
        return float(np.sum(capped_counts))
    
    def _curvature_in_y_range(self, poly_m: np.ndarray, y0: float, y1: float) -> float:
        """
        Measure navigation-polyline curvature within a y range.
        Larger values indicate sharper curvature.
        """
        if poly_m is None or len(poly_m) < 3:
            return 0.0

        try:
            P = utils.resample_poly_by_ds(poly_m, ds=0.5)
            if P is None or len(P) < 3:
                return 0.0

            mask = (P[:, 1] >= y0) & (P[:, 1] <= y1)
            P_win = P[mask]

            if len(P_win) < 3:
                return 0.0

            v = np.diff(P_win, axis=0)
            ds = np.linalg.norm(v, axis=1)

            valid = ds > 1e-6
            v = v[valid]
            ds = ds[valid]

            if len(v) < 2:
                return 0.0

            u = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

            cosang = np.sum(u[1:] * u[:-1], axis=1)
            ang = np.arccos(np.clip(cosang, -1.0, 1.0))

            ds_mid = 0.5 * (ds[1:] + ds[:-1])
            kappa = ang / (ds_mid + 1e-6)

            if len(kappa) == 0:
                return 0.0

            return float(np.percentile(kappa, 90))

        except Exception:
            return 0.0


    def check_turn_risk_by_curvature(self, nav_poly: np.ndarray):
        """
        Detect high-risk turn regions from near- and far-range curvature.
        Returns:
            is_turn_risk, near_kappa, far_kappa, diff_kappa, reason
        """
        near_kappa = self._curvature_in_y_range(
            nav_poly,
            self.turn_near_y0_m,
            self.turn_near_y1_m
        )

        far_kappa = self._curvature_in_y_range(
            nav_poly,
            self.turn_far_y0_m,
            self.turn_far_y1_m
        )

        diff_kappa = far_kappa - near_kappa

        # A sharp far section indicates an approaching turn.
        turn_ahead = (
            far_kappa >= self.turn_far_kappa_thr and
            diff_kappa >= self.turn_diff_kappa_thr
        )

        # A sharp near section indicates the vehicle is already turning.
        turning_now = near_kappa >= self.turn_near_kappa_thr

        is_turn_risk = turn_ahead or turning_now

        if turning_now:
            reason = "turning_now"
        elif turn_ahead:
            reason = "turn_ahead"
        else:
            reason = "normal"

        return is_turn_risk, near_kappa, far_kappa, diff_kappa, reason
    
    def update_turn_protection_state(self, raw_is_turn_risk: bool, raw_reason: str):
        """
        Apply hysteresis to raw curvature risk to prevent flicker within turns.
        
        Returns:
            turn_protect_active, state_reason
        """

        if raw_is_turn_risk:
            self.turn_on_count += 1
            self.turn_off_count = 0
        else:
            self.turn_off_count += 1
            self.turn_on_count = 0

        # Not currently in turn protection.
        if not self.in_turn_protection:
            if self.turn_on_count >= self.turn_enter_frames:
                self.in_turn_protection = True
                return True, f"enter_{raw_reason}"

            return False, "normal"

        # Hold protection for a minimum duration before allowing exit.
        if self.turn_skip_count < self.turn_min_hold_frames:
            if raw_is_turn_risk:
                return True, f"hold_{raw_reason}"
            else:
                return True, f"hold_wait_clear off_count={self.turn_off_count}"

        # Require consecutive clear frames after the minimum hold duration.
        if self.turn_off_count >= self.turn_exit_frames:
            self.in_turn_protection = False
            return False, f"exit_clear off_count={self.turn_off_count}"

        # Continue protection until enough clear frames accumulate.
        if raw_is_turn_risk:
            return True, f"stay_{raw_reason}"
        else:
            return True, f"wait_clear off_count={self.turn_off_count}"

    def process_one_frame(self, img_path: str, homography: np.ndarray,
                        frame: dict, idx: int,
                        en_route: torch.Tensor,
                        scene_id: str = "Unknown") -> Tuple[Dict[str, Any], List, int]:
        
        # Image processing
        img = cv2.imread(img_path)
        if img is None: return {}
        # Reuse the same edge and tangent data for search and visualization.
        edges, pts_img, theta_img = edges_and_tangent(img, self)

        if len(pts_img) == 0:
            return {}

        # Unified tangent angles are retained for visualization.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, self.gauss_ksize, self.gauss_sigma)
        tan_deg_img = compute_tangent_angles(gray_blur)
        tan_deg_unified = unify_tangent_angles(tan_deg_img, None, True)

        # BEV points used by the search.
        base_bev, unit_bev = image_dirs_to_bev_vecs(
            homography,
            pts_img,
            theta_img,
            step_pix=6.0,
            forward_only=True
        )
        # The caller selects either GPS or ground-truth GPS for initialization.
        gps = frame.get("gps_to_use") 
        if gps is None: return {}

        # Read vehicle-motion data from the frame.
        yaw_rate = frame.get("yaw_rate", 0.0) 
        speed = frame.get("speed", 0.0)
        time_step = frame.get("delta_t", 0.0)

        # GT mode uses raw GPS so lateral displacement remains visible.
        use_raw = (self.mode == 'gt')
        heading_gt_val = frame.get("heading_gt") if use_raw else None

        # =========================================================
        # Delay OSM initialization until sustained motion is detected.
        # =========================================================
        if self.mode == 'osm' and not self.is_started:
            if speed > self.START_SPEED_THRESHOLD:
                self.consecutive_moving_frames += 1
            else:
                self.consecutive_moving_frames = 0
            
            if self.consecutive_moving_frames >= self.REQUIRED_MOVING_FRAMES:
                self.is_started = True
                print(f"    [Init] Vehicle started moving at frame {idx}. Tracking initiated!")
            else:
                # Skip output and pose history until initialization is allowed.
                return {}
        # =========================================================
        # =======================================================
        # Dead reckoning
        # =======================================================
        if len(self.prev_poses) == 0 or self.mode == 'gt':
            # Initialize the first frame from GPS and map projection.
            gps = frame.get("gps_to_use") 
            if gps is None: return {}
            
            initial_pose, self.prev_index = pose_from_gps_on_en_route(
                en_route, self.prev_index, gps, BEVPose, 
                lookahead_points=self.heading_lookahead_points,
                use_raw_gps=use_raw,
                heading_gt=heading_gt_val 
            )
        else:
            # Predict subsequent OSM frames from yaw rate and speed.
            last_refined_pose = self.prev_poses[-1]
            initial_pose = calc_sensor_predicted_pose(
                last_refined_pose, yaw_rate, speed, time_step, BEVPose
            )
            
            # Update only the route-match index without snapping back to GPS.
            pred_gps = torch.tensor([initial_pose.position_x.item(), initial_pose.position_y.item()], dtype=torch.float64)
            self.prev_index, _ = utils.match_route(en_route, self.prev_index, pred_gps)
        # TODO: prev_poses -> prev_init_poses
        prev_pos = torch.tensor([initial_pose.position_x, initial_pose.position_y], dtype=torch.float64)
        sub_en, match_idx = clip_en_route(en_route, prev_pos, self.prev_index, self.subroute_length)
        
        # Build the initial navigation polyline.
        initial_veh_route = map_en_route_to_veh_route(sub_en, initial_pose)
        if initial_veh_route is not None:
            initial_nav_poly_full = clip_veh_route(initial_veh_route, walk_dist=self.lookahead_long_m).detach().cpu().numpy()
        else:
            initial_nav_poly_full = np.array([[0.0, 0.0], [0.0, self.lookahead_long_m]]) 
            
        # Use the full configured search distance.
        FWD_internal = self.fwd_search_m 
        FWD_disp = FWD_internal
        y0, y1 = self.roi_y_start_m, self.roi_y_start_m + FWD_internal
        bev_roi_poly = utils.bev_box_poly_lr_y(self.roi_half_width_m, y0, y1)

        # =========================================================
        # Skip lane search in risky turns, with hysteresis to prevent flicker.
        # =========================================================
        if self.mode == 'osm' and self.turn_protection_enabled:
            raw_is_turn_risk, near_kappa, far_kappa, diff_kappa, turn_reason = \
                self.check_turn_risk_by_curvature(initial_nav_poly_full)

            turn_protect_active, turn_state_reason = self.update_turn_protection_state(
                raw_is_turn_risk,
                turn_reason
            )

            if turn_protect_active:
                self.turn_skip_count += 1

                if not self.has_search_initialized:
                    if self.coarse_search_count < self.max_coarse_search_count:
                        self.need_coarse_reinit = True

                if self.turn_skip_count >= self.turn_skip_force_coarse_frames:
                    if self.coarse_search_count < self.max_coarse_search_count:
                        self.need_coarse_reinit = True

                refined_pose = initial_pose
                self.prev_poses.append(refined_pose)

                empty_lane_results = {
                    "best_left_lane": np.empty((0, 2)),
                    "best_right_lane": np.empty((0, 2)),
                    "best_score": 0,
                    "found_width": 0.0,
                    "matched_left_pts": np.empty((0, 2)),
                    "matched_right_pts": np.empty((0, 2)),
                    "viz_points": np.empty((0, 2)),
                    "force_empty_label": True
                }

                keep_mask_turn = filter_edges_by_nav_direction(
                    base_bev, unit_bev, initial_nav_poly_full, self, y0, y1
                )
                kept_pts_img_turn = pts_img[keep_mask_turn]
                kept_base_bev_turn = base_bev[keep_mask_turn]

                print(
                    f"    [Turn-Protect][{idx:04d}] "
                    f"SkipSearch raw={raw_is_turn_risk} "
                    f"raw_reason={turn_reason} "
                    f"state={turn_state_reason} "
                    f"near_kappa={near_kappa:.4f} "
                    f"far_kappa={far_kappa:.4f} "
                    f"diff={diff_kappa:.4f} "
                    f"skip_count={self.turn_skip_count} "
                    f"off_count={self.turn_off_count} "
                    f"need_coarse={self.need_coarse_reinit}"
                )

                gps_gt = frame.get("gps_gt")
                heading_gt = frame.get("heading_gt")

                gt_pose_obj = None
                if gps_gt is not None and heading_gt is not None:
                    gt_pose_obj = BEVPose(gps_gt[0], gps_gt[1], heading_gt, requires_grad=False)

                results = {
                    "original_image": img,
                    "edges": edges,
                    "tan_deg_unified": tan_deg_unified,
                    "nav_poly": initial_nav_poly_full,
                    "FWD_internal": FWD_internal,
                    "FWD_disp": FWD_disp,
                    "kept_pts_img": kept_pts_img_turn,
                    "kept_base_bev": kept_base_bev_turn,
                    "lane_search_results": empty_lane_results,
                    "bev_roi_poly": bev_roi_poly,
                    "debug_raw_route": initial_nav_poly_full,
                    "initial_pose": initial_pose,
                    "refined_pose": refined_pose,
                    "best_pose_offset": (0.0, 0.0, 0.0),
                    "gt_pose": gt_pose_obj
                }

                return results

            else:
                if self.turn_skip_count > 0:
                    print(
                        f"    [Turn-Recover][{idx:04d}] "
                        f"Exit turn protection after {self.turn_skip_count} skipped frames. "
                        f"state={turn_state_reason} "
                        f"near_kappa={near_kappa:.4f} "
                        f"far_kappa={far_kappa:.4f} "
                        f"diff={diff_kappa:.4f} "
                        f"need_coarse={self.need_coarse_reinit}"
                    )

                self.turn_skip_count = 0

        keep_mask = filter_edges_by_nav_direction(base_bev, unit_bev, initial_nav_poly_full, self, y0, y1)
        bev_subset_fixed = base_bev[keep_mask]
        
        path_len = 0.0
        if len(initial_nav_poly_full) > 1:
            diffs = np.diff(initial_nav_poly_full, axis=0)
            path_len = np.sum(np.linalg.norm(diffs, axis=1))
            
        max_forward_dist = np.max(initial_nav_poly_full[:, 1]) if len(initial_nav_poly_full) > 0 else 0.0
        MIN_FORWARD_DIST = 20.0  

        # Preserve trajectory continuity when quality filtering skips a frame.
        if path_len < 20.0 or max_forward_dist < MIN_FORWARD_DIST: 
            # print(f"[QC] Skip frame {idx}")
            self.prev_poses.append(initial_pose)
            return {}
        
        # Branch by operating mode.
        refined_pose = initial_pose

        if self.mode == 'osm':
            t_search_start = time.perf_counter()

            # Initialize search results.
            best_pose_offset_log, best_nav_poly, best_score, best_lane_search_results = (0.0, 0.0, 0.0), None, -1, {}
            debug_search_mode = "Unknown"

            should_run_coarse = (
                (len(self.prev_poses) == 0 or self.need_coarse_reinit)
                and self.coarse_search_count < self.max_coarse_search_count
            )
            if should_run_coarse:
                # ==========================================
                # Stage 1: broad coarse search.
                # ==========================================
                self.shift_match_tolerance_m = self.coarse_match_tolerance_m 
                self.pose_candidate_offsets_cfg = self.coarse_offsets_cfg
                print(f"    [Dynamic] Frame {idx}: INIT/REINIT Coarse-to-Fine...", end="")

                coarse_sorted_results = self.nav_path_refine(
                    initial_pose, en_route, self.prev_index, bev_subset_fixed, FWD_internal, fast_mode=True
                )

                self.shift_match_tolerance_m = 0.15 
                fine_cfg_set = set()

                top_k = min(self.top_k_coarse, len(coarse_sorted_results))
                for i in range(top_k):
                    c_dx, c_dy, c_dh = coarse_sorted_results[i][0]
                    fine_cfg_set.update(self.generate_fine_grid(c_dx, c_dy, c_dh))

                self.pose_candidate_offsets_cfg = list(fine_cfg_set)
                debug_search_mode = (
                    f"COARSE_TO_FINE #{self.coarse_search_count + 1}/"
                    f"{self.max_coarse_search_count} "
                    f"Top-{top_k} ({len(fine_cfg_set)} fine grids)"
                )

                print(f" -> FINE search (Top-{top_k}, {len(fine_cfg_set)} grids)...", end="")

                fine_sorted_results = self.nav_path_refine(
                    initial_pose, en_route, self.prev_index, bev_subset_fixed, FWD_internal
                )

                # Count the attempt now; update initialization state after quality checks.
                self.coarse_search_count += 1

            else:
                # ==========================================
                # Narrow Search
                # ==========================================
                self.shift_match_tolerance_m = 0.35 
                self.pose_candidate_offsets_cfg = self.generate_fine_grid(0.0, 0.0, 0.0)

                if self.need_coarse_reinit and self.coarse_search_count >= self.max_coarse_search_count:
                    debug_search_mode = (
                        f"NARROW Fine only "
                        f"(coarse budget exhausted {self.coarse_search_count}/"
                        f"{self.max_coarse_search_count})"
                    )
                else:
                    debug_search_mode = f"NARROW Fine only ({len(self.pose_candidate_offsets_cfg)} grids, fallback OFF)"

                print(f"    [Dynamic] Frame {idx}: {debug_search_mode}...", end="")

                fine_sorted_results = self.nav_path_refine(
                    initial_pose, en_route, self.prev_index, bev_subset_fixed, FWD_internal
                )
                
            t_search_end = time.perf_counter()
            search_time_ms = (t_search_end - t_search_start) * 1000.0
            
            # ==========================================
            # Stage 3: Candidate-level geometric sanity selection
            # ==========================================
            swap_event = "Normal"  # Keep the original log key name for compatibility

            def check_ego_center_sanity(lane_results):
                left_lane = lane_results.get("best_left_lane", np.empty((0, 2)))
                right_lane = lane_results.get("best_right_lane", np.empty((0, 2)))

                def get_lane_x_at_y(lane, y_query):
                    if lane is None or len(lane) < 2:
                        return None

                    lane = lane[np.argsort(lane[:, 1])]
                    y = lane[:, 1]
                    x = lane[:, 0]

                    if y_query < y.min() or y_query > y.max():
                        return None

                    return float(np.interp(y_query, y, x))

                CHECK_Y_M = 8.0
                MIN_SIDE_DIST_M = 0.8
                MAX_SIDE_DIFF_M = 1.2

                left_x = get_lane_x_at_y(left_lane, CHECK_Y_M)
                right_x = get_lane_x_at_y(right_lane, CHECK_Y_M)

                if left_x is None or right_x is None:
                    return False, "missing_lane_at_check_y"

                left_dist = abs(left_x)
                right_dist = abs(right_x)

                if not (left_x * right_x < 0):
                    return False, f"lanes_not_straddling_ego Lx={left_x:.2f} Rx={right_x:.2f}"

                if left_dist < MIN_SIDE_DIST_M or right_dist < MIN_SIDE_DIST_M:
                    return False, f"too_close_to_lane L={left_dist:.2f}m R={right_dist:.2f}m"

                if abs(left_dist - right_dist) > MAX_SIDE_DIFF_M:
                    return False, f"side_imbalance L={left_dist:.2f}m R={right_dist:.2f}m"

                return True, f"ego_center_ok L={left_dist:.2f}m R={right_dist:.2f}m"


            if fine_sorted_results:
                # Candidate-level ego-center sanity filtering.
                # Reject unreasonable ego-center candidates first, then select the highest-score valid one.
                ego_valid_results = []

                for cand_rank, cand in enumerate(fine_sorted_results):
                    cand_offset, cand_nav_poly, cand_score, cand_results = cand

                    ego_ok, ego_reason = check_ego_center_sanity(cand_results)

                    if ego_ok:
                        ego_valid_results.append((cand_rank + 1, cand, ego_reason))

                if len(ego_valid_results) > 0:
                    selected_rank, selected_cand, selected_reason = ego_valid_results[0]
                    best_pose_offset_log, best_nav_poly, best_score, best_lane_search_results = selected_cand

                    if selected_rank == 1:
                        swap_event = "Normal"
                    else:
                        swap_event = f"EgoCenter_Select_R{selected_rank} {selected_reason}"

                else:
                    # All candidates failed ego-center sanity.
                    # Keep the original rank-1 candidate here; the later final check will clear this frame.
                    best_pose_offset_log, best_nav_poly, best_score, best_lane_search_results = fine_sorted_results[0]
                    selected_rank = 1
                    selected_reason = "all_candidates_failed_ego_center"
                    swap_event = selected_reason
            
            # Select the state and refined pose.
            if best_score >= self.stable_score_threshold:
                debug_state = "OK"
                self.is_tracking_stable = True
            else:
                if len(self.prev_poses) == 0:
                    debug_state = "LOW_SCORE_INIT"
                else:
                    debug_state = "LOW_SCORE_NARROW_NO_FALLBACK"
                self.is_tracking_stable = False

            dx, dy, d_deg = best_pose_offset_log

            found_width_log = best_lane_search_results.get("found_width", 0.0) if best_lane_search_results else 0.0
            nav_kept_log = len(bev_subset_fixed) if bev_subset_fixed is not None else 0

            print(
                f" | State={debug_state} "
                f"Score={best_score:.2f} "
                f"Width={found_width_log:.2f}m "
                f"Offset=({dx:.2f},{dy:.2f},{d_deg:.1f}deg) "
                f"Time={search_time_ms:.1f}ms "
                f"nav_kept={nav_kept_log} "
                f"Event={swap_event}"
            )

            # Record per-frame statistics.
            if self.mode == 'osm':
                self.stats_log.append({
                    "Scene_ID": scene_id,
                    "Frame_ID": idx,
                    "Search_Mode": debug_search_mode,
                    "State": debug_state,
                    "Search_Time_ms": round(search_time_ms, 2),
                    "Offset_X_m": dx,
                    "Offset_Y_m": dy,
                    "Offset_Heading_deg": d_deg,
                    "Score": best_score,
                    "Found_Width_m": round(best_lane_search_results.get("found_width", 0), 2),
                    "Event": swap_event
                })

            candidate_refined_pose = BEVPose(
                initial_pose.position_x + dx,
                initial_pose.position_y + dy,
                initial_pose.heading + math.radians(d_deg),
                requires_grad=False
            )

            refined_pose = candidate_refined_pose
            final_search_results = best_lane_search_results
            final_nav_poly = best_nav_poly
            
        else:
            # GT mode
            search_time_ms = 0.0
            best_pose_offset_log = (0.0, 0.0, 0.0)
            best_score = 0.0 
            
            veh_route_now = map_en_route_to_veh_route(sub_en, initial_pose)
            final_nav_poly = clip_veh_route(veh_route_now, walk_dist=FWD_internal).detach().cpu().numpy()
            
            keep_mask = filter_edges_by_nav_direction(base_bev, unit_bev, final_nav_poly, self, y0, y1)
            bev_subset = base_bev[keep_mask]
            final_search_results = find_lanes_asymmetric(final_nav_poly, bev_subset, self)

            if final_search_results:
                best_score = final_search_results.get("best_score", 0.0)

        # =========================================================
        # [ACTION 4.5] Ego-lane center sanity check
        # Clear the frame if no trustworthy candidate remains.
        # =========================================================
        pose_update_allowed = True

        if self.mode == 'osm' and final_search_results:
            ego_ok_final, ego_reason_final = check_ego_center_sanity(final_search_results)

            if not ego_ok_final:
                final_search_results["best_left_lane"] = np.empty((0, 2))
                final_search_results["best_right_lane"] = np.empty((0, 2))

                pose_update_allowed = False

                print(f"    [Reject-EgoCenter][{idx:04d}] {ego_reason_final}")

        # =========================================================
        # Validate the two lane sides independently.
        # =========================================================
        
        # Reject the frame when its global score indicates an intersection or sparse features.
        REJECT_SCORE_THRESHOLD = 60 
        
        if best_score < REJECT_SCORE_THRESHOLD:
            # Suppress low-confidence pseudo-labels.
            final_search_results["best_left_lane"] = np.empty((0, 2))
            final_search_results["best_right_lane"] = np.empty((0, 2))

            pose_update_allowed = False
        else:
            # Reject short or sparse fragments on each side.
            MIN_SPAN_Y = 8.0
            MIN_DENSITY = 0.4
            
            has_left = "best_left_lane" in final_search_results and len(final_search_results["best_left_lane"]) > 0
            has_right = "best_right_lane" in final_search_results and len(final_search_results["best_right_lane"]) > 0
            
            pts_l = final_search_results.get("matched_left_pts", np.empty((0, 2)))
            pts_r = final_search_results.get("matched_right_pts", np.empty((0, 2)))
            
            def check_single_lane_quality(pts):
                if pts is None or len(pts) == 0: return False
                y_min, y_max = np.min(pts[:, 1]), np.max(pts[:, 1])
                span = y_max - y_min
                if span < MIN_SPAN_Y: return False
                
                density = len(pts) / (span + 1e-6)
                if density < MIN_DENSITY: return False
                
                return True

            # Clear either side that fails its geometry checks.
            if has_left and not check_single_lane_quality(pts_l):
                final_search_results["best_left_lane"] = np.empty((0, 2))

            if has_right and not check_single_lane_quality(pts_r):
                final_search_results["best_right_lane"] = np.empty((0, 2))

        # ======================================================
        # Adopt the searched pose only when final quality checks pass.
        # ======================================================
        if self.mode == 'osm':
            if pose_update_allowed and best_score >= REJECT_SCORE_THRESHOLD and swap_event != "all_candidates_failed_ego_center":
                # The search result is trustworthy.
                refined_pose = candidate_refined_pose

                self.has_search_initialized = True
                self.need_coarse_reinit = False
            else:
                # Fall back instead of applying an unreliable offset.
                refined_pose = initial_pose

                # Keep coarse reinitialization pending until initialization succeeds.
                if not self.has_search_initialized:
                    if self.coarse_search_count < self.max_coarse_search_count:
                        self.need_coarse_reinit = True
                    else:
                        self.need_coarse_reinit = False
        # ======================================================
        # Append the final pose only after all checks.
        # ======================================================
        self.prev_poses.append(refined_pose)

        # -----------------------------------------------------
        # Update lane-width history for smoothing.
        # -----------------------------------------------------
        curr_width = final_search_results.get("found_width", 0.0)
        remaining_left = len(final_search_results.get("best_left_lane", [])) > 0
        remaining_right = len(final_search_results.get("best_right_lane", [])) > 0
        
        if (remaining_left or remaining_right) and curr_width > 0:
            if len(self.width_history) == 0:
                if 2.5 <= curr_width <= 5.0: 
                    self.width_history.append(curr_width)
            else:
                self.width_history.append(curr_width)
                if len(self.width_history) > self.max_history_len:
                    self.width_history.pop(0)

        # =========================================================

        # Post-process and package results.
        final_kept_mask = filter_edges_by_nav_direction(base_bev, unit_bev, final_nav_poly, self, y0, y1)
        kept_pts_img = pts_img[final_kept_mask]
        kept_base_bev_for_combo = base_bev[final_kept_mask] 
        kept_base_bev = final_search_results.get('viz_points', kept_base_bev_for_combo)
        gps_gt = frame.get("gps_gt")
        heading_gt = frame.get("heading_gt")

        gt_pose_obj = None
        if gps_gt is not None and heading_gt is not None:
            # Build a non-differentiable pose for visualization only.
            gt_pose_obj = BEVPose(gps_gt[0], gps_gt[1], heading_gt, requires_grad=False)

        results = {
            "original_image": img, "edges": edges, "tan_deg_unified": tan_deg_unified,
            "nav_poly": final_nav_poly, "FWD_internal": FWD_internal, "FWD_disp": FWD_disp,
            "kept_pts_img": kept_pts_img, 
            "kept_base_bev": kept_base_bev, 
            "lane_search_results": final_search_results, 
            "bev_roi_poly": bev_roi_poly,
            "debug_raw_route": initial_nav_poly_full,
            "initial_pose": initial_pose,
            "refined_pose": refined_pose,
            "best_pose_offset": best_pose_offset_log,
            "gt_pose": gt_pose_obj
        }
        return results
    
    def export_statistics(self, out_dir: str, file_name: str = "all_scenes_stats.csv"):
        """Append statistics to a CSV shared across multiple scenes."""
        if not self.stats_log:
            return
        
        df = pd.DataFrame(self.stats_log)
        out_path = os.path.join(out_dir, file_name)
        
        # Write a header only when creating the file.
        file_exists = os.path.isfile(out_path)
        
        # Append this scene's rows.
        df.to_csv(out_path, mode='a', index=False, encoding='utf-8-sig', header=not file_exists)
        
        print(f"\n[Stats] Successfully appended {len(self.stats_log)} frames to: {out_path}")
        
        # Clear in-memory rows after writing to avoid duplicate output.
        self.stats_log = []

class BEVPose:
    def __init__(self, position_x, position_y, heading, requires_grad=True):
        def to_param(val):
            if isinstance(val, torch.Tensor): return nn.Parameter(val.clone().detach().double(), requires_grad=requires_grad)
            else: return nn.Parameter(torch.tensor(val, dtype=torch.float64), requires_grad=requires_grad)
        self.position_x, self.position_y, self.heading = to_param(position_x), to_param(position_y), to_param(heading)

def visualize_frame_results(cfg: SSLD, Hpx: np.ndarray, idx: int, results: Dict[str, Any],
                            vis_settings: object, arrows_per_frame: int,
                            prev_poses: list, en_route_en: torch.Tensor,
                            calib_data: Optional[Dict] = None): 
    """
    Visualize one processed frame.
    The right panel shows matched points rather than every candidate.
    """
    if not results: return

    # Unpack data.
    img = results["original_image"]
    edges = results["edges"]
    tan_deg_unified = results["tan_deg_unified"]
    nav_poly = results["nav_poly"]
    FWD_internal, FWD_disp = results["FWD_internal"], results["FWD_disp"]
    kept_pts_img = results["kept_pts_img"]
    kept_base_bev = results["kept_base_bev"]
    lane_search_results = results["lane_search_results"]
    bev_roi_poly = results["bev_roi_poly"]
    debug_raw_route = results.get("debug_raw_route")

    # -------------------------------------------------------------------------
    # Draw a BEV polyline in image space.
    # -------------------------------------------------------------------------
    def _draw_lane_on_image(canvas, bev_points_m, color=(255, 0, 255), thickness=3):
        if bev_points_m is None or len(bev_points_m) < 2: return
        min_y, max_x = cfg.roi_y_start_m, cfg.lr_m
        try:
            valid_mask = (bev_points_m[:, 1] >= min_y) & (np.abs(bev_points_m[:, 0]) <= max_x)
            valid_pts = bev_points_m[valid_mask]
            if len(valid_pts) < 2: return
            img_pts = vis.project_bev_poly_to_image(valid_pts, Hpx, cfg)
            if len(img_pts) < 2: return
            sort_idx = np.argsort(img_pts[:, 1])[::-1]
            cv2.polylines(canvas, [np.round(img_pts[sort_idx]).astype(np.int32)],
                          isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
        except Exception as e:
            print(f"[WARN] Draw lane failed: {e}")

    # -------------------------------------------------------------------------
    # Prepare the ROI and candidate-point canvas.
    # -------------------------------------------------------------------------
    roi_img_poly = vis.project_bev_poly_to_image(bev_roi_poly, Hpx, cfg)
    img_with_bev_roi = vis.draw_img_polygon(img, roi_img_poly, color=(127, 127, 0), alpha=0.35, closed=True, thick=2)
    # Add status text.
    cv2.putText(img_with_bev_roi, f"FWD: {FWD_disp:.1f}m", (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img_with_bev_roi, f"FWD: {FWD_disp:.1f}m", (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    
    # Draw all candidate points in green.
    img_with_bev_roi = vis.draw_points_on_image(img_with_bev_roi, kept_pts_img, color=(0, 255, 0))

    # -------------------------------------------------------------------------
    # Prepare image-space views.
    # -------------------------------------------------------------------------

    # Dashboard camera view with pseudo-label lines only.
    cam_view_with_pseudo = img.copy()

    # Side-by-side view with matched points and pseudo-label lines.
    vis_fit_only_img = img.copy()
    
    # Draw only matched points in the right panel.
    if lane_search_results:
        # Collect matched points from both sides.
        pts_list = []
        if 'matched_left_pts' in lane_search_results: pts_list.append(lane_search_results['matched_left_pts'])
        if 'matched_right_pts' in lane_search_results: pts_list.append(lane_search_results['matched_right_pts'])
        
        for bev_pts in pts_list:
            if bev_pts is not None and len(bev_pts) > 0:
                img_pts = vis.project_bev_poly_to_image(bev_pts, Hpx, cfg)
                for pt in img_pts:
                    cv2.circle(vis_fit_only_img, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1, cv2.LINE_AA)

    # Draw selected lane boundaries in purple.
    if lane_search_results:
        best_left = lane_search_results.get('best_left_lane')
        best_right = lane_search_results.get('best_right_lane')
        
        # ROI and candidate-point view.
        _draw_lane_on_image(img_with_bev_roi, best_left)
        _draw_lane_on_image(img_with_bev_roi, best_right)

        # Matched-point view.
        _draw_lane_on_image(vis_fit_only_img, best_left)
        _draw_lane_on_image(vis_fit_only_img, best_right)

        # Dashboard camera view.
        _draw_lane_on_image(cam_view_with_pseudo, best_left)
        _draw_lane_on_image(cam_view_with_pseudo, best_right)

    # -------------------------------------------------------------------------
    # Prepare BEV visualization.
    # -------------------------------------------------------------------------
    bev_img = cv2.warpPerspective(img, Hpx, (cfg.bev_w, cfg.bev_h), flags=cv2.INTER_LINEAR)
    viz_setting_base = {
        'width': cfg.bev_w, 'height': cfg.bev_h,
        'forward_meters': cfg.forward_m, 'backward_meters': cfg.backward_m,
        'polygons': [{'points': bev_roi_poly, 'color': (255, 255, 0), 'alpha': 0.35, 'border_color': (0, 140, 255), 'thickness': 2, 'closed': True}],
        'polylines': [], 'poses': [],
        'texts': [{'text': f"FWD display={FWD_disp:.1f}m (internal={FWD_internal:.1f}m)", 'pos': (12, 28), 'color': (255, 255, 255), 'bg_color': (0, 0, 0)}]
    }
    if nav_poly is not None: viz_setting_base['polylines'].append({'points': nav_poly, 'color': (0, 0, 255), 'thickness': 3})
    
    if lane_search_results:
        viz_setting_base['polylines'].append({'points': lane_search_results.get('best_left_lane', np.empty((0,2))), 'color': (0, 255, 255), 'thickness': 2})
        viz_setting_base['polylines'].append({'points': lane_search_results.get('best_right_lane', np.empty((0,2))), 'color': (0, 255, 255), 'thickness': 2})
        
        for pt in lane_search_results.get('matched_left_pts', []): 
            viz_setting_base['poses'].append({'pos': pt, 'radius': 2, 'color': (0, 165, 255), 'heading_is_valid': False})
        for pt in lane_search_results.get('matched_right_pts', []): 
            viz_setting_base['poses'].append({'pos': pt, 'radius': 2, 'color': (255, 0, 0), 'heading_is_valid': False})

        viz_setting_base['texts'].append({'text': f"Found Width: {lane_search_results.get('found_width', 0):.2f}m", 'pos': (12, 52), 'color': (0, 255, 255), 'bg_color': (0,0,0)})
        viz_setting_base['texts'].append({'text': f"Score: {lane_search_results.get('best_score', 0):.2f}", 'pos': (12, 76), 'color': (255, 255, 255), 'bg_color': (0,0,0)})

    poses = viz_setting_base.get('poses', [])
    if len(kept_base_bev) > 0:
        ids = np.arange(len(kept_base_bev)); np.random.shuffle(ids)
        for i in ids[:arrows_per_frame]: 
            poses.append({'type': 'lane_candidate', 'pos': kept_base_bev[i], 'radius': 1, 'color': (0, 255, 0), 'heading_is_valid': False})
    viz_setting_base['poses'] = poses
    
    bev_overlay = vis.draw_vehicle_bev(copy.deepcopy(viz_setting_base), base_image=bev_img)
    bev_grid = vis.draw_vehicle_bev(copy.deepcopy(viz_setting_base), base_image=None)

    # Save individual views.
    scene_token = utils.norm_scene_id(vis_settings.scene_id)
    fn = lambda name: os.path.join(vis_settings.out_dir, f"{name}_{scene_token}_{idx:04d}.png")
    utils.ensure_dir(vis_settings.out_dir)
    
    cv2.imwrite(fn("b_match_guided_img"), img_with_bev_roi) 
    # Save the original/matched-point side-by-side view.
    side_by_side = cv2.hconcat([img, vis_fit_only_img])
    cv2.imwrite(fn("e_side_by_side_fit"), side_by_side)

    # bins (Optional)
    bins = vis.angle_bins_visual(edges, tan_deg_unified, vis_settings.bin_deg, half_circle=True)
    # Initialize the map view for graceful fallback.
    map_bev_image = None

    # Map-centric BEV visualization.
    if prev_poses: 
        try:            
            refined_pose = results.get("refined_pose")
            curr_init_pose = results.get("initial_pose")
            gt_pose = results.get("gt_pose")
            
            # Use the complete route extent.
            route_pts = en_route_en.detach().cpu().numpy()
            
            # Fix the map center to the route bounds to prevent view jitter.
            route_min = np.min(route_pts, axis=0)
            route_max = np.max(route_pts, axis=0)
            route_center = (route_min + route_max) / 2.0
            
            # Fit the complete route within the 1000-pixel canvas.
            route_range = np.max(route_max - route_min)
            # Leave a margin around the route.
            auto_ppm = (1000.0 * 0.8) / max(route_range, 1.0) 

            map_bev_setting = {
                'width': 1000, 'height': 1000, 
                'pixel_per_meter': auto_ppm,
                'center_pose_world': (route_center[0], route_center[1]),
                'route': {
                    'points': route_pts, 
                    'color': [100, 100, 100], 
                    'radius': 0, 'thickness': 2, 'name': 'Global Route'
                },
                'poses': []
            }

            # Historical poses in light green.
            for k, pose_obj in enumerate(prev_poses[:-1]):
                map_bev_setting['poses'].append({
                    'pose': (pose_obj.position_x.item(), pose_obj.position_y.item(), pose_obj.heading.item()),
                    'color': [150, 255, 150], 'radius': 3, 'heading_is_valid': 0, 'name': 'History'
                })

            # Initial GPS pose in orange.
            map_bev_setting['poses'].append({
                'pose': (curr_init_pose.position_x.item(), curr_init_pose.position_y.item(), curr_init_pose.heading.item()),
                'color': [0, 165, 255], # Orange
                'radius': 6, 'heading_is_valid': 1, 'name': 'Initial GPS'
            })

            # Refined pose in cyan.
            if refined_pose:
                map_bev_setting['poses'].append({
                    'pose': (refined_pose.position_x.item(), refined_pose.position_y.item(), refined_pose.heading.item()),
                    'color': [255, 255, 0], # Cyan
                    'radius': 8, 'heading_is_valid': 1, 'name': 'Refined Pose'
                })

            # ==========================================
            # Ground-truth pose in magenta.
            # ==========================================
            if gt_pose is not None:
                # Accept either a BEVPose or a plain coordinate sequence.
                gt_x = gt_pose.position_x.item() if hasattr(gt_pose, 'position_x') else gt_pose[0]
                gt_y = gt_pose.position_y.item() if hasattr(gt_pose, 'position_y') else gt_pose[1]
                # Default to zero when heading is unavailable.
                gt_h = gt_pose.heading.item() if hasattr(gt_pose, 'heading') else (gt_pose[2] if len(gt_pose) > 2 else 0.0)

                map_bev_setting['poses'].append({
                    'pose': (gt_x, gt_y, gt_h),
                    'color': [255, 0, 255],  # Magenta in BGR format
                    'radius': 7, 'heading_is_valid': 1, 'name': 'GT Pose'
                })

            map_bev_image = vis.draw_map_bev(map_bev_setting)
            cv2.imwrite(fn("map_bev"), map_bev_image)
            
        except Exception as e:
            print(f"[WARN] Failed to generate map_bev visual for frame {idx}: {e}")
    # ===================================================================
    # ========= Custom Dashboard Visualization ==========================
    # ===================================================================

    # Use a blank map when map rendering is unavailable.
    if map_bev_image is None:
        map_bev_image = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
        cv2.putText(
            map_bev_image,
            "map_bev unavailable",
            (260, 500),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )

    # Compose camera, map-centric, and vehicle-centric views.
    dashboard = vis.combine_custom_dashboard(
        cam_view=cam_view_with_pseudo,
        veh_bev=bev_grid,
        map_bev=map_bev_image,
        cam_lanes_only=img
    )

    cv2.imwrite(fn("z_combo"), dashboard)

def edges_and_tangent(img_bgr: np.ndarray, cfg: SSLD) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, cfg.gauss_ksize, cfg.gauss_sigma)
    edges = cv2.Canny(gray, cfg.canny_low, cfg.canny_high)
    roi = vis.handle_image_roi(gray.shape, cfg, mode='mask')
    edges &= roi
    ys, xs = np.where(edges > 0)
    if len(xs) == 0: return edges, np.empty((0, 2)), np.empty(0)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    theta_grad = np.arctan2(gy, gx)
    theta_tan = theta_grad + (math.pi / 2.0)
    return edges, np.stack([xs, ys], axis=1).astype(np.float64), theta_tan[ys, xs]

def find_lanes_by_symmetric_expansion(nav_poly: np.ndarray, bev_points: np.ndarray, cfg: SSLD, fast_mode: bool = False) -> Optional[dict]:
    if nav_poly is None or len(nav_poly) < 2 or len(bev_points) == 0:
        return None
    path = utils.resample_poly_by_ds(nav_poly, ds=0.25)
    if len(path) < 2: return None
    normals = []
    vecs = np.diff(path, axis=0)
    seg_normals = np.c_[-vecs[:, 1], vecs[:, 0]]
    seg_normals /= (np.linalg.norm(seg_normals, axis=1, keepdims=True) + 1e-9)
    normals.append(seg_normals[0])
    normals.extend(seg_normals)
    normals = np.array(normals)
    nav_path_tree = cKDTree(path)
    distances_to_center, _ = nav_path_tree.query(bev_points)
    valid_indices = np.where(distances_to_center > cfg.exclusion_radius_m)[0]
    filtered_bev_points = bev_points[valid_indices]
    if len(filtered_bev_points) == 0: return None
    filtered_points_tree = cKDTree(filtered_bev_points)

    best_score = -1
    best_half_width = (cfg.lane_width_min_m + cfg.lane_width_max_m) / 4.0
    if fast_mode:
        # Representative narrow, normal, and wide half-widths.
        half_width_range = [1.4, 1.75, 2.1]
    else:
        half_width_range = np.arange(cfg.lane_width_min_m / 2.0, cfg.lane_width_max_m / 2.0, cfg.lane_width_step_m / 2.0)

    # ==========================================
    # Allow slight centerline straddling while rejecting lanes on the same side.
    # ==========================================
    STRADDLE_MARGIN = 0.3

    # Width-penalty parameters.
    OPTIMAL_WIDTH_MAX = 3.8
    PENALTY_RATE_PER_METER = 0.30

    # Evaluate straddling at the first visible ROI point, outside the front blind spot.
    idx_roi = np.searchsorted(path[:, 1], cfg.roi_y_start_m)
    if idx_roi >= len(path): idx_roi = len(path) - 1

    for half_width in half_width_range:
        left_candidate = path + normals * half_width
        right_candidate = path - normals * half_width

        # Check whether the vehicle straddles the candidate boundaries.
        left_start_x = left_candidate[idx_roi, 0]
        right_start_x = right_candidate[idx_roi, 0]
        
        if min(left_start_x, right_start_x) > STRADDLE_MARGIN: continue
        if max(left_start_x, right_start_x) < -STRADDLE_MARGIN: continue

        dist_left, _ = filtered_points_tree.query(left_candidate)
        dist_right, _ = filtered_points_tree.query(right_candidate)
        
        score_left = cfg._calculate_segment_score(dist_left, cfg.shift_match_tolerance_m)
        score_right = cfg._calculate_segment_score(dist_right, cfg.shift_match_tolerance_m)
        
        raw_total_score = score_left + score_right

        # ==========================================
        # Penalize excessively wide candidates.
        # ==========================================
        current_width = half_width * 2.0
        penalty_factor = 1.0
        if current_width > OPTIMAL_WIDTH_MAX:
            # Retain at least 40% of the score.
            excess_m = current_width - OPTIMAL_WIDTH_MAX
            penalty_factor = max(0.4, 1.0 - (excess_m * PENALTY_RATE_PER_METER))

        total_score = raw_total_score * penalty_factor

        if total_score > best_score:
            best_score = total_score
            best_half_width = half_width
    
    best_left_lane = path + normals * best_half_width
    best_right_lane = path - normals * best_half_width
    found_width = best_half_width * 2.0
    if len(best_left_lane) > 0:
        left_lane_tree = cKDTree(best_left_lane)
        dist_to_final_left, _ = left_lane_tree.query(filtered_bev_points)
        matched_left_pts = filtered_bev_points[dist_to_final_left < cfg.shift_match_tolerance_m]
    else: matched_left_pts = np.empty((0, 2))
    if len(best_right_lane) > 0:
        right_lane_tree = cKDTree(best_right_lane)
        dist_to_final_right, _ = right_lane_tree.query(filtered_bev_points)
        matched_right_pts = filtered_bev_points[dist_to_final_right < cfg.shift_match_tolerance_m]
    else: matched_right_pts = np.empty((0, 2))
    viz_points = np.concatenate([matched_left_pts, matched_right_pts]) if len(matched_left_pts) > 0 or len(matched_right_pts) > 0 else np.empty((0, 2))

    return {"best_left_lane": best_left_lane, "best_right_lane": best_right_lane, "best_score": int(best_score),
            "found_width": found_width, "matched_left_pts": matched_left_pts, "matched_right_pts": matched_right_pts,
            "viz_points": viz_points}

# ===================================================================
# ========= Asymmetric lane search for Ground Truth mode =============
# ===================================================================

def find_lanes_asymmetric(nav_poly: np.ndarray, bev_points: np.ndarray, cfg: SSLD) -> Optional[dict]:
    """
    Search left and right widths independently in GT mode.
    """
    if nav_poly is None or len(nav_poly) < 2 or len(bev_points) == 0:
        return {}
    
    path = utils.resample_poly_by_ds(nav_poly, ds=0.25)
    if len(path) < 2: return {}
    
    # Build path normals.
    vecs = np.diff(path, axis=0)
    seg_normals = np.c_[-vecs[:, 1], vecs[:, 0]]
    seg_normals /= (np.linalg.norm(seg_normals, axis=1, keepdims=True) + 1e-9)
    normals = np.vstack([seg_normals[0], seg_normals])
    
    # Prepare the spatial index.
    nav_path_tree = cKDTree(path)
    distances_to_center, _ = nav_path_tree.query(bev_points)
    valid_indices = np.where(distances_to_center > cfg.exclusion_radius_m)[0]
    filtered_bev_points = bev_points[valid_indices]
    if len(filtered_bev_points) == 0: return {}
    
    filtered_points_tree = cKDTree(filtered_bev_points)

    # Find the best left width.
    best_score_L = -1
    best_width_L = cfg.lane_width_min_m
    width_range = np.arange(cfg.lane_width_min_m / 2.0, cfg.lane_width_max_m / 2.0, cfg.lane_width_step_m / 2.0)
    
    for w in width_range:
        left_candidate = path + normals * w
            
        dist, _ = filtered_points_tree.query(left_candidate)
        score = cfg._calculate_segment_score(dist, cfg.shift_match_tolerance_m)
        if score > best_score_L:
            best_score_L = score
            best_width_L = w
            
    best_left_lane = path + normals * best_width_L
    
    # Find the best right width.
    best_score_R = -1
    best_width_R = cfg.lane_width_min_m
    
    for w in width_range:
        right_candidate = path - normals * w

        dist, _ = filtered_points_tree.query(right_candidate)
        score = cfg._calculate_segment_score(dist, cfg.shift_match_tolerance_m)
        if score > best_score_R:
            best_score_R = score
            best_width_R = w
            
    best_right_lane = path - normals * best_width_R
    
    # Collect matched points.
    if len(best_left_lane) > 0:
        left_lane_tree = cKDTree(best_left_lane)
        dist_L, _ = left_lane_tree.query(filtered_bev_points)
        matched_left_pts = filtered_bev_points[dist_L < cfg.shift_match_tolerance_m]
    else: matched_left_pts = np.empty((0, 2))
        
    if len(best_right_lane) > 0:
        right_lane_tree = cKDTree(best_right_lane)
        dist_R, _ = right_lane_tree.query(filtered_bev_points)
        matched_right_pts = filtered_bev_points[dist_R < cfg.shift_match_tolerance_m]
    else: matched_right_pts = np.empty((0, 2))
    
    viz_points = np.concatenate([matched_left_pts, matched_right_pts]) if len(matched_left_pts) > 0 or len(matched_right_pts) > 0 else np.empty((0, 2))

    return {
        "best_left_lane": best_left_lane, 
        "best_right_lane": best_right_lane, 
        "best_score": int(best_score_L + best_score_R),
        "found_width": (best_width_L + best_width_R),
        "matched_left_pts": matched_left_pts, 
        "matched_right_pts": matched_right_pts,
        "viz_points": viz_points
    }

def compute_inverse_perspective_homography(bev_to_img_rot, bev_to_img_trans, calib_mat) -> np.ndarray:
    bev_to_img_extrinsic = torch.ones(3, 4, dtype=torch.float64)
    bev_to_img_extrinsic[:3, :3] = bev_to_img_rot
    bev_to_img_extrinsic[:3, 3] = bev_to_img_trans
    bev_to_img_xform = calib_mat @ bev_to_img_extrinsic
    bev_to_img_xform_del = torch.column_stack((bev_to_img_xform[:, 0], bev_to_img_xform[:, 2], bev_to_img_xform[:, 3]))
    return torch.linalg.inv(bev_to_img_xform_del).detach().cpu().numpy()

def apply_inverse_perspective_mapping(homography_torch: torch.Tensor, img_lanes: List[torch.Tensor]) -> List[torch.Tensor]:
    bev_lanes = []
    for img_lane in img_lanes:
        ones = torch.ones(1, img_lane.shape[0], requires_grad=False, device=img_lane.device, dtype=img_lane.dtype)
        img_lane_h = torch.cat([img_lane.T, ones], dim=0)
        bev_lane_h = homography_torch @ img_lane_h
        bev_lane = (bev_lane_h[:2] / bev_lane_h[2]).T
        bev_lanes.append(bev_lane)
    return bev_lanes

def img2bev_Hpx(homography_m: np.ndarray, cfg: SSLD) -> Tuple[np.ndarray, float]:
    PPM = cfg.bev_h / (cfg.forward_m + cfg.backward_m)
    origin_px = cfg.bev_w // 2
    origin_py = int(cfg.forward_m * PPM)
    A = np.array([[PPM, 0., origin_px], [0., -PPM, origin_py], [0., 0., 1.]], np.float64)
    return A @ homography_m, PPM

def ipm_points(homography_m: np.ndarray, xy_img: np.ndarray) -> np.ndarray:
    if len(xy_img) == 0: return np.empty((0, 2), np.float64)
    homography_torch = torch.tensor(homography_m, dtype=torch.float64)
    t = torch.tensor(xy_img, dtype=torch.float64)
    return apply_inverse_perspective_mapping(homography_torch, [t])[0].detach().cpu().numpy()

def compute_tangent_angles(gray_image_uint8) -> np.ndarray:
    gx = cv2.Sobel(gray_image_uint8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_image_uint8, cv2.CV_32F, 0, 1, ksize=3)
    return np.rad2deg(np.arctan2(gy, gx) + math.pi / 2.0)

def unify_tangent_angles(tan_deg_img: np.ndarray, ref_dir_deg: Optional[float] = None, half_circle=True) -> np.ndarray:
    ang = tan_deg_img % 360.0
    if ref_dir_deg is None: return ang % 180.0 if half_circle else ang
    rd = ref_dir_deg % 360.0
    diff = ((ang - rd + 540.0) % 360.0) - 180.0
    ang[np.abs(diff) > 90.0] += 180.0
    ang %= 360.0
    return ang

def image_dirs_to_bev_vecs(homography_m: np.ndarray, pts_img: np.ndarray, theta_img: np.ndarray,
                           step_pix: float, forward_only=True) -> Tuple[np.ndarray, np.ndarray]:
    if len(pts_img) == 0: return np.empty((0, 2)), np.empty((0, 2))
    step = np.stack([np.cos(theta_img), np.sin(theta_img)], axis=1) * step_pix
    base = ipm_points(homography_m, pts_img)
    fwd = ipm_points(homography_m, pts_img + step)
    v = fwd - base
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    unit = v / n
    if forward_only:
        unit[unit[:, 1] < 0.0] *= -1.0
    return base, unit

def pose_from_gps_on_en_route(en_route, prev_index, gps_xy, BEVPose_class, 
                              lookahead_points=1, use_raw_gps=False, 
                              heading_gt=None):
    """
    Locate a pose along ``en_route`` from GPS.
    GT mode may use raw GPS and a supplied ground-truth heading.
    """
    gps = torch.tensor(gps_xy, dtype=torch.float64)
    
    # Project GPS onto the route to obtain the route index.
    idx, proj = match_route(en_route, prev_index, gps)
    
    # Determine heading.
    if use_raw_gps and heading_gt is not None:
        # Prefer the IMU ground truth in GT mode.
        heading = torch.tensor(heading_gt, dtype=torch.float64)
        # NuScenes heading is expressed in the global frame.
    else:
        # Fall back to the route tangent in OSM mode or when heading is missing.
        i_start = min(max(idx, 0), len(en_route) - 2)
        i_end = min(i_start + lookahead_points, len(en_route) - 1)
        p_start = en_route[i_start]
        p_end = en_route[i_end]
        disp = p_end - p_start
        
        if torch.norm(disp) < 1e-3: 
            i_fallback = min(max(idx, 0), len(en_route) - 2)
            disp = en_route[i_fallback + 1] - en_route[i_fallback]

        heading = torch.atan2(disp[0], disp[1])
    
    # Select the vehicle position.
    if use_raw_gps:
        pose_x, pose_y = gps[0], gps[1]
    else:
        pose_x, pose_y = proj[0], proj[1]
    
    # Construct the pose.
    pose = BEVPose_class(pose_x, pose_y, heading, requires_grad=False)
    
    new_prev_index = idx 
    final_prev_index = max(0, new_prev_index) 
    
    return pose, final_prev_index

def calc_sensor_predicted_pose(prev_pose, yaw_rate, speed, time_step, BEVPose_class):
    """Predict the next pose from the vehicle-motion model."""
    heading_prev = prev_pose.heading
    heading_s = heading_prev + yaw_rate * time_step  

    if yaw_rate == 0:
        y_s = prev_pose.position_y + speed * torch.cos(heading_prev) * time_step
        x_s = prev_pose.position_x + speed * torch.sin(heading_prev) * time_step
    else:
        y_s = prev_pose.position_y + speed / yaw_rate * (torch.sin(heading_s) - torch.sin(heading_prev))
        x_s = prev_pose.position_x - speed / yaw_rate * (torch.cos(heading_s) - torch.cos(heading_prev))

    # BEVPose handles tensor conversion internally.
    return BEVPose_class(x_s, y_s, heading_s, requires_grad=False)

def filter_edges_by_nav_direction(base_bev: np.ndarray, unit_bev: np.ndarray,
                                  nav_poly: np.ndarray, cfg: SSLD,
                                  y0: float, y1: float) -> np.ndarray:
    if len(base_bev) == 0: 
        return np.zeros(len(base_bev), dtype=bool)
    
    # Spatial ROI filtering.
    mask_roi = (base_bev[:, 0] >= -cfg.roi_half_width_m) & \
               (base_bev[:, 0] <= cfg.roi_half_width_m) & \
               (base_bev[:, 1] >= y0) & \
               (base_bev[:, 1] <= y1)
    
    return mask_roi
