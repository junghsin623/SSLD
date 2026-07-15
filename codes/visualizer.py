# filename: visualizer.py
# -*- coding: utf-8 -*-
"""
Visualization module for vehicle- and map-centric views.
Uses ``viz_setting`` dictionaries and pose lists for flexible rendering.
"""
import os
import cv2
import numpy as np
from typing import List, Tuple

# Utility imports
from utils import norm_scene_id, ensure_dir, H_inv
from tools.visualizer import draw_grid_lines, truncate_route, combine_custom_dashboard

def draw_legend(width, type, viz_setting, image, start_index=0):
    """
    [MODIFIED]
    Draw a legend in the upper-right corner without overlapping prior entries.
    """
    legend_font = cv2.FONT_HERSHEY_SIMPLEX
    legend_scale = 0.6
    legend_thickness = 1

    legend_x = width - 180
    legend_y = 20
    line_height = 25
    
    items_to_draw = []
    
    # Combine legend entries from different sources.
    if type == 'poses':
        items_to_draw = viz_setting.get('poses', [])
    elif type == 'routes':
        if 'route' in viz_setting: items_to_draw.append(viz_setting['route'])
        if 'sub_route' in viz_setting: items_to_draw.append(viz_setting['sub_route'])
    
    # Deduplicate entries by name.
    unique_legends = {}
    for info in items_to_draw:
        name = info.get('name', None)
        if name:
            color = tuple(info.get('color', [0, 255, 0]))
            unique_legends[name] = color
    
    i = start_index
    for name, color in unique_legends.items():
        # Draw the color swatch.
        box_x1, box_y1 = legend_x, legend_y + i * line_height
        box_x2, box_y2 = box_x1 + 15, box_y1 + 15
        cv2.rectangle(image, (box_x1, box_y1), (box_x2, box_y2), color, -1)

        # Draw the label.
        cv2.putText(image, name, (box_x2 + 5, box_y2 - 2),
                    legend_font, legend_scale, (0, 0, 0), legend_thickness, cv2.LINE_AA)
        i += 1
        
    return i

# ===================================================================
# ========= [MODIFIED] SSLD Vehicle-Centric BEV =====================
# ===================================================================

def draw_vehicle_bev(viz_setting: dict, base_image: np.ndarray = None) -> np.ndarray:
    """
    [MODIFIED]
    Draw polygons, polylines, poses, and text in vehicle-centric BEV space.
    """
    # Read canvas and BEV parameters.
    width = int(viz_setting.get('width', 600))
    height = int(viz_setting.get('height', 800))
    forward_m = viz_setting.get('forward_meters', 60.0)
    backward_m = viz_setting.get('backward_meters', 20.0)
    PPM = height / (forward_m + backward_m)

    # Compute the vehicle origin in pixels.
    origin_px = width // 2
    origin_py = int(forward_m * PPM)

    # Create a canvas or reuse the supplied base image.
    if base_image is None:
        image = np.ones((height, width, 3), dtype=np.uint8) * 255
        # Vehicle-centric views are always centered at the origin.
        draw_grid_lines(width, height, PPM, origin_px, origin_py, image, center_x=0, center_y=0)
    else:
        image = base_image.copy()

    # Convert meter coordinates to pixels.
    def _bev_meter_to_pixel(x_m, y_m):
        return int(round(origin_px + x_m * PPM)), int(round(origin_py - y_m * PPM))

    # Draw polygons.
    for poly_info in viz_setting.get('polygons', []):
        points_m = poly_info.get('points', np.empty((0, 2)))
        if points_m.shape[0] < 3: continue
        points_px = np.array([_bev_meter_to_pixel(x, y) for x, y in points_m], dtype=np.int32)
        alpha = poly_info.get('alpha', 0.5)
        if alpha > 0:
            overlay = image.copy()
            cv2.fillPoly(overlay, [points_px], tuple(poly_info.get('color', (255, 255, 0))))
            image = cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)
        if 'border_color' in poly_info:
            cv2.polylines(image, [points_px], poly_info.get('closed', True), tuple(poly_info.get('border_color')), poly_info.get('thickness', 2), cv2.LINE_AA)

    # Draw polylines.
    for line_info in viz_setting.get('polylines', []):
        points_m = line_info.get('points', np.empty((0, 2)))
        if points_m.shape[0] < 2: continue
        points_px = np.array([_bev_meter_to_pixel(x, y) for x, y in points_m], dtype=np.int32)
        cv2.polylines(image, [points_px], False, tuple(line_info.get('color', (0, 0, 255))), line_info.get('thickness', 3), cv2.LINE_AA)

    # Draw poses with optional heading arrows.
    for pose in viz_setting.get('poses', []):
        pos_m = pose.get('pos')
        if pos_m is None: continue
        
        # Draw the pose marker.
        px, py = _bev_meter_to_pixel(pos_m[0], pos_m[1])
        cv2.circle(image, (px, py), pose.get('radius', 1), tuple(pose.get('color', (0, 255, 0))), -1, cv2.LINE_AA)
                   
        # Draw an arrow only when heading is valid.
        if pose.get('heading_is_valid', False):
            vec = pose.get('heading_vec')
            if vec is None: continue
            
            arrow_len_m = pose.get('arrow_len_m', 1.0)
            p_end_m = (pos_m[0] + vec[0] * arrow_len_m, pos_m[1] + vec[1] * arrow_len_m)
            p1_px = _bev_meter_to_pixel(p_end_m[0], p_end_m[1])
            
            # Default the arrow to the marker color.
            arrow_color = tuple(pose.get('arrow_color', pose.get('color')))
            cv2.arrowedLine(image, (px, py), p1_px, arrow_color, pose.get('arrow_thickness', 1), tipLength=pose.get('tipLength', 0.4))

    # Draw text overlays.
    for text_info in viz_setting.get('texts', []):
        text, pos = text_info.get('text', ""), text_info.get('pos', (10, 30))
        bg_color = text_info.get('bg_color', None)
        if bg_color is not None:
            cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, bg_color, 3, cv2.LINE_AA)
        cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_info.get('color', (255,255,255)), 1, cv2.LINE_AA)
        
    return image

# ===================================================================
# ================= SSLD image-space rendering and saving =================
# ===================================================================

def handle_image_roi(img_shape: Tuple[int, int], cfg, mode='mask', draw_on_img: np.ndarray = None,
                     show_params: bool = True) -> np.ndarray:
    """
    Generate or visualize the initial image-space ROI.
    """
    h, w = img_shape
    y0 = int(h * cfg.roi_y_min_ratio)
    x0 = int(w * cfg.roi_x_min_ratio)
    x1 = int(w * cfg.roi_x_max_ratio)
    
    # Define ROI polygon vertices.
    roi_poly = np.array([
        [x0, y0],
        [x1, y0],
        [x1, h - 1],
        [x0, h - 1]
    ], dtype=np.int32)

    if mode == 'poly':
        return roi_poly

    elif mode == 'mask':
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [roi_poly], 255)
        return mask

    elif mode == 'visualize':
        if draw_on_img is None:
            raise ValueError("'draw_on_img' is required in 'visualize' mode.")
        
        vis_img = draw_on_img.copy()
        overlay = vis_img.copy()
        # Draw the translucent ROI fill.
        cv2.fillPoly(overlay, [roi_poly], (0, 255, 255))
        vis_img = cv2.addWeighted(overlay, 0.25, vis_img, 0.75, 0)
        # Draw the ROI border.
        cv2.polylines(vis_img, [roi_poly], True, (0, 140, 255), 2, cv2.LINE_AA)
        
        # Optionally show ROI parameters.
        if show_params:
            txt = f"Canny [{cfg.canny_low},{cfg.canny_high}], blur={cfg.gauss_ksize}/{cfg.gauss_sigma}"
            cv2.putText(vis_img, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis_img, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return vis_img
        
    else:
        raise ValueError(f"Invalid mode '{mode}'; choose 'poly', 'mask', or 'visualize'.")


def angle_bins_visual(edges: np.ndarray, angle_deg: np.ndarray, bin_deg: int, half_circle=True) -> List[Tuple[int, int, np.ndarray]]:
    H, W = edges.shape[:2]
    angle_img = angle_deg.astype(np.float32)
    angle_img[edges == 0] = np.nan

    if half_circle:
        angle_img = np.mod(angle_img, 180.0); total_range = 180
    else:
        angle_img = np.mod(angle_img, 360.0); total_range = 360

    valid = ~np.isnan(angle_img)
    nbins = int(np.ceil(total_range / bin_deg))
    bins = []
    for b in range(nbins):
        st = b * bin_deg; ed = st + bin_deg
        m = valid & (angle_img >= st) & (angle_img < ed)
        bins.append((st, ed, (m.astype(np.uint8) * 255)))
    return bins

def project_bev_poly_to_image(poly_bev_m: np.ndarray, Hpx: np.ndarray, cfg) -> np.ndarray:
    Hpx_inv = H_inv(Hpx)
    PPM = cfg.bev_w / (2 * cfg.lr_m)
    pts_bev_px = [[(cfg.bev_w // 2) + xm * PPM, int(cfg.forward_m * PPM) - ym * PPM] for (xm, ym) in poly_bev_m]
    pts = np.array(pts_bev_px, np.float32).reshape(1, -1, 2)
    pts_img = cv2.perspectiveTransform(pts, Hpx_inv).reshape(-1, 2)
    return pts_img

def draw_img_polygon(img: np.ndarray, poly: np.ndarray, color=(0, 255, 255), alpha=0.35, closed=True, thick=2):
    overlay = img.copy()
    poly_i = np.round(poly).astype(np.int32)
    cv2.fillPoly(overlay, [poly_i], color)
    out = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    cv2.polylines(out, [poly_i], closed, (0, 140, 255), thick, cv2.LINE_AA)
    return out

def draw_points_on_image(img, points, color=(0,255,0), radius=1):
    vis_img = img.copy()
    for (x, y) in points.astype(int):
        cv2.circle(vis_img, (x, y), radius, color, -1, cv2.LINE_AA)
    return vis_img

def save_all_visuals(cfg, idx, vis_lane_img, bev_overlay, bev_grid, bins, original_img, fwd_internal, fwd_disp):
    fn = lambda name: os.path.join(cfg.out_dir, f"{name}_{norm_scene_id(cfg.scene_id)}_{idx:04d}.png")
    ensure_dir(cfg.out_dir)
    
    cv2.imwrite(fn("b_match_guided_img"), vis_lane_img)
    cv2.imwrite(fn("c_bev_overlay"), bev_overlay)
    cv2.imwrite(fn("d_bev_grid"), bev_grid)
    
    bin_dir = os.path.join(cfg.out_dir, f"bins_{idx:04d}")
    ensure_dir(bin_dir)
    for (st, ed, m) in bins:
        vis = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
        cv2.putText(vis, f"bin [{st},{ed}) deg", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, f"bin [{st},{ed}) deg", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(bin_dir, f"bin_{st:03d}_{ed:03d}.jpg"), vis)

    # Read the height directly from the supplied BEV image.
    h_bev = bev_overlay.shape[0]
    r = h_bev / original_img.shape[0]
    col2 = cv2.resize(vis_lane_img, (int(original_img.shape[1] * r), h_bev))
    combo = cv2.hconcat([col2, bev_overlay, bev_grid])
    cv2.imwrite(fn("z_combo"), combo)

# ===================================================================
# ========= [NEW] Functions for Map-Centric BEV (from OLRA) =========
# ===================================================================

def _draw_map_route(type, viz_setting, image, pixel_per_meter, origin_px, origin_py, center_x, center_y):
    """
    Draw one route or sub-route.
    """
    route_info = viz_setting.get(type, {})
    points = route_info.get('points', [])
    color = tuple(route_info.get('color', [255, 0, 0]))
    thickness = int(route_info.get('thickness', 2))
    radius = int(route_info.get('radius', 3))

    if len(points) >= 2:
        pts = []
        for x, y in points:
            # Convert world coordinates relative to the view center.
            rel_x = (x - center_x) * pixel_per_meter
            rel_y = (y - center_y) * pixel_per_meter
            px = int(origin_px + rel_x)
            py = int(origin_py - rel_y)  # Invert the pixel y-axis.
            pts.append((px, py))
            if radius > 0:
                cv2.circle(image, (px, py), radius, color, -1)

        for i in range(1, len(pts)):
            cv2.line(image, pts[i-1], pts[i], color, thickness, cv2.LINE_AA)

def _draw_map_poses(viz_setting, image, pixel_per_meter, origin_px, origin_py, arrow_length_scale, center_x=0, center_y=0):
    """
    Draw a list of poses.
    """
    for pose_info in viz_setting.get('poses', []):
        x, y, heading = pose_info['pose']
        color  = tuple(pose_info.get('color', [0, 255, 255]))
        radius = int(pose_info.get('radius', 5))

        # Convert world coordinates relative to the view center.
        rel_x = (x - center_x) * pixel_per_meter
        rel_y = (y - center_y) * pixel_per_meter
        px = int(origin_px + rel_x)
        py = int(origin_py - rel_y)  # Invert the pixel y-axis.

        cv2.circle(image, (px, py), radius, color, -1, cv2.LINE_AA)
        
        if pose_info.get('heading_is_valid', 0) == 1:
            arrow_len_px = arrow_length_scale * pixel_per_meter
            
            # SSLD heading uses atan2(dx, dy), so zero points along positive y.
            dx_world_comp = np.sin(heading)
            dy_world_comp = np.cos(heading)
            
            # Pixel x follows world x, while pixel y follows negative world y.
            p_end_x = int(px + dx_world_comp * arrow_len_px)
            p_end_y = int(py - dy_world_comp * arrow_len_px)

            cv2.arrowedLine(image, (px, py), (p_end_x, p_end_y),
                            color, 2, tipLength=0.3, line_type=cv2.LINE_AA)

def draw_map_bev(viz_setting: dict) -> np.ndarray:
    """
    Draw a map-centric BEV image with non-overlapping legends.
    """
    # Read canvas dimensions and resolution.
    width = int(viz_setting.get('width', 800))
    height = int(viz_setting.get('height', 1000))
    pixel_per_meter = viz_setting.get('pixel_per_meter', 2.0) 
    arrow_length_scale = 5.0 
    
    # Place the map-centric origin at the canvas center.
    origin_px = width // 2
    origin_py = height // 2

    # Create a white canvas.
    image = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Read the view center in world coordinates.
    center_pose = viz_setting.get('center_pose_world', [0, 0])
    center_x, center_y = center_pose[0], center_pose[1]

    # Draw grid lines.
    draw_grid_lines(width, height, pixel_per_meter, origin_px, origin_py, image, center_x, center_y)

    # Draw the route.
    _draw_map_route('route', viz_setting, image, pixel_per_meter, origin_px, origin_py, center_x, center_y)

    # Draw the optional sub-route.
    _draw_map_route('sub_route', viz_setting, image, pixel_per_meter, origin_px, origin_py, center_x, center_y)

    # Draw poses.
    _draw_map_poses(viz_setting, image, pixel_per_meter, origin_px, origin_py, arrow_length_scale, center_x, center_y)

    # Chain legend calls to prevent overlap.
    last_idx = draw_legend(width, 'poses', viz_setting, image, start_index=0)
    draw_legend(width, 'routes', viz_setting, image, start_index=last_idx)

    return image

# ===================================================================
# ========= [NEW] 3D Carpet Visualization (from OLRA) ===============
# ===================================================================
def draw_route_carpet_3d(image, route_points, calib_data, width=3.5, alpha=0.4, color=(255, 100, 0)):
    """
    Render a route carpet using the legacy coordinate and extrinsic conventions.
    """
    if route_points is None or len(route_points) < 2:
        return image

    rot = calib_data.get('rot')
    trans = calib_data.get('trans')
    calib_mat = calib_data.get('mat')
    
    import torch

    if torch.is_tensor(rot): rot = rot.cpu().numpy()
    if torch.is_tensor(trans): trans = trans.cpu().numpy()
    if torch.is_tensor(calib_mat): calib_mat = calib_mat.cpu().numpy()
    
    # Build the vehicle-to-camera extrinsic matrix.
    # Translation is defined in the ego frame and rotated into the camera frame.
    camera_extrinsic = np.eye(4)
    camera_extrinsic[:3, :3] = rot
    camera_extrinsic[:3, 3] = rot @ trans
    
    # Prepare the overlay.
    cam_img = image.copy()
    overlay = cam_img.copy()
    
    # Clip the route to avoid rendering behind the camera.
    truncated_list = truncate_route(route_points.tolist(), clip_dist=1.0)
    route = np.array(truncated_list)
    if route.shape[0] < 2: return image

    # Build 3D left and right boundaries as [lateral, height, forward, 1].
    device_path = np.hstack([
        route[:, 0].reshape(-1, 1), # x
        np.zeros((len(route), 1)),
        route[:, 1].reshape(-1, 1),
        np.ones((len(route), 1))
    ])
    
    device_path_l = device_path.copy(); device_path_l[:, 0] -= width / 2
    device_path_r = device_path.copy(); device_path_r[:, 0] += width / 2

    # Transform into camera coordinates.
    t_l = (camera_extrinsic @ device_path_l.T).T
    t_r = (camera_extrinsic @ device_path_r.T).T
    
    # Retain points with positive camera depth.
    valid = (t_l[:, 2] > 0.1) & (t_r[:, 2] > 0.1)
    t_l, t_r = t_l[valid], t_r[valid]

    if len(t_l) < 2: return image

    # Apply camera intrinsics.
    p_l = calib_mat @ t_l[:, :3].T
    p_r = calib_mat @ t_r[:, :3].T
    
    # 6. Normalize
    p_l = p_l / (p_l[2:3, :] + 1e-9)
    p_r = p_r / (p_r[2:3, :] + 1e-9)
    
    # Convert to pixels.
    img_pts_l = np.column_stack((p_l[0, :], p_l[1, :])).astype(np.int32)
    img_pts_r = np.column_stack((p_r[0, :], p_r[1, :])).astype(np.int32)
    
    # Draw the carpet fill.
    for i in range(1, len(img_pts_l)):
        quad = np.array([
            img_pts_l[i-1], img_pts_r[i-1],
            img_pts_r[i],   img_pts_l[i]
        ], dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(overlay, [quad], color)
        
    # Blend the overlay.
    cv2.addWeighted(overlay, alpha, cam_img, 1 - alpha, 0, cam_img)
    
    # Draw boundary lines for debugging.
    cv2.polylines(cam_img, [img_pts_l], False, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.polylines(cam_img, [img_pts_r], False, (0, 0, 255), 1, cv2.LINE_AA)
    
    return cam_img
