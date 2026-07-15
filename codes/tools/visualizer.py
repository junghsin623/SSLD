import numpy as np
import cv2

def truncate_route(points, clip_dist=5.0):
    """
    將路徑 points 截斷到 clip_dist (公尺) 開始，並在 clip_dist 處插入交點。
    points: list of [x, y]，y 為車頭前進方向距離
    clip_dist: 截斷門檻，保留 y >= clip_dist 的部分
    返回新的截斷後路徑 list of [x, y]
    """
    truncated = []
    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        # 若跨越 clip_dist，先加上一個交點
        if y0 < clip_dist and y1 >= clip_dist:
            t = (clip_dist - y0) / (y1 - y0)
            xc = x0 + (x1 - x0) * t
            yc = clip_dist
            truncated.append([xc, yc])
        # 若在 clip_dist 之後，保留點
        if y1 >= clip_dist:
            truncated.append([x1, y1])
    return truncated

def draw_legend(width, type, viz_setting, image):
    # 6. Draw legend in the top-right corner
    legend_font = cv2.FONT_HERSHEY_SIMPLEX
    legend_scale = 0.3
    legend_thickness = 1

    legend_x = width - 150   # left margin of legend area
    legend_y = 20            # starting y position
    line_height = 20

    for i, info in enumerate(viz_setting.get(type, [])):
        color = tuple(info.get('color', [0, 255, 0]))
        if type == 'poses':
            name  = info.get('name', f'Pose {i+1}')
        else:
            name  = info.get('name', f'Route {i+1}')

        # Draw color box
        box_x1, box_y1 = legend_x, legend_y + i * line_height
        box_x2, box_y2 = box_x1 + 15, box_y1 + 15
        cv2.rectangle(image, (box_x1, box_y1), (box_x2, box_y2), color, -1)

        # Draw label text
        cv2.putText(image, name, (box_x2 + 5, box_y2 - 2),
                    legend_font, legend_scale, (0, 0, 0), legend_thickness)

def draw_grid_lines(width, height, pixel_per_meter, origin_px, origin_py, image, center_x=0, center_y=0):
    grid_color = (200, 200, 200)
    font_color = (100, 100, 100)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_thickness = 1
    meter_step = 5
    pixel_step = int(pixel_per_meter * meter_step)

    # Horizontal grid lines with labels
    for y in range(0, height, pixel_step):
        cv2.line(image, (0, y), (width, y), grid_color, 1)
        world_y = (origin_py - y) / pixel_per_meter + center_y
        if abs(world_y) > 0.1:
            cv2.putText(image, f'{int(world_y)}m', (origin_px + 3, y - 3), font, font_scale, font_color, font_thickness)

    # Vertical grid lines with labels
    for x in range(0, width, pixel_step):
        cv2.line(image, (x, 0), (x, height), grid_color, 1)
        world_x = (x - origin_px) / pixel_per_meter + center_x
        if abs(world_x) > 0.1:
            cv2.putText(image, f'{int(world_x)}m', (x + 2, origin_py - 2), font, font_scale, font_color, font_thickness)

def draw_route(type, viz_setting, image, pixel_per_meter, origin_px, origin_py, center_x, center_y):
    route_info = viz_setting.get(type, {})
    points = route_info.get('points', [])
    color = tuple(route_info.get('color', [255, 0, 0]))
    thickness = int(route_info.get('thickness', 2))
    radius = int(route_info.get('radius', 3))

    if len(points) >= 2:
        pts = []
        for x, y in points:
            rel_x = (x - center_x) * pixel_per_meter
            rel_y = (y - center_y) * pixel_per_meter
            px = int(origin_px + rel_x)
            py = int(origin_py - rel_y)
            pts.append((px, py))
            cv2.circle(image, (px, py), radius, color, -1)

        for i in range(1, len(pts)):
            cv2.line(image, pts[i-1], pts[i], color, thickness)

def draw_poses(viz_setting, image, pixel_per_meter, origin_px, origin_py, arrow_length_scale, center_x=0, center_y=0, type='map'):
    for pose_info in viz_setting.get('poses', []):
        x, y, heading = pose_info['pose']
        color  = tuple(pose_info.get('color', [0, 255, 255]))
        radius = int(pose_info.get('radius', 5))

        rel_x = (x - center_x) * pixel_per_meter
        rel_y = (y - center_y) * pixel_per_meter
        px = int(origin_px + rel_x)
        py = int(origin_py - rel_y)

        cv2.circle(image, (px, py), radius, color, -1)
        if pose_info.get('heading_is_valid', 0) == 1:
            arrow_px = arrow_length_scale * pixel_per_meter
            dx = int(np.cos(heading) * arrow_px)
            dy = int(np.sin(heading) * arrow_px)
            if type == 'map':
                temp = dx
                dx = dy
                dy = temp
            cv2.arrowedLine(image, (px, py),
                            (px + dx, py - dy),
                            color, 2, tipLength=0.3)

def draw_map_bev(viz_setting):
        # Assume there are multiple poses and only one route
        # Setting format:
        # {
        #     'poses': [
        #       {'pose': <x, y, heading of 1st pose>,
        #       'color': <[r, g, b] of 1st pose>,
        #       'radius': <Radius of 1st pose>},
        #       'heading_is_valid': 1 # Point with valid heading
        #       ...
        #       {'pose': <x, y, heading of Nth pose>,
        #       'color': <[r, g, b] of Nth pose>,
        #       'radius': <R of Nth pose>},
        #       'heading_is_valid': 0 # Pure point
        #     ]
        #     'route': {
        #         'points': [<x, y of 1st point>, ... <x, y of Mth point>],
        #         'color': <[r, g, b] of route>,
        #         'radius': <Radius of route points>
        #         'thickness': <Thickness of route>
        #     }
        #     'width': <image width>
        #     'height': <image height>
        # }
    width = int(viz_setting.get('width', 800))
    height = int(viz_setting.get('height', 600))
    pixel_per_meter = viz_setting.get('pixel_per_meter', 1.0)
    arrow_length_scale = 10
    origin_px = width // 2
    origin_py = height // 2

    image = np.ones((height, width, 3), dtype=np.uint8) * 255

    # 🚩 Step 1. 設定中心點（世界座標）
    center_pose = viz_setting.get('center_pose', [0, 0])  # 預設中心在 (0,0)
    center_x, center_y = center_pose[0], center_pose[1]

    # 🚩 Step 2. 畫格線
    draw_grid_lines(width, height, pixel_per_meter, origin_px, origin_py, image, center_x, center_y)

    # Step 3. 畫 route
    draw_route('route', viz_setting, image, pixel_per_meter, origin_px, origin_py, center_x, center_y)

    # Step 4. 畫 sub_route
    draw_route('sub_route', viz_setting, image, pixel_per_meter, origin_px, origin_py, center_x, center_y)

    # Step 5. 畫 poses
    draw_poses(viz_setting, image, pixel_per_meter, origin_px, origin_py, arrow_length_scale, center_x, center_y)

    # 6. Draw legend in the top-right corner
    draw_legend(width, 'poses', viz_setting, image)

    return image

def _draw_loss_weight(image, viz_setting, debug_info, center_x=0, center_y=0):
    if not debug_info:
        return
    
    #read size and define visible distance(meters)
    height = int(viz_setting.get('height', 600))
    width = int(viz_setting.get('width', 800))
    forward_m = viz_setting.get('forward_meters', 70)
    backward_m = viz_setting.get('backward_meters', 10)

    pixel_per_meter = height / (forward_m + backward_m)
    origin_py = int(forward_m * pixel_per_meter)

    x_diffs = debug_info.get('x_diffs', [])
    weights = debug_info.get('weights', [])

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_color = (0, 0, 0)
    thickness = 1
    
    step_m = 2.0

    for i, cur_x_diff in enumerate(x_diffs):
        cur_weight = weights[i] if i < len(weights) else 0.0

        dist_m = 8 + i * step_m
        pixel_y = int(origin_py - (dist_m * pixel_per_meter))
        pixel_x = width - 250

        x_diff_str = f"{cur_x_diff:.2e}" if cur_x_diff < 0.001 and cur_x_diff > 0 else f"{cur_x_diff:.3f}"
        weight_str = f"{cur_weight:.2e}" if cur_weight < 0.01 and cur_weight > 0 else f"{cur_weight:.2f}"
        
        # if cur_loss == 0: loss_str = "0.0"
        # if cur_weight == 0: weight_str = "0.0"

        text = f"{int(dist_m)}m|X_diff:{x_diff_str} W:{weight_str}"
        cv2.putText(image, text, (pixel_x, pixel_y), font, font_scale, font_color, thickness)

    return image

def draw_vehicle_bev(viz_setting, vehicle_position=(0,0,0), debug_mode=False, debug_info=None):
    # 1. Read canvas size
    width = int(viz_setting.get('width', 800))
    height = int(viz_setting.get('height', 600))

    # 2. Define forward/backward visible distance (meters)
    backward_m = viz_setting.get('backward_meters', 10)   # 10 m behind the vehicle
    forward_m  = viz_setting.get('forward_meters',  70)   # 70 m ahead of the vehicle

    # 3. Compute pixel-per-meter scale based on canvas height
    pixel_per_meter = height / (forward_m + backward_m)
    arrow_length_scale = 10

    # 4. Compute vehicle origin in pixel coordinates
    origin_px = width  // 2
    origin_py = int(forward_m * pixel_per_meter)

    # 5. Create a white background image
    image = np.ones((height, width, 3), dtype=np.uint8) * 255

    # 6. Draw grid lines every 5 meters
    draw_grid_lines(width, height, pixel_per_meter, origin_px, origin_py, image)
    
    # # 7. Draw given poses if available
    draw_poses(viz_setting, image, pixel_per_meter, origin_px, origin_py, arrow_length_scale, type='veh')

    # 8. Draw truncated routes
    clip_dist = viz_setting.get('clip_distance', 5.0)
    routes = viz_setting.get('routes')

    for route_info in routes:
        original_points = route_info.get('points', [])
        truncated = truncate_route(original_points, clip_dist)
        if len(truncated) < 2:
            continue

        color     = tuple(route_info.get('color', [0, 255, 0]))
        thickness = int(route_info.get('thickness', 2))
        radius    = int(route_info.get('radius', 3))

        pix_pts = []
        for x, y in truncated:
            rel_x = x * pixel_per_meter
            rel_y = y * pixel_per_meter
            px = int(origin_px + rel_x)
            py = int(origin_py - rel_y)
            pix_pts.append((px, py))
            cv2.circle(image, (px, py), radius, color, -1)
        for i in range(1, len(pix_pts)):
            cv2.line(image, pix_pts[i-1], pix_pts[i], color, thickness)

    # 9. Draw legend in the top-right corner
    draw_legend(width, 'routes', viz_setting, image)

    if debug_mode and debug_info is not None:
        _draw_loss_weight(image, viz_setting, debug_info)

    return image

def draw_cam_view(image, veh_to_cam_rot, veh_to_cam_trans, calib_mat, viz_setting):
        # Assume there are multiple routes in vehicle BEV
        # Setting format: 
        # {
        #     'routes': [
        #       {'route': [<x, y of 1st point>, ... <x, y of last point of 1st route>], 
        #       'color': <[r, g, b] of 1st route>, 
        #       ...
        #       {'route': [<x, y of 1st point>, ... <x, y of last point of Nth route>], 
        #       'color': <[r, g, b] of Nth route>,
        #     ]
        #     'width': <width of vBEV carpet>,
        #     'radius': <Radius of image route point>,
        # }
        # Project vehicle BEV routes into image by camera parameters 
        # (veh_to_cam_rot, veh_to_cam_trans, calib_mat)
        
    # 計算 camera_extrinsic（如原本）
    camera_extrinsic = np.eye(4)
    camera_extrinsic[:3, :3] = veh_to_cam_rot
    camera_extrinsic[:3, 3] = veh_to_cam_rot @ veh_to_cam_trans  # 修正為矩陣相乘

    # 取回畫寬度與點半徑
    width  = viz_setting.get('width', 1)
    radius = viz_setting.get('radius', 5)
    max_distance = viz_setting.get('max_distance', 30)  # 加入最大距離設定（預設 30 公尺）
    alpha = viz_setting.get('alpha', 0.4)  # 加入透明度設定（預設 0.4）

    # 先在 cam_view 上畫車道線
    cam_img = image.copy()
    overlay = cam_img.copy()  # 建立 overlay 圖層

    for route_info in viz_setting.get('routes', []):
        route = np.array(route_info['route'])   # N x 2 世界座標
        # —— 在這裡做截斷 —— 
        clip_dist = viz_setting.get('clip_distance', 5.0)
        truncated = truncate_route(route.tolist(), clip_dist)
        route = np.array(truncated)              # 只保留從 5m 開始的點，並含交點
        if route.shape[0] < 2:
            continue   # 截斷完沒長度就跳過
        color = tuple(route_info.get('color', [0,255,0]))

        # 1. 建立左右邊界的齊次座標：
        device_path = np.hstack([
            route[:,0].reshape(-1,1), 
            np.zeros((len(route),1)), 
            route[:,1].reshape(-1,1), 
            np.ones((len(route),1))
        ])
        device_path_l = device_path.copy(); device_path_l[:,0] -= width/2
        device_path_r = device_path.copy(); device_path_r[:,0] += width/2

        # 2. 轉到相機座標系，再過濾 z>0：
        t_l = (camera_extrinsic @ device_path_l.T).T
        t_r = (camera_extrinsic @ device_path_r.T).T
        valid = (t_l[:,2]>0) & (t_r[:,2]>0)
        t_l, t_r = t_l[valid], t_r[valid]

        if len(t_l)==0 or len(t_r)==0:
            continue

        # 3. 投影到影像平面（齊次座標）
        p_l = calib_mat @ t_l[:,:3].T     # shape (3, N)
        p_r = calib_mat @ t_r[:,:3].T
        
        # 4. 安全地做 homogeneous normalization（先 clone 分母，再做非 in-place 除法）
        den_l = p_l[2:3, :].clone()      # shape (1, N)
        den_r = p_r[2:3, :].clone()
        p_l = p_l / den_l                 # shape (3, N)
        p_r = p_r / den_r
        
        # 5. 轉成整數 pixel 座標
        import torch

        # 如果 p_l, p_r 是 torch.Tensor，就先 .detach().cpu().numpy()
        if isinstance(p_l, torch.Tensor):
            p_l_np = p_l.detach().cpu().numpy()
            p_r_np = p_r.detach().cpu().numpy()
        else:
            p_l_np = p_l
            p_r_np = p_r
        
        # 四捨五入後轉 int
        x_l = p_l_np[0].round().astype(int)
        y_l = p_l_np[1].round().astype(int)
        x_r = p_r_np[0].round().astype(int)
        y_r = p_r_np[1].round().astype(int)
        
        # 組成 (x,y) tuple list
        img_pts_l = list(zip(x_l, y_l))
        img_pts_r = list(zip(x_r, y_r))
        
        # 用 fillPoly 畫「地毯」：左右邊界之間填滿色塊
        m = min(len(img_pts_l), len(img_pts_r))
        for i in range(1, m):
            quad = np.array([
                img_pts_l[i-1], img_pts_r[i-1],
                img_pts_r[i],   img_pts_l[i]
            ], dtype=np.int32).reshape(-1,1,2)

            # 在 overlay 上畫地毯
            cv2.fillPoly(overlay, [quad], color)

    # 所有地毯畫完後，一次性做透明疊合
    cv2.addWeighted(overlay, alpha, cam_img, 1 - alpha, 0, cam_img)

    return cam_img


def draw_lane(image, lanes, color=(255, 0, 0), radius=3, thickness=2):
    """
    lanes: List of lanes, each lane is a list of (x, y) tuples (in pixel coordinates)
    image: The image to draw lanes on
    color: Color of the lane (default yellow)
    radius: Radius of each point
    thickness: Thickness of connecting lines
    """
    for lane in lanes:
        if len(lane) < 2:
            continue  # 至少需要兩個點才能畫線
        # 畫點
        for (x, y) in lane:
            cv2.circle(image, (int(x), int(y)), radius, color, -1)
        # 畫線
        for i in range(1, len(lane)):
            pt1 = (int(lane[i - 1][0]), int(lane[i - 1][1]))
            pt2 = (int(lane[i][0]), int(lane[i][1]))
            cv2.line(image, pt1, pt2, color, thickness)
    return image

# Add labels with optional color
def add_label(image, text, color=(255, 0, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    position = (10, 30)
    labeled = image.copy()
    cv2.putText(labeled, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    return labeled


def combine_images(cam_view, veh_bev, map_bev):
    """
    改版需求：
    1) 為 Map BEV 與 Vehicle BEV 影像加上灰色的邊界框。
    2) Map BEV 與 Vehicle BEV 垂直堆疊，若寬度不同，以 Veh BEV 寬度為準縮放 Map BEV。
    3) 將(2)的堆疊結果縮放，使其高度與 Cam View 的高度一致，最後與 Cam View 水平並排。
    """
    if cam_view is None or veh_bev is None or map_bev is None:
        raise ValueError("❌ 有圖片讀取失敗，請確認檔案存在")

    def _add_border(img, color=(150,150,150), thickness=2):
        bordered = img.copy()
        h, w = bordered.shape[:2]
        cv2.rectangle(bordered, (0,0), (w-1, h-1), color, thickness)
        return bordered

    vb_h, vb_w = veh_bev.shape[:2]
    mb_h, mb_w = map_bev.shape[:2]

    # Map BEV 寬度對齊 veh_bev
    if mb_w != vb_w:
        scale = vb_w / float(mb_w)
        new_h = max(1, int(round(mb_h * scale)))
        map_bev_resized = cv2.resize(map_bev, (vb_w, new_h))
    else:
        map_bev_resized = map_bev

    # 加上邊框與標籤
    map_bev_labeled = add_label(map_bev_resized, "map_bev")
    veh_bev_labeled = add_label(veh_bev, "veh_bev")
    map_bev_bordered = _add_border(map_bev_labeled)
    veh_bev_bordered = _add_border(veh_bev_labeled)

    # 垂直堆疊
    bev_column = np.vstack((map_bev_bordered, veh_bev_bordered))

    # 調整高度與 cam_view 相同
    cam_h, cam_w = cam_view.shape[:2]
    col_h, col_w = bev_column.shape[:2]
    if col_h != cam_h:
        scale = cam_h / float(col_h)
        new_w = max(1, int(round(col_w * scale)))
        bev_column_resized = cv2.resize(bev_column, (new_w, cam_h))
    else:
        bev_column_resized = bev_column

    cam_labeled = add_label(cam_view, "cam_view", color=(255, 255, 0))

    # 最終左右並排
    final_image = np.hstack((cam_labeled, bev_column_resized))
    return final_image

def combine_cam_view_veh_bev(cam_view, veh_bev):
    height, width = cam_view.shape[:2]

    # Resize BEV images
    bev_resized = cv2.resize(veh_bev, (width // 2, height))

    # Add texts
    bev_labeled = add_label(bev_resized, "veh_bev")                     # 預設黃色
    cam_labeled = add_label(cam_view, "cam_view", color=(255, 255, 0))  # 淺藍黃色（青色偏黃）

    # Adjust the height to be the same as that of cam_view
    cam_h, cam_w = cam_view.shape[:2]
    col_h, col_w = bev_labeled.shape[:2]
    if col_h != cam_h:
        scale = cam_h / float(col_h)
        new_w = max(1, int(round(col_w * scale)))
        bev_labeled = cv2.resize(bev_labeled, (new_w, cam_h))


    # Combine images
    final_image = np.hstack((cam_labeled, bev_labeled))

    return final_image

def combine_custom_dashboard(cam_view, veh_bev, map_bev, cam_lanes_only):
    """
    自訂排版 Dashboard：
    - 左上：完整的 cam_view
    - 左下：兩個正方形並排 [cam_lanes_only (左), map_bev (右)]
    - 右側：佔滿全部高度的直長條 veh_bev
    """
    if cam_view is None or veh_bev is None or map_bev is None or cam_lanes_only is None:
        raise ValueError("有圖片讀取失敗或遺失，請確認輸入影像")

    # 內部小工具：加上邊框
    def _add_border(img, color=(150, 150, 150), thickness=2):
        bordered = img.copy()
        h, w = bordered.shape[:2]
        cv2.rectangle(bordered, (0, 0), (w-1, h-1), color, thickness)
        return bordered

    # 1. 取得 Map BEV 的尺寸作為基準正方形 (S x S)
    mb_h, mb_w = map_bev.shape[:2]
    S = mb_w  # 假設 map_bev 已是正方形 (例如 600x600)

    # 2. 處理左下半部的兩個正方形
    # 將「只顯示車道線的影像」強制縮放為正方形 S x S 以符合幾何排版
    cam_lanes_sq = cv2.resize(cam_lanes_only, (S, S))
    map_bev_sq = cv2.resize(map_bev, (S, S))

    # 加上文字標籤與邊框
    cam_lanes_sq = _add_border(add_label(cam_lanes_sq, "lanes_only", color=(0, 255, 0)))
    map_bev_sq = _add_border(add_label(map_bev_sq, "map_bev", color=(255, 0, 0)))

    # 水平拼接左下半部 (變成寬 2S, 高 S)
    bottom_left = np.hstack((cam_lanes_sq, map_bev_sq))

    # 3. 處理左上半部 (cam_view 主畫面)
    # 將主畫面的寬度等比例縮放至對齊底部的 2S
    cam_h, cam_w = cam_view.shape[:2]
    target_cam_w = 2 * S
    scale_cam = target_cam_w / float(cam_w)
    target_cam_h = int(cam_h * scale_cam)
    
    cam_view_resized = cv2.resize(cam_view, (target_cam_w, target_cam_h))
    cam_view_resized = _add_border(add_label(cam_view_resized, "cam_view", color=(255, 255, 0)))

    # 4. 垂直拼接形成完整的「左半部」
    left_column = np.vstack((cam_view_resized, bottom_left))

    # 5. 處理右半部 (V_BEV)
    # 將 V_BEV 的高度等比例縮放至對齊左半部的總高度
    left_total_h = left_column.shape[0]
    vb_h, vb_w = veh_bev.shape[:2]
    
    scale_veh = left_total_h / float(vb_h)
    target_veh_w = int(vb_w * scale_veh)
    
    veh_bev_resized = cv2.resize(veh_bev, (target_veh_w, left_total_h))
    veh_bev_resized = _add_border(add_label(veh_bev_resized, "veh_bev", color=(0, 255, 255)))

    # 6. 最終組合：將左半部與右半部水平拼接
    final_dashboard = np.hstack((left_column, veh_bev_resized))

    return final_dashboard
