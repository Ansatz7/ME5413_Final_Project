#!/usr/bin/env python3
"""
box_counter.py

订阅相机图像和 Velodyne 3D 点云，用 YOLOv8 识别箱子数字，
将检测结果定位到世界坐标并做去重计数。

深度来源：/mid/points（Velodyne 16线点云），按水平角索引，
覆盖范围远优于单线 2D LiDAR，彻底解决 NO_POS 问题。

发布:
  /me5413/box_count    (std_msgs/String)      JSON格式计数结果
  /me5413/box_markers  (visualization_msgs/MarkerArray)  RViz可视化
  /me5413/debug_image  (sensor_msgs/Image)    标注图：YOLO框+距离+状态

位置计算模式（~use_ground_truth 参数）：
  False（默认）: TF链 velodyne→map，正式运行时用
  True          : 直接用 /gazebo/ground_truth/state，传送测试时用
"""

import math
import json
import os

import rospy
import tf2_ros
import tf2_geometry_msgs  # 注册 PointStamped TF 变换支持，必须保留
import cv2
import numpy as np
from cv_bridge import CvBridge
from ultralytics import YOLO

import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Bool, String
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray

_DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), '..', 'models', 'box_detector.pt')

COLOR_NEW   = (0, 220, 0)
COLOR_DUP   = (0, 200, 255)
COLOR_NOPOS = (0, 0, 220)

# 点云水平角索引分辨率（弧度），用于快速查表
_CLOUD_BIN_RES = math.radians(0.5)   # 0.5° per bin
_CLOUD_BINS    = int(2 * math.pi / _CLOUD_BIN_RES) + 1


class BoxCounter:
    def __init__(self):
        rospy.init_node('box_counter_node')

        self.cloud_topic      = rospy.get_param('~cloud_topic',       '/mid/points')
        self.image_topic      = rospy.get_param('~image_topic',       '/front/image_raw')
        self.map_frame        = rospy.get_param('~map_frame',         'map')
        self.dedup_dist       = rospy.get_param('~dedup_dist',        1.5)
        self.min_conf         = rospy.get_param('~min_conf',          0.4)
        self.use_ground_truth = rospy.get_param('~use_ground_truth',  False)
        model_path = rospy.get_param('~model_path', _DEFAULT_MODEL)
        self._yolo = YOLO(model_path)
        rospy.loginfo("YOLO 模型已加载: %s", model_path)

        # 相机内参
        self.image_width = 640
        self.camera_hfov = 2 * math.atan(self.image_width / (2 * 554.254))

        self.bridge       = CvBridge()
        self.latest_image = None
        self.gt_pose      = None
        # 点云水平角→最近距离查找表，在点云回调中预计算
        self._cloud_dist  = None   # np.ndarray shape (_CLOUD_BINS,)，inf表示无数据
        self._cloud_pts   = None   # np.ndarray shape (_CLOUD_BINS, 3)，最近点的xyz（velodyne帧）
        self.counter      = {}
        self.counted      = []
        self._last_count_state = ''

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.pub_count   = rospy.Publisher('/me5413/box_count',    String,      queue_size=1)
        self.pub_markers = rospy.Publisher('/me5413/box_markers',  MarkerArray, queue_size=1)
        self.pub_done    = rospy.Publisher('/me5413/scan_done',    Bool,        queue_size=1)
        self.pub_debug   = rospy.Publisher('/me5413/debug_image',  Image,       queue_size=1)
        self.save_debug  = rospy.get_param('~save_debug', False)
        self.debug_dir   = rospy.get_param('~debug_dir',  '/tmp/me5413_debug')
        self._frame_idx  = 0
        if self.save_debug:
            import os
            os.makedirs(self.debug_dir, exist_ok=True)
            rospy.loginfo("调试图保存到: %s", self.debug_dir)

        rospy.Subscriber(self.cloud_topic, PointCloud2, self._cloud_cb,
                         queue_size=1, buff_size=2**24)
        rospy.Subscriber(self.image_topic, Image, self._image_cb,
                         queue_size=1, buff_size=2**24)
        rospy.Subscriber('/me5413/scan_trigger', Bool, self._trigger_cb, queue_size=1)

        if self.use_ground_truth:
            rospy.Subscriber('/gazebo/ground_truth/state', Odometry,
                             self._gt_cb, queue_size=1)
            rospy.loginfo("BoxCounter 启动（GT模式+Velodyne）| HFOV=%.1f° dedup=%.1fm",
                          math.degrees(self.camera_hfov), self.dedup_dist)
        else:
            rospy.loginfo("BoxCounter 启动（TF模式+Velodyne, map=%s）| HFOV=%.1f° dedup=%.1fm",
                          self.map_frame, math.degrees(self.camera_hfov), self.dedup_dist)
        rospy.loginfo("等待 /me5413/scan_trigger 触发信号...")

    # ── 回调 ─────────────────────────────────────────────

    def _image_cb(self, msg):
        self.latest_image = msg

    def _gt_cb(self, msg):
        pos = msg.pose.pose.position
        q   = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.gt_pose = (pos.x, pos.y, yaw)

    def _cloud_cb(self, msg):
        """
        将 Velodyne 点云预处理为"水平角→最近距离"查找表。
        只保留合理高度范围的点（过滤地面和天花板），提升深度精度。
        """
        try:
            pts = np.array(list(
                pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
            ), dtype=np.float32)
        except Exception:
            return
        if len(pts) == 0:
            return

        # 高度过滤：保留箱子所在高度层（velodyne 帧，z 约 -0.4~1.5m 对应地面到箱顶）
        mask = (pts[:, 2] > -0.5) & (pts[:, 2] < 1.5)
        pts  = pts[mask]
        if len(pts) == 0:
            return

        # 水平角 & 水平距离
        h_angles = np.arctan2(pts[:, 1], pts[:, 0])          # [-π, π]
        h_dists  = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)

        # 建查找表：每个角度 bin 存储最近距离及其对应点
        dist_table = np.full(_CLOUD_BINS, np.inf, dtype=np.float32)
        pts_table  = np.zeros((_CLOUD_BINS, 3), dtype=np.float32)

        bin_idx = (np.floor((h_angles + math.pi) / _CLOUD_BIN_RES)
                   .astype(np.int32).clip(0, _CLOUD_BINS - 1))

        # 遍历所有点，保留每个 bin 里最近的
        order = np.argsort(h_dists)
        for i in order:
            b = bin_idx[i]
            if h_dists[i] < dist_table[b]:
                dist_table[b] = h_dists[i]
                pts_table[b]  = pts[i]

        self._cloud_dist = dist_table
        self._cloud_pts  = pts_table
        self._cloud_frame = msg.header.frame_id

    # ── 触发处理 ─────────────────────────────────────────

    BOX_X_MIN, BOX_X_MAX = -20.0,  0.0
    BOX_Y_MIN, BOX_Y_MAX = -10.0, 10.0

    def _in_box_zone(self, wx, wy):
        return (self.BOX_X_MIN <= wx <= self.BOX_X_MAX and
                self.BOX_Y_MIN <= wy <= self.BOX_Y_MAX)

    def _trigger_cb(self, msg):
        if not msg.data:
            return
        if self._cloud_dist is None or self.latest_image is None:
            rospy.logwarn("触发时传感器数据未就绪，跳过")
            self.pub_done.publish(Bool(data=True))
            return
        if self.use_ground_truth and self.gt_pose is None:
            rospy.logwarn("地面真值尚未收到，跳过")
            self.pub_done.publish(Bool(data=True))
            return

        try:
            cv_img = self.bridge.imgmsg_to_cv2(self.latest_image, desired_encoding='bgr8')
        except Exception as e:
            rospy.logwarn("图像转换失败: %s", e)
            self.pub_done.publish(Bool(data=True))
            return

        detections  = self._detect_digits(cv_img)
        annotations = []

        if detections:
            for (digit, cx, _cy, bbox, conf) in detections:
                dist, world_pos = self._pixel_to_world(cx)
                if world_pos is None:
                    annotations.append((bbox, digit, conf, dist, 'nopos', None))
                    continue
                wx, wy = world_pos
                if not self._in_box_zone(wx, wy):
                    annotations.append((bbox, digit, conf, dist, 'nopos', None))
                    continue
                if self._already_counted(wx, wy):
                    annotations.append((bbox, digit, conf, dist, 'dup', world_pos))
                    continue
                self.counted.append((wx, wy, digit))
                self.counter[digit] = self.counter.get(digit, 0) + 1
                annotations.append((bbox, digit, conf, dist, 'new', world_pos))
                rospy.loginfo("[新箱子] 数字=%s 位置=(%.2f, %.2f)  统计: %s",
                              digit, wx, wy, self.counter)
            self._publish_results()

        self._publish_debug_image(cv_img, annotations)
        self.pub_done.publish(Bool(data=True))

    # ── YOLO 检测 ─────────────────────────────────────────

    def _detect_digits(self, cv_img):
        """
        用 YOLOv8 检测箱子，返回与原 EasyOCR 接口相同的列表：
          [(digit_str, cx_px, cy_px, bbox_polygon, conf), ...]
        bbox_polygon 为顺时针四点列表，与 _publish_debug_image 兼容。
        """
        results = self._yolo(cv_img, conf=self.min_conf, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            digit = str(cls + 1)          # class 0→'1', class 8→'9'
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            bbox = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            detections.append((digit, cx, cy, bbox, conf))
        return detections

    # ── 像素 → 世界坐标（Velodyne 点云版） ───────────────

    def _pixel_to_world(self, pixel_x):
        """
        用 Velodyne 点云查找表获取深度，取代 2D SICK 单线测距。
        返回 (dist, world_pos)；world_pos=None 表示无法定位。
        """
        normalized = (pixel_x - self.image_width / 2.0) / (self.image_width / 2.0)
        angle_cam  = normalized * (self.camera_hfov / 2.0)   # 相机水平角（rad）

        # 在查找表中搜索 angle_cam ± 10° 内最近的有效点
        SEARCH_DEG  = 10
        search_bins = int(math.radians(SEARCH_DEG) / _CLOUD_BIN_RES)
        center_bin  = int((angle_cam + math.pi) / _CLOUD_BIN_RES)
        center_bin  = max(0, min(center_bin, _CLOUD_BINS - 1))

        best_dist, best_pt = np.inf, None
        for offset in range(-search_bins, search_bins + 1):
            b = (center_bin + offset) % _CLOUD_BINS
            if self._cloud_dist[b] < best_dist:
                best_dist = self._cloud_dist[b]
                best_pt   = self._cloud_pts[b]

        if best_pt is None or not np.isfinite(best_dist):
            return None, None

        dist = float(best_dist)

        if self.use_ground_truth:
            # 直接用地面真值位姿 + angle_cam 计算世界坐标
            rx, ry, ryaw = self.gt_pose
            world_angle  = ryaw + angle_cam
            return dist, (rx + dist * math.cos(world_angle),
                          ry + dist * math.sin(world_angle))
        else:
            # 将 Velodyne 帧中的点通过 TF 变换到 map 帧
            p = PointStamped()
            p.header.frame_id = self._cloud_frame
            p.header.stamp    = rospy.Time(0)
            p.point.x, p.point.y, p.point.z = float(best_pt[0]), float(best_pt[1]), float(best_pt[2])
            try:
                map_p = self.tf_buffer.transform(p, self.map_frame, rospy.Duration(0.2))
                return dist, (map_p.point.x, map_p.point.y)
            except tf2_ros.TransformException as e:
                rospy.logdebug("TF失败: %s", e)
                return dist, None

    # ── 去重 ─────────────────────────────────────────────

    def _already_counted(self, wx, wy):
        return any(math.hypot(wx - cx, wy - cy) < self.dedup_dist
                   for (cx, cy, _) in self.counted)

    # ── 调试图 ───────────────────────────────────────────

    def _publish_debug_image(self, cv_img, annotations):
        if self.pub_debug.get_num_connections() == 0 and not self.save_debug:
            return

        vis = cv_img.copy()
        for (bbox, digit, conf, dist, status, world_pos) in annotations:
            color = COLOR_NEW if status == 'new' else (COLOR_DUP if status == 'dup' else COLOR_NOPOS)
            label = {'new': 'NEW', 'dup': 'DUP', 'nopos': 'NO_POS'}[status]

            pts = np.array([[int(p[0]), int(p[1])] for p in bbox], dtype=np.int32)
            cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)

            x0  = int(min(p[0] for p in bbox))
            y0  = int(min(p[1] for p in bbox)) - 6
            ds  = f'{dist:.2f}m' if dist is not None else '?m'
            ps  = f'({world_pos[0]:.1f},{world_pos[1]:.1f})' if world_pos else ''
            cv2.putText(vis, f'{digit} {conf:.2f} {ds} [{label}]',
                        (x0, max(y0, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            if ps:
                cv2.putText(vis, ps, (x0, max(y0+14, 26)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

        summary = 'Count: ' + json.dumps(self.counter, sort_keys=True)
        cv2.putText(vis, summary, (6, 18),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(vis, summary, (5, 17),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),       1, cv2.LINE_AA)

        try:
            self.pub_debug.publish(self.bridge.cv2_to_imgmsg(vis, encoding='bgr8'))
        except Exception as e:
            rospy.logwarn("调试图发布失败: %s", e)

        if self.save_debug:
            cv2.imwrite(f'{self.debug_dir}/frame_{self._frame_idx:04d}.jpg', vis)
            self._frame_idx += 1

    # ── 结果发布 ─────────────────────────────────────────

    def _publish_results(self):
        state = json.dumps(self.counter, sort_keys=True)
        if state == self._last_count_state:
            return
        self._last_count_state = state
        self.pub_count.publish(String(data=state))

        frame = 'world' if self.use_ground_truth else self.map_frame
        markers = MarkerArray()
        stamp   = rospy.Time.now()
        for i, (wx, wy, digit) in enumerate(self.counted):
            m = Marker()
            m.header.frame_id = frame
            m.header.stamp    = stamp
            m.ns, m.id        = 'counted_boxes', i
            m.type            = Marker.TEXT_VIEW_FACING
            m.action          = Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = wx, wy, 1.5
            m.pose.orientation.w = 1.0
            m.scale.z  = 0.5
            m.color.g, m.color.b, m.color.a = 1.0, 0.2, 1.0
            m.text     = digit
            m.lifetime = rospy.Duration(0)
            markers.markers.append(m)
        self.pub_markers.publish(markers)


if __name__ == '__main__':
    try:
        BoxCounter()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
