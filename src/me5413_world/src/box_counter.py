#!/usr/bin/env python3
"""
box_counter.py

订阅相机图像和激光扫描，用 EasyOCR 识别箱子上的数字，
利用激光雷达将检测结果定位到世界坐标，并做去重计数。

发布:
  /me5413/box_count    (std_msgs/String)      JSON格式计数结果
  /me5413/box_markers  (visualization_msgs/MarkerArray)  RViz可视化
  /me5413/debug_image  (sensor_msgs/Image)    标注图：OCR框+距离+状态

位置计算模式（~use_ground_truth 参数）：
  False（默认）: TF链 tim551→map，正式运行时用
  True          : 直接用 /gazebo/ground_truth/state，传送测试时用
"""

import math
import json

import rospy
import tf2_ros
import tf2_geometry_msgs  # 注册 PointStamped TF 变换支持，必须保留
import cv2
import numpy as np
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Bool, String
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray

import easyocr

READER = easyocr.Reader(['en'], gpu=False, verbose=False)
VALID_DIGITS = {'1', '2', '3', '4', '5', '6', '7', '8', '9'}

# 调试图颜色 (BGR)
COLOR_NEW   = (0, 220, 0)    # 绿：新箱子
COLOR_DUP   = (0, 200, 255)  # 黄：重复
COLOR_NOPOS = (0, 0, 220)    # 红：无法定位


class BoxCounter:
    def __init__(self):
        rospy.init_node('box_counter_node')

        self.laser_topic      = rospy.get_param('~laser_topic',       '/front/scan')
        self.image_topic      = rospy.get_param('~image_topic',       '/front/image_raw')
        self.map_frame        = rospy.get_param('~map_frame',         'map')
        self.dedup_dist       = rospy.get_param('~dedup_dist',        1.5)
        self.min_conf         = rospy.get_param('~min_conf',          0.5)
        self.same_frame_px    = rospy.get_param('~same_frame_px',     150)
        self.use_ground_truth = rospy.get_param('~use_ground_truth',  False)

        self.image_width = 640
        self.camera_hfov = 2 * math.atan(self.image_width / (2 * 554.254))

        self.bridge       = CvBridge()
        self.scan         = None
        self.latest_image = None
        self.gt_pose      = None
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

        rospy.Subscriber(self.laser_topic, LaserScan, self._scan_cb,  queue_size=1)
        rospy.Subscriber(self.image_topic, Image,     self._image_cb, queue_size=1,
                         buff_size=2**24)
        rospy.Subscriber('/me5413/scan_trigger', Bool, self._trigger_cb, queue_size=1)

        if self.use_ground_truth:
            rospy.Subscriber('/gazebo/ground_truth/state', Odometry,
                             self._gt_cb, queue_size=1)
            rospy.loginfo("BoxCounter 启动（地面真值模式）| HFOV=%.1f° dedup=%.1fm",
                          math.degrees(self.camera_hfov), self.dedup_dist)
        else:
            rospy.loginfo("BoxCounter 启动（TF模式, map=%s）| HFOV=%.1f° dedup=%.1fm",
                          self.map_frame, math.degrees(self.camera_hfov), self.dedup_dist)
        rospy.loginfo("等待 /me5413/scan_trigger 触发信号...")

    # ── 回调 ─────────────────────────────────────────────

    def _scan_cb(self, msg):
        self.scan = msg

    def _image_cb(self, msg):
        self.latest_image = msg

    def _gt_cb(self, msg):
        pos = msg.pose.pose.position
        q   = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.gt_pose = (pos.x, pos.y, yaw)

    # 箱子生成范围（含边界余量）；用于过滤打到墙/上层区域的误检
    BOX_X_MIN, BOX_X_MAX = -20.0,  0.0
    BOX_Y_MIN, BOX_Y_MAX = -10.0, 10.0

    def _in_box_zone(self, wx, wy):
        return (self.BOX_X_MIN <= wx <= self.BOX_X_MAX and
                self.BOX_Y_MIN <= wy <= self.BOX_Y_MAX)

    def _trigger_cb(self, msg):
        if not msg.data:
            return
        if self.scan is None or self.latest_image is None:
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

        detections = self._detect_digits(cv_img)
        # annotations: list of (bbox, digit, conf, dist, status, world_pos)
        annotations = []

        if detections:
            for (digit, cx, _cy, bbox, conf) in detections:
                dist, world_pos = self._pixel_to_world_with_dist(cx)
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

    # ── OCR ──────────────────────────────────────────────

    def _detect_digits(self, cv_img):
        """
        对多种预处理结果跑 OCR 并合并，提高召回率。
        返回 (text, cx, cy, bbox, conf) 列表。
        """
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        # 三种预处理：对比度增强、原始灰度、自适应二值化
        enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants = [enhanced, gray, binary]

        raw = []
        for img_var in variants:
            for result in READER.readtext(img_var,
                                          allowlist='123456789',
                                          min_size=15,
                                          text_threshold=self.min_conf - 0.1):
                bbox, text, conf = result
                text = text.strip()
                if text not in VALID_DIGITS:
                    continue
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                cx = int(sum(xs) / 4)
                cy = int(sum(ys) / 4)
                raw.append((text, cx, cy, bbox, conf))

        # 合并：同一数字且像素距离近的保留置信度最高的那个
        merged = []
        used = [False] * len(raw)
        for i, (d1, cx1, cy1, b1, c1) in enumerate(raw):
            if used[i]:
                continue
            best = (d1, cx1, cy1, b1, c1)
            for j, (d2, cx2, cy2, b2, c2) in enumerate(raw):
                if i == j or used[j]:
                    continue
                if d1 == d2 and math.hypot(cx1 - cx2, cy1 - cy2) < self.same_frame_px:
                    used[j] = True
                    if c2 > best[4]:
                        best = (d2, cx2, cy2, b2, c2)
            used[i] = True
            merged.append(best)
        return merged

    # ── 同帧内合并 ────────────────────────────────────────

    def _merge_intraframe(self, detections):
        """合并同帧内相同数字且像素距离近的检测，返回 (text, cx, cy, bbox, conf)。"""
        merged = []
        used   = [False] * len(detections)
        for i, (d1, cx1, cy1, bbox1, conf1) in enumerate(detections):
            if used[i]:
                continue
            group_x, group_y = [cx1], [cy1]
            best_conf, best_bbox = conf1, bbox1
            for j, (d2, cx2, cy2, bbox2, conf2) in enumerate(detections):
                if i == j or used[j]:
                    continue
                if d1 == d2 and math.hypot(cx1 - cx2, cy1 - cy2) < self.same_frame_px:
                    group_x.append(cx2)
                    group_y.append(cy2)
                    used[j] = True
                    if conf2 > best_conf:
                        best_conf, best_bbox = conf2, bbox2
            used[i] = True
            merged.append((d1,
                           int(sum(group_x) / len(group_x)),
                           int(sum(group_y) / len(group_y)),
                           best_bbox, best_conf))
        return merged

    # ── 像素 → 世界坐标 ──────────────────────────────────

    def _pixel_to_world_with_dist(self, pixel_x):
        """返回 (dist, world_pos)；world_pos 为 None 表示无法定位。"""
        scan       = self.scan
        normalized = (pixel_x - self.image_width / 2.0) / (self.image_width / 2.0)
        angle_cam  = normalized * (self.camera_hfov / 2.0)

        if not (scan.angle_min <= angle_cam <= scan.angle_max):
            return None, None

        center_idx = int((angle_cam - scan.angle_min) / scan.angle_increment)
        center_idx = max(0, min(center_idx, len(scan.ranges) - 1))

        # 在 angle_cam 附近 ±5 条激光线中找最近的有效读数
        # 避免激光线恰好从箱子边缘缝隙穿过导致 inf
        dist = None
        for offset in range(0, 6):
            for sign in ([0] if offset == 0 else [+1, -1]):
                idx = center_idx + sign * offset
                if not (0 <= idx < len(scan.ranges)):
                    continue
                d = scan.ranges[idx]
                if math.isfinite(d) and scan.range_min <= d <= scan.range_max:
                    dist = d
                    break
            if dist is not None:
                break

        if dist is None:
            return None, None

        if self.use_ground_truth:
            return dist, self._gt_to_world(dist, angle_cam)
        else:
            return dist, self._tf_to_world(dist, angle_cam, scan.header.frame_id)

    def _gt_to_world(self, dist, angle_cam):
        rx, ry, ryaw = self.gt_pose
        world_angle  = ryaw + angle_cam
        return (rx + dist * math.cos(world_angle),
                ry + dist * math.sin(world_angle))

    def _tf_to_world(self, dist, angle_cam, sensor_frame):
        p = PointStamped()
        p.header.frame_id = sensor_frame
        p.header.stamp    = rospy.Time(0)
        p.point.x = dist * math.cos(angle_cam)
        p.point.y = dist * math.sin(angle_cam)
        p.point.z = 0.0
        try:
            map_p = self.tf_buffer.transform(p, self.map_frame, rospy.Duration(0.1))
            return (map_p.point.x, map_p.point.y)
        except tf2_ros.TransformException as e:
            rospy.logdebug("TF 变换失败 (%s→%s): %s", sensor_frame, self.map_frame, e)
            return None

    # ── 去重 ─────────────────────────────────────────────

    def _already_counted(self, wx, wy):
        return any(math.hypot(wx - cx, wy - cy) < self.dedup_dist
                   for (cx, cy, _) in self.counted)

    # ── 调试图 ───────────────────────────────────────────

    def _publish_debug_image(self, cv_img, annotations):
        """在图像上绘制所有检测框并发布。"""
        if self.pub_debug.get_num_connections() == 0 and not self.save_debug:
            return

        vis = cv_img.copy()
        for (bbox, digit, conf, dist, status, world_pos) in annotations:
            if status == 'new':
                color = COLOR_NEW
                label = 'NEW'
            elif status == 'dup':
                color = COLOR_DUP
                label = 'DUP'
            else:
                color = COLOR_NOPOS
                label = 'NO_POS'

            # 画 OCR 边界框
            pts = np.array([[int(p[0]), int(p[1])] for p in bbox], dtype=np.int32)
            cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)

            # 标注文字
            x0 = int(min(p[0] for p in bbox))
            y0 = int(min(p[1] for p in bbox)) - 6
            dist_str = f'{dist:.2f}m' if dist is not None else '?m'
            if world_pos is not None:
                pos_str = f'({world_pos[0]:.1f},{world_pos[1]:.1f})'
            else:
                pos_str = ''
            line1 = f'{digit} {conf:.2f} {dist_str} [{label}]'
            line2 = pos_str

            cv2.putText(vis, line1, (x0, max(y0, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            if line2:
                cv2.putText(vis, line2, (x0, max(y0 + 14, 26)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

        # 左上角显示当前计数
        summary = 'Count: ' + json.dumps(self.counter, sort_keys=True)
        cv2.putText(vis, summary, (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(vis, summary, (5, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        try:
            self.pub_debug.publish(
                self.bridge.cv2_to_imgmsg(vis, encoding='bgr8'))
        except Exception as e:
            rospy.logwarn("调试图发布失败: %s", e)

        if self.save_debug:
            path = f'{self.debug_dir}/frame_{self._frame_idx:04d}.jpg'
            cv2.imwrite(path, vis)
            self._frame_idx += 1

    # ── 结果发布 ─────────────────────────────────────────

    def _publish_results(self):
        state = json.dumps(self.counter, sort_keys=True)
        if state == self._last_count_state:
            return
        self._last_count_state = state
        self.pub_count.publish(String(data=state))

        frame = 'world' if self.use_ground_truth else self.map_frame
        marker_array = MarkerArray()
        stamp = rospy.Time.now()
        for i, (wx, wy, digit) in enumerate(self.counted):
            m = Marker()
            m.header.frame_id    = frame
            m.header.stamp       = stamp
            m.ns                 = 'counted_boxes'
            m.id                 = i
            m.type               = Marker.TEXT_VIEW_FACING
            m.action             = Marker.ADD
            m.pose.position.x    = wx
            m.pose.position.y    = wy
            m.pose.position.z    = 1.5
            m.pose.orientation.w = 1.0
            m.scale.z            = 0.5
            m.color.g            = 1.0
            m.color.b            = 0.2
            m.color.a            = 1.0
            m.text               = digit
            m.lifetime           = rospy.Duration(0)
            marker_array.markers.append(m)
        self.pub_markers.publish(marker_array)


if __name__ == '__main__':
    try:
        BoxCounter()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
