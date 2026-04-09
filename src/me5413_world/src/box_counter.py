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

import re
from sklearn.cluster import DBSCAN
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Bool, String
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelStates
from visualization_msgs.msg import Marker, MarkerArray

_BOX_NAME_RE = re.compile(r'^number(\d)_\d+$')

_DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), '..', 'models', 'box_detector.pt')

COLOR_NEW     = (0, 220, 0)
COLOR_DUP     = (0, 200, 255)
COLOR_NOPOS   = (0, 0, 220)
COLOR_PENDING = (0, 165, 255)   # 橙色：候选中（等待多视角确认）

# 点云水平角索引分辨率（弧度），用于快速查表
_CLOUD_BIN_RES = math.radians(0.5)   # 0.5° per bin
_CLOUD_BINS    = int(2 * math.pi / _CLOUD_BIN_RES) + 1


class BoxCounter:
    def __init__(self):
        rospy.init_node('box_counter_node')

        self.cloud_topic      = rospy.get_param('~cloud_topic',       '/mid/points')
        self.image_topic      = rospy.get_param('~image_topic',       '/front/image_raw')
        self.map_frame        = rospy.get_param('~map_frame',         'map')
        self.robot_frame      = rospy.get_param('~robot_frame',       'base_link')
        self.min_conf         = rospy.get_param('~min_conf',          0.9)
        self.use_ground_truth = rospy.get_param('~use_ground_truth',  False)
        # 去重阈值：同数字用1.2m，不同数字用0.4m
        self.dedup_same_digit = rospy.get_param('~dedup_same_digit',  1.2)
        self.dedup_any_digit  = rospy.get_param('~dedup_any_digit',   0.4)
        # bbox内有效点最少数量，不足则认为深度不可信
        self.min_bbox_pts     = rospy.get_param('~min_bbox_pts',      1)
        # 连续检测模式：手动驾驶时使用，不依赖 scan_trigger
        # continuous_hz > 0 时启用定时器，每隔 move_dist_min 米触发一次
        self.continuous_hz    = rospy.get_param('~continuous_hz',     0.0)
        self.move_dist_min    = rospy.get_param('~move_dist_min',     0.3)
        # 多观测确认：同一位置至少观测 min_obs 次且来自 min_obs_dist 米外的不同位置才计入
        self.min_obs          = rospy.get_param('~min_obs',           2)
        self.min_obs_dist     = rospy.get_param('~min_obs_dist',      0.4)
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
        self._cloud_dist  = None   # 保留兼容，不再使用
        self._cloud_pts   = None
        self._cloud_frame = 'velodyne'
        self._box_clusters = []    # 每帧更新的箱子候选cluster列表
        self._last_cloud_time = None
        self.counter      = {}
        # 已确认箱子：[{'digit':str, 'wx':float, 'wy':float,
        #               'sum_x':float, 'sum_y':float, 'n':int}]
        # 确认后继续接收观测精炼位置（不再冻结）
        self.counted      = []
        self._last_count_state = ''
        # 多观测候选池：{id: {'digit':str, 'wx':float, 'wy':float,
        #                     'obs':[(robot_x, robot_y)], 'sum_x':float, 'sum_y':float, 'count':int}}
        self._candidates  = {}
        self._cand_next_id = 0
        # 连续模式：上次触发时机器人位置
        self._last_trigger_pos = None
        # GT参考（debug用）
        self.show_gt      = rospy.get_param('~show_gt', False)
        self._gt_boxes    = []   # [(digit_str, wx, wy), ...]
        # Gazebo world → map 帧的偏移（由机器人在两帧的初始位置算出）
        self._gz_to_map_offset = None   # (dx, dy)
        self._gt_robot_gz_pos  = None   # 机器人在Gazebo world的位置（从ground_truth读取）

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.pub_count   = rospy.Publisher('/me5413/box_count',    String,      queue_size=1)
        self.pub_markers = rospy.Publisher('/me5413/box_markers',  MarkerArray, queue_size=1)
        self.pub_done    = rospy.Publisher('/me5413/scan_done',    Bool,        queue_size=1)
        self.pub_debug   = rospy.Publisher('/me5413/debug_image',  Image,       queue_size=1)
        self.pub_gt      = rospy.Publisher('/me5413/gt_box_markers', MarkerArray, queue_size=1)
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
            rospy.loginfo("BoxCounter 启动（GT模式+Velodyne）| HFOV=%.1f° same=%.1fm any=%.1fm",
                          math.degrees(self.camera_hfov), self.dedup_same_digit, self.dedup_any_digit)
        else:
            rospy.loginfo("BoxCounter 启动（TF模式+Velodyne, map=%s）| HFOV=%.1f° same=%.1fm any=%.1fm",
                          self.map_frame, math.degrees(self.camera_hfov),
                          self.dedup_same_digit, self.dedup_any_digit)

        if self.show_gt:
            rospy.Subscriber('/gazebo/model_states', ModelStates, self._gt_models_cb, queue_size=1)
            rospy.Subscriber('/gazebo/ground_truth/state', Odometry, self._gt_robot_cb, queue_size=1)
            rospy.Timer(rospy.Duration(0.5), self._publish_gt_markers)
            rospy.loginfo("GT参考模式已开启，红色marker = 箱子真实位置（自动对齐map帧）")

        if self.continuous_hz > 0:
            rospy.Timer(rospy.Duration(1.0 / self.continuous_hz), self._continuous_cb)
            rospy.loginfo("连续检测模式启动：%.1f Hz，最小移动距离 %.2f m，"
                          "最少观测 %d 次（间距 >= %.1f m）",
                          self.continuous_hz, self.move_dist_min,
                          self.min_obs, self.min_obs_dist)
        else:
            rospy.loginfo("等待 /me5413/scan_trigger 触发信号...")

    # ── 回调 ─────────────────────────────────────────────

    def _image_cb(self, msg):
        self.latest_image = msg

    def _gt_robot_cb(self, msg):
        """记录机器人在Gazebo world的位置，用于计算坐标系偏移。"""
        self._gt_robot_gz_pos = (msg.pose.pose.position.x,
                                 msg.pose.pose.position.y)
        # 一旦有了机器人GT位置，尝试从TF获取机器人在map帧的位置，计算偏移
        if self._gz_to_map_offset is None:
            map_pos = self._get_robot_pos_map()
            if map_pos is not None:
                gz_x, gz_y = self._gt_robot_gz_pos
                dx = map_pos[0] - gz_x
                dy = map_pos[1] - gz_y
                self._gz_to_map_offset = (dx, dy)
                rospy.loginfo("GT坐标系偏移已校准: dx=%.3f dy=%.3f "
                              "(map原点=Gazebo(%.2f,%.2f))", dx, dy, gz_x, gz_y)

    def _gt_models_cb(self, msg):
        boxes = []
        for name, pose in zip(msg.name, msg.pose):
            m = _BOX_NAME_RE.match(name)
            if m:
                boxes.append((m.group(1), pose.position.x, pose.position.y))
        self._gt_boxes = boxes

    def _publish_gt_markers(self, _):
        if not self._gt_boxes:
            return
        # 偏移未校准时跳过（等TF就绪）
        if self._gz_to_map_offset is None:
            return
        dx, dy = self._gz_to_map_offset
        markers = MarkerArray()
        stamp   = rospy.Time.now()
        for i, (digit, gz_x, gz_y) in enumerate(self._gt_boxes):
            wx = gz_x + dx
            wy = gz_y + dy
            m = Marker()
            m.header.frame_id = self.map_frame
            m.header.stamp    = stamp
            m.ns, m.id        = 'gt_boxes', i
            m.type            = Marker.TEXT_VIEW_FACING
            m.action          = Marker.ADD
            m.pose.position.x = wx
            m.pose.position.y = wy
            m.pose.position.z = 2.0
            m.pose.orientation.w = 1.0
            m.scale.z  = 0.6
            m.color.r, m.color.a = 1.0, 1.0   # 红色
            m.text     = digit
            m.lifetime = rospy.Duration(1.0)
            markers.markers.append(m)
            # 红色圆圈标记位置
            m2 = Marker()
            m2.header = m.header
            m2.ns, m2.id   = 'gt_circles', i
            m2.type        = Marker.CYLINDER
            m2.action      = Marker.ADD
            m2.pose.position.x = wx
            m2.pose.position.y = wy
            m2.pose.position.z = 0.4
            m2.pose.orientation.w = 1.0
            m2.scale.x = m2.scale.y = 0.8
            m2.scale.z = 0.05
            m2.color.r, m2.color.a = 1.0, 0.4
            m2.lifetime = rospy.Duration(1.0)
            markers.markers.append(m2)
        self.pub_gt.publish(markers)

    def _continuous_cb(self, _):
        """连续模式定时器：机器人移动超过 move_dist_min 才触发一次检测。"""
        robot_pos = self._get_robot_pos_map()
        if robot_pos is None:
            return
        rx, ry = robot_pos
        if self._last_trigger_pos is not None:
            dist = math.hypot(rx - self._last_trigger_pos[0],
                              ry - self._last_trigger_pos[1])
            if dist < self.move_dist_min:
                return
        self._last_trigger_pos = (rx, ry)
        self._run_detection(robot_pos)

    def _get_robot_pos_map(self):
        """从TF查询机器人在map帧中的当前位置，失败返回None。"""
        if self.use_ground_truth:
            if self.gt_pose is None:
                return None
            return (self.gt_pose[0], self.gt_pose[1])
        try:
            t = self.tf_buffer.lookup_transform(
                self.map_frame, self.robot_frame, rospy.Time(0),
                rospy.Duration(0.1))
            return (t.transform.translation.x, t.transform.translation.y)
        except tf2_ros.TransformException:
            return None

    def _gt_cb(self, msg):
        pos = msg.pose.pose.position
        q   = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.gt_pose = (pos.x, pos.y, yaw)

    def _cloud_cb(self, msg):
        """
        Velodyne点云处理：
        1. 过滤地面和天花板
        2. DBSCAN聚类 → 找到候选箱子的3D中心（velodyne帧）
        3. 按cluster尺寸过滤（保留符合箱子大小的）
        """
        self._last_cloud_time = msg.header.stamp
        self._cloud_frame = msg.header.frame_id
        try:
            pts = np.array(list(
                pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
            ), dtype=np.float32)
        except Exception:
            return
        if len(pts) == 0:
            return

        # 步骤1：高度过滤
        # 地面约在z=-0.5m（velodyne安装高度约0.5m），箱子高0.8m
        # 保留z∈(0.05, 1.2)：去掉地面、保留箱子侧面点
        mask = (pts[:, 2] > 0.05) & (pts[:, 2] < 1.2)
        pts = pts[mask]
        if len(pts) < 5:
            self._box_clusters = []
            return

        # 步骤2：只保留前方一定距离内的点（减少计算量）
        h_dists = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        pts = pts[h_dists < 15.0]
        if len(pts) < 5:
            self._box_clusters = []
            return

        # 步骤3：DBSCAN聚类（水平面xy做聚类，忽略z）
        db = DBSCAN(eps=0.25, min_samples=3).fit(pts[:, :2])
        labels = db.labels_

        clusters = []
        for label in set(labels):
            if label == -1:  # 噪点
                continue
            cluster_pts = pts[labels == label]

            # 计算cluster的水平尺寸
            x_extent = cluster_pts[:, 0].max() - cluster_pts[:, 0].min()
            y_extent = cluster_pts[:, 1].max() - cluster_pts[:, 1].min()
            max_extent = max(x_extent, y_extent)
            min_extent = min(x_extent, y_extent)

            # 箱子0.8×0.8m，允许部分遮挡，保留0.2~2.0m范围的cluster
            if max_extent < 0.2 or max_extent > 2.0:
                continue
            # 过滤极细长物体（长墙壁等）：仅当两边都有一定宽度时才卡比例
            # 正面看箱子时min_extent可能很小（<0.05m），不能严格卡
            if min_extent > 0.1 and max_extent / (min_extent + 1e-6) > 6.0:
                continue

            center = cluster_pts.mean(axis=0)  # (cx, cy, cz) in velodyne frame
            h_dist = math.sqrt(center[0]**2 + center[1]**2)
            clusters.append({'center': center, 'h_dist': h_dist,
                              'n_pts': len(cluster_pts)})

        self._box_clusters = clusters
        if clusters:
            rospy.loginfo_throttle(2.0, "[cluster] 找到 %d 个箱子候选 (共%d个过滤后点)",
                                   len(clusters), len(pts))

    # ── 触发处理 ─────────────────────────────────────────

    BOX_X_MIN, BOX_X_MAX = -30.0, 30.0
    BOX_Y_MIN, BOX_Y_MAX = -20.0, 20.0

    def _in_box_zone(self, wx, wy):
        return (self.BOX_X_MIN <= wx <= self.BOX_X_MAX and
                self.BOX_Y_MIN <= wy <= self.BOX_Y_MAX)

    def _trigger_cb(self, msg):
        """巡逻节点触发模式（兼容旧接口）：收到信号后立即检测一帧。"""
        if not msg.data:
            return
        if self.latest_image is None:
            rospy.logwarn("触发时图像未就绪，跳过")
            self.pub_done.publish(Bool(data=True))
            return
        if self.use_ground_truth and self.gt_pose is None:
            rospy.logwarn("地面真值尚未收到，跳过")
            self.pub_done.publish(Bool(data=True))
            return

        # 等待点云刷新（Velodyne 10Hz，最多等0.5s）
        wait_start = rospy.Time.now()
        rate = rospy.Rate(50)
        while (self._last_cloud_time is None or
               (rospy.Time.now() - self._last_cloud_time).to_sec() > 0.15):
            if (rospy.Time.now() - wait_start).to_sec() > 0.5:
                rospy.logwarn("等待点云超时，跳过")
                self.pub_done.publish(Bool(data=True))
                return
            rate.sleep()

        robot_pos = self._get_robot_pos_map()
        self._run_detection(robot_pos)
        self.pub_done.publish(Bool(data=True))

    def _run_detection(self, robot_pos):
        """
        核心检测逻辑：处理当前图像帧，将结果加入多观测候选池并发布调试图。
        robot_pos: (rx, ry) in map frame，连续模式下提供，可为 None（跳过多观测距离校验）。
        """
        if self.latest_image is None or self._last_cloud_time is None:
            return
        try:
            cv_img = self.bridge.imgmsg_to_cv2(self.latest_image, desired_encoding='bgr8')
        except Exception as e:
            rospy.logwarn("图像转换失败: %s", e)
            return

        detections  = self._detect_digits(cv_img)
        annotations = []
        changed     = False

        for det in detections:
            digit, bbox, conf = det[0], det[3], det[4]
            dist, world_pos = self._bbox_to_world(bbox)
            if world_pos is None:
                annotations.append((bbox, digit, conf, dist, 'nopos', None))
                continue
            wx, wy = world_pos
            if not self._in_box_zone(wx, wy):
                rospy.logwarn_throttle(2.0, "[zone外] 数字=%s 位置=(%.2f, %.2f) zone=[%.1f~%.1f, %.1f~%.1f]",
                                       digit, wx, wy,
                                       self.BOX_X_MIN, self.BOX_X_MAX,
                                       self.BOX_Y_MIN, self.BOX_Y_MAX)
                annotations.append((bbox, digit, conf, dist, 'nopos', world_pos))
                continue
            if self._already_counted(wx, wy, digit):
                annotations.append((bbox, digit, conf, dist, 'dup', world_pos))
                continue

            # 加入候选池，尝试多观测确认
            status = self._add_observation(digit, wx, wy, robot_pos)
            annotations.append((bbox, digit, conf, dist, status, world_pos))
            if status == 'new':
                changed = True

        if changed:
            self._publish_results()
        self._publish_debug_image(cv_img, annotations)

    def _add_observation(self, digit, wx, wy, robot_pos):
        """
        将一次检测加入候选池，实现多视角交叉验证。
        返回：'new'（刚达到确认条件，计入计数）
              'pending'（候选中，等待更多独立观测）
              'dup'（坐标已在confirmed列表里，属重复）
        min_obs=1 时退化为原先的单帧计数行为。
        """
        # 在候选池里查找同数字且坐标接近的候选
        matched_id = None
        for cid, cand in self._candidates.items():
            if cand['digit'] != digit:
                continue
            if math.hypot(wx - cand['wx'], wy - cand['wy']) < self.dedup_same_digit:
                matched_id = cid
                break

        if matched_id is None:
            # 首次见到这个位置的这个数字，建新候选
            cid = self._cand_next_id
            self._cand_next_id += 1
            obs = [robot_pos] if robot_pos is not None else []
            self._candidates[cid] = {
                'digit': digit,
                'wx': wx, 'wy': wy,
                'sum_x': wx, 'sum_y': wy,
                'count': 1,
                'obs': obs,
            }
            rospy.logdebug("[候选+] id=%d 数字=%s (%.2f,%.2f) 独立观测=%d/%d",
                           cid, digit, wx, wy, len(obs), self.min_obs)
            if self.min_obs <= 1:
                return self._confirm_candidate(cid)
            return 'pending'

        cand = self._candidates[matched_id]

        # 更新位置均值
        cand['count'] += 1
        cand['sum_x'] += wx
        cand['sum_y'] += wy
        cand['wx'] = cand['sum_x'] / cand['count']
        cand['wy'] = cand['sum_y'] / cand['count']

        # 仅当观测位置距已有观测足够远时才算新的独立观测
        if robot_pos is not None:
            too_close = any(
                math.hypot(robot_pos[0] - p[0], robot_pos[1] - p[1]) < self.min_obs_dist
                for p in cand['obs']
            )
            if not too_close:
                cand['obs'].append(robot_pos)

        distinct = len(cand['obs'])
        rospy.logdebug("[候选~] id=%d 数字=%s (%.2f,%.2f) 独立观测=%d/%d",
                       matched_id, digit, cand['wx'], cand['wy'], distinct, self.min_obs)

        if distinct >= self.min_obs:
            return self._confirm_candidate(matched_id)
        return 'pending'

    def _confirm_candidate(self, cid):
        """将候选升级为已确认，加入计数。"""
        cand = self._candidates.pop(cid)
        digit, wx, wy = cand['digit'], cand['wx'], cand['wy']
        self.counted.append({
            'digit': digit, 'wx': wx, 'wy': wy,
            'sum_x': cand['sum_x'], 'sum_y': cand['sum_y'], 'n': cand['count']
        })
        self.counter[digit] = self.counter.get(digit, 0) + 1
        rospy.loginfo("[新箱子✓] 数字=%s 位置=(%.2f,%.2f) 独立观测=%d  统计: %s",
                      digit, wx, wy, len(cand['obs']), self.counter)
        return 'new'

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

    # ── bbox → 世界坐标（点云聚类法） ────────────────────

    def _bbox_to_world(self, bbox):
        """
        用DBSCAN cluster中心匹配YOLO bbox：
        1. 把每个cluster中心投影到图像坐标
        2. 找投影点最接近bbox中心且在bbox内的cluster
        3. 用该cluster的3D中心做TF变换得世界坐标
        返回 (dist, (wx, wy)) 或 (None, None)。
        """
        FX = 554.254
        FY = 554.254
        CX = 320.0
        CY = 256.0

        x1 = min(p[0] for p in bbox)
        x2 = max(p[0] for p in bbox)
        y1 = min(p[1] for p in bbox)
        y2 = max(p[1] for p in bbox)
        cx_px = (x1 + x2) / 2.0
        cy_px = (y1 + y2) / 2.0

        # 扩大bbox范围（允许cluster投影点略超出框）
        margin = max((x2 - x1) * 0.3, 20)

        if not self._box_clusters:
            return None, None

        best_cluster = None
        best_dist_px = float('inf')

        for clu in self._box_clusters:
            vx, vy, vz = clu['center']
            if vx <= 0.3:   # 在机器人后方或太近
                continue

            # velodyne → 相机像素坐标
            # cam_Z=vx(前), cam_X=-vy(右=负y), cam_Y=-vz(下=负z)
            u = FX * (-vy) / vx + CX
            v = FY * (-vz) / vx + CY

            # 检查投影点是否在bbox范围内（含margin）
            if not (x1 - margin <= u <= x2 + margin):
                continue
            if not (y1 - margin <= v <= y2 + margin):
                continue

            # 找到投影点最接近bbox中心的cluster
            d_px = math.hypot(u - cx_px, v - cy_px)
            if d_px < best_dist_px:
                best_dist_px = d_px
                best_cluster = clu

        if best_cluster is None:
            return None, None

        center = best_cluster['center']
        dist = best_cluster['h_dist']

        rospy.loginfo("[cluster] 匹配 dist=%.2fm center=(%.2f,%.2f,%.2f) px_err=%.1f",
                      dist, center[0], center[1], center[2], best_dist_px)

        if self.use_ground_truth:
            if self.gt_pose is None:
                return dist, None
            rx, ry, ryaw = self.gt_pose
            # velodyne ≈ base_link，直接旋转到世界坐标
            wx = rx + center[0] * math.cos(ryaw) - center[1] * math.sin(ryaw)
            wy = ry + center[0] * math.sin(ryaw) + center[1] * math.cos(ryaw)
            return dist, (wx, wy)

        # TF模式：velodyne帧 → map帧
        p = PointStamped()
        p.header.frame_id = self._cloud_frame
        p.header.stamp    = rospy.Time(0)
        p.point.x = float(center[0])
        p.point.y = float(center[1])
        p.point.z = float(center[2])
        try:
            map_p = self.tf_buffer.transform(p, self.map_frame, rospy.Duration(0.1))
            return dist, (map_p.point.x, map_p.point.y)
        except tf2_ros.TransformException as e:
            rospy.logwarn_throttle(2.0, "TF失败: %s", e)
            return dist, None

    # ── 去重（数字+坐标联合） ────────────────────────────

    def _already_counted(self, wx, wy, digit):
        """
        检查是否已计数，同时精炼已确认箱子的位置（滑动平均）。
        返回 True 表示已计数（跳过），False 表示全新箱子。
        """
        for entry in self.counted:
            d = math.hypot(wx - entry['wx'], wy - entry['wy'])
            if d < self.dedup_any_digit:
                # 极近匹配：精炼位置
                entry['sum_x'] += wx; entry['sum_y'] += wy; entry['n'] += 1
                entry['wx'] = entry['sum_x'] / entry['n']
                entry['wy'] = entry['sum_y'] / entry['n']
                return True
            if d < self.dedup_same_digit and entry['digit'] == digit:
                # 同数字近距离匹配：精炼位置
                entry['sum_x'] += wx; entry['sum_y'] += wy; entry['n'] += 1
                entry['wx'] = entry['sum_x'] / entry['n']
                entry['wy'] = entry['sum_y'] / entry['n']
                return True
        return False

    # ── 调试图 ───────────────────────────────────────────

    def _publish_debug_image(self, cv_img, annotations):
        if self.pub_debug.get_num_connections() == 0 and not self.save_debug:
            return

        vis = cv_img.copy()
        for (bbox, digit, conf, dist, status, world_pos) in annotations:
            color = {'new': COLOR_NEW, 'dup': COLOR_DUP,
                     'pending': COLOR_PENDING, 'nopos': COLOR_NOPOS}.get(status, COLOR_NOPOS)
            label = {'new': 'NEW', 'dup': 'DUP',
                     'pending': 'PEND', 'nopos': 'NO_POS'}.get(status, status)

            pts = np.array([[int(p[0]), int(p[1])] for p in bbox], dtype=np.int32)
            cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)

            x0  = int(min(p[0] for p in bbox))
            y0  = int(min(p[1] for p in bbox)) - 6
            ds  = f'{dist:.2f}m' if dist is not None else '?m'
            ps  = f'({world_pos[0]:.1f},{world_pos[1]:.1f})' if world_pos else ''
            # pending时显示当前独立观测数/目标数
            if status == 'pending':
                obs_cnt = next((len(c['obs']) for c in self._candidates.values()
                                if c['digit'] == digit), 0)
                label = f'PEND {obs_cnt}/{self.min_obs}'
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

        frame = 'odom' if self.use_ground_truth else self.map_frame
        markers = MarkerArray()
        stamp   = rospy.Time.now()
        for i, entry in enumerate(self.counted):
            wx, wy, digit = entry['wx'], entry['wy'], entry['digit']
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
