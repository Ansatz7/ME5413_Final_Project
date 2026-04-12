#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auto_navigator.py — 二楼自主导航状态机

流程（从一楼任务完成后接手）:
  1. 等待 /me5413/level1_done 信号
  2. 前往 leave_level_1 → 发布 /cmd_unblock 开坡道
  3. 前往坡道起点 → 爬坡 → 二楼
  4. 出口选择（尝试出口1，失败切出口2）
  5. 二楼走廊巡航 → 精准停靠

坐标说明（map frame，与一楼巡逻相同坐标系）:
  出生点 → map(0, 0)，Gazebo 偏移 (+22.5, +7.5)
  坡道出口约 map x=40，二楼走廊 x∈[26, 40]

原作者: 队友 (BrandonSoong6)
改动: 加 /me5413/level1_done 等待，融入 level1_patrol 框架
"""

import json
import rospy
import actionlib
import math
import tf.transformations as tf_trans
import tf
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus
from std_msgs.msg import Bool, String
from std_srvs.srv import Empty
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray


class AutoNavigator:
    def __init__(self):
        rospy.init_node('auto_navigator')

        # ── 等待 move_base ────────────────────────────────────────────
        rospy.loginfo('[auto_navigator] 等待 move_base...')
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.client.wait_for_server()
        rospy.loginfo('[auto_navigator] move_base 已就绪')

        # ── 发布器 ────────────────────────────────────────────────────
        self.unblock_pub = rospy.Publisher(
            '/cmd_unblock', Bool, queue_size=1, latch=True)
        self.pub_markers = rospy.Publisher(
            '/auto_waypoints', MarkerArray, queue_size=1, latch=True)

        # ── TF ───────────────────────────────────────────────────────
        self.tf_listener = tf.TransformListener()

        # ── level1_done / level2_start 订阅 ──────────────────────────
        self._level1_done  = False
        self._level2_start = False   # level2_quick_nav 用：跳过阶段1-3
        rospy.Subscriber('/me5413/level1_done',   Bool, self._cb_level1_done)
        rospy.Subscriber('/me5413/level2_start',  Bool, self._cb_level2_start)

        # ── 任务一：箱子计数 ──────────────────────────────────────────
        # _box_count: 实时更新的计数（整个任务周期）
        # _level1_count: level1_done 时的快照，用于比对二楼新增检测
        # _target_digit: 一楼计数最少的数字
        self._box_count    = {}
        self._level1_count = {}
        self._target_digit = None
        rospy.Subscriber('/me5413/box_count', String, self._cb_box_count)

        # ── 任务二：前置激光雷达（出口障碍检测） ─────────────────────
        self._latest_scan = None
        rospy.Subscriber('/front/scan', LaserScan, self._cb_scan)

        # ── 停稳检测：监听里程计线速度 ───────────────────────────────
        self._linear_vel = 1.0   # 初始非零，保证未收到数据前不误判为停稳
        from nav_msgs.msg import Odometry
        rospy.Subscriber('/odometry/filtered', Odometry, self._cb_odom)

        # ── 任务一（二楼）：YOLO 原始识别结果（当前帧，无需 cluster）──
        self._yolo_raw = []   # list of digit strings seen in current frame
        rospy.Subscriber('/me5413/yolo_raw', String, self._cb_yolo_raw)

        # ── 巡逻点（map frame） ───────────────────────────────────────
        self.wp = {
            # 一楼交接点（level1_patrol 最后一个点附近）
            "leave_level_1":   ( 8.0,  -3.5, -math.pi / 2),
            # 坡道引导
            "start_slope":     (10.0,  -4.0,  0.0),
            "slope1":          (30.0,  -3.2,  0.0),
            "slope2":          (32.0,  -5.2,  0.0),
            "slope3":          (34.3,  -5.2,  0.0),
            "slope4":          (35.3,  -3.2,  0.0),
            "end_slope":       (40.5,  -3.3,  math.pi / 2),
            # 二楼入口
            "level_2_1":       (40.6,  16.0,  math.pi / 2),
            "level_2_2":       (37.0,   7.3,  math.pi),
            # 出口侦察点：朝北（π/2），出口在正左侧，用左侧激光检测锥桶
            "decision_point_1": (36.2, 12.0,  math.pi / 2),
            "decision_point_2": (36.2,  2.3,  math.pi / 2),
            # 出口通过点
            "l2_exit1":        (33.5,  12.0,  math.pi),
            "l2_exit2":        (33.5,   2.3,  math.pi),
        }

        # 走廊巡航点（x=29，靠近房间门，右侧朝西可监测动态障碍）
        self.corridor_points = [
            (29.3, 14.6, -math.pi/2),
            (29.3,  9.5, -math.pi/2),
            (29.3,  4.5, -math.pi/2),
            (29.3,  0.0, -math.pi/2),
        ]
        # 最终停靠点（x=26 列，y 由计数结果动态决定）
        self.end_points = [
            (26.0, 14.6, math.pi),
            (26.0,  9.5, math.pi),
            (26.0,  4.5, math.pi),
            (26.0,  0.0, math.pi),
        ]

    # ── 可视化 ───────────────────────────────────────────────────────

    def _publish_markers(self):
        """把所有导航点发布为 MarkerArray，在 RViz 中显示。"""
        # 收集所有点：(x, y, yaw_rad, 标签, 颜色rgb)
        entries = []
        color_map = {
            'level1': (0.2, 0.6, 1.0),   # 蓝：一楼出口区
            'slope':  (1.0, 0.6, 0.1),   # 橙：坡道
            'level2': (0.2, 0.9, 0.2),   # 绿：二楼
            'exit':   (1.0, 0.2, 0.8),   # 紫：出口选择
            'end':    (1.0, 0.2, 0.2),   # 红：终点
        }

        named = [
            ('leave_level_1', 'level1'),
            ('start_slope',   'slope'),
            ('slope1',        'slope'),
            ('slope2',        'slope'),
            ('slope3',        'slope'),
            ('slope4',        'slope'),
            ('end_slope',     'slope'),
            ('level_2_1',     'level2'),
            ('level_2_2',     'level2'),
            ('l2_exit1',      'exit'),
            ('l2_exit2',      'exit'),
        ]
        for name, ctype in named:
            x, y, yaw = self.wp[name]
            entries.append((x, y, yaw, name, color_map[ctype]))

        for i, pt in enumerate(self.corridor_points):
            entries.append((pt[0], pt[1], pt[2], f'corridor_{i}', color_map['level2']))

        for i, pt in enumerate(self.end_points):
            col = color_map['end'] if i == len(self.end_points)-1 else color_map['level2']
            entries.append((pt[0], pt[1], pt[2], f'end_{i}', col))

        array = MarkerArray()
        stamp = rospy.Time.now()
        mid = 0

        for (x, y, yaw, label, (r, g, b)) in entries:
            # 球
            s = Marker()
            s.header.frame_id = 'map'
            s.header.stamp = stamp
            s.ns, s.id = 'auto_sphere', mid
            s.type = Marker.SPHERE
            s.action = Marker.ADD
            s.pose.position.x, s.pose.position.y, s.pose.position.z = x, y, 0.3
            s.pose.orientation.w = 1.0
            s.scale.x = s.scale.y = s.scale.z = 0.35
            s.color.r, s.color.g, s.color.b, s.color.a = r, g, b, 0.85
            s.lifetime = rospy.Duration(0)
            array.markers.append(s)
            mid += 1

            # 箭头
            a = Marker()
            a.header.frame_id = 'map'
            a.header.stamp = stamp
            a.ns, a.id = 'auto_arrow', mid
            a.type = Marker.ARROW
            a.action = Marker.ADD
            a.pose.position.x, a.pose.position.y, a.pose.position.z = x, y, 0.3
            a.pose.orientation.z = math.sin(yaw / 2)
            a.pose.orientation.w = math.cos(yaw / 2)
            a.scale.x, a.scale.y, a.scale.z = 0.6, 0.08, 0.08
            a.color.r, a.color.g, a.color.b, a.color.a = 1.0, 0.9, 0.0, 0.9
            a.lifetime = rospy.Duration(0)
            array.markers.append(a)
            mid += 1

            # 文字
            t = Marker()
            t.header.frame_id = 'map'
            t.header.stamp = stamp
            t.ns, t.id = 'auto_text', mid
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x, t.pose.position.y, t.pose.position.z = x, y, 0.75
            t.pose.orientation.w = 1.0
            t.scale.z = 0.35
            t.color.r = t.color.g = t.color.b = t.color.a = 1.0
            t.text = f'{label}\n({x:.1f},{y:.1f})'
            t.lifetime = rospy.Duration(0)
            array.markers.append(t)
            mid += 1

        self.pub_markers.publish(array)
        rospy.loginfo('[auto_navigator] 导航点已发布到 /auto_waypoints')

    # ── 回调 ─────────────────────────────────────────────────────────

    def _cb_level2_start(self, msg):
        """level2_quick_nav 信号：跳过阶段1-3，直接从阶段4开始。"""
        if msg.data and not self._level2_start:
            rospy.loginfo('[auto_navigator] 收到 level2_start，快照计数并跳到阶段4')
            self._level1_count = dict(self._box_count)
            if self._level1_count:
                self._target_digit = min(
                    self._level1_count, key=lambda d: self._level1_count[d])
                rospy.loginfo('[auto_navigator] 计数快照 %s → 目标数字: %s',
                              self._level1_count, self._target_digit)
            self._level1_done  = True   # 解除 level1_done 等待循环
            self._level2_start = True

    def _cb_level1_done(self, msg):
        if msg.data and not self._level1_done:
            rospy.loginfo('[auto_navigator] 收到 level1_done，准备接手！')
            # 拍快照：此时的计数即为一楼最终结果
            self._level1_count = dict(self._box_count)
            if self._level1_count:
                self._target_digit = min(
                    self._level1_count, key=lambda d: self._level1_count[d])
                rospy.loginfo('[auto_navigator] 一楼计数快照 %s → 目标数字: %s',
                              self._level1_count, self._target_digit)
            else:
                rospy.logwarn('[auto_navigator] 一楼计数为空，将用默认停靠点')
            self._level1_done = True

    def _cb_box_count(self, msg):
        """实时更新计数（box_counter 全程发布）。"""
        try:
            self._box_count = json.loads(msg.data)
        except Exception:
            pass

    def _cb_odom(self, msg):
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self._linear_vel = math.hypot(vx, vy)

    def _wait_until_stopped(self, vel_thresh=0.05, timeout=3.0):
        """等待机器人线速度低于 vel_thresh m/s，最多等 timeout 秒。"""
        rate  = rospy.Rate(20)
        start = rospy.Time.now()
        while not rospy.is_shutdown():
            if self._linear_vel < vel_thresh:
                rospy.loginfo('[auto_navigator] 已停稳 (vel=%.3f m/s)', self._linear_vel)
                return
            if (rospy.Time.now() - start).to_sec() > timeout:
                rospy.logwarn('[auto_navigator] 停稳等待超时 (vel=%.3f m/s)，强制继续', self._linear_vel)
                return
            rate.sleep()

    def _cb_scan(self, msg):
        """任务二：缓存最新激光扫描数据。"""
        self._latest_scan = msg

    def _cb_yolo_raw(self, msg):
        """任务一（二楼）：缓存 YOLO 当前帧原始识别数字列表。"""
        try:
            self._yolo_raw = json.loads(msg.data)
        except Exception:
            pass

    # ── 激光工具函数 ─────────────────────────────────────────────────

    def _get_min_scan_range(self, angle_min_deg=-135.0, angle_max_deg=135.0):
        """在指定角度窗口内取激光最小距离。
        角度以机器人正前方为 0，左正右负（ROS 标准）。
        返回 inf 表示无有效数据。
        """
        if self._latest_scan is None:
            return float('inf')
        scan = self._latest_scan
        n = len(scan.ranges)
        a_min = max(math.radians(angle_min_deg), scan.angle_min)
        a_max = min(math.radians(angle_max_deg), scan.angle_max)
        i0 = max(0,   int((a_min - scan.angle_min) / scan.angle_increment))
        i1 = min(n-1, int((a_max - scan.angle_min) / scan.angle_increment))
        valid = [r for r in scan.ranges[i0:i1+1]
                 if not math.isinf(r) and not math.isnan(r) and r > 0.05]
        return min(valid) if valid else float('inf')

    # ── 任务二：出口障碍检测 ────────────────────────��────────────────

    def _left_ratio(self):
        """返回左侧 75~105° 扇区 3m 内的射线占比（0.0~1.0）。
        无数据返回 1.0（保守，当作墙）。
        """
        if self._latest_scan is None:
            return 1.0
        scan = self._latest_scan
        n = len(scan.ranges)
        a0 = max(math.radians(75),  scan.angle_min)
        a1 = min(math.radians(105), scan.angle_max)
        i0 = max(0,   int((a0 - scan.angle_min) / scan.angle_increment))
        i1 = min(n-1, int((a1 - scan.angle_min) / scan.angle_increment))
        valid   = [r for r in scan.ranges[i0:i1+1]
                   if not math.isinf(r) and not math.isnan(r) and r > 0.05]
        if not valid:
            return 1.0
        blocked = [r for r in valid if r < 3.0]
        return len(blocked) / len(valid)

    def _navigate_and_detect_exit(self, dest_x, dest_y, dest_yaw):
        """导航到侦察点，持续采样左侧占比，连续稳定后判断。

        判断规则（滑动窗口 5 帧，波动 < 10% 视为稳定）：
          稳定且均值 <  5%  → 畅通（返回 False）
          稳定且均值 < 80%  → 障碍（返回 True）
          均值 >= 80%       → 仍在墙壁旁，继续采样
        """
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = 'map'
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = dest_x
        goal.target_pose.pose.position.y = dest_y
        q = tf_trans.quaternion_from_euler(0, 0, dest_yaw)
        goal.target_pose.pose.orientation.x = q[0]
        goal.target_pose.pose.orientation.y = q[1]
        goal.target_pose.pose.orientation.z = q[2]
        goal.target_pose.pose.orientation.w = q[3]
        self.client.send_goal(goal)
        rospy.loginfo('[auto_navigator] 前往侦察点 (%.1f, %.1f)，持续监测左侧占比...', dest_x, dest_y)

        rate    = rospy.Rate(5)   # 5 Hz 采样
        window  = []
        WINDOW  = 5
        STABLE  = 0.10   # 窗口内 max-min < 10% → 稳定
        OPEN    = 0.05   # 稳定均值 < 5%  → 畅通
        WALL    = 0.80   # 均值 >= 80%    → 仍是墙，继续
        timeout = rospy.Time.now() + rospy.Duration(30.0)

        while not rospy.is_shutdown():
            if self.client.wait_for_result(rospy.Duration(0.1)):
                rospy.loginfo('[auto_navigator] 导航完成，就地判断')
                break
            if rospy.Time.now() > timeout:
                rospy.logwarn('[auto_navigator] 侦察超时，就地判断')
                break

            ratio = self._left_ratio()
            window.append(ratio)
            if len(window) > WINDOW:
                window.pop(0)

            rospy.loginfo('[auto_navigator] 左侧占比 %.0f%%', ratio * 100)

            if len(window) == WINDOW:
                r_mean  = sum(window) / WINDOW
                r_range = max(window) - min(window)
                if r_range < STABLE and r_mean < WALL:
                    verdict = '畅通' if r_mean < OPEN else '障碍'
                    rospy.loginfo('[auto_navigator] 稳定均值 %.0f%% (波动 %.0f%%) → %s',
                                  r_mean * 100, r_range * 100, verdict)
                    break

            rate.sleep()

        avg = sum(window) / len(window) if window else self._left_ratio()
        rospy.loginfo('[auto_navigator] 最终判定：%.0f%% → %s',
                      avg * 100, '堵塞' if avg >= OPEN else '畅通')
        return avg >= OPEN

    # ── 任务三：动态障碍等待 + 冲入房间 ────────────────────────────

    # 障碍物活动范围以外的等候 X 坐标（走廊侧，障碍在 x≈26-29 之间运动）
    _WAIT_X = 30.0

    def _wait_obstacle_min_then_enter(self, target_x, target_y, target_yaw,
                                      timeout=30.0):
        """在当前位置（走廊检查点）等待动态障碍物经过最近点后立刻冲入房间。

        小车朝南（yaw=-π/2），右侧（-120°~-60°）正对障碍物活动轴（西方）。
        状态机：
          waiting  → 右侧 min_range < 3.0m → approaching（障碍物进入监测圈）
          approaching → min_range 开始增大（刚过最近点）→ 立刻导航冲入
        """
        rate      = rospy.Rate(20)
        start     = rospy.Time.now()
        phase     = 'waiting'
        prev_r    = float('inf')
        MIN_DIST  = 3.0   # 进入监测圈阈值

        rospy.loginfo('[auto_navigator] 原地等待障碍物最近点，右侧监测中...')

        while not rospy.is_shutdown():
            if (rospy.Time.now() - start).to_sec() > timeout:
                rospy.logwarn('[auto_navigator] 等待超时，直接冲入')
                break

            # 右侧 -120°~-60°（朝南时 = 西方，障碍物活动方向）
            cur_r = self._get_min_scan_range(-120, -60)
            rospy.loginfo_throttle(1.0, '[auto_navigator] 右侧最近距离 %.2fm', cur_r)

            if phase == 'waiting':
                if cur_r < MIN_DIST:
                    phase   = 'approaching'
                    prev_r  = cur_r
                    rospy.loginfo('[auto_navigator] 障碍物进入监测圈 %.2fm', cur_r)

            elif phase == 'approaching':
                if cur_r > prev_r + 0.05:   # 距离开始增大 → 刚过最近点
                    rospy.loginfo('[auto_navigator] 障碍物已过最近点 (%.2f→%.2fm)，冲入！',
                                  prev_r, cur_r)
                    break
                prev_r = min(prev_r, cur_r)  # 持续记录最近值

            rate.sleep()

        # 冲入房间
        state = self.send_goal(target_x, target_y, target_yaw,
                               early_stop_dist=0.15, timeout=30.0)
        if state != GoalStatus.SUCCEEDED:
            rospy.logwarn('[auto_navigator] 冲入失败，等下一个周期重试...')
            self._wait_obstacle_min_then_enter(target_x, target_y, target_yaw,
                                               timeout=20.0)

    # ── 任务一：走廊扫描 + YOLO 识别目标行 ──────────────────────────

    def _scan_corridor_for_target(self):
        """依次导航到每个走廊点（x=32，朝南，右摄像头朝西对准展示箱）。
        停稳后采样 /me5413/yolo_raw 2s，检测到 target_digit 即锁定该行。
        返回对应 end_point；全程未检测到则返回 end_points[-1] 兜底。
        """
        if self._target_digit is None:
            rospy.logwarn('[auto_navigator] 无目标数字，走廊全程后用默认停靠点')
            for pt in self.corridor_points:
                self.send_goal(*pt)
            return self.end_points[-1]

        td = self._target_digit
        rospy.loginfo('[auto_navigator] 走廊扫描开始，目标数字=%s', td)

        for i, corridor_pt in enumerate(self.corridor_points):
            self.send_goal(*corridor_pt)
            rospy.sleep(0.5)  # 等车停稳

            # 在此走廊点采样 2s，看 YOLO 当前帧能否看到 target_digit
            deadline = rospy.Time.now() + rospy.Duration(2.0)
            found = False
            rate = rospy.Rate(10)
            while rospy.Time.now() < deadline and not rospy.is_shutdown():
                if td in self._yolo_raw:
                    found = True
                    break
                rate.sleep()

            rospy.loginfo('[auto_navigator] 走廊点%d：yolo_raw=%s  目标%s %s',
                          i, self._yolo_raw, td, '✓ 找到' if found else '未见')
            if found:
                return self.end_points[i]

        rospy.logwarn('[auto_navigator] 走廊全程未检测到目标数字%s，使用末端点兜底', td)
        return self.end_points[-1]

    # ── 导航到单点（实时距离监控，丝滑过弯） ─────────────────────────

    def send_goal(self, x, y, yaw, early_stop_dist=0.4, timeout=90.0):
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = 'map'
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        q = tf_trans.quaternion_from_euler(0, 0, yaw)
        goal.target_pose.pose.orientation.x = q[0]
        goal.target_pose.pose.orientation.y = q[1]
        goal.target_pose.pose.orientation.z = q[2]
        goal.target_pose.pose.orientation.w = q[3]

        rospy.loginfo('[auto_navigator] → (%.1f, %.1f, %.2f rad)', x, y, yaw)
        self.client.send_goal(goal)
        start = rospy.Time.now()

        while not rospy.is_shutdown():
            if self.client.wait_for_result(rospy.Duration(0.05)):
                return self.client.get_state()

            if (rospy.Time.now() - start).to_sec() > timeout:
                rospy.logwarn('[auto_navigator] 超时 (%.0fs)，强制取消', timeout)
                self.client.cancel_goal()
                return GoalStatus.ABORTED

            try:
                (trans, _) = self.tf_listener.lookupTransform(
                    'map', 'base_link', rospy.Time(0))
                dist = math.hypot(trans[0] - x, trans[1] - y)
                if dist < early_stop_dist:
                    rospy.loginfo('[auto_navigator] 丝滑过弯 dist=%.2fm', dist)
                    self.client.cancel_goal()
                    return GoalStatus.SUCCEEDED
            except Exception as e:
                rospy.logwarn_throttle(5.0, '[auto_navigator] TF 查询失败: %s', e)

    # ── 主流程 ────────────────────────────────────────────────────────

    def run(self):
        # ── 发布导航点可视化 ──────────────────────────────────────────
        self._publish_markers()

        # ── 等待一楼完成 ──────────────────────────────────────────────
        rospy.loginfo('[auto_navigator] 等待 /me5413/level1_done ...')
        rate = rospy.Rate(5)
        while not rospy.is_shutdown() and not self._level1_done:
            rate.sleep()
        if rospy.is_shutdown():
            return

        if self._level2_start:
            # ── 快速测试模式：跳过阶段1-3 ────────────────────────────
            rospy.loginfo('[auto_navigator] ===== 快速测试模式：跳过阶段1-3 =====')
            rospy.loginfo('[auto_navigator] 机器人已传送到二楼，直接从阶段4开始')
            try:
                rospy.wait_for_service('/move_base/clear_costmaps', timeout=3.0)
                rospy.ServiceProxy('/move_base/clear_costmaps', Empty)()
            except Exception:
                pass
        else:
            # ── 阶段1: 开坡道 + 穿锥桶 ──────────────────────────────
            # 机器人此时在点13 (7.5,-1.5)，锥桶在 (7.5,-2.5)
            # 必须先发 /cmd_unblock 让锥桶消失，再导航通过该坐标
            rospy.loginfo('[auto_navigator] ===== 阶段1: 离开一楼 =====')
            rospy.logwarn('[auto_navigator] 发布 /cmd_unblock（10s窗口开始）...')
            self.unblock_pub.publish(Bool(data=True))
            rospy.sleep(0.3)  # 等 Gazebo 删锥桶

            rospy.loginfo('[auto_navigator] 清空 costmap 残留...')
            try:
                rospy.wait_for_service('/move_base/clear_costmaps', timeout=3.0)
                rospy.ServiceProxy('/move_base/clear_costmaps', Empty)()
                rospy.loginfo('[auto_navigator] costmap 已清空')
            except Exception as e:
                rospy.logwarn('[auto_navigator] clear_costmaps 失败: %s', e)

            # 锥桶已消失，依次通过各检查点
            self.send_goal(*self.wp['leave_level_1'])   # (8.0,-3.5) 锥桶后方
            self.send_goal(*self.wp['start_slope'])     # 坡道起点

            # ── 阶段2: 爬坡（move_base 规划） ────────────────────────
            rospy.loginfo('[auto_navigator] ===== 阶段2: 爬坡 =====')
            try:
                rospy.wait_for_service('/move_base/clear_costmaps', timeout=3.0)
                rospy.ServiceProxy('/move_base/clear_costmaps', Empty)()
                rospy.loginfo('[auto_navigator] 爬坡前清空 costmap')
            except Exception as e:
                rospy.logwarn('[auto_navigator] clear_costmaps 失败: %s', e)
            self.send_goal(*self.wp['slope1'],  early_stop_dist=1.0, timeout=120.0)
            self.send_goal(*self.wp['slope2'],  early_stop_dist=1.0, timeout=60.0)
            self.send_goal(*self.wp['slope3'],  early_stop_dist=1.0, timeout=60.0)
            self.send_goal(*self.wp['slope4'],  early_stop_dist=1.0, timeout=60.0)
            self.send_goal(*self.wp['end_slope'], early_stop_dist=0.5, timeout=60.0)

            # ── 阶段3: 进入二楼 ──────────────────────────────────────
            rospy.loginfo('[auto_navigator] ===== 阶段3: 二楼入口 =====')
            self.send_goal(*self.wp['level_2_1'])
            self.send_goal(*self.wp['level_2_2'])

        # ── 阶段4: 出口选择（激光侦察） ──────────────────────────────
        rospy.loginfo('[auto_navigator] ===== 阶段4: 出口选择 =====')
        # 先停在侦察点，用激光检测正前方是否有锥桶
        if self._navigate_and_detect_exit(*self.wp['decision_point_1']):
            rospy.logwarn('[auto_navigator] 出口1有障碍，切换出口2')
            self.send_goal(*self.wp['decision_point_2'], early_stop_dist=0.30)
            self.send_goal(*self.wp['l2_exit2'])
        else:
            rospy.loginfo('[auto_navigator] 出口1畅通，直接通过')
            self.send_goal(*self.wp['l2_exit1'])

        # ── 阶段5: 走廊扫描（YOLO识别目标行） ───────────────────────
        rospy.loginfo('[auto_navigator] ===== 阶段5: 走廊扫描 =====')
        target = self._scan_corridor_for_target()
        rospy.loginfo('[auto_navigator] 目标停靠点: (%.1f, %.1f)', target[0], target[1])

        # ── 阶段6: 等候障碍时机 + 冲入房间 ─────────────────────────
        rospy.loginfo('[auto_navigator] ===== 阶段6: 等候时机 + 冲入目标房间 =====')
        self._wait_obstacle_min_then_enter(*target)

        rospy.loginfo('[auto_navigator] ===== 全程任务完成 =====')


if __name__ == '__main__':
    try:
        AutoNavigator().run()
    except rospy.ROSInterruptException:
        pass
