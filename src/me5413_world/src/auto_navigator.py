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

import rospy
import actionlib
import math
import tf.transformations as tf_trans
import tf
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus
from std_msgs.msg import Bool
from std_srvs.srv import Empty
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

        # ── level1_done 订阅 ──────────────────────────────────────────
        self._level1_done = False
        rospy.Subscriber('/me5413/level1_done', Bool, self._cb_level1_done)

        # ── 巡逻点（map frame） ───────────────────────────────────────
        self.wp = {
            # 一楼交接点（level1_patrol 最后一个点附近）
            "leave_level_1": (8.0,  -3.0, -math.pi / 2),
            # 坡道引导
            "start_slope":   (10.0, -4.0,  0.0),
            "slope1":        (30.0, -3.2,  0.0),
            "slope2":        (32.0, -5.2,  0.0),
            "slope3":        (34.3, -5.2,  0.0),
            "slope4":        (35.3, -3.2,  0.0),
            "end_slope":     (40.5, -3.3,  math.pi / 2),
            # 二楼
            "level_2_1":     (40.6, 16.0,  math.pi / 2),
            "level_2_2":     (37.0,  7.3,  math.pi),
            # 出口选择
            "l2_exit1":      (34.7, 12.0,  math.pi),   # 出口1
            "l2_exit2":      (34.7,  2.3,  math.pi),   # 出口2（备选）
        }

        # 走廊巡航点（x=32 列）
        self.corridor_points = [
            (32.0, 14.6, math.pi),
            (32.0,  9.5, math.pi),
            (32.0,  4.5, math.pi),
            (32.0,  0.0, math.pi),
        ]
        # 最终停靠点（x=26 列，由箱子计数结果决定，暂用最后一个）
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

    def _cb_level1_done(self, msg):
        if msg.data and not self._level1_done:
            rospy.loginfo('[auto_navigator] 收到 level1_done，准备接手！')
            self._level1_done = True

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
            except Exception:
                pass

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

        # ── 阶段1: 开坡道 + 穿锥桶 ───────────────────────────────────
        # 机器人此时在点13 (7.5,-1.5)，锥桶在 (7.5,-2.5)
        # 必须先发 /cmd_unblock 让锥桶消失，再导航通过该坐标
        rospy.loginfo('[auto_navigator] ===== 阶段1: 离开一楼 =====')
        rospy.logwarn('[auto_navigator] 发布 /cmd_unblock（10s窗口开始）...')
        self.unblock_pub.publish(Bool(data=True))
        rospy.sleep(0.5)  # 等 Gazebo 删锥桶

        rospy.loginfo('[auto_navigator] 清空 costmap 残留...')
        try:
            rospy.wait_for_service('/move_base/clear_costmaps', timeout=3.0)
            rospy.ServiceProxy('/move_base/clear_costmaps', Empty)()
            rospy.loginfo('[auto_navigator] costmap 已清空')
        except Exception as e:
            rospy.logwarn('[auto_navigator] clear_costmaps 失败: %s', e)

        # 锥桶已消失，依次通过各检查点
        self.send_goal(*self.wp['leave_level_1'])   # (8.0,-3.0) 锥桶后方
        self.send_goal(*self.wp['start_slope'])     # 坡道起点

        # ── 阶段2: 爬坡（move_base 规划） ────────────────────────────
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

        # ── 阶段3: 进入二楼 ──────────────────────────────────────────
        rospy.loginfo('[auto_navigator] ===== 阶段3: 二楼入口 =====')
        self.send_goal(*self.wp['level_2_1'])
        self.send_goal(*self.wp['level_2_2'])

        # ── 阶段4: 出口选择 ──────────────────────────────────────────
        rospy.loginfo('[auto_navigator] ===== 阶段4: 出口选择 =====')
        state = self.send_goal(*self.wp['l2_exit1'])
        if state != GoalStatus.SUCCEEDED:
            rospy.logwarn('[auto_navigator] 出口1失败，切换出口2')
            self.send_goal(*self.wp['l2_exit2'])

        # ── 阶段5: 走廊巡航 ──────────────────────────────────────────
        rospy.loginfo('[auto_navigator] ===== 阶段5: 走廊巡航 =====')
        for pt in self.corridor_points:
            self.send_goal(*pt)

        # ── 阶段6: 精准停靠（暂停靠末端，后续接箱子计数结果） ────────
        rospy.loginfo('[auto_navigator] ===== 阶段6: 精准停靠 =====')
        self.send_goal(*self.end_points[-1], early_stop_dist=0.1)

        rospy.loginfo('[auto_navigator] ===== 全程任务完成 =====')


if __name__ == '__main__':
    try:
        AutoNavigator().run()
    except rospy.ROSInterruptException:
        pass
