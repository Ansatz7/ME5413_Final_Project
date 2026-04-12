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

        # ── TF ───────────────────────────────────────────────────────
        self.tf_listener = tf.TransformListener()

        # ── level1_done 订阅 ──────────────────────────────────────────
        self._level1_done = False
        rospy.Subscriber('/me5413/level1_done', Bool, self._cb_level1_done)

        # ── 巡逻点（map frame） ───────────────────────────────────────
        self.wp = {
            # 一楼交接点（level1_patrol 最后一个点附近）
            "leave_level_1": (7.5,  -2.0, -math.pi / 2),
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
        # ── 等待一楼完成 ──────────────────────────────────────────────
        rospy.loginfo('[auto_navigator] 等待 /me5413/level1_done ...')
        rate = rospy.Rate(5)
        while not rospy.is_shutdown() and not self._level1_done:
            rate.sleep()
        if rospy.is_shutdown():
            return

        # ── 阶段1: 前往交接点 + 开坡道 ───────────────────────────────
        # 锥桶位于 map(7.5, -2.5)，机器人从 level1_patrol 末端 ~(7.5,-1.5) 出发
        # 先到达紧邻锥桶的位置，再发 /cmd_unblock，然后立刻穿过，确保10s内通过
        rospy.loginfo('[auto_navigator] ===== 阶段1: 离开一楼 =====')
        self.send_goal(*self.wp['leave_level_1'])  # 到 (7.5,-2.0)，紧邻锥桶

        rospy.logwarn('[auto_navigator] 发布 /cmd_unblock，移除路障（10s窗口开始）...')
        self.unblock_pub.publish(Bool(data=True))
        rospy.sleep(0.5)  # 等 Gazebo 删锥桶

        rospy.loginfo('[auto_navigator] 清空 costmap 残留...')
        try:
            rospy.wait_for_service('/move_base/clear_costmaps', timeout=3.0)
            rospy.ServiceProxy('/move_base/clear_costmaps', Empty)()
            rospy.loginfo('[auto_navigator] costmap 已清空')
        except Exception as e:
            rospy.logwarn('[auto_navigator] clear_costmaps 失败: %s', e)

        # 立刻穿过锥桶原位（7.5,-2.5），用小 early_stop 确保快速通过
        self.send_goal(8.5, -3.0, 0.0, early_stop_dist=0.5)   # 穿越点：锥桶正后方
        self.send_goal(*self.wp['start_slope'])                  # 坡道起点

        # ── 阶段2: 爬坡 ──────────────────────────────────────────────
        rospy.loginfo('[auto_navigator] ===== 阶段2: 坡道 =====')
        for name in ['slope1', 'slope2', 'slope3', 'slope4', 'end_slope']:
            self.send_goal(*self.wp[name])

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
