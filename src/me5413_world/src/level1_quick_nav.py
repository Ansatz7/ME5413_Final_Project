#!/usr/bin/env python3
"""
level1_quick_nav.py — 快速跑到交接点，触发二楼测试

流程:
  1. 重生成箱子（可选）
  2. 两点导航：入口 → leave_level_1（point 13）
  3. 到达后发布 /cmd_unblock（开坡道 10s）
  4. 发布 /me5413/level1_done = true

用途：隔离测试 auto_navigator.py 二楼流程，跳过完整一楼巡逻
"""

import math
import rospy
import actionlib
import tf2_ros
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Quaternion
from std_msgs.msg import Bool, Int16

ARRIVAL_DIST  = 0.4   # m，距目标此距离内视为到达
GOAL_TIMEOUT  = 90.0  # 单点超时
RESPAWN_BOXES = True  # 是否重生成箱子


def yaw_to_quat(yaw_deg):
    y = math.radians(yaw_deg)
    return Quaternion(x=0, y=0, z=math.sin(y/2), w=math.cos(y/2))


def make_goal(x, y, yaw_deg):
    g = MoveBaseGoal()
    g.target_pose.header.frame_id = 'map'
    g.target_pose.header.stamp = rospy.Time.now()
    g.target_pose.pose.position.x = x
    g.target_pose.pose.position.y = y
    g.target_pose.pose.orientation = yaw_to_quat(yaw_deg)
    return g


class QuickNav:
    def __init__(self):
        rospy.init_node('level1_quick_nav')

        self.pub_respawn = rospy.Publisher(
            '/rviz_panel/respawn_objects', Int16, queue_size=1, latch=True)
        self.pub_done = rospy.Publisher(
            '/me5413/level1_done', Bool, queue_size=1, latch=True)

        self._tf_buf = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buf)

        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo('[quick_nav] 等待 move_base...')
        self.client.wait_for_server()
        rospy.loginfo('[quick_nav] move_base 就绪')

    def _robot_xy(self):
        try:
            t = self._tf_buf.lookup_transform(
                'map', 'base_link', rospy.Time(0), rospy.Duration(0.1))
            return t.transform.translation.x, t.transform.translation.y
        except Exception:
            return None

    def _go(self, x, y, yaw_deg, label):
        rospy.loginfo('[quick_nav] → %s (%.1f, %.1f)', label, x, y)
        self.client.send_goal(make_goal(x, y, yaw_deg))
        rate = rospy.Rate(5)
        start = rospy.Time.now()
        while not rospy.is_shutdown():
            state = self.client.get_state()
            if state == actionlib.GoalStatus.SUCCEEDED:
                rospy.loginfo('[quick_nav] 到达 %s ✓ (SUCCEEDED)', label)
                return True
            if state in (actionlib.GoalStatus.ABORTED,
                         actionlib.GoalStatus.REJECTED,
                         actionlib.GoalStatus.PREEMPTED):
                rospy.logwarn('[quick_nav] 导航失败 state=%d, 跳过 %s', state, label)
                return False
            pos = self._robot_xy()
            if pos and math.hypot(pos[0]-x, pos[1]-y) < ARRIVAL_DIST:
                rospy.loginfo('[quick_nav] 到达 %s ✓ (dist)', label)
                self.client.cancel_goal()
                rospy.sleep(0.3)
                return True
            if (rospy.Time.now() - start).to_sec() > GOAL_TIMEOUT:
                rospy.logwarn('[quick_nav] 超时！跳过 %s', label)
                self.client.cancel_goal()
                rospy.sleep(0.5)
                return False
            rate.sleep()

    def run(self):
        if RESPAWN_BOXES:
            # 重生成箱子（8s）与定位收敛（8s）并行，节省等待时间
            rospy.loginfo('[quick_nav] 重生成箱子 + 等待定位收敛（并行）...')
            self.pub_respawn.publish(Int16(data=0))
            rospy.sleep(2.0)
            self.pub_respawn.publish(Int16(data=1))
            rospy.sleep(6.0)
            rospy.loginfo('[quick_nav] 箱子完成，定位已收敛')
        else:
            rospy.loginfo('[quick_nav] 等待定位收敛 (8s)...')
            rospy.sleep(8.0)

        # 到交接点
        # self._go(3.5, 1.0,  90, '入口')
        self._go(7.5, -1.5, -90, 'leave_level_1')

        # 只发 level1_done，/cmd_unblock 由 auto_navigator 负责
        # （auto_navigator 会在紧邻锥桶时再发，确保10s内通过）
        rospy.loginfo('[quick_nav] 发布 /me5413/level1_done')
        self.pub_done.publish(Bool(data=True))

        rospy.loginfo('[quick_nav] 完成，auto_navigator 应已接手')
        rospy.spin()


if __name__ == '__main__':
    try:
        QuickNav().run()
    except rospy.ROSInterruptException:
        pass
