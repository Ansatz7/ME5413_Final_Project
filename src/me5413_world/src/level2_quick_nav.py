#!/usr/bin/env python3
"""
level2_quick_nav.py — 把机器人直接传送到二楼，跳过爬坡

流程:
  1. /gazebo/set_model_state 传送到 level_2_2 (map 37.0, 7.3, yaw=π)
  2. 等 HDL 定位收敛（8s）
  3. 发布 FAKE_BOX_COUNT + /me5413/level2_start
     → auto_navigator 跳过阶段1-3，直接从阶段4（出口选择）开始

Gazebo 坐标 = map 坐标 - (22.5, 7.5)
二楼地面 z = 2.5，机器人落在 z = 2.6
"""

import json
import math
import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Bool, String

# 与 level1_quick_nav.py 保持一致
FAKE_BOX_COUNT = {"1": 1, "2": 2, "3": 3, "4": 4,
                  "5": 5, "6": 6, "7": 7, "8": 8, "9": 9}

# level_2_2: map (37.0, 7.3, yaw=π)
_MAP_X, _MAP_Y, _YAW = 37.0, 7.3, math.pi
_GZ_X  = _MAP_X - 22.5   # 14.5
_GZ_Y  = _MAP_Y -  7.5   # -0.2
_GZ_Z  = 2.6              # 二楼地面 2.5 + 小车高度余量


class Level2QuickNav:
    def __init__(self):
        rospy.init_node('level2_quick_nav')

        self.pub_box_count    = rospy.Publisher(
            '/me5413/box_count',    String,                   queue_size=1, latch=True)
        self.pub_level2_start = rospy.Publisher(
            '/me5413/level2_start', Bool,                     queue_size=1, latch=True)
        self.pub_initialpose  = rospy.Publisher(
            '/initialpose',         PoseWithCovarianceStamped, queue_size=1)

    def _teleport(self):
        rospy.loginfo('[level2_quick_nav] 等待 /gazebo/set_model_state ...')
        rospy.wait_for_service('/gazebo/set_model_state')
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        s = ModelState()
        s.model_name = 'jackal'
        s.pose.position.x = _GZ_X
        s.pose.position.y = _GZ_Y
        s.pose.position.z = _GZ_Z
        # yaw = π → quaternion (0, 0, 1, 0)
        s.pose.orientation.x = 0.0
        s.pose.orientation.y = 0.0
        s.pose.orientation.z = math.sin(_YAW / 2)
        s.pose.orientation.w = math.cos(_YAW / 2)
        s.reference_frame = 'world'

        rospy.loginfo('[level2_quick_nav] 传送到二楼 Gazebo(%.1f, %.1f, %.1f) yaw=π',
                      _GZ_X, _GZ_Y, _GZ_Z)
        set_state(s)

    def _set_initial_pose(self):
        """向 /initialpose 发布 map 坐标 (37.0, 7.3, yaw=π)，
        告知 HDL localization 在哪里重新收敛。"""
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = rospy.Time.now()
        msg.pose.pose.position.x = _MAP_X
        msg.pose.pose.position.y = _MAP_Y
        msg.pose.pose.position.z = _GZ_Z   # 二楼高度，与 Gazebo 传送 z 一致
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = math.sin(_YAW / 2)
        msg.pose.pose.orientation.w = math.cos(_YAW / 2)
        # 对角协方差（位置 0.5m²，角度 0.1rad²）
        msg.pose.covariance[0]  = 0.5
        msg.pose.covariance[7]  = 0.5
        msg.pose.covariance[35] = 0.1
        self.pub_initialpose.publish(msg)
        rospy.loginfo('[level2_quick_nav] /initialpose 已发布 map(%.1f, %.1f) yaw=π',
                      _MAP_X, _MAP_Y)

    def run(self):
        self._teleport()

        # 等 Gazebo 物理稳定，再等 HDL localization 节点 subscriber 就绪
        rospy.sleep(1.0)

        # 重复发 3 次，确保 HDL 收到（publisher 建立时可能有延迟）
        for _ in range(3):
            self._set_initial_pose()
            rospy.sleep(0.3)

        rospy.loginfo('[level2_quick_nav] 等待 HDL 定位收敛 (10s)...')
        rospy.sleep(10.0)

        # 发布模拟一楼计数（auto_navigator 快照用）
        fake_str = json.dumps(FAKE_BOX_COUNT)
        rospy.loginfo('[level2_quick_nav] 发布模拟计数 %s  目标数字: %s',
                      fake_str, min(FAKE_BOX_COUNT, key=FAKE_BOX_COUNT.get))
        self.pub_box_count.publish(String(data=fake_str))
        rospy.sleep(0.1)

        # 触发 auto_navigator 从阶段4开始
        rospy.loginfo('[level2_quick_nav] 发布 /me5413/level2_start，auto_navigator 接手')
        self.pub_level2_start.publish(Bool(data=True))

        rospy.loginfo('[level2_quick_nav] 完成，auto_navigator 应已接手')
        rospy.spin()


if __name__ == '__main__':
    try:
        Level2QuickNav().run()
    except rospy.ROSInterruptException:
        pass
