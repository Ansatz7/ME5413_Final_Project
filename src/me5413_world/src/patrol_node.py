#!/usr/bin/env python3
"""
patrol_node.py

传送式巡航节点（用 Gazebo set_model_state 直接传送小车到各检查点）：
  1. 重生成箱子
  2. 依次传送到各检查点
  3. 每个检查点朝向房间内侧，旋转 90°（6步×15°），每步停稳后触发一次OCR
  4. 所有检查点完成后发布 /cmd_unblock

每个角的扫描起始朝向已对齐房间边墙，旋转 90° CCW 后恰好覆盖该象限全部箱子：
  西南角(yaw=  0°): 从东→北   西北角(yaw=-90°): 从南→东
  东北角(yaw=180°): 从西→南   东南角(yaw= 90°): 从北→西

等 LIO-SAM + move_base 就绪后，只需把 _teleport() 换成 move_base 目标点，
其余触发逻辑完全不变。
"""

import math
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Int16
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState

# ── 检查点 ───────────────────────────────────────────────────────────
# (x, y, yaw_deg_start, 说明)
# yaw_deg_start = 扫描起始朝向；机器人从此朝向 CCW 旋转 SCAN_TOTAL_DEG 度
# 南北外墙在 y≈±10，箱子最南/北到 y=±8，故 y=±9 距两者均有 1m 安全距离
WAYPOINTS = [
    # Room A 四角
    (-19.0, -9.0,    0, 'Room A 西南角'),   # 东→北，覆盖东北象限
    (-19.0,  9.0,  -90, 'Room A 西北角'),   # 南→东，覆盖东南象限
    (-11.0,  9.0,  180, 'Room A 东北角'),   # 西→南，覆盖西南象限
    (-11.0, -9.0,   90, 'Room A 东南角'),   # 北→西，覆盖西北象限
    # Room B 四角
    ( -9.0,  9.0,  -90, 'Room B 西北角'),   # 南→东，覆盖东南象限
    ( -9.0, -9.0,    0, 'Room B 西南角'),   # 东→北，覆盖东北象限
    ( -1.0, -9.0,   90, 'Room B 东南角'),   # 北→西，覆盖西北象限
    ( -1.0,  9.0,  180, 'Room B 东北角'),   # 西→南，覆盖西南象限
]

# ── 运动参数 ─────────────────────────────────────────────────────────
ANGULAR_SPEED  = 0.4   # rad/s 原地旋转速度
SCAN_STEP_DEG  = 9     # 每步旋转角度（°）—— 分辨率
SCAN_TOTAL_DEG = 90    # 每个检查点扫描总角度（°）
SCAN_STEPS     = SCAN_TOTAL_DEG // SCAN_STEP_DEG   # = 10 步
STOP_WAIT      = 0.5   # 旋转一步后静止等待时间（秒）
SCAN_TIMEOUT   = 5.0   # 等待 scan_done 的超时时间（秒）

# ────────────────────────────────────────────────────────────────────


class PatrolNode:
    def __init__(self):
        rospy.init_node('patrol_node')

        self.pub_cmd_vel = rospy.Publisher('/cmd_vel',                   Twist, queue_size=1)
        self.pub_trigger = rospy.Publisher('/me5413/scan_trigger',       Bool,  queue_size=1)
        self.pub_unblock = rospy.Publisher('/cmd_unblock',               Bool,  queue_size=1)
        self.pub_respawn = rospy.Publisher('/rviz_panel/respawn_objects', Int16, queue_size=1)

        self.scan_done = False
        rospy.Subscriber('/me5413/scan_done', Bool, self._scan_done_cb, queue_size=1)

        rospy.wait_for_service('/gazebo/set_model_state', timeout=10.0)
        self._set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        rospy.loginfo("PatrolNode 启动，3秒后开始...")
        rospy.sleep(3.0)

    # ── 回调 ────────────────────────────────────────────────────────

    def _scan_done_cb(self, msg):
        if msg.data:
            self.scan_done = True

    # ── 传送 ────────────────────────────────────────────────────────

    def _teleport(self, x, y, yaw_deg):
        """直接传送小车到指定位置，不依赖任何定位。"""
        yaw = math.radians(yaw_deg)
        state = ModelState()
        state.model_name        = 'jackal'
        state.reference_frame   = 'world'
        state.pose.position.x   = x
        state.pose.position.y   = y
        state.pose.position.z   = 0.1
        state.pose.orientation.z = math.sin(yaw / 2)
        state.pose.orientation.w = math.cos(yaw / 2)
        try:
            self._set_state(state)
            rospy.sleep(1.0)   # 等物理引擎稳定
        except Exception as e:
            rospy.logwarn("传送失败: %s", e)

    # ── 运动原语 ────────────────────────────────────────────────────

    def _stop(self):
        self.pub_cmd_vel.publish(Twist())
        rospy.sleep(STOP_WAIT)

    def _rotate_step(self):
        """旋转一步（SCAN_STEP_DEG°），停稳后触发OCR，等待完成。"""
        angle    = math.radians(SCAN_STEP_DEG)
        duration = angle / ANGULAR_SPEED

        msg = Twist()
        msg.angular.z = ANGULAR_SPEED
        end  = rospy.Time.now() + rospy.Duration(duration)
        rate = rospy.Rate(20)
        while rospy.Time.now() < end and not rospy.is_shutdown():
            self.pub_cmd_vel.publish(msg)
            rate.sleep()
        self._stop()

        # 触发 box_counter 识别当前帧
        self.scan_done = False
        self.pub_trigger.publish(Bool(data=True))

        # 等待 box_counter 完成
        timeout = rospy.Time.now() + rospy.Duration(SCAN_TIMEOUT)
        rate    = rospy.Rate(10)
        while not self.scan_done and rospy.Time.now() < timeout:
            rate.sleep()
        if not self.scan_done:
            rospy.logwarn("scan_done 超时，继续下一步")

    def _scan_sector(self, label=''):
        """从当前朝向 CCW 旋转 SCAN_TOTAL_DEG°，每 SCAN_STEP_DEG° 触发一次OCR。"""
        rospy.loginfo("开始扫描: %s  (%d°, 共%d步×%d°)",
                      label, SCAN_TOTAL_DEG, SCAN_STEPS, SCAN_STEP_DEG)
        for step in range(SCAN_STEPS):
            rospy.loginfo("  步骤 %d/%d", step + 1, SCAN_STEPS)
            self._rotate_step()
        rospy.loginfo("扫描完成: %s", label)

    # ── 初始化 ──────────────────────────────────────────────────────

    def _respawn_boxes(self):
        rospy.loginfo("清除并重生成箱子...")
        self.pub_respawn.publish(Int16(data=0))
        rospy.sleep(2.0)
        self.pub_respawn.publish(Int16(data=1))
        rospy.sleep(6.0)   # spawner 实际需要约 4.6s 完成，留 1.4s 余量
        rospy.loginfo("箱子重生成完成")

    # ── 主流程 ──────────────────────────────────────────────────────

    def run(self):
        self._respawn_boxes()

        for (x, y, yaw_deg, label) in WAYPOINTS:
            rospy.loginfo("传送到 %s (%.1f, %.1f, 起始朝向%d°)", label, x, y, yaw_deg)
            self._teleport(x, y, yaw_deg)
            self._scan_sector(label)

        rospy.loginfo("所有检查点完成，发布 /cmd_unblock")
        self.pub_unblock.publish(Bool(data=True))
        rospy.loginfo("任务完成！")


if __name__ == '__main__':
    try:
        PatrolNode().run()
    except rospy.ROSInterruptException:
        pass
