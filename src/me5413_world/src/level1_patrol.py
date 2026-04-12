#!/usr/bin/env python3
"""
level1_patrol.py — 一楼自主巡逻节点

流程：
  1. （可选）重生成箱子
  2. 依次向 move_base 发送巡逻点
  3. 箱子识别由 box_counter_node 在后台自动完成（无需手动触发）
  4. 所有点完成后停在 leave_level_1，发布 /me5413/level1_done 信号

坐标系说明（基于源码分析 + 门洞几何验证）：
  Gazebo → map 偏移约 (+22.5, +7.5)
  出生点:  Gazebo(-22.5,-7.5)  →  map(0, 0)
  入口门洞: map x=2.5, y∈[-0.82, 0.68]，宽 1.5m
  房间 A:  map x∈[4.5, 10.5], y∈[-0.5, 15.5]
  走廊隔墙: map x∈[10.5, 14.5]（连通口1下/连通口2上）
  房间 B:  map x∈[14.5, 20.5], y∈[-0.5, 15.5]

可视化：节点启动后在 RViz 中添加 Topic /patrol_waypoints (MarkerArray) 即可看到所有巡逻点。
验证坐标：在 RViz 工具栏选「Publish Point」(十字准星)，点击地图上的墙角，终端会打印出 map frame 坐标。
"""

import math
import rospy
import actionlib
import tf2_ros
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Bool, Int16


# ── 巡逻点（ROS map frame，单位 m）──────────────────────────────────
# (x, y, yaw_deg, 说明)
# yaw_deg: 到达后朝向（0=东, 90=北, 180=西, -90=南）

PATROL_WAYPOINTS = [
    # ── #1  入口穿越（门洞 y∈[-0.82,0.68]，宽1.5m）─────────────────
    ( 3.5,  1.0,  90, '0 入口'),          # 出生口

    # ── #3~#5  房间 A ────────────────────────────────────────────────
    ( 4.0, 16.0,   0, '1 A-左上'),          # A左上
    (11.5, 15.5, -90, '2 A-右上'),          # A右上
    (11.5,  7.5, -90, '3 A-右中'),          # A右中

    (13.0,  2.0,  45, '4 通道中'),          # A进B
    (14.0, 16.0,   0, '5 B-左上'),          # B左上
    (21.0, 15.5, -90, '6 B-右上'),          # B右上


    # ── #10~#13  房间 B 下半 → 穿越下连通口 ─────────────────────────
    (21.0,  1.0, -90, '7 B-右墙下（转角前）'), # B右下前
    (19.0, -1.0, 180, '8 B-右墙下（转角后）'), # B右下后
    (13.5, -0.5,  90, '9 B-左下'),           # B左下
    (13.5,  7.5,  90, '10 A-右中'),          # B左中
    (12.0, 13.0,-135, '11 通道中'),          # B进A
    (11.0, -1.0, 180, '12 A-右下'),          # A右下

    # ── #14  交接点 ──────────────────────────────────────────────────
    ( 7.5, -1.8, -90, '13 leave_level_1'),  # 交接点


]


# ── 参数 ─────────────────────────────────────────────────────────────
GOAL_TIMEOUT   = 60.0   # 单点导航超时（秒）
ARRIVAL_DIST   = 0.4    # 距目标 xy 小于此值（m）
ARRIVAL_YAW    = 2.0    # 朝向误差小于此值（rad，约114°）——两者同时满足才视为到达
ARRIVE_PAUSE   = 0.0    # 到达即走，无停顿（box_counter 行进中持续采样）
RESPAWN_BOXES  = True   # 启动时是否重生成箱子


def yaw_to_quat(yaw_deg):
    yaw = math.radians(yaw_deg)
    return Quaternion(
        x=0.0,
        y=0.0,
        z=math.sin(yaw / 2.0),
        w=math.cos(yaw / 2.0),
    )


def make_goal(x, y, yaw_deg):
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = 'map'
    goal.target_pose.header.stamp    = rospy.Time.now()
    goal.target_pose.pose.position.x  = x
    goal.target_pose.pose.position.y  = y
    goal.target_pose.pose.position.z  = 0.0
    goal.target_pose.pose.orientation = yaw_to_quat(yaw_deg)
    return goal


def build_waypoint_markers():
    """把 PATROL_WAYPOINTS 打包成 MarkerArray，发布到 /patrol_waypoints。"""
    array = MarkerArray()
    stamp = rospy.Time.now()

    for i, (x, y, yaw_deg, label) in enumerate(PATROL_WAYPOINTS):
        yaw = math.radians(yaw_deg)

        # ── 圆球：巡逻点位置 ─────────────────────────────────────────
        sphere = Marker()
        sphere.header.frame_id = 'map'
        sphere.header.stamp    = stamp
        sphere.ns              = 'patrol_sphere'
        sphere.id              = i
        sphere.type            = Marker.SPHERE
        sphere.action          = Marker.ADD
        sphere.pose.position.x = x
        sphere.pose.position.y = y
        sphere.pose.position.z = 0.3
        sphere.pose.orientation.w = 1.0
        sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.3
        # 颜色：绿色→普通点，蓝色→特殊点（入口/走廊），红色→leave_level_1
        if label == 'leave_level_1':
            sphere.color.r, sphere.color.g, sphere.color.b = 1.0, 0.2, 0.2
        elif '穿越' in label or '入口' in label:
            sphere.color.r, sphere.color.g, sphere.color.b = 0.2, 0.5, 1.0
        else:
            sphere.color.r, sphere.color.g, sphere.color.b = 0.2, 0.9, 0.2
        sphere.color.a = 0.85
        sphere.lifetime = rospy.Duration(0)
        array.markers.append(sphere)

        # ── 箭头：到达后朝向 ─────────────────────────────────────────
        arrow = Marker()
        arrow.header.frame_id = 'map'
        arrow.header.stamp    = stamp
        arrow.ns              = 'patrol_arrow'
        arrow.id              = i
        arrow.type            = Marker.ARROW
        arrow.action          = Marker.ADD
        arrow.pose.position.x = x
        arrow.pose.position.y = y
        arrow.pose.position.z = 0.3
        arrow.pose.orientation.z = math.sin(yaw / 2.0)
        arrow.pose.orientation.w = math.cos(yaw / 2.0)
        arrow.scale.x = 0.5   # 箭头长度
        arrow.scale.y = 0.08
        arrow.scale.z = 0.08
        arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = 1.0, 0.8, 0.0, 0.9
        arrow.lifetime = rospy.Duration(0)
        array.markers.append(arrow)

        # ── 文字标签：序号 + 名称 ────────────────────────────────────
        text = Marker()
        text.header.frame_id = 'map'
        text.header.stamp    = stamp
        text.ns              = 'patrol_text'
        text.id              = i
        text.type            = Marker.TEXT_VIEW_FACING
        text.action          = Marker.ADD
        text.pose.position.x = x
        text.pose.position.y = y
        text.pose.position.z = 0.7
        text.pose.orientation.w = 1.0
        text.scale.z = 0.35
        text.color.r = text.color.g = text.color.b = 1.0
        text.color.a = 1.0
        text.text = f'#{i} {label}\n({x:.1f},{y:.1f})'
        text.lifetime = rospy.Duration(0)
        array.markers.append(text)

    return array


class Level1Patrol:
    def __init__(self):
        rospy.init_node('level1_patrol')

        self.pub_respawn  = rospy.Publisher('/rviz_panel/respawn_objects',
                                            Int16, queue_size=1, latch=True)
        self.pub_done     = rospy.Publisher('/me5413/level1_done',
                                            Bool,  queue_size=1, latch=True)
        self.pub_markers  = rospy.Publisher('/patrol_waypoints',
                                            MarkerArray, queue_size=1, latch=True)

        self._tf_buf = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buf)

        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo('[level1_patrol] 等待 move_base 就绪...')
        self.client.wait_for_server()


        # 立刻发布 Marker，不需要等定位收敛（方便在 RViz 里对比墙角）
        self.pub_markers.publish(build_waypoint_markers())
        rospy.loginfo('[level1_patrol] 巡逻点已发布到 /patrol_waypoints（RViz 添加 MarkerArray 可见）')
        rospy.loginfo('[level1_patrol] move_base 就绪，等待 run() 启动')

    # ── 箱子重生成 ────────────────────────────────────────────────────

    def _respawn_boxes(self):
        rospy.loginfo('[level1_patrol] 清除并重生成箱子...')
        self.pub_respawn.publish(Int16(data=0))
        rospy.sleep(2.0)
        self.pub_respawn.publish(Int16(data=1))
        rospy.sleep(6.0)
        rospy.loginfo('[level1_patrol] 箱子重生成完成')

    # ── 导航到单个目标点 ──────────────────────────────────────────────

    def _robot_pose(self):
        """返回机器人当前 map 坐标 (x, y, yaw_rad)，失败返回 None。"""
        try:
            t = self._tf_buf.lookup_transform(
                'map', 'base_link', rospy.Time(0), rospy.Duration(0.1))
            tr = t.transform.translation
            q  = t.transform.rotation
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw  = math.atan2(siny, cosy)
            return tr.x, tr.y, yaw
        except Exception:
            return None

    @staticmethod
    def _yaw_err(a, b):
        """返回两角之间的最小绝对差（rad），结果在 [0, π]。"""
        d = (a - b) % (2 * math.pi)
        if d > math.pi:
            d = 2 * math.pi - d
        return d

    def _navigate_to(self, x, y, yaw_deg, label):
        rospy.loginfo('[level1_patrol] → %s  (%.1f, %.1f, %d°)',
                      label, x, y, yaw_deg)
        goal = make_goal(x, y, yaw_deg)
        self.client.send_goal(goal)

        rate = rospy.Rate(5.0)
        start = rospy.Time.now()

        while not rospy.is_shutdown():
            # ── 1. actionlib 已有结果 ────────────────────────────────
            state = self.client.get_state()
            if state == actionlib.GoalStatus.SUCCEEDED:
                rospy.loginfo('[level1_patrol] 到达 %s ✓ (SUCCEEDED)', label)
                rospy.sleep(ARRIVE_PAUSE)
                return True
            if state in (actionlib.GoalStatus.ABORTED,
                         actionlib.GoalStatus.REJECTED,
                         actionlib.GoalStatus.PREEMPTED):
                rospy.logwarn('[level1_patrol] 导航失败（state=%d），跳过 %s',
                              state, label)
                return False

            # ── 2. 距离+朝向检查：同时满足才视为到达 ────────────────
            pose = self._robot_pose()
            if pose is not None:
                rx, ry, ryaw = pose
                dist    = math.hypot(rx - x, ry - y)
                yawerr  = self._yaw_err(ryaw, math.radians(yaw_deg))
                if dist < ARRIVAL_DIST and yawerr < ARRIVAL_YAW:
                    rospy.loginfo(
                        '[level1_patrol] 到达 %s ✓ (dist=%.2fm yaw_err=%.2frad)',
                        label, dist, yawerr)
                    self.client.cancel_goal()
                    rospy.sleep(0.3)
                    return True

            # ── 3. 超时保底 ──────────────────────────────────────────
            if (rospy.Time.now() - start).to_sec() > GOAL_TIMEOUT:
                rospy.logwarn('[level1_patrol] 超时！取消目标 %s，继续下一点',
                              label)
                try:
                    self.client.cancel_goal()
                except Exception:
                    pass
                rospy.sleep(0.5)
                return False

            rate.sleep()

    # ── 主流程 ────────────────────────────────────────────────────────

    def run(self):
        rospy.loginfo('[level1_patrol] ===== 一楼巡逻开始 =====')

        if RESPAWN_BOXES:
            # 重生成箱子（8s）与定位收敛（8s）并行
            rospy.loginfo('[level1_patrol] 重生成箱子 + 等待定位收敛（并行）...')
            self._respawn_boxes()
            rospy.loginfo('[level1_patrol] 箱子完成，定位已收敛，开始导航')
        else:
            rospy.loginfo('[level1_patrol] 等待定位收敛 (8s)...')
            rospy.sleep(8.0)
            rospy.loginfo('[level1_patrol] 开始导航')

        total   = len(PATROL_WAYPOINTS)
        success = 0
        for i, (x, y, yaw_deg, label) in enumerate(PATROL_WAYPOINTS):
            if rospy.is_shutdown():
                break
            rospy.loginfo('[level1_patrol] [%d/%d] 前往 %s', i + 1, total, label)
            ok = self._navigate_to(x, y, yaw_deg, label)
            if ok:
                success += 1

        rospy.loginfo('[level1_patrol] ===== 一楼巡逻结束（%d/%d 点成功）=====',
                      success, total)
        rospy.loginfo('[level1_patrol] 发布 /me5413/level1_done，交接给队友流程...')
        self.pub_done.publish(Bool(data=True))


if __name__ == '__main__':
    try:
        Level1Patrol().run()
    except rospy.ROSInterruptException:
        pass
