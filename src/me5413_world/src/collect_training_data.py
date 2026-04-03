#!/usr/bin/env python3
"""
collect_training_data.py

通过 Gazebo 地面真值 3D 投影自动生成 YOLO 训练数据：
  - 订阅 /gazebo/model_states 获取所有箱子的世界坐标
  - 订阅 /gazebo/ground_truth/state 获取机器人真实位姿
  - 收到 /me5413/scan_trigger 时，将每个箱子的 8 个角投影到相机图像
  - 生成覆盖整个箱子的 YOLO 格式边界框（class cx cy w h，归一化）
  - 数字标签直接从模型名称提取（"number5_2" → 5），100% 准确

采集完后运行：
  yolo train data=/tmp/me5413_dataset/dataset.yaml \
             model=yolov8n.pt epochs=50 imgsz=640 batch=16
"""

import math
import os
import re

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import Bool

# ── 相机内参（640×512，fx=fy=554.254，cx=320，cy=256） ───────────────
IMG_W  = 640
IMG_H  = 512
FX     = 554.254
FY     = 554.254
CX     = 320.0
CY     = 256.0
CAM_Z_OFFSET = 0.3   # 相机安装高度（机器人底盘上方 0.3m）
CAM_X_OFFSET = 0.2   # 相机在机器人前方 0.2m（近似，对投影影响很小）

# ── 箱子参数 ─────────────────────────────────────────────────────────
BOX_HALF = 0.4        # 箱子半边长 0.8m / 2
BOX_Z_CENTER = 0.4    # 箱子中心高度（底面 z=0，顶面 z=0.8，中心 z=0.4）

# 8 个角相对于箱子中心的偏移
CORNERS_REL = np.array([
    [s * BOX_HALF, t * BOX_HALF, u * BOX_HALF]
    for s in (-1, 1) for t in (-1, 1) for u in (-1, 1)
], dtype=np.float32)

# ── 过滤参数 ─────────────────────────────────────────────────────────
MIN_DEPTH    = 0.8    # 箱子中心距相机最近距离（m）
MAX_DEPTH    = 18.0   # 箱子中心距相机最远距离（m）
MIN_BOX_PX   = 5      # bbox 最小边长（像素），太小跳过
MARGIN_PX    = 5      # 允许 bbox 超出图像边界的像素数（用于部分可见箱子）
ROOM_WALL_X  = -10.0  # 两个房间之间的隔墙 x 坐标（墙后箱子不可见，不标注）

# 一楼箱子模型名格式：number{digit}_{index}，例如 number5_2
BOX_NAME_RE = re.compile(r'^number(\d)_\d+$')


def _quat_to_yaw(q: Quaternion) -> float:
    """四元数 → yaw 角（弧度）。"""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _project_box(wx, wy, rx, ry, ryaw):
    """
    将箱子的 8 个角从世界坐标投影到图像坐标，返回像素级 bbox (u_min, v_min, u_max, v_max)。
    若箱子在相机后方或完全在图像外则返回 None。
    """
    # 相机中心在世界坐标中的位置
    cam_wx = rx + CAM_X_OFFSET * math.cos(ryaw)
    cam_wy = ry + CAM_X_OFFSET * math.sin(ryaw)

    us, vs = [], []
    for corner in CORNERS_REL:
        # 角点世界坐标
        cx_w = wx + corner[0]
        cy_w = wy + corner[1]
        cz_w = BOX_Z_CENTER + corner[2]

        # 世界 → 机器人机体坐标系
        dx = cx_w - cam_wx
        dy = cy_w - cam_wy
        dz = cz_w - CAM_Z_OFFSET

        dx_body =  dx * math.cos(ryaw) + dy * math.sin(ryaw)
        dy_body = -dx * math.sin(ryaw) + dy * math.cos(ryaw)
        dz_body = dz

        # 机体 → 相机光学坐标系（z 前，x 右，y 下）
        cam_Z =  dx_body
        cam_X = -dy_body
        cam_Y = -dz_body

        if cam_Z <= 0.1:
            return None   # 箱子在相机后方

        u = FX * cam_X / cam_Z + CX
        v = FY * cam_Y / cam_Z + CY
        us.append(u)
        vs.append(v)

    u_min, u_max = min(us), max(us)
    v_min, v_max = min(vs), max(vs)

    # 过滤：bbox 必须与图像有交叠（允许少量出界）
    if u_max < -MARGIN_PX or u_min > IMG_W + MARGIN_PX:
        return None
    if v_max < -MARGIN_PX or v_min > IMG_H + MARGIN_PX:
        return None

    # 裁剪到图像边界
    u_min = max(0.0, u_min)
    u_max = min(float(IMG_W), u_max)
    v_min = max(0.0, v_min)
    v_max = min(float(IMG_H), v_max)

    if u_max - u_min < MIN_BOX_PX or v_max - v_min < MIN_BOX_PX:
        return None

    return u_min, v_min, u_max, v_max


class DataCollector:
    def __init__(self):
        rospy.init_node('collect_training_data')

        self.out_dir = rospy.get_param('~out_dir', os.path.join(os.path.dirname(__file__), '..', '..', 'me5413_dataset'))
        self.img_dir = os.path.join(self.out_dir, 'images', 'train')
        self.lbl_dir = os.path.join(self.out_dir, 'labels', 'train')
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.lbl_dir, exist_ok=True)

        self.bridge        = CvBridge()
        self.latest_image  = None
        self.robot_pose    = None   # (rx, ry, ryaw)
        self.box_list      = []     # [(digit_cls, wx, wy), ...]
        # 从已有帧数续号，避免覆盖之前的数据
        self.frame_idx     = len([f for f in os.listdir(self.img_dir) if f.endswith('.jpg')])
        self.total_labels  = 0
        if self.frame_idx > 0:
            rospy.loginfo("已有 %d 帧，从 frame_%05d 续号", self.frame_idx, self.frame_idx)

        rospy.Subscriber('/front/image_raw',          Image,       self._image_cb,      queue_size=1, buff_size=2**24)
        rospy.Subscriber('/gazebo/ground_truth/state', Odometry,   self._gt_cb,         queue_size=1)
        rospy.Subscriber('/gazebo/model_states',       ModelStates, self._models_cb,     queue_size=1)
        rospy.Subscriber('/me5413/scan_trigger',       Bool,        self._trigger_cb,    queue_size=1)
        self.pub_done = rospy.Publisher('/me5413/scan_done', Bool, queue_size=1)

        rospy.loginfo("DataCollector (GT投影模式) 启动，等待巡逻触发信号...")
        rospy.loginfo("数据将保存到: %s", self.out_dir)

    # ── 回调 ────────────────────────────────────────────────────────

    def _image_cb(self, msg):
        self.latest_image = msg

    def _gt_cb(self, msg):
        q   = msg.pose.pose.orientation
        yaw = _quat_to_yaw(q)
        self.robot_pose = (msg.pose.pose.position.x,
                           msg.pose.pose.position.y,
                           yaw)

    def _models_cb(self, msg):
        boxes = []
        for name, pose in zip(msg.name, msg.pose):
            m = BOX_NAME_RE.match(name)
            if m is None:
                continue
            digit = int(m.group(1))          # 1–9
            cls   = digit - 1                 # 0–8
            wx    = pose.position.x
            wy    = pose.position.y
            boxes.append((cls, wx, wy))
        self.box_list = boxes

    # ── 触发：投影 + 保存 ────────────────────────────────────────────

    def _trigger_cb(self, msg):
        if not msg.data or self.latest_image is None:
            self.pub_done.publish(Bool(data=True))
            return
        if self.robot_pose is None:
            rospy.logwarn("尚未收到机器人真值位姿，跳过")
            self.pub_done.publish(Bool(data=True))
            return
        if not self.box_list:
            rospy.logwarn("尚未收到 model_states，跳过")
            self.pub_done.publish(Bool(data=True))
            return

        try:
            cv_img = self.bridge.imgmsg_to_cv2(self.latest_image, desired_encoding='bgr8')
        except Exception as e:
            rospy.logwarn("图像转换失败: %s", e)
            self.pub_done.publish(Bool(data=True))
            return

        rx, ry, ryaw = self.robot_pose

        # 判断机器人在哪个房间（隔墙在 x≈-10）
        robot_in_room_a = rx < ROOM_WALL_X
        labels = []
        for (cls, wx, wy) in self.box_list:
            # 同房间过滤：墙对面的箱子不可见，不标注
            box_in_room_a = wx < ROOM_WALL_X
            if robot_in_room_a != box_in_room_a:
                continue

            # 深度过滤：箱子中心距机器人
            dist = math.hypot(wx - rx, wy - ry)
            if dist < MIN_DEPTH or dist > MAX_DEPTH:
                continue

            bbox = _project_box(wx, wy, rx, ry, ryaw)
            if bbox is None:
                continue

            u_min, v_min, u_max, v_max = bbox
            cx_n = (u_min + u_max) / 2.0 / IMG_W
            cy_n = (v_min + v_max) / 2.0 / IMG_H
            bw_n = (u_max - u_min) / IMG_W
            bh_n = (v_max - v_min) / IMG_H
            labels.append(f"{cls} {cx_n:.6f} {cy_n:.6f} {bw_n:.6f} {bh_n:.6f}")

        if not labels:
            self.pub_done.publish(Bool(data=True))
            return

        name = f"frame_{self.frame_idx:05d}"
        cv2.imwrite(os.path.join(self.img_dir, f"{name}.jpg"), cv_img)
        with open(os.path.join(self.lbl_dir, f"{name}.txt"), 'w') as f:
            f.write('\n'.join(labels))

        self.frame_idx    += 1
        self.total_labels += len(labels)
        rospy.loginfo("已收集 %d 帧 / %d 个标注  (本帧 %d 个箱子，机器人 %.1f,%.1f yaw=%.1f°)",
                      self.frame_idx, self.total_labels, len(labels),
                      rx, ry, math.degrees(ryaw))
        self.pub_done.publish(Bool(data=True))

    def save_yaml(self):
        content = (
            f"path: {self.out_dir}\n"
            "train: images/train\n"
            "val:   images/train\n\n"
            "nc: 9\n"
            "names: ['1','2','3','4','5','6','7','8','9']\n"
        )
        with open(os.path.join(self.out_dir, 'dataset.yaml'), 'w') as f:
            f.write(content)
        rospy.loginfo("dataset.yaml 已生成: %s/dataset.yaml", self.out_dir)

    def generate_preview(self):
        """将标注框画回图片，保存到 preview/ 供人工验证。"""
        import glob
        COLORS = [(0,0,255),(0,128,255),(0,255,255),(0,255,0),(255,255,0),
                  (255,128,0),(255,0,0),(128,0,255),(200,200,200)]
        prev_dir = os.path.join(self.out_dir, 'preview')
        os.makedirs(prev_dir, exist_ok=True)
        for img_path in sorted(glob.glob(os.path.join(self.img_dir, '*.jpg'))):
            name = os.path.splitext(os.path.basename(img_path))[0]
            img  = cv2.imread(img_path)
            h, w = img.shape[:2]
            lbl_path = os.path.join(self.lbl_dir, name + '.txt')
            for line in open(lbl_path):
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
                x1 = int((cx - bw / 2) * w); y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w); y2 = int((cy + bh / 2) * h)
                color = COLORS[cls % len(COLORS)]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, str(cls + 1), (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imwrite(os.path.join(prev_dir, name + '.jpg'), img)
        rospy.loginfo("Preview 已生成: %s (%d 张)", prev_dir, self.frame_idx)


if __name__ == '__main__':
    node = None
    try:
        node = DataCollector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        if node is not None:
            node.save_yaml()
            node.generate_preview()
            rospy.loginfo("采集结束：%d 帧，%d 个标注", node.frame_idx, node.total_labels)
            if node.frame_idx > 0:
                rospy.loginfo("========== 训练命令 ==========")
                rospy.loginfo("yolo train data=%s/dataset.yaml "
                              "model=yolov8n.pt epochs=50 imgsz=640 batch=16",
                              node.out_dir)
