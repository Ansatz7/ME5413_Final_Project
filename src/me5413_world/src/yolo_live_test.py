#!/usr/bin/env python3
"""
yolo_live_test.py

实时 YOLO 检测可视化：订阅相机图像，推理后把框画回去发布到
/me5413/yolo_live，用 rqt_image_view 查看。

rosrun me5413_world yolo_live_test.py
rqt_image_view /me5413/yolo_live
"""

import os
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO

_DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), '..', 'models', 'box_detector.pt')


class YoloLiveTest:
    def __init__(self):
        rospy.init_node('yolo_live_test')
        model_path = rospy.get_param('~model_path', _DEFAULT_MODEL)
        self.conf  = rospy.get_param('~conf', 0.9)
        self.model = YOLO(model_path)
        self.bridge = CvBridge()

        self.pub = rospy.Publisher('/me5413/yolo_live', Image, queue_size=1)
        rospy.Subscriber('/front/image_raw', Image, self._cb, queue_size=1, buff_size=2**24)
        rospy.loginfo("YOLO live test 启动，模型: %s  置信度阈值: %.2f", model_path, self.conf)
        rospy.loginfo("用 rqt_image_view /me5413/yolo_live 查看")

    def _cb(self, msg):
        if self.pub.get_num_connections() == 0:
            return
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logwarn("图像转换失败: %s", e)
            return

        results = self.model(img, conf=self.conf, verbose=False)[0]
        vis = results.plot()   # ultralytics 自带画框+标签

        try:
            self.pub.publish(self.bridge.cv2_to_imgmsg(vis, encoding='bgr8'))
        except Exception as e:
            rospy.logwarn("发布失败: %s", e)


if __name__ == '__main__':
    try:
        YoloLiveTest()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
