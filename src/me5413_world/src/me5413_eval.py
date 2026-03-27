#!/usr/bin/env python3
import math

import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32


class MappingEvaluator:
    def __init__(self):
        rospy.init_node("me5413_eval_node")

        self.gt_topic = rospy.get_param("~ground_truth_topic", "/gazebo/ground_truth/state")
        self.estimate_topic = rospy.get_param("~estimate_topic", "/odom")
        self.error_topic = rospy.get_param("~error_topic", "/me5413/mapping_error")

        self.gt_pose = None
        self.estimate_pose = None

        self.sub_gt = rospy.Subscriber(self.gt_topic, Odometry, self.gt_callback, queue_size=10)
        self.sub_estimate = rospy.Subscriber(self.estimate_topic, Odometry, self.estimate_callback, queue_size=10)
        self.pub_error = rospy.Publisher(self.error_topic, Float32, queue_size=10)

        rospy.loginfo(
            "Mapping evaluator started. Comparing '%s' against '%s' and publishing to '%s'.",
            self.estimate_topic,
            self.gt_topic,
            self.error_topic,
        )
        rospy.logwarn(
            "This node currently measures online planar pose error only. "
            "Using /odom reflects odometry drift, not full SLAM accuracy."
        )

    def gt_callback(self, msg):
        self.gt_pose = msg.pose.pose.position

    def estimate_callback(self, msg):
        self.estimate_pose = msg.pose.pose.position
        self.calculate_error()

    def calculate_error(self):
        if self.gt_pose is None or self.estimate_pose is None:
            return

        error = math.hypot(
            self.gt_pose.x - self.estimate_pose.x,
            self.gt_pose.y - self.estimate_pose.y,
        )

        self.pub_error.publish(Float32(data=error))
        rospy.loginfo_throttle(
            1.0,
            "Current planar pose error between %s and %s: %.4f m",
            self.estimate_topic,
            self.gt_topic,
            error,
        )


if __name__ == "__main__":
    try:
        MappingEvaluator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
