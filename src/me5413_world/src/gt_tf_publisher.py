#!/usr/bin/env python3
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry


class GroundTruthTfPublisher:
    def __init__(self):
        rospy.init_node("ground_truth_tf_publisher")

        self.parent_frame = rospy.get_param("~parent_frame", "world")
        self.child_frame = rospy.get_param("~child_frame", "ground_truth")
        self.gt_topic = rospy.get_param("~ground_truth_topic", "/gazebo/ground_truth/state")

        self.broadcaster = tf2_ros.TransformBroadcaster()
        self.subscriber = rospy.Subscriber(self.gt_topic, Odometry, self.callback, queue_size=10)

        rospy.loginfo(
            "Publishing ground-truth TF from topic '%s' as %s -> %s",
            self.gt_topic,
            self.parent_frame,
            self.child_frame,
        )

    def callback(self, msg):
        transform = TransformStamped()
        transform.header.stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()
        transform.header.frame_id = self.parent_frame
        transform.child_frame_id = self.child_frame

        transform.transform.translation.x = msg.pose.pose.position.x
        transform.transform.translation.y = msg.pose.pose.position.y
        transform.transform.translation.z = msg.pose.pose.position.z
        transform.transform.rotation = msg.pose.pose.orientation

        self.broadcaster.sendTransform(transform)


if __name__ == "__main__":
    try:
        GroundTruthTfPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
