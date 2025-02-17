import rospy
from duckietown_msgs.msg import WheelsCmdStamped
from std_msgs.msg import Header


def publish_wheel_velocity():
    # Initialize the ROS node
    rospy.init_node('publish_wheel_velocity', anonymous=True)

    # Create a publisher object
    pub = rospy.Publisher('/paperino/wheels_driver_node/wheels_cmd', WheelsCmdStamped, queue_size=10)

    # Create a WheelsCmdStamped message
    msg = WheelsCmdStamped()

    # Create and assign a header with timestamp
    msg.header = Header()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = "base_link"  # You can change this based on your setup

    # Set velocities for left and right wheels
    msg.vel_left = 0.5  # Left wheel velocity (m/s)
    msg.vel_right = 0.5  # Right wheel velocity (m/s)

    # Publish the message at 10 Hz
    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown():
        pub.publish(msg)
        rospy.loginfo(f"Publishing: Left Vel: {msg.vel_left}, Right Vel: {msg.vel_right}")
        rate.sleep()


if __name__ == '__main__':
    try:
        publish_wheel_velocity()
    except rospy.ROSInterruptException:
        pass