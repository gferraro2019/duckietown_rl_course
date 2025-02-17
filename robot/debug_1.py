# Import the necessary message type for wheels control
from duckietown_msgs.msg import WheelsCmdStamped
import rospy
from pynput.keyboard import Key, Listener

# Initialize the ROS node
rospy.init_node('keyboard_control', anonymous=True)

# Publisher for wheels commands
wheels_cmd_pub = rospy.Publisher('/paperino/wheels_driver_node/wheels_cmd', WheelsCmdStamped, queue_size=10)

# Define the wheels command message
wheels_cmd = WheelsCmdStamped()

# Control the Duckiebot
def move_forward():
    wheels_cmd.vel_left = 0.0  # Speed for left wheel
    wheels_cmd.vel_right = 0.0  # Speed for right wheel
    wheels_cmd_pub.publish(wheels_cmd)
    rospy.loginfo("Moving forward")

def move_backward():
    wheels_cmd.vel_left = -1.0  # Speed for left wheel (reverse)
    wheels_cmd.vel_right = -1.0  # Speed for right wheel (reverse)
    wheels_cmd_pub.publish(wheels_cmd)
    rospy.loginfo("Moving backward")

def turn_left():
    wheels_cmd.vel_left = -0.5  # Turn left by slowing down left wheel
    wheels_cmd.vel_right = 0.5   # Turn right by speeding up right wheel
    wheels_cmd_pub.publish(wheels_cmd)
    rospy.loginfo("Turning left")

def turn_right():
    wheels_cmd.vel_left = 0.5   # Turn left by speeding up left wheel
    wheels_cmd.vel_right = -0.5  # Turn right by slowing down right wheel
    wheels_cmd_pub.publish(wheels_cmd)
    rospy.loginfo("Turning right")

def stop():
    wheels_cmd.vel_left = 0.0  # Stop left wheel
    wheels_cmd.vel_right = 0.0  # Stop right wheel
    wheels_cmd_pub.publish(wheels_cmd)
    rospy.loginfo("Stopping")

# Handle key press events
def on_press(key):
    try:
        if key.char == 'w':  # Move forward on 'w' key
            move_forward()
        elif key.char == 's':  # Move backward on 's' key
            move_backward()
        elif key.char == 'a':  # Turn left on 'a' key
            turn_left()
        elif key.char == 'd':  # Turn right on 'd' key
            turn_right()
        elif key == Key.esc:  # Stop on 'esc' key
            stop()
            return False  # Stop listener
    except AttributeError:
        pass

# Start listening to keyboard events
with Listener(on_press=on_press) as listener:
    listener.join()
