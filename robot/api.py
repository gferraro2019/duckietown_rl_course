
"""
API between your code and the duckiebot ros topics.
"""


import os
import time
import random
import socket
import curses
import numpy as np
import rospy
from duckietown_msgs.msg import WheelsCmdStamped
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from std_msgs.msg import Header


class DuckiebotAPI(object):
    """
    API between the code and the duckiebot ros topics.
    This class is an interface that defines usefull functions, used by the discrete actions and continuous
    actions environments.
    """

    def __init__(self, **params):
        print()
        print("    ______________________________________________________    ")
        print()
        print("   ___                 _            _   _       ____  _     _ ")
        print("  |_ _|_ __         __| | ___ _ __ | |_| |__   |  _ \| |   | |")
        print("   | || '_ \ _____ / _` |/ _ \ '_ \| __| '_ \  | |_) | |   | |")
        print("   | || | | |_____| (_| |  __/ |_) | |_| | | | |  _ <| |___|_|")
        print("  |___|_| |_|      \__,_|\___| .__/ \__|_| |_| |_| \_\_____(_)")
        print("                             |_|                              ")
        print("    ______________________________________________________    ")
        print()
        print()
        self.robot_name = params.get("robot_name", "paperino")      # Duckiebot name

        # Init a node for this api
        #self.node = rospy.init_node('api_from_' + socket.gethostname(), anonymous=True)

        print("  > Initializing node...")
        self.node = rospy.init_node('api', anonymous=True)
        print("  > Node initialized.")

        # Setup ros command publisher
        self.commands_publisher = rospy.Publisher('/' + str(self.robot_name) + '/wheels_driver_node/wheels_cmd',
                                                  WheelsCmdStamped, queue_size=10)
        print("  > Commands publisher initialized.")

        # Set up the observation update process
        # - The robot will send us a lot of images. Because we don't want when the get_observation method will be
        #   called, we will update the last observation every time, and return it when get_observation is called.
        self.last_observation = None    # Not important, it will be instantiated when we receive the first observation.
        self.observation_subscriber = rospy.Subscriber(
            f"/{self.robot_name}/camera_node/image/compressed",
            CompressedImage,
            self.observation_callback
        )

        time.sleep(0.5)  # Wait for the publisher and subscriber to be registered.
        print("  > Api initialized.")

    def observation_callback(self, observation_message):
        """
        This function is called everytime an observation is received.
        Returns: None
        """
        try:
            observation = np.frombuffer(observation_message.data, np.uint8)     # Parse the image
            self.last_observation = observation                                 # Save the image
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

    def set_velocity_raw(self, left_wheel_velocity=0.0, right_wheel_velocity=0.0):
        msg = WheelsCmdStamped()
        print("Which mean left:", left_wheel_velocity, ", right:", right_wheel_velocity)

        # Set message parameters
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "base_link"
        msg.vel_left = left_wheel_velocity
        msg.vel_right = right_wheel_velocity

        # Publish the message at 10 Hz
        rate = rospy.Rate(10)
        if not rospy.is_shutdown():
            self.commands_publisher.publish(msg)
            rate.sleep()

    def set_velocity(self, linear_velocity=0.0, angular_velocity=0.0):
        print("Setting velocity to", linear_velocity, ",", angular_velocity, ":")
        print("Which mean left:", linear_velocity - angular_velocity, ", right:", linear_velocity + angular_velocity)
        self.set_velocity_raw(
            linear_velocity - angular_velocity,
            linear_velocity + angular_velocity)

    def apply_action(self, linear_velocity=0.0, angular_velocity=0.0):
        self.set_velocity(linear_velocity=linear_velocity, angular_velocity=angular_velocity)  # Send the action
        time.sleep(self.action_duration)    # Let the action run for a fixed duration
        self.set_velocity()             # Stop the robot at the end of the timer (default velocities are 0.0)
        time.sleep(0.1)                     # Makes sure this action is done before sending a new one.

    def stop_robot(self):
        self.set_velocity(0.0, 0.0)

    def get_observation(self):
        return self.last_observation


from gymnasium.spaces import Discrete, Box
from enum import Enum


class DuckieBotDiscrete(DuckiebotAPI):

    """
    DuckieBot environment with discrete actions.
    """

    class Actions(Enum):
        FORWARD = 0
        BACKWARD = 1
        LEFT = 2
        RIGHT = 3


    def __init__(self, **params):
        """
        Instantiate a discrete action environment.
        Args:
            robot_name (str): The name of the robot you are using.
            fixed_linear_velocity (float): The linear velocity that will be used for linear actions (forward, backward).
                The velocity used for the backward action is - fixed_linear_velocity.
                Default = 0.1.
            fixed_angular_velocity (float): The angular velocity that will be used for angular actions (left, right).
                The velocity used for the left action is - fixed_angular_velocity.
                Default = 0.1.
            action_duration (float): The duration of the action in seconds.
                Default = 1.
            stochasticity (float): The probability to take a discrete action. If a different actions is chosen, the
                action actually taken is chosen uniformly among the remaining actions.
                Default = 0.
        """

        print("  > Initializing environment... ")
        super().__init__(**params)
        self.observation_space = Box(low=0, high=255, shape=(100, 200, 3), dtype=np.uint8)  # Images observation space
        self.action_space = Discrete(4)     # Action space with four possible actions (from 0 to 3 included)

        self.fixed_linear_velocity: float = params.get("fixed_linear_velocity", 0.1)
        self.fixed_angular_velocity: float = params.get("fixed_angular_velocity", 0.1)
        self.action_duration: float = params.get("action_duration", 1.)
        self.stochasticity: float = params.get("stochasticity", 0.0)      # Probability to take a different action,

        print("  > Environment initialized.")

    def step(self, action):

        assert self.action_space.contains(action)
        original_action = action
        if self.stochasticity > 0.0 and random.random() < self.stochasticity:
            available_actions = list(set(range(self.action_space.n)) - {int(action)})
            action = random.choice(available_actions)

        print(f"Action chosen: {original_action} -> {action} (after stochasticity)")  # Debugging log

        if action == self.Actions.FORWARD.value:
            self.apply_action(linear_velocity=self.fixed_linear_velocity)
        if action == self.Actions.BACKWARD.value:
            self.apply_action(linear_velocity=-self.fixed_linear_velocity)
        if action == self.Actions.LEFT.value:
            self.apply_action(angular_velocity=self.fixed_angular_velocity)
        if action == self.Actions.RIGHT.value:
            self.apply_action(angular_velocity=-self.fixed_angular_velocity)

        # Now the action is performed, return the observation
        # NB: Because you have to design your own reward, the reward isn't computed here.
        # Compute it after the call to this function.
        return self.get_observation()

    def reset(self):
        print("    ########################################    ")
        print("    ###      RESET FUNCTION CALLED.      ###    ")
        print("    ###  Pick up the robot and place it  ###    ")
        print("    ###  in a valid initial position     ###    ")
        print("    ###  and then press any key to       ###    ")
        print("    ###  continue.                       ###    ")
        print("    ########################################    ")
        input("Press any key to continue ...")
        time.sleep(0.2)
        return self.get_observation()



# class DuckieBotContinuous(DuckiebotAPI):
#
#     """
#     DuckieBot environment with continuous actions.
#     """
#
#     def __init__(self, **params):
#         """
#         Instantiate a discrete action environment.
#         Args:
#             robot_name (str): The name of the robot you are using.
#             fixed_linear_velocity (float): The linear velocity that will be used for linear actions (forward, backward).
#                 The velocity used for the backward action is - fixed_linear_velocity.
#                 Default = 0.1.
#             fixed_angular_velocity (float): The angular velocity that will be used for angular actions (left, right).
#                 The velocity used for the left action is - fixed_angular_velocity.
#                 Default = 0.1.
#             action_duration (float): The duration of the action in seconds.
#                 Default = 1.
#             stochasticity (float): The probability to take a discrete action. If a different actions is chosen, the
#                 action actually taken is chosen uniformly among the remaining actions.
#                 Default = 0.
#         """
#         super().__init__()
#         self.api = DuckiebotAPI(robot_name=params.get("robot_name", "paperino"))      # Instantiate robot api
#         self.observation_space = Box(low=0, high=255, shape=(100, 200, 3), dtype=np.uint8)  # Images observation space
#         self.action_space = Discrete(4)     # Action space with four possible actions (from 0 to 3 included)
#
#         self.fixed_linear_velocity: float = params.get("fixed_linear_velocity", 0.1)
#         self.fixed_angular_velocity: float = params.get("fixed_angular_velocity", 0.1)
#         self.action_duration: float = params.get("action_duration", 1.)
#         self.stochasticity: float = params.get("stochasticity", 0.0)      # Probability to take a different action,
#
#     def step(self, action):
#
#         assert self.action_space.contains(action)
#
#         if action == self.Actions.FORWARD:
#             self.apply_action(linear_velocity=self.fixed_linear_velocity)
#         if action == self.Actions.BACKWARD:
#             self.apply_action(linear_velocity=-self.fixed_linear_velocity)
#         if action == self.Actions.LEFT:
#             self.apply_action(angular_velocity=self.fixed_angular_velocity)
#         if action == self.Actions.RIGHT:
#             self.apply_action(angular_velocity=-self.fixed_angular_velocity)
#
#         # Now the action is performed, return the observation
#         # NB: Because you have to design your own reward, the reward isn't computed here.
#         # Compute it after the call to this function.
#         return self.api.get_observation()


import keyboard
import time


def read_arrow_keys(stdscr):
    # Disable cursor and enable keypad
    curses.curs_set(0)
    stdscr.nodelay(1)

    while True:
        key = stdscr.getch()
        if key == curses.KEY_UP:
            return 0
        elif key == curses.KEY_DOWN:
            return 1
        elif key == curses.KEY_LEFT:
            return 2
        elif key == curses.KEY_RIGHT:
            return 3
        elif key == 67 or key == 99:  # 'C' key (uppercase)
            return -1
        time.sleep(0.1)


if __name__ == '__main__':
    # Initialize the environment
    env = DuckieBotDiscrete(robot_name="paperino", fixed_linear_velocity=0.3, fixed_angular_velocity=0.1,
                            action_duration=0.3, stochasticity=0.1)

    # Reset the environment and get the initial observation
    observation = env.reset()

    # Print the initial observation
    print("Initial Observation:", observation)

    # Main loop to select actions
    while True:
        try:
            # Start the curses screen
            action = curses.wrapper(read_arrow_keys)
            print(f"Action selected: {action}")
            if action == -1:
                print("\nExiting...")
                break
            observation = env.step(action)

            # Optionally: you can add a small sleep to control the action speed
            time.sleep(0.5)

        except KeyboardInterrupt:
            print("\nExiting...")

    # api = DuckiebotAPI()
    # action_duration = 1
    # std_linear_velocity = 0.3
    # std_angular_velocity = 0.3
    #
    # # Send linear velocity
    # api.set_velocity(0.2, 0.0)
    # time.sleep(action_duration)
    #
    # api.stop_robot()
