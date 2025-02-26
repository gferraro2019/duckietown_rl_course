from gymnasium.spaces import Discrete, Box
from enum import Enum
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
from cv_bridge import CvBridge
from gymnasium import Env
from std_msgs.msg import Int32
from cv_bridge import CvBridge
import threading


class DuckieBotDiscrete(Env):

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

        print("  > Initializing environment... ")
        super().__init__()
        self.robot_name: str = params.get("robot_name", "paperino")
        
        # Ros stuff
        rospy.init_node('robot_discrete_environment', anonymous=True)   # Initialise the ros node
        self.actions_publisher = rospy.Publisher('/' + str(self.robot_name) + '/discrete_action', Int32, queue_size=10) # Create a publisher actions
        self._image_bridge = CvBridge()
        self.last_observation = None    # Not important, it will be instantiated when we receive the first observation.
        self.observation_subscriber = rospy.Subscriber(
            f"/{self.robot_name}/observations",
            CompressedImage,
            self.observation_callback
        )
        self.observation_event = threading.Event()  # Used to wait for an observation message

        # Env stuff
        self.observation_space = Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8)  # Images observation space
        self.action_space = Discrete(4)     # Action space with four possible actions (from 0 to 3 included)

        self.stochasticity: float = params.get("stochasticity", 0.0)      # Probability to take a different action,

        print("  > Environment initialized.")


    def observation_callback(self, observation_message):
        """
        This function is called everytime an observation is received.
        Returns: None
        """
        try:
            self.last_observation = self._image_bridge.compressed_imgmsg_to_cv2(observation_message)[150:]
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

    def step(self, action):

        assert self.action_space.contains(action)
        original_action = action
        if self.stochasticity > 0.0 and random.random() < self.stochasticity:
            available_actions = list(set(range(self.action_space.n)) - {int(action)})
            action = random.choice(available_actions)

        # print(f"Action chosen: {original_action} -> {action} (after stochasticity)")  # Debugging log
        if isinstance(action, np.ndarray):
            action = int(action)
        
        self.actions_publisher.publish(action)

        # Wait until a new observation is received
        self.observation_event.wait()  # Blocks execution until the event is set

        return self.last_observation, 0, False, False, {}

    def reset(self, seed=None, options=None):
        print("    ########################################    ")
        print("    ###      RESET FUNCTION CALLED.      ###    ")
        print("    ###  Pick up the robot and place it  ###    ")
        print("    ###  in a valid initial position     ###    ")
        print("    ###  and then press any key to       ###    ")
        print("    ###  continue.                       ###    ")
        print("    ########################################    ")
        input("Press any key to continue ...")
        time.sleep(0.2)
        return self.last_observation, {}

