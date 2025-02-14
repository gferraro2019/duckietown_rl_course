
"""
API between your code and the duckiebot ros topics.
"""


import os
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped


class DuckiebotAPI(object):
    """
    API between the code and the duckiebot ros topics.
    This class is an interface that defines usefull functions, used by the discrete actions and continuous
    actions environments.
    """

    def __init__(self, **params):
        self.hostname = params.get("hostname", "paperino")      # Duckiebot name

    def set_velocity(self, linear_velocity, angular_velocity):

    def set_linear_velocity(self, linear_velocity):

    def set_angular_velocity(self, angular_velocity)


class DuckieBotDiscrete(DuckiebotAPI):

    """
    API between the code and the duckiebot ros topics.
    Instantiate this API leads to a discrete actions environment
    """

    def __init__(self,
                 discrete_actions: bool = True,

                 # If discrete_actions is True
                 fixed_linear_velocity: float = 0.1,            # Only used if discrete_actions is True
                 fixed_angular_velocity: float = 0.1,           # Only used if discrete_actions is True

                 # If discrete_actions is False
                 maximal_linear_velocity: float = 0.1,          # Only used if discrete_actions is False
                 maximal_angular_velocity: float = 0.1,         # Only used if discrete_actions is False

                 stochasticity: int = 0,                        # A percentage:
                 ):
    def action(self):

