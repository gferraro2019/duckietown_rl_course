"""
API between your code and the duckiebot ros topics with PID wheel control.
"""

import os
import time
import random
import socket
import curses
import numpy as np
import rospy
from enum import Enum
from duckietown_msgs.msg import WheelsCmdStamped
from duckietown_msgs.msg import WheelEncoderStamped
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Header, Int32, Float32

class PIDController:
    """
    Implémentation personnalisée d'un contrôleur PID
    """
    def __init__(self, kp, ki, kd, setpoint=0, output_limits=None, type=''):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        self.type = type
        
        self.error_sum = 0
        self.last_error = 0
        self.last_time = rospy.Time.now()
    
    def compute(self, current_value):
        # Calculer le temps écoulé
        current_time = rospy.Time.now()
        dt = (current_time - self.last_time).to_sec()
        self.last_time = current_time
        
        if dt <= 0:
            return 0
            
        # Calculer l'erreur
        error = (self.setpoint - current_value)/135.0
        
        # Terme proportionnel
        p_term =  error
        
        # Terme intégral
        self.error_sum += error * dt
        i_term = self.error_sum
        
        # Terme dérivé (seulement si dt est significatif)
        d_term = 0
        if dt > 0:
            d_term =  (error - self.last_error) / dt
        self.last_error = error
        
        # Calculer la sortie
        output = self.kp*p_term + self.ki*i_term + self.kd*d_term
        
        print('name : ', self.type, 'p_term : ', p_term , 'i_term : ', i_term , 'd_term : ', d_term)
        print('output : ', output)

        
        
        # Appliquer les limites si définies
        if self.output_limits is not None:
            output = max(self.output_limits[0], min(output, self.output_limits[1]))
        
        return output
    
    def reset(self):
        """Réinitialiser l'intégrateur et l'erreur précédente"""
        self.error_sum = 0
        self.last_error = 0

class DuckieBotAPI(object):
    """
    API between the code and the duckiebot ros topics.
    This class is an interface that defines usefull functions, used by the discrete actions and continuous
    actions environments. Enhanced with PID control for wheel movement.
    """

    class Actions(Enum):
        FORWARD = 0
        BACKWARD = 1
        LEFT = 2
        RIGHT = 3
        STOP = 4  # Action pour arrêter le robot

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
        self.fixed_linear_velocity: float = params.get("fixed_linear_velocity", 0.4)
        self.fixed_angular_velocity: float = params.get("fixed_angular_velocity", 0.2)
        self.control_time: float = params.get("control_time", 1.0)  # Temps de contrôle pour chaque action

        # Init a node for this api
        print("  > Initializing node...")
        self.node = rospy.init_node('actions_converter', anonymous=True)
        print("  > Node initialized.")

        # PID control parameters
        self.TICKS_PER_REV = 135  # Nombre de ticks par tour de roue
        self.UPDATE_RATE = 10     # Hz
        
        # PID setup - Implémentation personnalisée
        kp = 0.1
        ki = 0.01
        kd = 0.01
        self.pid_left = PIDController(
            kp=kp, 
            ki=ki, 
            kd=kd, 
            setpoint=0,
            output_limits=(-1, 1),
            type='left'
        )
        self.pid_right = PIDController(
            kp=kp, 
            ki=ki, 
            kd=kd, 
            setpoint=0,
            output_limits=(-1, 1),
            type='right'
        )
        
        # Position actuelle des roues en ticks
        self.current_ticks_left = 0
        self.current_ticks_right = 0
        self.prev_ticks_left = 0
        self.prev_ticks_right = 0
        
        # Vitesses cibles en ticks/s pour chaque roue
        self.target_speed_left = 0.0
        self.target_speed_right = 0.0
        
        # État du contrôle PID
        self.pid_active = True  # Par défaut, le PID est actif
        
        # Gestion du timeout directement dans pid_control
        self.last_action_time = rospy.Time.now()
        self.action_active = False
        
        # Actions possibles
        self.ACTION_MAPPING = {
            0: (0.0, 0.0),    # avancer: les deux roues tournent vers l'avant
            1: (-1.0, -1.0),  # reculer: les deux roues tournent vers l'arrière
            2: (1.0, -1.0),   # droite: roue gauche vers l'avant, roue droite vers l'arrière
            3: (-1.0, 1.0),   # gauche: roue gauche vers l'arrière, roue droite vers l'avant
            4: (1.0, 1.0)     # stop: les deux roues s'arrêtent
        }
        
        # Temps pour le calcul de vitesse
        self.last_time = rospy.Time.now()
        
        # Setup ros command publisher
        self.commands_publisher = rospy.Publisher('/' + str(self.robot_name) + '/wheels_driver_node/wheels_cmd',
                                                  WheelsCmdStamped, queue_size=10)
        print("  > Commands publisher initialized.")

        # Setup ros command publisher
        self.observations_publisher = rospy.Publisher('/' + str(self.robot_name) + '/observation', CompressedImage, queue_size=10)
        print("  > Observation publisher initialized.")
        self.last_observation_message = None

        # Set up the observation update process
        self.last_observation_message = None
        self.observation_subscriber = rospy.Subscriber(
            f"/{self.robot_name}/camera_node/image/compressed",
            CompressedImage,
            self.observation_callback
        )

        # Setup action listener
        self.actions_subscriber = rospy.Subscriber('/' + str(self.robot_name) + '/discrete_action', Int32, self.actions_callback)
        
        # Setup wheel tick subscribers for PID control
        rospy.Subscriber('/' + str(self.robot_name) + '/left_wheel_encoder_node/tick', WheelEncoderStamped, self.left_tick_callback)
        rospy.Subscriber('/' + str(self.robot_name) + '/right_wheel_encoder_node/tick', WheelEncoderStamped, self.right_tick_callback)
        
        # Timer for PID control
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.UPDATE_RATE), self.pid_control)

        # Première mise à l'arrêt du robot
        self.stop_robot_direct()
        
        time.sleep(0.5)  # Wait for the publisher and subscriber to be registered.
        print("  > Api initialized with custom PID wheel control.")
        rospy.spin()

    def observation_callback(self, observation_message):
        """
        This function is called everytime an observation is received.
        Returns: None
        """
        try:
            self.last_observation_message = observation_message
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")
    
    def left_tick_callback(self, msg):
        """Callback pour les ticks de la roue gauche"""
        self.current_ticks_left = msg.data
    
    def right_tick_callback(self, msg):
        """Callback pour les ticks de la roue droite"""
        self.current_ticks_right = msg.data
    
    def stop_robot_direct(self):
        """Arrêter le robot directement sans PID"""
        # Désactiver le PID temporairement
        self.pid_active = False
        
        # Envoyer directement la commande d'arrêt
        self.set_velocity_raw(0.0, 0.0)
        
        # Réinitialiser les intégrateurs PID
        self.pid_left.reset()
        self.pid_right.reset()
        
        # Mettre à jour les setpoints à zéro
        self.pid_left.setpoint = 0.0
        self.pid_right.setpoint = 0.0
        
        self.action_active = False
        rospy.loginfo("Robot stopped directly")
        
        # Réactiver le PID après l'arrêt
        # self.pid_active = True
    
    def publish_observation(self):
        """Publier l'observation actuelle"""
        if self.last_observation_message:
            if not rospy.is_shutdown():
                self.observations_publisher.publish(self.last_observation_message)
    
    def actions_callback(self, data):
        """Callback pour traiter les actions reçues"""
        action = int(data.data)
        print("    [api] Received action", action)
        if isinstance(action, np.ndarray):
            action = int(action)
        
        # Appliquer l'action
        self.apply_action(action)
        
    def apply_action(self, action):
        """Appliquer une action spécifique"""
        # Traiter la nouvelle action
        if action in self.ACTION_MAPPING and action!=0:
            left_dir, right_dir = self.ACTION_MAPPING[action]
            
            # Définir une vitesse cible en ticks/s
            target_speed = 135  # ticks/s
            self.target_speed_left = left_dir * target_speed
            self.target_speed_right = right_dir * target_speed
            
            # Réactiver le PID si nécessaire
            self.pid_active = True
            
            # Mettre à jour les setpoints des PIDs
            self.pid_left.setpoint = self.target_speed_left
            self.pid_right.setpoint = self.target_speed_right
            
            # Enregistrer le temps de début de l'action et marquer comme active
            self.last_action_time = rospy.Time.now()
            self.action_active = True
            
            rospy.loginfo(f"New target speeds - Left: {self.target_speed_left}, Right: {self.target_speed_right}")
        else:
            rospy.logwarn(f"Unknown action received: {action}")
            self.stop_robot_direct()

    def pid_control(self, event):
        """Fonction principale d'asservissement PID appelée périodiquement"""
        current_time = rospy.Time.now()
        
        # Vérification du timeout de l'action
        if self.action_active:
            elapsed_time = (current_time - self.last_action_time).to_sec()
            if elapsed_time >= self.control_time:
                print('ACTION TIMEOUT ###################################################')
                self.stop_robot_direct()
                self.publish_observation()
        
        # Ne pas calculer le PID si le PID est désactivé
        if not self.pid_active:
            return
            
        dt = (current_time - self.last_time).to_sec()
        
        if dt > 0:
            # Calculer les vitesses actuelles en ticks/s
            speed_left = (self.current_ticks_left - self.prev_ticks_left) / dt
            speed_right = (self.current_ticks_right - self.prev_ticks_right) / dt
            print('speed_left', speed_left, 'speed_right', speed_right)
            
            # Calculer les commandes PID avec notre implémentation personnalisée
            cmd_left = self.pid_left.compute(speed_left)
            cmd_right = self.pid_right.compute(speed_right)
            # print('cmd_left:', cmd_left, 'cmd_right:', cmd_right)
            
            # Appliquer les commandes aux roues
            self.set_velocity_raw(cmd_left, cmd_right)
            
            # Sauvegarder les valeurs pour le prochain cycle
            self.prev_ticks_left = self.current_ticks_left
            self.prev_ticks_right = self.current_ticks_right
            self.last_time = current_time
            
            rospy.logdebug(f"Speed - Left: {speed_left:.2f}, Right: {speed_right:.2f}")
            rospy.logdebug(f"Command - Left: {cmd_left:.2f}, Right: {cmd_right:.2f}")

    def set_velocity_raw(self, left_wheel_velocity=0.0, right_wheel_velocity=0.0):
        # print("    [api] setting vel raw to ", left_wheel_velocity, ", ", right_wheel_velocity) 
        msg = WheelsCmdStamped()

        # Set message parameters
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "base_link"
        msg.vel_left = left_wheel_velocity
        msg.vel_right = right_wheel_velocity

        # Publish the message
        if not rospy.is_shutdown():
            self.commands_publisher.publish(msg)
            
    def set_velocity(self, linear_velocity=0.0, angular_velocity=0.0):
        # Convertir vitesse linéaire/angulaire en commandes de roues
        left_vel = linear_velocity - angular_velocity
        right_vel = linear_velocity + angular_velocity
        
        # Définir les setpoints des PIDs proportionnels à la vitesse demandée
        # Facteur de conversion à ajuster selon les caractéristiques du robot
        conversion_factor = 100  # ticks/s par unité de vitesse
        
        self.target_speed_left = left_vel * conversion_factor
        self.target_speed_right = right_vel * conversion_factor
        
        self.pid_left.setpoint = self.target_speed_left
        self.pid_right.setpoint = self.target_speed_right
        
        print(f"    [api] setting velocity setpoints - Left: {self.target_speed_left}, Right: {self.target_speed_right}")

if __name__ == "__main__":
    DuckieBotAPI(robot_name="gastone")
