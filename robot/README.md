# Welcome to the DuckieTown Robots page
First of all, DuckieTown has its own shell that you can invoke with the command `dts`.

We have 2 robots whose usernames are: paperino, and gastone.

The Robots use ROS1.

# Instruction for Robot's use
1. Connect the router to the laptop with the USB and LAN cables.
2. Turn on the robot.
3. After the wifi dongle start blinking, check if the robot is visible:<br>
    `dts fleet discover`
4. To create an ssh conncection with a robot:<br>
    `ssh duckie@paperino.local` or `ssh duckie@gastone.local` <br>
    `pwd: quackquack`
5. To check the robot's status:<br>
    http://paperino.local or http://gastone.local


# Let's pilot the robot manually
`dts duckiebot keyboard_control paperino`<br>
`dts duckiebot keyboard_control gastone`

# To control the LEDs manually
`dts duckiebot led_control paperino`<br>
`dts duckiebot led_control gastone`

# To create a DTProject, choose a template on github and follow the instructions:
    https://docs.duckietown.com/daffy/devmanual-software/beginner/dtproject/templates.html#project-templates     
 
# To visualize the Topic list
1. [Optional] if you use the ros:noetic Docker image, start the container:
`docker run --network host -it ros:noetic`
2. run the following command wherever you have ros1 intalled:
`export ROS_MASTER_URI=http://192.168.1.173:11311`

# to connect with ros contained on the duckie bot:
1. `export IP_LOCAL= the ip of the local machine`
2. `docker run -it --rm --network host --add-host paperino.local:192.168.1.10 -e ROS_MASTER_URI=http://paperino.local:11311 -e ROS_HOSTNAME=$IP_LOCAL duckietown/dt-ros-commons:daffy-amd64 bash`
