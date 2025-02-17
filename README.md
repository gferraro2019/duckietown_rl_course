# Welcome to the DuckieTown Robots page
First of all, DuckieTown has its own shell that you can invoke with the command dts.

We have 2 robots whose usernames are: paperino, and gastone.

The Robots use ROS1.

# Instruction for RObot's use
1. connect the router
2. turn on the robot
3. after the wifi dongle start blinking, check if the robot is visible:
    `dts fleet discover`
4. to create an ssh conncection with a robot:
    `ssh duckie@paperino.local`
    `pwd: quackquack`
5. to check the robot status:
    http://paperino.local or http://gastone.local


# Let's pilot the robot manually
`dts duckiebot keyboard_control paperino`
`dts duckiebot keyboard_control gastone`

# To control the LDEs manually
`dts duckiebot led_control paperino`
`dts duckiebot led_control gastone`

# To create a DTProject, choose a template on github and follow the instructions:
    https://docs.duckietown.com/daffy/devmanual-software/beginner/dtproject/templates.html#project-templates     
 