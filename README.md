# Welcome to the DuckieTown Robots page
First of all, DuckieTown has its own shell that you can invoke with the command dts.

We have 2 robots whose usernames are: paperino, and gastone.

The Robots use ROS1.


# Setup

ROS 1 is REQUIRED to communicate with the robot.
You can:
 - Use ros1 on your machine to communicate with it,
 - If your system can't run ros 1 (like ubuntu 22 or ubuntu 24 apparently), create a container with ros1, and use it to 
run your code.

## If you have a distribution that do not support ros1 (ubuntu 22/24 for example)

Create a docker container with ubuntu 20.04 and ros noetic, and use it to run your code.
We recommend using the duckietown container as it already have a lot of useful tools installed.
We also recommand to make sure that:
 - The container recognise your screen so you can plot stuff directly with matplotlib,
 - The container recognise your gpu so you can use cuda,
 - Your code is mounted as a volume inside your container. You can run your code in your container using your favourite 
ide, but you will need to tell it to source ros. (You can look how do so, but I found easier to mount the code.) 

```shell
docker run -it \
  --network host --add-host paperino.local:192.168.1.10 -e ROS_MASTER_URI=http://paperino.local:11311 -e ROS_HOSTNAME=$IP_LOCAL \
  --gpus all \
  --env="DISPLAY" --env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  -v "/location/":/destination \
  --name container_name \
  duckietown/dt-ros-commons:daffy-amd64  \
  bash
```

```shell
docker run -it \
  --network host --add-host paperino.local:192.168.1.10 -e ROS_MASTER_URI=http://paperino.local:11311 -e ROS_HOSTNAME=$IP_LOCAL \
  --gpus all \
  --env="DISPLAY" --env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  -v "$HOME/computing/containers_workdirs/donaldpool/":/workdir \
  --name test_cont \
  duckietown/dt-ros-commons:daffy-amd64 \
  bash
```

Explanation per line:
 -  `--network ...` is to make sure the docker container can communicate with the duckiebot (don't ask me how I just copy-pasted it).
 - `--gpus all` is to make sure the container can use your gpus for computation.
 - `--env="DISPLAY" ...` if for plot on your computer screen.
 - `-v "/location/":/destination` mount the local directory "/location/" as a volume in your container, as a directory name "/destination".
Replace the location and destination of course, make sure to put absolute path for both, idk if "$HOME" or "~" work 
just test it if you want or put "/home/..." instead.
Of course, modify anything inside "/location/" will modify it into "/destination" in the container, which will not be the case with a copy.
 - `--name container_name ` set the name of your container. Choose whatever you want.
 - `duckietown/dt-ros-commons:daffy-amd64 ` is the TAG of the IMAGE we recommend to use. It will be downloaded if you don't have it locally.
 - `bash` is the command to run at the creation of the container.

> **NB**: If you want to plot stuff on your screen, you will have to enter the command `xhost +local:` on your computer, at any moment but before to run plt.show() on your container.
> You can also remove the `--env="DISPLAY" ...` line because it is not necessary otherwise.

This will create a container from the used image. The container can then be closed.

You can check your container status with:
```shell
docker ps -a --format "table {{.Names}}\t{{.Status}}" | grep <YOUR_CONTAINER_NAME>
```

If the status is "Exited", you should start it with:
```shell
docker start -i <YOUR_CONTAINER_NAME>
```

If the status is "Up", you should ask it to run a shell **interactively** (using bash which is the default shell in this container).
```shell
docker exec -it test_cont /bin/bash
```

## Setup the robot

At his point, you should have a container working, or a working ros1 on your computer.
You will need "dts" (duckietown-shell) to interact with the robot.
If you used the container image proposed in the previous section, it is already installed in the container.
Otherwise, check how to install it [here](https://docs.duckietown.com/daffy/opmanual-duckiebot/setup/setup_laptop/setup_dt_shell.html).

**steps to use the robot:**
1. connect the router
2. turn on the robot
3. after the wifi dongle start blinking, check if the robot is visible:
```bash
dts fleet discover
```

From here, there is a lot of things you can test.
For all the commands bellow, replace "paperino" by your robot name. There is only two robots: "paperino" and "gastone".
The name of the robot is writen on the top of each one.

To create an ssh connection with a robot (password: quackquack):
```bash
ssh duckie@paperino.local
```

To go on the robot webpage:
 - Paperino: http://paperino.local
 - Gastone: http://gastone.local

To control the robot with keyboard (arrow keys):
```bash
dts duckiebot keyboard_control paperino
```
then wait for the interface to show up.

To control the LDEs manually:
```bash
dts duckiebot led_control paperino
```

To create a DTProject, choose a template on github and follow the instructions:
https://docs.duckietown.com/daffy/devmanual-software/beginner/dtproject/templates.html#project-templates     
 