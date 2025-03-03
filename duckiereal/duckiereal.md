# Duckiebot in Real


# Objective: Complete three laps of the circuit as quickly as possible with the real robot.

The aim of this course is to give you the opportunity to test a reinforcement learning algorithm on a real robot. You won't be evaluated completely on the technical result obtained, as what you're trying to achieve is difficult (see readme file for evaluation). However, your reasoning, approach and investment will condition the grade and also your chances of success. 

In the simulation section, you will have been able to work with the simulated version of the robots, which is easy to query and, above all, very sample efficient (ability to generate data quickly in this precise context). The real robot is not, of course. So you're going to have to think about the different possibilities available to you to get a model that works on the real robot. 

We recommend that you start from one of the following three directions: 
* [Domain_randomization](./domain_randomization/domain_randomization.md) 
* [Imitation_learning](./imitation_learning/behavioral_cloning.md) 
* [Model predictive control](./mpc/mpc.md)

The aim is not to follow these algorithms to the letter, but rather to draw inspiration from them. You can hybridize methods between them and even start from other approaches in which you believe. 
We leave you free to make your own technical choices. The aim is for you to propose a rational scientific approach, which you will present to us at the end with results that confirm or refute your hypotheses.

















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

First, install nvidia-docker2 on your machine to use your gpu in the container:

```shell
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
  && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
  && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

Then you can create a container with the following command:



```shell
docker run -it \
  --network host -e ROS_MASTER_URI=http://paperino.local:11311 -e ROS_HOSTNAME=$(hostname -I | awk '{print $1}') \
  --gpus all \
  --env="DISPLAY" --env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  -v "/location/":/destination \
  --name container_name \
  duckietown/dt-ros-commons:daffy-amd64  \
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

### Setup the container

Run the following commands in your container to make it up to date:
```shell
apt update &&
  pip3 install --upgrade pip
```

If you want to use matplotlib (can be done at any moment):
```shell
apt install -y python3-tk
```

You will need torch at some point, so install it when you have some time because it could be a bit long:
```shell
pip3 install numpy torch
```

## Duckietown-shell (dts)

Duckietown-shell (dts) contain usefull tools. 
 - Scan duckiebots on the network
 - Control duckiebots with your keyboard

Install **dts** on your local machine if you want to use them:

> **WARNING** `echo "export PATH=~/.local/bin:${PATH}" >> ~/.bashrc &&` add permanently dts to your path. Don't forget 
> to remove it later or simply run `export PATH=~/.local/bin:${PATH}` in any terminal you launch.

```shell
pip3 install --no-cache-dir --user --upgrade duckietown-shell &&
  echo "export PATH=~/.local/bin:${PATH}" >> ~/.bashrc &&
  source ~/.bashrc
```

## Setup the robot

At his point, you should have a container working, or a working ros1 on your computer.
You will need "dts" (duckietown-shell) to interact with the robot, so make sure you completed the above section 
without issue.

**steps to use the robot:**
1. connect the router
2. turn on the robot
3. after the wifi dongle start blinking, check if the robot is visible:
```bash
dts fleet discover
```

> **WARNING:** For all the commands bellow, replace "paperino" by your robot name. There is only two robots: "paperino" and "gastone".
The name of the robot is writen on the top of each one.

### \> To create an ssh connection with a robot (password: quackquack):

   1. get the robot ip **from your machine**:
       ```shell
       ssh duckie@paperino.local
       # password: quackquack
       ```
       Then on the duckiebot
       ```shell
       hostname -I  # Take the first IP
       ```
   2. put this ip in your container:
       ```shell
       echo "<robot_ip> paperino.local" >> /etc/hosts 
       ```
   3. test ssh connection **from the container**:
       ```shell
       ssh duckie@paperino.local
       # password: quackquack
       ```

### \> To go on the robot webpage:
It contains useful information like robot camera rendering;
 - Paperino: http://paperino.local
 - Gastone: http://gastone.local

### \> To control the robot with keyboard (arrow keys):
```bash
dts duckiebot keyboard_control paperino
```
then wait for the interface to show up.

To control the LDEs manually:
```bash
dts duckiebot led_control paperino
```

### \> To create a DTProject, choose a template on github and follow the instructions:
https://docs.duckietown.com/daffy/devmanual-software/beginner/dtproject/templates.html#project-templates     

## Before to use the robot api code

Verify if everything is working:
 - Check if the robot is connected using dts or ssh 🠉🠉🠉🠉
   - Check if the ros topics are up:

     1. get the robot ip (check the section "ssh connection" above)
     2. Set up the ros master host:
         ```shell
         export ROS_MASTER_URI=http://<robot_ip>:11311
         ```
     3. Then you can list the topics:
         ```shell
         rostopic list
         ```
     4. Makes sure the necessary topics are up (if nothing is shown, you're good to go. Otherwise, good luck! (call me))
         ```shell
         topics=(
           "/paperino/camera_node/image/compressed"
           "/paperino/wheels_driver_node/wheels_cmd"
         )
         # Check each topic
         for topic in "${topics[@]}"; do
           if ! echo "$(rostopic list)" | grep -q "$topic"; then
             echo "Missing topic: $topic"
           fi
         done
         ```

# to connect with ros contained on the duckie bot:
1. `export IP_LOCAL= the ip of the local machine`
2. `docker run -it --rm --network host --add-host paperino.local:192.168.1.10 -e ROS_MASTER_URI=http://paperino.local:11311 -e ROS_HOSTNAME=$IP_LOCAL duckietown/dt-ros-commons:daffy-amd64 bash`