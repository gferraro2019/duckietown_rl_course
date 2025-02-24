CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

docker run -it \
  --network host -e ROS_MASTER_URI=http://paperino.local:11311 -e ROS_HOSTNAME=$(hostname -I | awk '{print $1}') \
  --gpus all \
  --env="DISPLAY" --env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  -v $CURRENT_DIR:/$(basename "$CURRENT_DIR") \
  --name indepth_rl_container \
  duckietown/dt-ros-commons:daffy-amd64  \
  bash

# Wait a moment to ensure the container is fully up
sleep 2

# Run the command inside the container
docker exec indepth_rl_container touch /$(basename "$CURRENT_DIR")/test.txt


