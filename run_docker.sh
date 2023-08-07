#!/bin/sh
#sudo docker run --rm -it --userns="host" /
#xhost +

## Select device

user_id=$(id -u)        # User id number
group_id=$(id -g)       # User group id number
container_name="igashybrik"

docker run -it --rm -u $user_id:$group_id \
    --gpus '"device=4,5"' --shm-size 8G \
    -e DISPLAY=$DISPLAY --name=$container_name --net="host" \
    -v /tmp/.X11-unix/:/tmp/.X11-unix --volume="$HOME/.Xauthority:/root/.Xauthority:rw" \
    -v /home/VICOMTECH/igastesi/SURFER/HybrIK/:/usr/src/app/ \
    -v /home/VICOMTECH/igastesi/SURFER/data/:/usr/src/data/ \
    igastesi/pytorch_container bash

