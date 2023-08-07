#!/bin/sh

user_id=$(id -u)        # User id number
group_id=$(id -g)       # User group id number

docker build --no-cache\
  --build-arg USER_ID=$user_id \
  --build-arg GROUP_ID=$group_id \
  --file Dockerfile \
  -t igastesi/pytorch_container .
