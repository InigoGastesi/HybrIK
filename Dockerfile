# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:21.08-py3
ENV DEBIAN_FRONTEND noninteractive
# Install linux packages
RUN apt update && apt install -y zip htop screen libgl1-mesa-glx build-essential libsm6 libxext6 libxrender-dev ffmpeg

# Install python dependencies
RUN python -m pip install --upgrade pip
# Create working directory
RUN mkdir -p /usr/src/app
RUN mkdir -p /usr/src/data
WORKDIR /usr/src/app

## User id definition
# Define two arguments to be called by the build command
ARG USER_ID
ARG GROUP_ID
# Add a user with the data defined in the arguments
RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user
