sudo docker build \
    -t jdurrant/deeplig . 

# --shm-size="2g" \

sudo docker run \
    --gpus all \
    -it --rm \
    --shm-size="2g" \
    --ipc=host \
    -v $(realpath ../../../):/mnt jdurrant/deeplig

# Note that --ipc=host means host shared memory same as docker image. So
# shm-size above doesn't really do anything...
