sudo docker build -t jdurrant/deeplig . 
sudo docker run --gpus all --ipc=host -it --rm -v $(realpath ../):/mnt jdurrant/deeplig
