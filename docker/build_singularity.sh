# Unlikely to work on CRC. Try on bills, as root.

pip3 install spython
spython recipe Dockerfile &> Singularity.def
sudo singularity build Singularity.sif Singularity.def
