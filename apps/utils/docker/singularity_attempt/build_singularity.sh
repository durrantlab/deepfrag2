# Unlikely to work on CRC. Try on bills, as root.

rm -f Singularity.sif  Singularity.def
pip3 install spython
spython recipe Dockerfile &> Singularity.def
sudo singularity build Singularity.sif Singularity.def

echo
echo "Now copy Singularity.sif to remote system, where you can use start_singularity.sh"
