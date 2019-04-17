###############################################
# environment setup for reinforcement learning
# should work on ubuntu 18.04
###############################################

# make sure we are up-to-date
apt-get update
sudo apt-get --assume-yes install tmux

###############################################
# anaconda
###############################################
# not needed... doing this all with pip...
# curl -O https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
# bash Anaconda3-2019.03-Linux-x86_64.sh -b -p ~/anconda3

###############################################
# pytorch
###############################################
# conda install -c pytorch pytorch 

###############################################
# tensorflow
###############################################
# conda install -c conda-forge tensorflow 

###############################################
# vizdoom (from https://github.com/mwydmuch/ViZDoom/blob/master/doc/Building.md)
###############################################

# ZDoom dependencies
sudo apt-get install build-essential zlib1g-dev libsdl2-dev libjpeg-dev \
nasm tar libbz2-dev libgtk2.0-dev cmake git libfluidsynth-dev libgme-dev \
libopenal-dev timidity libwildmidi-dev unzip

# Boost libraries
sudo apt-get install libboost-all-dev

# Python 3 dependencies
sudo apt-get install python3-dev python3-pip
pip3 install numpy

# install vizdoom
sudo pip3 install vizdoom

# other stuff
pip3 install jupyter
pip3 install sklearn
pip3 install scikit-image
pip3 install tqdm

pip3 install torch torchvision
