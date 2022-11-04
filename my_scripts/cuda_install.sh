#!/bin/bash
## This gist contains instructions about cuda v10.1 and cudnn 7.6 installation in Ubuntu 18.04 for Tensorflow 2.1.0

### steps ####
# verify the system has a cuda-capable gpu
# download and install the nvidia cuda toolkit and cudnn
# setup environmental variables
# verify the installation
###

### If you have previous installation remove it first. 
apt-get purge nvidia*
apt remove nvidia-*
rm /etc/apt/sources.list.d/cuda*
apt-get autoremove && apt-get autoclean
rm -rf /usr/local/cuda*


### to verify your gpu is cuda enable check
lspci | grep -i nvidia

### gcc compiler is required for development using the cuda toolkit. to verify the version of gcc install enter
gcc --version

# system update
apt-get update
apt-get upgrade


# install other import packages
apt-get install g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev


# first get the PPA repository driver
add-apt-repository ppa:graphics-drivers/ppa
apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" | tee /etc/apt/sources.list.d/cuda.list
apt-get update

# installing CUDA-11.0
apt-get -o Dpkg::Options::="--force-overwrite" install cuda-11-0 cuda-drivers


# setup your paths
echo 'export PATH=/usr/local/cuda-11.0/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
ldconfig

# install cuDNN v8.0.3
# in order to download cuDNN you have to be regeistered here https://developer.nvidia.com/developer-program/signup
# then download cuDNN v8.0.3 for cuda 11.0 form https://developer.nvidia.com/cudnn

wget http://people.cs.uchicago.edu/~kauffman/nvidia/cudnn/cudnn-11.0-linux-x64-v8.0.3.33.tgz
CUDNN_TAR_FILE="cudnn-11.0-linux-x64-v8.0.3.33.tgz"
tar -xzvf ${CUDNN_TAR_FILE}

# Copy the following files into the cuda toolkit directory.
cp -P cuda/include/cudnn* /usr/local/cuda-11.0/include
cp -P cuda/lib64/libcudnn* /usr/local/cuda-11.0/lib64/
chmod a+r /usr/local/cuda-11.0/lib64/libcudnn*

# Finally, to verify the installation, check
nvidia-smi
nvcc -V

# pip install torch==1.4.0+cu100 -f https://download.pytorch.org/whl/torch_stable.html
