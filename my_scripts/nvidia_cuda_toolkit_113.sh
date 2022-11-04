wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda-repo-ubuntu1804-11-3-local_11.3.0-465.19.01-1_amd64.deb
dpkg -i cuda-repo-ubuntu1804-11-3-local_11.3.0-465.19.01-1_amd64.deb
apt-key add /var/cuda-repo-ubuntu1804-11-3-local/7fa2af80.pub
apt-get update
apt-get -y install cuda


apt install nvidia-cuda-toolkit-11.3
ln -s /usr/local/cuda-11.3 /usr/local/cuda