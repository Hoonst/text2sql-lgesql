wget https://github.com/Kitware/CMake/releases/download/v3.18.2/cmake-3.18.2.tar.gz 
tar -xvf cmake-3.18.2.tar.gz 

cd cmake-3.18.2
./bootstrap
make
make install