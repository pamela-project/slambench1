# GLUT, Qt and X via Aptitude

sudo apt-get install qt5-default freeglut3-dev libxmu-dev libxi-dev

# TooN

cd /code/ext
git clone https://github.com/edrosten/TooN.git
cd TooN/
./configure --prefix=$PWD/install
make
make install
# export INCLUDE_PATH=$PWD/install/include:$INCLUDE_PATH

# SLAMBench

cd ~/projects
hg clone https://bitbucket.org/pgk/sycl-slambench
mkdir build
cd build

export CXX=g++ && cmake -DTOON_INCLUDE_PATH=$HOME/code/ext/TooN/install/include ..
# scene2raw can be used to construct the living_room_traj2_loop.raw file fromxi
# the tar.zip.
make scene2raw
make kfusion-benchmark-cpp
make kfusion-qt-cpp
./kfusion/kfusion-benchmark-cpp -i ../../sycl-slambench_old/living_room_traj2_loop.raw -s 4.8 -p 0.34,0.5,0.24 -z 4 -c 2 -r 1 -k 481.2,480,320,240

export CXX=syclcc && cmake -DTOON_INCLUDE_PATH=$HOME/code/ext/TooN/install/include ..
make kfusion-benchmark-sycl
make kfusion-qt-sycl
./kfusion/kfusion-benchmark-sycl -i ../living_room_traj2_loop.raw -s 4.8 -p 0.34,0.5,0.24 -z 4 -c 2 -r 1 -k 481.2,480,320,240
