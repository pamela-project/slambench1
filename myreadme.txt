mkdir build
cd build

export CXX=g++ && make kfusion-benchmark-cpp
make kfusion-qt-cpp
make kfusion-benchmark-cpp
./kfusion/kfusion-benchmark-cpp -i ../living_room_traj2_loop.raw -s 4.8 -p 0.34,0.5,0.24 -z 4 -c 2 -r 1 -k 481.2,480,320,240

export CXX=syclcc && cmake ..
make kfusion-qt-sycl
make kfusion-benchmark-sycl
./kfusion/kfusion-benchmark-sycl -i ../living_room_traj2_loop.raw -s 4.8 -p 0.34,0.5,0.24 -z 4 -c 2 -r 1 -k 481.2,480,320,240
