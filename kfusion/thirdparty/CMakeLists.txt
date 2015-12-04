cmake_minimum_required(VERSION 2.8)

if (OPENNI2_FOUND)

     include_directories(${OPENNI2_INCLUDE_PATHS})
    add_executable(oni2raw oni2raw.cpp)
    target_link_libraries(oni2raw ${OPENNI2_LIBRARIES})

endif(OPENNI2_FOUND)

if (OPENCL_FOUND)
 
    include_directories(${OPENCL_INCLUDE_DIRS})
    add_library(oclwrapper SHARED OCLWrapper.cpp)
    target_link_libraries(oclwrapper ${OPENCL_LIBRARIES})

endif(OPENCL_FOUND)

include_directories(.)

add_library(lodepng SHARED lodepng.cpp)
add_executable(scene2raw scene2raw.cpp)
target_link_libraries(scene2raw lodepng)
