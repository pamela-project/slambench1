cmake_minimum_required(VERSION 2.8)

set(appname kfusion)

find_package(TooN   REQUIRED)
find_package(OpenCL )
find_package(CUDA   )
find_package(GLUT   )
find_package(OpenGL )
find_package(OpenNI2)
find_package(PAPI)

find_package(PkgConfig)
find_package(Qt5Core)
find_package(Qt5Widgets)
find_package(Qt5Gui)
find_package(Qt5OpenGL)
find_package(Qt5PrintSupport)

include_directories(./include ${TOON_INCLUDE_PATH} ./thirdparty)


if (CUDA_FOUND) 
 SET( CUDA_PROPAGATE_HOST_FLAGS OFF )
endif(CUDA_FOUND)

 # ---- PREPARE COMMON DEPENDENCIES  ------------ 
if(APPLE)
    set(common_libraries     stdc++ pthread)
else(APPLE)
    set(common_libraries     rt stdc++ pthread)
endif(APPLE)


set(common_compile_flags "")


 # ----------------- CPP VERSION ----------------- 

add_library(${appname}-cpp  src/cpp/kernels.cpp)
target_link_libraries(${appname}-cpp   ${common_libraries})	
SET_TARGET_PROPERTIES(${appname}-cpp PROPERTIES COMPILE_FLAGS " -D__KFUSION_VERSION__=\\\"cpp\\\" ")
add_version(${appname} cpp "" "")

 # ----------------- OPENMP VERSION ----------------- 

add_library(${appname}-openmp  src/cpp/kernels.cpp)
target_link_libraries(${appname}-openmp   ${common_libraries})	
SET_TARGET_PROPERTIES(${appname}-openmp PROPERTIES COMPILE_FLAGS "-fopenmp -D__KFUSION_VERSION__=\\\"openmp\\\" ")
add_version(${appname} openmp "-fopenmp" "-fopenmp")


 #  ----------------- OCL VERSION ----------------- 
 
if (OPENCL_FOUND) 
   include_directories(${OPENCL_INCLUDE_DIRS})
    add_library(${appname}-opencl  src/opencl/kernels.cpp src/opencl/common_opencl.cpp)
    target_link_libraries(${appname}-opencl   ${common_libraries} ${OPENCL_LIBRARIES})	
    SET_TARGET_PROPERTIES(${appname}-opencl PROPERTIES COMPILE_FLAGS " -D__KFUSION_VERSION__=\\\"opencl\\\" ")
    add_version(${appname} opencl "" "" )
endif(OPENCL_FOUND)


#  ----------------- thirdparty ----------------- 

add_subdirectory( thirdparty    )


