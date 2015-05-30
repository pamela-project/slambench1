cmake_minimum_required(VERSION 2.8)

set(appname kfusion)

find_package(TooN   REQUIRED)
find_package(OpenCL )
find_package(SYCL   REQUIRED)
find_package(GLUT   )
find_package(OpenGL )
find_package(OpenNI2)

find_package(PkgConfig)
find_package(Qt5Core)
find_package(Qt5Widgets)
find_package(Qt5Gui)
find_package(Qt5OpenGL)
find_package(Qt5PrintSupport)

include_directories(../include ${TOON_INCLUDE_PATHS} ../thirdparty)


 # ---- PREPARE COMMON DEPENDENCIES  ------------ 
if(APPLE)
    set(common_libraries     stdc++ pthread)
else(APPLE)
    set(common_libraries     rt stdc++ pthread)
endif(APPLE)


set(common_compile_flags "")

 #  ----------------- SYCL VERSION ----------------- 

message("SYCL was found: ${SYCL_CC}/syclone")
add_library(${appname}-sycl  ../src/sycl/kernels.cpp)
target_link_libraries(${appname}-sycl   ${common_libraries})	
add_version(${appname} sycl "" "") # Paths 

