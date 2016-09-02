cmake_minimum_required(VERSION 2.8)

project(pamela_testcases)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O3 -std=gnu++11   ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -std=gnu++11  ")

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O3      -Wall -Wno-unknown-pragmas -Wextra -Werror ")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3  -Wall -Wno-unknown-pragmas -Wextra -Werror ")

if(APPLE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D__APPLE__")
endif(APPLE)

# Set the directory for PAPI_PREFIX if you have trouble finding PAPI
# think this is the right place or else do it in the FindPAPI.cmake
# this is because I needed Haswell support and the installed version in 
# /usr was 5.3.0 and the defualt installation via download of the latest version
# was 5.4.0 and this went into /usr/local
set(PAPI_PREFIX "/usr/local") 


 # ---- PREPARE COMMON DEPENDENCIES  ------------ 


function(add_version appname version compile_flags link_flags)

if(APPLE)
    set(main_common_libraries     ${link_flags} stdc++ pthread)
else(APPLE)
    set(main_common_libraries     ${link_flags} rt stdc++ pthread)
endif(APPLE)

set(main_common_compile_flags ${compile_flags})
if (PAPI_FOUND)
        include_directories(${PAPI_INCLUDE_DIRS})
	set(main_common_compile_flags "-DPAPI_MONITORING ${main_common_compile_flags}")
	set(main_common_libraries   "${main_common_libraries} ${PAPI_LIBRARIES}")

endif(PAPI_FOUND)

if (OPENNI2_FOUND)
    include_directories(${OPENNI2_INCLUDE_PATHS})
    set(main_common_libraries ${main_common_libraries} ${OPENNI2_LIBRARIES})
    set(main_common_compile_flags "-DDO_OPENNI ${main_common_compile_flags}")
endif(OPENNI2_FOUND)



  #  ------ BENCHMARK --------
  add_executable(${appname}-benchmark-${version} src/benchmark.cpp src/PowerMonitor.cpp)
  target_link_libraries(${appname}-benchmark-${version} ${appname}-${version} ${main_common_libraries})
  SET_TARGET_PROPERTIES(${appname}-benchmark-${version}
                             PROPERTIES COMPILE_FLAGS "${main_common_compile_flags}")


#  ------ MAIN --------
  if (GLUT_FOUND)
    add_executable(${appname}-main-${version} src/mainQt.cpp src/PowerMonitor.cpp src/reader.cpp)
      SET_TARGET_PROPERTIES(${appname}-main-${version}
                             PROPERTIES COMPILE_FLAGS "${main_common_compile_flags}")


if(APPLE)
    target_link_libraries(${appname}-main-${version} ${appname}-${version} ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES}  ${main_common_libraries} )
else(APPLE)
    target_link_libraries(${appname}-main-${version} ${appname}-${version} ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES}  ${main_common_libraries} GL )
endif(APPLE)
  endif(GLUT_FOUND)
  #  ------ QT --------
  if (Qt5OpenGL_FOUND)

    file(GLOB qt_srcs qt/*)

    QT5_ADD_RESOURCES( ResourceSources  "qt/images.qrc")
    set(qt_srcs ${qt_srcs} ${ResourceSources} include/PowerMonitor.h thirdparty/perfstats.h)
    SET(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "")
    SET(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS "")
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_INCLUDE_CURRENT_DIR ON)

    include_directories(qt)
    add_definitions( -D_FILE_OFFSET_BITS=64 -fPIC )

    add_executable(${appname}-qt-${version} src/mainQt.cpp src/PowerMonitor.cpp ${qt_srcs} src/reader.cpp)

    ### qt5_use_modules is not available with cmake 2.8.7 ###
    #qt5_use_modules(qtgui-${version} Core Gui Widgets OpenGL PrintSupport)

    add_definitions(${Qt5Widgets_DEFINITIONS})
    add_definitions(${Qt5OpenGL_DEFINITIONS})
    add_definitions(${Qt5PrintSuppoer_DEFINITIONS})
    include_directories(${Qt5Widgets_INCLUDE_DIRS})
    include_directories(${Qt5OpenGL_INCLUDE_DIRS})
    include_directories(${Qt5PrintSupport_INCLUDE_DIRS})
    include_directories(${Qt5Gui_INCLUDE_DIRS})
    
    SET_TARGET_PROPERTIES(${appname}-qt-${version} PROPERTIES COMPILE_FLAGS "${main_common_compile_flags} -std=c++0x  -D__QT__")

if(APPLE)
    target_link_libraries(${appname}-qt-${version} Qt5::Core Qt5::Gui Qt5::Widgets Qt5::OpenGL Qt5::PrintSupport 
			    ${main_common_libraries} ${appname}-${version}  ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES}   )  
else(APPLE) 
    target_link_libraries(${appname}-qt-${version} Qt5::Core Qt5::Gui Qt5::Widgets Qt5::OpenGL Qt5::PrintSupport 
			    ${main_common_libraries} ${appname}-${version}  ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES}  GL )  
endif(APPLE)
    
    
  endif(Qt5OpenGL_FOUND)

endfunction()


add_subdirectory( kfusion )


