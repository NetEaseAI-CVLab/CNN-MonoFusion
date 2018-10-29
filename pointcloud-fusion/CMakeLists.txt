cmake_minimum_required(VERSION 2.8)
project(CNN_MonoFusion)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

SET(CMAKE_BUILD_TYPE Release)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -W0 -bigobj")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W0 -bigobj")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
SET(DBUILD_SHARED true)

# PCL LIBRARY
# find_package(PCL 1.2 REQUIRED)
# include_directories(${PCL_INCLUDE_DIRS})
# link_directories(${PCL_LIBRARY_DIRS})
# add_definitions(${PCL_DEFINITIONS})

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV 3.0 QUIET)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()

find_package(Eigen3 REQUIRED)

# set(KINECT_LIBRARIRY "C:/Program Files/Microsoft SDKs/Kinect/v2.0_1409/Lib/x64/Kinect20.lib")
set(KINECT_INCLUDE "C:/Program Files/Microsoft SDKs/Kinect/v2.0_1409/inc")
set(PANGOLIN_INCLUDE ${PROJECT_SOURCE_DIR}/Thirdparty/Pangolin/include)
set(PANGOLIN_LIB ${PROJECT_SOURCE_DIR}/Thirdparty/Pangolin/pangolin.lib)
set(PANGOLIN_external ${PROJECT_SOURCE_DIR}/Thirdparty/Pangolin/external)
set(PANGOLIN_BUILD_GENERATE ${PROJECT_SOURCE_DIR}/Thirdparty/Pangolin/build_generate)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${PROJECT_SOURCE_DIR}/src/dense_map
${EIGEN3_INCLUDE_DIR}
${PANGOLIN_INCLUDE}
${KINECT_INCLUDE}
${PANGOLIN_external}/glew/include
${PANGOLIN_external}/libjpeg/include
${PANGOLIN_external}/libpng/include
${PANGOLIN_external}/zlib/include
${PANGOLIN_BUILD_GENERATE}/include
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)


file(GLOB ORBSLAM_SRCS src/*.h include/*.h src/*.cpp src/*.cc)
file(GLOB DenseMap_SRCS src/dense_map/*.h src/dense_map/*.cpp src/dense_map/*.cc)

add_library(${PROJECT_NAME} STATIC
${ORBSLAM_SRCS}
${DenseMap_SRCS}
)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/Release/DBoW2.lib
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/bin/Release/g2o.lib
${KINECT_LIBRARIRY}
${PANGOLIN_external}/libpng/lib/libpng16_static.lib
${PANGOLIN_external}/zlib/lib/zlibstatic.lib
${PANGOLIN_external}/libjpeg/lib/jpeg.lib
${PANGOLIN_BUILD_GENERATE}/Release/pangolin.lib
${PANGOLIN_external}/glew/lib/glew.lib
glu32.lib
opengl32.lib
# ${PCL_LIBRARIES}
)


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/rgbd_monodepth)
file(GLOB SRCS Examples/rgbd_monodepth/*.h Examples/rgbd_monodepth/*.cpp Examples/rgbd_monodepth/*.cc)
add_executable(rgbd_monodepth Examples/rgbd_monodepth/rgbd_monodepth.cc ${SRCS})
target_link_libraries(rgbd_monodepth ${PROJECT_NAME})


