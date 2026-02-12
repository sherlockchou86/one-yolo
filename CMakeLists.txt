cmake_minimum_required(VERSION 3.10)
project(one-yolo VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fPIC -w -fdiagnostics-color=always -pthread")

option(BUILD_WITH_ORT "enable onnxruntime?" OFF)
option(BUILD_WITH_OVN "enable openvino?" OFF)
option(BUILD_WITH_TRT "enable tensorrt?" OFF)
option(BUILD_WITH_RKN "enable rknn?" OFF)

# OpenCV required
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("include")

##########################
# build one-yolo library #
##########################
file(GLOB BASE_SRCS "src/*.cpp")
file(GLOB_RECURSE TRACK_SRCS "src/track/*.cpp")
file(GLOB_RECURSE ORT_SRCS "src/ort/*.cpp")
file(GLOB_RECURSE OVN_SRCS "src/ovn/*.cpp")
list(APPEND ONE_YOLO_DEPEND_SRCS ${BASE_SRCS} ${TRACK_SRCS})
list(APPEND ONE_YOLO_DEPEND_LIBS ${OpenCV_LIBS})

# enable onnxruntime, 
# make sure onnxruntime installed and include & libs path could be found by cmake
if(BUILD_WITH_ORT)
    add_definitions(-DBUILD_WITH_ORT)
    list(APPEND ONE_YOLO_DEPEND_SRCS ${ORT_SRCS})
    list(APPEND ONE_YOLO_DEPEND_LIBS onnxruntime)
endif()
# enable openvino, 
# make sure openvino installed and include & libs path could be found by cmake
if(BUILD_WITH_OVN)
    find_package(OpenVINO REQUIRED)
    add_definitions(-DBUILD_WITH_OVN)
    list(APPEND ONE_YOLO_DEPEND_SRCS ${OVN_SRCS})
    list(APPEND ONE_YOLO_DEPEND_LIBS openvino::runtime)
endif()

add_library(${PROJECT_NAME} SHARED ${ONE_YOLO_DEPEND_SRCS})
target_link_libraries(${PROJECT_NAME} ${ONE_YOLO_DEPEND_LIBS})

##########################
# build one-yolo samples #
##########################
add_subdirectory(samples)