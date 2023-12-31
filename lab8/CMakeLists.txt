cmake_minimum_required(VERSION 2.8.3)
project(lab_8)

add_compile_options(-std=c++14)
#include_directories("$ENV{HOME}/vnav_ws/devel/include")
#LIST(APPEND CMAKE_PREFIX_PATH "/opt/ros/$ENV{ROS_DISTRO};$ENV{HOME}/vnav_ws/devel")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  actionlib
  tf
  tf_conversions
  image_transport
  message_filters
  cv_bridge
  darknet_ros
  opencv3_catkin
  gflags_catkin
  glog_catkin
  gtsam_catkin
  pgo
)

# Boost - same requirement as gtsam
find_package(Boost 1.50 REQUIRED)

find_package(Eigen3 REQUIRED)
message(STATUS "Eigen Version: ${EIGEN3_VERSION_STRING} ${EIGEN3_VERSION}")

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES
 CATKIN_DEPENDS roscpp std_msgs opencv3_catkin gflags_catkin glog_catkin gtsam_catkin actionlib pgo 
)

include_directories(include ${Boost_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${catkin_INCLUDE_DIRS})

add_library(
  detectors
  src/yolo_detector.cpp
)

target_link_libraries(detectors ${catkin_LIBRARIES})

add_library(
  orb_feature_tracker
  src/feature_tracker.cpp 
  src/orb_feature_tracker.cpp
)

target_link_libraries(orb_feature_tracker ${catkin_LIBRARIES})

add_executable(deliverable_2 src/deliverable_2.cpp)
target_link_libraries(deliverable_2 detectors orb_feature_tracker ${Boost_LIBRARIES}  ${catkin_LIBRARIES} -lffi -ltiff)

add_executable(deliverable_4 src/deliverable_4.cpp)
target_link_libraries(deliverable_4 ${catkin_LIBRARIES})
