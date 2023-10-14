cmake_minimum_required(VERSION 3.0.2)
project(pgo)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
  gtsam_catkin
  message_generation
  opencv3_catkin
  tf
  tf_conversions
  roscpp
  rospy
  std_msgs
  geometry_msgs
)

## System dependencies are found with CMake's conventions
find_package(Boost 1.50 REQUIRED)
find_package(Eigen3 REQUIRED)

#msg files
add_message_files(
  FILES
  OptimizationRequest.msg
  OptimizationResult.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES
  CATKIN_DEPENDS roscpp std_msgs opencv3_catkin gtsam_catkin message_runtime
)

include_directories(include ${Boost_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${catkin_INCLUDE_DIRS})

add_library(optimizers src/pose_graph.cpp)

target_link_libraries(optimizers ${catkin_LIBRARIES}) 

add_executable(pgo_node src/pgo.cpp)
target_link_libraries(pgo_node optimizers ${Boost_LIBRARIES} -ltbbmalloc ${catkin_LIBRARIES})


## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
# target_link_libraries(${PROJECT_NAME}_node
#   ${catkin_LIBRARIES}
# )
