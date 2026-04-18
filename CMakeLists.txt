cmake_minimum_required(VERSION 3.8)
project(optical_flow_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(mavros_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
# added this
add_executable(farneback_flow_node src/opt_flow.cpp)

# Link the libraries to your node
target_include_directories(farneback_flow_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_link_libraries(farneback_flow_node
  ${OpenCV_LIBRARIES}
)

ament_target_dependencies(farneback_flow_node
  rclcpp
  sensor_msgs
  geometry_msgs
  mavros_msgs
  cv_bridge
)

install(TARGETS
  farneback_flow_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()
