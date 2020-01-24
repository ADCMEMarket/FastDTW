cmake_minimum_required(VERSION 3.11)

project(fast_dtw)
set (CMAKE_CXX_STANDARD 11)

# set(CMAKE_BUILD_TYPE Debug)

# set(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS}")
# find_package(catkin REQUIRED COMPONENTS   roscpp   rospy )

# catkin_package()

include_directories(include)
set(SOURCES src/ColMajorCell.cpp src/FastDTW.cpp src/SearchWindow.cpp src/WarpPath.cpp)
add_library(fast_dtw ${SOURCES} )


add_executable(example_fast_dtw examples/example_fast_dtw.cpp )
target_link_libraries( example_fast_dtw fast_dtw ${catkin_LIBRARIES} )
