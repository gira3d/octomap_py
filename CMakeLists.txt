cmake_minimum_required(VERSION 2.9)
project(octomap_py)

set(octomap_DIR "${CMAKE_SOURCE_DIR}/../octomap/lib/cmake/octomap")

find_package(pybind11 REQUIRED)
find_package(octomap REQUIRED)
find_package(Eigen3 REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(
  ${OCTOMAP_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

# pybind modules
pybind11_add_module(octomap_py src/octomap_py.cpp)
target_link_libraries(octomap_py PRIVATE ${PYTHON_LIBRARIES} ${OCTOMAP_LIBRARIES})
set_target_properties(octomap_py PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_INSTALL_PREFIX})

install(TARGETS DESTINATION ${CMAKE_INSTALL_PREFIX})
