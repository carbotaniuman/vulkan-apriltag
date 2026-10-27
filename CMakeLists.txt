cmake_minimum_required(VERSION 3.22)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

project(vulktan_apriltag VERSION 1.0 LANGUAGES C CXX)

include(cmake/Shaders.cmake)

find_package(Vulkan 1.3 REQUIRED)
find_package(Threads REQUIRED)

set(BUILD_PYTHON_WRAPPER OFF CACHE BOOL "" FORCE)
include(FetchContent)
FetchContent_Declare(
  apriltag
  GIT_REPOSITORY https://github.com/carbotaniuman/apriltag.git
  GIT_TAG        04ebca938bb16bd453965ddfdf30554745245045
)
FetchContent_MakeAvailable(apriltag)

add_library(apriltag_vk STATIC
  src/support/buffer.cpp
  src/support/context.cpp
  src/support/device.cpp
  src/support/pipeline.cpp
  src/support/timer.cpp
  src/vk/threshold.cpp
  src/vk/segmentation.cpp
  src/vk/boundaries.cpp
  src/vk/cluster_grouping.cpp
  src/vk/fit_quad.cpp
  src/vk/apriltag.cpp
  src/image_impl.cpp
)
target_include_directories(apriltag_vk PUBLIC include)
target_compile_features(apriltag_vk PUBLIC cxx_std_20)
set_target_properties(apriltag_vk PROPERTIES CXX_EXTENSIONS OFF)
# For designated initializers.
target_compile_definitions(apriltag_vk PUBLIC VULKAN_HPP_NO_CONSTRUCTORS)
target_link_libraries(apriltag_vk PUBLIC Vulkan::Headers Vulkan::Vulkan ${CMAKE_DL_LIBS})

apriltag_add_shaders(apriltag_vk
  INCLUDE_DIRS include
  SHADERS
    shaders/threshold.slang
    shaders/segmentation/label.slang
    shaders/segmentation/final.slang
    shaders/boundaries/reset.slang
    shaders/boundaries/find_points.slang
    shaders/boundaries/accumulate.slang
    shaders/cluster_grouping/scan.slang
    shaders/cluster_grouping/scatter.slang
    shaders/fit_quad/sort.slang
    shaders/fit_quad/scan.slang
    shaders/fit_quad/maxima.slang
    shaders/fit_quad/search.slang
)

add_executable(avk_devices src/vk/devices.cpp)
target_link_libraries(avk_devices PRIVATE apriltag_vk)

add_executable(vulktan_apriltag src/vk/apriltag_main.cpp src/quad_decoding.cpp)
target_link_libraries(vulktan_apriltag PRIVATE apriltag_vk apriltag Threads::Threads)
