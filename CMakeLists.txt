cmake_minimum_required(VERSION 3.10)

project(reactionWheelRNN_NEW)

set(CMAKE_PREFIX_PATH  "/home/arms/libtorch_101_1/libtorch/share/cmake/Torch")

set(CMAKE_CXX_STANDARD 14)

#include_directories(TORCH{PATH}/share/cmake/Torch)

find_package(Torch REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR} "/home/arms/build/dxl_sdk_linux_v1_01/include" "/usr/local/natinst/nidaqmxbase/include")
link_directories("/usr/local/natinst/nidaqmxbase/lib64/")

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -march=native -mtune=native -fpermissive -std=c++11 -g")
set(SMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}" )


add_executable(${PROJECT_NAME} main.cpp pid.cpp EKF.cpp EKF.h)
target_link_libraries(${PROJECT_NAME}  nidaqmxbase nidaqmxbaselv "${TORCH_LIBRARIES}" )

#set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)


if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET ${PROJECT_NAME}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif (MSVC)