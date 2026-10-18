cmake_minimum_required(VERSION 3.13)
project(26VisionGroupRound2)

find_package(OpenCV REQUIRED)


add_executable(calib Src/Calib.cpp)

target_link_libraries(calib PRIVATE ${OpenCV_LIBS})

add_executable(armor Src/Armor.cpp)

target_link_libraries(armor PRIVATE ${OpenCV_LIBS})