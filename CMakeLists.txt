cmake_minimum_required(VERSION 3.20)

project(FitnessCounter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# This matches the OpenCV 4.12.0 installation already configured on this PC.
set(OPENCV_ROOT "D:/common models/opencv460" CACHE PATH "OpenCV installation root")
set(OPENCV_INCLUDE_DIR "${OPENCV_ROOT}/include")
set(OPENCV_LIB_DIR "${OPENCV_ROOT}/lib")
set(OPENCV_BIN_DIR "${OPENCV_ROOT}/bin")

if(NOT EXISTS "${OPENCV_INCLUDE_DIR}/opencv2/opencv.hpp")
    message(FATAL_ERROR "OpenCV headers were not found: ${OPENCV_INCLUDE_DIR}")
endif()

add_executable(FitnessCounter
    A/demo/main.cpp
    A/scr/PoseDetector.cpp
    A/include/BodyPose.h
    A/include/PoseDetector.h
    B/include/PoseFeatures.h
    B/include/PoseFeatureExtractor.h
    B/src/PoseFeatureExtractor.cpp
    B/include/ActionCounter.h
    B/src/ActionCounter.cpp
    B/include/AutoExerciseClassifier.h
    B/src/AutoExerciseClassifier.cpp
)

set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT FitnessCounter)

target_include_directories(FitnessCounter PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/A/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/B/include"
    "${OPENCV_INCLUDE_DIR}"
)

target_compile_definitions(FitnessCounter PRIVATE
    $<$<CONFIG:Debug>:_DEBUG>
)

target_link_libraries(FitnessCounter PRIVATE
    debug "${OPENCV_LIB_DIR}/opencv_world4120d.lib"
    optimized "${OPENCV_LIB_DIR}/opencv_world4120.lib"
)

add_executable(PoseFeatureExtractorTest
    B/tests/PoseFeatureExtractorTest.cpp
    B/src/PoseFeatureExtractor.cpp
    A/include/BodyPose.h
    B/include/PoseFeatures.h
    B/include/PoseFeatureExtractor.h
)

add_executable(ActionCounterTest
    B/tests/ActionCounterTest.cpp
    B/src/ActionCounter.cpp
    B/include/ActionCounter.h
    B/include/PoseFeatures.h
    A/include/BodyPose.h
)

target_include_directories(ActionCounterTest PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/A/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/B/include"
    "${OPENCV_INCLUDE_DIR}"
)

target_link_libraries(ActionCounterTest PRIVATE
    debug "${OPENCV_LIB_DIR}/opencv_world4120d.lib"
    optimized "${OPENCV_LIB_DIR}/opencv_world4120.lib"
)

target_include_directories(PoseFeatureExtractorTest PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/A/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/B/include"
    "${OPENCV_INCLUDE_DIR}"
)

target_compile_definitions(PoseFeatureExtractorTest PRIVATE
    $<$<CONFIG:Debug>:_DEBUG>
)

target_link_libraries(PoseFeatureExtractorTest PRIVATE
    debug "${OPENCV_LIB_DIR}/opencv_world4120d.lib"
    optimized "${OPENCV_LIB_DIR}/opencv_world4120.lib"
)

# The updated A demo uses "A/models/...", relative to the project root.
set_target_properties(FitnessCounter PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

# Keep the required runtime DLLs next to the generated executable.
add_custom_command(TARGET FitnessCounter POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OPENCV_BIN_DIR}/opencv_world4120.dll"
        "$<TARGET_FILE_DIR:FitnessCounter>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OPENCV_BIN_DIR}/opencv_world4120d.dll"
        "$<TARGET_FILE_DIR:FitnessCounter>"
)
