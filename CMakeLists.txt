cmake_minimum_required(VERSION 3.0)

# Policy to address @foo@ variable expansion
if(POLICY CMP0053)
    cmake_policy(SET CMP0053 NEW)
endif()

# Set the project name and basic settings
project(APES LANGUAGES CXX VERSION 1.0.0)
include(CMake/StandardProjectSettings.cmake)

# Link this 'library' to set the c++ standard / compile-time options requested
# Additionally, link to get include and external dependencies
add_library(project_options INTERFACE)
target_compile_features(project_options INTERFACE cxx_std_17)
target_include_directories(project_options INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

# Link this 'library' to use the warnings specified in CompilerWarnings.cmake
add_library(project_warnings INTERFACE)

# Standard compiler warnings
include(CMake/CompilerWarnings.cmake)
set_project_warnings(project_warnings)

# Sanitizer options if supported by compiler
include(CMake/Sanitizers.cmake)
enable_sanitizers(project_options)

# Allow for static analysis options
include(CMake/StaticAnalyzers.cmake)

option(BUILD_SHARED_LIBS "Enable compilation of shared libraries" ON)
option(ENABLE_TESTING "Enable Test Builds" OFF)

# Very basic PCH example
option(ENABLE_PCH "Enable Precompiled Headers" OFF)
if(ENABLE_PCH)
    # This sets a global PCH parameter, each project will build its own PCH,
    # which is a good idea if any #define's change
    # These should be headers included in many places
    target_precompile_headers(project_options INTERFACE <vector> <string> <map>)
endif()

# Add dependencies
include(CMake/CPM.cmake)
add_subdirectory(external)

# Testing
if(ENABLE_TESTING)
    if(COVERAGE) 
        target_compile_options(project_options INTERFACE --coverage) 
        target_link_libraries(project_options INTERFACE gcov)
    endif()
    target_compile_definitions(project_options INTERFACE TESTING)
    enable_testing()
    message(STATUS "Building Tests.")
    add_subdirectory(test)
endif()

# Main code
add_subdirectory(src)
