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
option(ENABLE_MPI "Enable MPI builds" OFF)

# Very basic PCH example
option(ENABLE_PCH "Enable Precompiled Headers" OFF)
if(ENABLE_PCH)
    # This sets a global PCH parameter, each project will build its own PCH,
    # which is a good idea if any #define's change
    # These should be headers included in many places
    target_precompile_headers(project_options INTERFACE <vector> <string> <map>)
endif()

if(ENABLE_MPI)
    find_package(MPI REQUIRED)

    if(NOT TARGET MPI::MPI_CXX)
        add_library(MPI::MPI_CXX IMPORTED INTERFACE)

        set_property(TARGET MPI::MPI_CXX
                     PROPERTY INTERFACE_COMPILE_OPTIONS ${MPI_CXX_COMPILE_FLAGS})
        set_property(TARGET MPI::MPI_CXX
                     PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${MPI_CXX_INCLUDE_PATH}")
        set_property(TARGET MPI::MPI_CXX
                     PROPERTY INTERFACE_LINK_LIBRARIES ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES})
    endif() 
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

# Install CMake find_package files
include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_SOURCE_DIR}/CMake/apes-config.cmake.in
    ${CMAKE_BINARY_DIR}/CMake/apes-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/apes
    PATH_VARS CMAKE_INSTALL_INCLUDEDIR
              CMAKE_INSTALL_LIBDIR
)

configure_package_config_file(
  ${CMAKE_SOURCE_DIR}/CMake/apes-config.in
  ${CMAKE_CURRENT_BINARY_DIR}/bin/apes-config
  INSTALL_DESTINATION ${CMAKE_INSTALL_BINDIR}
  PATH_VARS CMAKE_INSTALL_INCLUDEDIR
            CMAKE_INSTALL_LIBDIR)

write_basic_package_version_file(
    ${CMAKE_BINARY_DIR}/CMake/apes-config-version.cmake
    VERSION ${ACHILLES_VERSION}
    COMPATIBILITY AnyNewerVersion
)

install(
    FILES
        ${CMAKE_BINARY_DIR}/CMake/apes-config.cmake
        ${CMAKE_BINARY_DIR}/CMake/apes-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/apes
)

install(
    FILES
        ${CMAKE_BINARY_DIR}/bin/apes-config
    DESTINATION ${CMAKE_INSTALL_BINDIR}
)
