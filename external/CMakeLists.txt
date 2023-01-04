CPMFindPackage(
    NAME fmt
    GIT_TAG 7.1.3
    GITHUB_REPOSITORY fmtlib/fmt
)
add_library(fmt::fmt ALIAS fmt)

install(TARGETS fmt
  EXPORT fmt
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY ${FMT_SOURCE_DIR}/include/fmt
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

CPMFindPackage(
    NAME spdlog
    GIT_TAG "v1.8.5"
    GITHUB_REPOSITORY gabime/spdlog
    OPTIONS
        "SPDLOG_COMPILED_LIB ON"
        "SPDLOG_FMT_EXTERNAL ON"
        "SPDLOG_BUILD_SHARED ON"
)
add_library(spdlog::spdlog ALIAS spdlog)

install(TARGETS spdlog
  EXPORT spdlog
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY ${spdlog_SOURCE_DIR}/include/spdlog
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})


CPMAddPackage(
    NAME docopt
    GIT_TAG v0.6.3
    GITHUB_REPOSITORY docopt/docopt.cpp
)
add_library(docopt::docopt ALIAS docopt)
install(TARGETS docopt
  EXPORT docopt
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

CPMAddPackage(
    NAME yaml-cpp
    GITHUB_REPOSITORY jbeder/yaml-cpp
    GIT_TAG 4edff1fa5dbfca16fc72d89870841bee89f8ef89
    OPTIONS
        "YAML_CPP_BUILD_TESTS OFF"
        "YAML_CPP_BUILD_CONTRIB OFF"
        "YAML_CPP_BUILD_TOOLS OFF"
)

install(TARGETS yaml-cpp
  EXPORT yaml-cpp
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY ${YAML_CPP_SOURCE_DIR}/include/yaml-cpp
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Install testing framework
if(ENABLE_TESTING OR ENABLE_FUZZING)
    # Catch2
    CPMAddPackage(
        NAME catch2
        GITHUB_REPOSITORY catchorg/Catch2
        GIT_TAG "v2.13.7"
    )
    add_library(Catch2::Catch2 ALIAS Catch2)

    # Trompeloeil
    CPMAddPackage(
        NAME trompeloeil
        GITHUB_REPOSITORY rollbear/trompeloeil
        VERSION 37
    )
endif()
