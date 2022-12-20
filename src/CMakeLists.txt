add_subdirectory(Tools)
add_subdirectory(Integrator)
add_subdirectory(Channel)

add_executable(example example.cc)
target_link_libraries(example PRIVATE project_options project_warnings
                              PUBLIC Channel)

find_package(PkgConfig REQUIRED)
pkg_check_modules(LHAPDF REQUIRED IMPORTED_TARGET lhapdf>=6.5)
link_directories(${LHAPDF_LIBDIR})

find_package(blockgen REQUIRED)

add_executable(combine combine.cc)
target_link_libraries(combine PRIVATE project_options project_warnings PUBLIC Channel)
target_link_libraries(combine PUBLIC blockgen)
target_link_libraries(combine PUBLIC ${LHAPDF_LIBRARIES})
target_include_directories(combine PUBLIC ${LHAPDF_INCLUDE_DIRS})

add_executable(test test_apes.cc)
target_link_libraries(test PRIVATE project_options project_warnings PUBLIC Channel)
target_link_libraries(test PUBLIC blockgen)

add_executable(example2 example2.cc)
target_link_libraries(example2 PRIVATE project_options project_warnings)

add_library(apes SHARED $<TARGET_OBJECTS:Channel>)
target_link_libraries(apes PUBLIC Channel)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
set_target_properties(Tools Integrator Channel apes spdlog fmt yaml-cpp PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
