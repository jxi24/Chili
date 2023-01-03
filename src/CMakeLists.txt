add_subdirectory(Tools)
add_subdirectory(Integrator)
add_subdirectory(Channel)

add_executable(example example.cc)
target_link_libraries(example PRIVATE project_options project_warnings
                              PUBLIC Channel)

add_executable(example2 example2.cc)
target_link_libraries(example2 PRIVATE project_options project_warnings)

add_library(apes SHARED $<TARGET_OBJECTS:Channel>)
target_link_libraries(apes PUBLIC Channel)
target_link_libraries(apes PUBLIC Tools)
target_link_libraries(apes PUBLIC Integrator)

# install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
#   DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS apes
  EXPORT apes
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

set_target_properties(Tools Integrator Channel apes spdlog fmt yaml-cpp PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
