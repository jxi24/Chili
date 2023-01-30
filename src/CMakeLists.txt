add_subdirectory(Tools)
add_subdirectory(Integrator)
add_subdirectory(Channel)
if(ENABLE_PYTHON OR ENABLE_TENSORFLOW)
    add_subdirectory(Interfaces)
endif()

add_executable(example example.cc)
target_link_libraries(example PRIVATE project_options project_warnings
                              PUBLIC Channel)

add_executable(example2 example2.cc)
target_link_libraries(example2 PRIVATE project_options project_warnings)

add_library(apes SHARED $<TARGET_OBJECTS:Channel>)
target_link_libraries(apes PUBLIC Channel)
target_link_libraries(apes PUBLIC Tools)
target_link_libraries(apes PUBLIC Integrator)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS apes
  EXPORT apes
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
#   DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
# #  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

set_target_properties(Tools Integrator Channel apes spdlog fmt yaml-cpp PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
if(MPI_FOUND)
  add_executable(example3 example_mpi.cc)
  target_link_libraries(example3 PRIVATE project_options project_warnings
    PUBLIC MPI::MPI_CXX Tools yaml-cpp Channel Integrator)
endif()

if(ENABLE_TENSORFLOW)
    add_library(apes_tf_test example_tf.cc)
    target_link_libraries(apes_tf_test PRIVATE project_options
                                       PUBLIC apes_tf)
endif()
