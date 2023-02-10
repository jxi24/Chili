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
target_link_libraries(apes PUBLIC Channel Tools Integrator)

install(TARGETS apes Tools Integrator Channel spdlog fmt yaml-cpp
        EXPORT ApesTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

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
