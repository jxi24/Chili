add_subdirectory(Cuda)
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

add_executable(example3 example.cu)
target_link_libraries(example3 PRIVATE project_options project_warnings
                      PUBLIC IntegratorCuda cudainterface)

add_library(chili SHARED $<TARGET_OBJECTS:Channel>)
target_link_libraries(chili PUBLIC Channel Tools Integrator stdgpu::stdgpu)

set(chili_targets chili Tools Integrator Channel example example2)

if(MPI_FOUND)
    add_executable(example3 example_mpi.cc)
    target_link_libraries(example3 PRIVATE project_options project_warnings
        PUBLIC MPI::MPI_CXX Tools yaml-cpp Channel Integrator)
    list(APPEND chili_targets example3)
endif()

if(ENABLE_TENSORFLOW)
    add_library(chili_tf_test example_tf.cc)
    target_link_libraries(chili_tf_test PRIVATE project_options
                                       PUBLIC chili_tf fmt spdlog yaml-cpp)
    list(APPEND chili_targets chili_tf_test)
endif()

set_target_properties(${chili_targets} PROPERTIES # fortran_interface_f
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

install(TARGETS ${chili_targets} spdlog fmt yaml-cpp
        EXPORT ChiliTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
