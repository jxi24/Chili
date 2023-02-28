add_library(Channel ChannelNode.cc Channel.cc MultiChannel.cc)
target_link_libraries(Channel PRIVATE project_options project_warnings
  PUBLIC Tools Integrator spdlog::spdlog)
if(TARGET MPI::MPI_CXX)
    target_link_libraries(Channel PUBLIC MPI::MPI_CXX)
    target_compile_definitions(Channel PUBLIC ENABLE_MPI)
endif()
install(TARGETS Channel
  EXPORT Channel
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

