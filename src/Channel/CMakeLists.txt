add_library(Channel ChannelNode.cc Channel.cc MultiChannel.cc)
target_link_libraries(Channel PRIVATE project_options project_warnings
  PUBLIC Tools Integrator spdlog::spdlog)
install(TARGETS Channel
  EXPORT Channel
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

