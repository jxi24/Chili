add_library(Channel ChannelNode.cc Channel.cc)
target_link_libraries(Channel PRIVATE project_options project_warnings
                              PUBLIC Integrator spdlog::spdlog)
