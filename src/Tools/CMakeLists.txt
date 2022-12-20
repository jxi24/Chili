add_library(Tools ThreeVector.cc FourVector.cc Poincare.cc ChannelElements.cc JetCluster.cc)
target_link_libraries(Tools PRIVATE project_options project_warnings
                            PUBLIC spdlog::spdlog)
