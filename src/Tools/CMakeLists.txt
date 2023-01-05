add_library(Tools ThreeVector.cc FourVector.cc Poincare.cc ChannelElements.cc JetCluster.cc)
target_link_libraries(Tools PRIVATE project_options project_warnings
  PUBLIC spdlog::spdlog)
install(TARGETS Tools
  EXPORT Tools
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

