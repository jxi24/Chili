add_library(Integrator AdaptiveMap.cc Vegas.cc)
target_link_libraries(Integrator PRIVATE project_options project_warnings
  PUBLIC fmt::fmt yaml-cpp spdlog::spdlog)
install(TARGETS Integrator
  EXPORT Integrator
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

