add_library(Integrator AdaptiveMap.cc Vegas.cc)
target_link_libraries(Integrator PRIVATE project_options project_warnings
  PUBLIC fmt::fmt yaml-cpp spdlog::spdlog)
if(TARGET MPI::MPI_CXX)
    target_link_libraries(Integrator PUBLIC MPI::MPI_CXX)
    target_compile_definitions(Integrator PUBLIC ENABLE_MPI)
endif()
install(TARGETS Integrator
  EXPORT Integrator
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

