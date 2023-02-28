add_library(Integrator AdaptiveMap.cc Vegas.cc)
target_link_libraries(Integrator PRIVATE project_options project_warnings
                                 PUBLIC fmt::fmt yaml-cpp spdlog::spdlog)

add_library(IntegratorCuda AdaptiveMap.cu Vegas.cu)
set_target_properties(IntegratorCuda PROPERTIES CUDA_ARCHITECTURES native)
set_property(TARGET IntegratorCuda PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(IntegratorCuda PRIVATE project_options
                                     PUBLIC stdgpu::stdgpu cudainterface)
if(TARGET MPI::MPI_CXX)
    target_link_libraries(Integrator PUBLIC MPI::MPI_CXX)
    target_compile_definitions(Integrator PUBLIC ENABLE_MPI)
endif()
install(TARGETS Integrator
  EXPORT Integrator
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
