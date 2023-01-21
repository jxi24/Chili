add_library(Tools ThreeVector.cc FourVector.cc Poincare.cc ChannelElements.cc JetCluster.cc MPI.cc)
target_link_libraries(Tools PRIVATE project_options project_warnings
    PUBLIC spdlog::spdlog yaml-cpp)
if(TARGET MPI::MPI_CXX)
    target_link_libraries(Tools PUBLIC MPI::MPI_CXX)
    target_compile_definitions(Tools PUBLIC ENABLE_MPI)
endif()
install(TARGETS Tools
  EXPORT Tools
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

